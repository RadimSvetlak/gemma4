# Minimal Gemma4 audio -> text transcription for all .flac files in current folder
# Verze s metrikami: casy zpracovani, RTF, tokeny/s a CUDA pamet.

from pathlib import Path
from math import gcd
import csv
import time

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig

MODEL_DIR = r".\gemma4_2B_it"
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30
MAX_NEW_TOKENS = 128
USE_4BIT_ON_CUDA = True
LOCAL_FILES_ONLY = True
SKIP_QUANT_MODULES = ["model.audio_tower", "model.embed_audio", "model.embed_vision", "lm_head"]

METRICS_CSV = "transcription_metrics.csv"


def sync_cuda():
    """Zpresni mereni casu na CUDA: GPU operace jsou jinak asynchronni."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def now():
    return time.perf_counter()


def fmt_seconds(value: float) -> str:
    return f"{value:.3f} s"


def load_audio(path: str, target_sr: int = TARGET_SAMPLE_RATE, max_seconds: int = MAX_AUDIO_SECONDS):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio soubor neexistuje: {path.resolve()}")

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)

    if sr != target_sr:
        divisor = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // divisor, sr // divisor).astype(np.float32)
        sr = target_sr

    audio = np.clip(audio, -1.0, 1.0)

    if max_seconds:
        audio = audio[: int(max_seconds * sr)]

    return audio, sr


def load_model_and_processor(model_dir: str):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Složka modelu neexistuje: {model_dir.resolve()}")

    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=LOCAL_FILES_ONLY)

    if torch.cuda.is_available() and USE_4BIT_ON_CUDA:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=SKIP_QUANT_MODULES,
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            str(model_dir),
            local_files_only=LOCAL_FILES_ONLY,
            quantization_config=quantization_config,
            dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
    else:
        device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}
        model = AutoModelForMultimodalLM.from_pretrained(
            str(model_dir),
            local_files_only=LOCAL_FILES_ONLY,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )

    model.eval()
    return processor, model


def move_inputs_to_model(inputs, model):
    device = next(p.device for p in model.parameters() if p.device.type != "meta")
    dtype = getattr(model, "dtype", torch.float16 if device.type == "cuda" else torch.float32)

    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def make_inputs(processor, audio, sampling_rate):
    prompt = (
        "Transcribe the following speech segment in its original language. "
        "Follow these specific instructions for formatting the answer:\n"
        "* Only output the transcription, with no newlines.\n"
        "* When transcribing numbers, write digits, for example write 3 instead of three."
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "audio", "audio": audio}]}]

    try:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception:
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return processor(text=text, audio=[audio], sampling_rate=sampling_rate, return_tensors="pt")


def cuda_memory_metrics():
    if not torch.cuda.is_available():
        return {
            "cuda_max_allocated_mb": 0.0,
            "cuda_max_reserved_mb": 0.0,
        }

    return {
        "cuda_max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
        "cuda_max_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024 / 1024, 2),
    }


def transcribe_one(processor, model, audio_path: Path):
    total_start = now()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 1) Nacteni + priprava audia
    t0 = now()
    audio, sr = load_audio(str(audio_path))
    audio_load_seconds = now() - t0
    audio_seconds = len(audio) / sr if sr else 0.0
    print(f"Audio připraveno: {audio_seconds:.2f} s, {sr} Hz, mono float32")

    # 2) Processor / chat template / tokenizace + audio features
    t0 = now()
    inputs = make_inputs(processor, audio, sr)
    input_prepare_seconds = now() - t0
    input_len = inputs["input_ids"].shape[-1]

    # 3) Presun vstupu na model/GPU
    t0 = now()
    inputs = move_inputs_to_model(inputs, model)
    sync_cuda()
    input_move_seconds = now() - t0

    # 4) Samotna generace
    sync_cuda()
    generation_start = now()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    sync_cuda()
    generation_seconds = now() - generation_start

    # 5) Dekodovani vystupu
    t0 = now()
    new_tokens = outputs[:, input_len:]
    generated_tokens = int(new_tokens.shape[-1])
    text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()
    decode_seconds = now() - t0

    total_seconds = now() - total_start
    realtime_factor = total_seconds / audio_seconds if audio_seconds > 0 else 0.0
    generation_realtime_factor = generation_seconds / audio_seconds if audio_seconds > 0 else 0.0
    tokens_per_second = generated_tokens / generation_seconds if generation_seconds > 0 else 0.0

    metrics = {
        "file": audio_path.name,
        "audio_seconds": round(audio_seconds, 3),
        "sample_rate": sr,
        "input_tokens": int(input_len),
        "generated_tokens": generated_tokens,
        "audio_load_seconds": round(audio_load_seconds, 3),
        "input_prepare_seconds": round(input_prepare_seconds, 3),
        "input_move_seconds": round(input_move_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "decode_seconds": round(decode_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "realtime_factor": round(realtime_factor, 3),
        "generation_realtime_factor": round(generation_realtime_factor, 3),
        "tokens_per_second": round(tokens_per_second, 2),
    }
    metrics.update(cuda_memory_metrics())

    return text, metrics


def print_file_metrics(metrics: dict):
    print("\nMETRIKY:")
    print(f"- délka audia: {metrics['audio_seconds']:.3f} s")
    print(f"- celkem: {fmt_seconds(metrics['total_seconds'])}")
    print(f"- generování: {fmt_seconds(metrics['generation_seconds'])}")
    print(f"- příprava vstupu: {fmt_seconds(metrics['input_prepare_seconds'])}")
    print(f"- načtení audia: {fmt_seconds(metrics['audio_load_seconds'])}")
    print(f"- RTF celkem: {metrics['realtime_factor']:.3f}x  (1.0 = stejně dlouho jako audio)")
    print(f"- RTF generování: {metrics['generation_realtime_factor']:.3f}x")
    print(f"- tokeny: input {metrics['input_tokens']}, generated {metrics['generated_tokens']}")
    print(f"- rychlost generování: {metrics['tokens_per_second']:.2f} tok/s")
    if torch.cuda.is_available():
        print(f"- CUDA peak allocated: {metrics['cuda_max_allocated_mb']:.2f} MB")
        print(f"- CUDA peak reserved: {metrics['cuda_max_reserved_mb']:.2f} MB")


def write_metrics_csv(metrics_rows, path: Path):
    if not metrics_rows:
        return

    fieldnames = list(metrics_rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)


def print_total_metrics(metrics_rows, total_wall_seconds):
    if not metrics_rows:
        return

    audio_total = sum(row["audio_seconds"] for row in metrics_rows)
    generation_total = sum(row["generation_seconds"] for row in metrics_rows)
    generated_tokens_total = sum(row["generated_tokens"] for row in metrics_rows)
    avg_rtf = total_wall_seconds / audio_total if audio_total > 0 else 0.0
    avg_generation_rtf = generation_total / audio_total if audio_total > 0 else 0.0
    avg_tok_s = generated_tokens_total / generation_total if generation_total > 0 else 0.0

    print("\n" + "=" * 60)
    print("SOUHRNNÉ METRIKY:")
    print(f"- počet souborů: {len(metrics_rows)}")
    print(f"- celková délka audia: {audio_total:.3f} s")
    print(f"- celkový čas běhu: {total_wall_seconds:.3f} s")
    print(f"- celkový čas generování: {generation_total:.3f} s")
    print(f"- RTF celkem: {avg_rtf:.3f}x")
    print(f"- RTF generování: {avg_generation_rtf:.3f}x")
    print(f"- vygenerované tokeny celkem: {generated_tokens_total}")
    print(f"- průměrná rychlost generování: {avg_tok_s:.2f} tok/s")


def transcribe():
    script_start = now()

    print("Gemma4 minimal FLAC -> text")
    print(f"Model: {Path(MODEL_DIR).resolve()}")

    flac_files = sorted(Path(".").glob("*.flac"))
    if not flac_files:
        raise FileNotFoundError(f"V aktuální složce nejsou žádné .flac soubory: {Path('.').resolve()}")

    print(f"Nalezeno FLAC souborů: {len(flac_files)}")

    load_start = now()
    processor, model = load_model_and_processor(MODEL_DIR)
    sync_cuda()
    model_load_seconds = now() - load_start
    print(f"Model načten. CUDA: {torch.cuda.is_available()}. Čas načtení modelu: {model_load_seconds:.3f} s")

    results = {}
    metrics_rows = []

    for index, audio_path in enumerate(flac_files, start=1):
        print("\n" + "=" * 60)
        print(f"[{index}/{len(flac_files)}] Audio: {audio_path.resolve()}")

        text, metrics = transcribe_one(processor, model, audio_path)
        results[audio_path.name] = text
        metrics_rows.append(metrics)

        print("\nPŘEPIS:")
        print(text)
        print_file_metrics(metrics)

        txt_path = audio_path.with_suffix(".txt")
        txt_path.write_text(text + "\n", encoding="utf-8")
        print(f"Uloženo: {txt_path.resolve()}")

    summary_path = Path("transcriptions.txt")
    with summary_path.open("w", encoding="utf-8") as f:
        for filename, transcription in results.items():
            f.write(f"## {filename}\n")
            f.write(transcription.strip() + "\n\n")

    metrics_path = Path(METRICS_CSV)
    write_metrics_csv(metrics_rows, metrics_path)

    total_wall_seconds = now() - script_start
    print_total_metrics(metrics_rows, total_wall_seconds)

    print("\n" + "=" * 60)
    print(f"Hotovo. Souhrn uložen: {summary_path.resolve()}")
    print(f"Metriky uloženy: {metrics_path.resolve()}")
    return results


if __name__ == "__main__":
    transcribe()
