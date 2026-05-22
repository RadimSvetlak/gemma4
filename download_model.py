# Stazeni modelu Gemma 4 E2B-it do aktualni slozky
# Bez parametru: spustis jen `python 35_download_gemma4_to_current_folder.py`
# Vysledek: ./gemma4_2B_it

from pathlib import Path
import os
import sys

MODEL_ID = "google/gemma-4-E2B-it"
TARGET_DIR = Path.cwd() / "gemma4_2B_it"


def main():
    print("Gemma 4 model downloader")
    print(f"Model: {MODEL_ID}")
    print(f"Cilova slozka: {TARGET_DIR.resolve()}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("\nCHYBI KNIHOVNA: huggingface_hub")
        print("Nainstaluj ji prikazem:")
        print("  pip install -U huggingface_hub")
        sys.exit(1)

    # Pro gated/licencovane modely je casto nutne byt prihlaseny:
    #   huggingface-cli login
    # nebo mit token v promenne prostredi HF_TOKEN / HUGGINGFACE_HUB_TOKEN.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStahuji model. Pri prvnim spusteni to muze trvat dele...")
    print("Pokud stazeni selze kvuli opravneni/licenci, spust:")
    print("  huggingface-cli login")
    print("a zkontroluj, ze mas na Hugging Face odsouhlasenou licenci modelu.\n")

    try:
        snapshot_path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(TARGET_DIR),
            token=token,
            resume_download=True,
        )
    except TypeError:
        # Fallback pro starsi/novejsi verze huggingface_hub, kde se podpis funkce muze lisit.
        snapshot_path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(TARGET_DIR),
            token=token,
        )
    except Exception as exc:
        print("\nStazeni selhalo.")
        print(f"Chyba: {type(exc).__name__}: {exc}")
        print("\nNejcastejsi priciny:")
        print("1) Nejsi prihlaseny na Hugging Face: huggingface-cli login")
        print("2) Nemas odsouhlasenou licenci / pristup k modelu")
        print("3) Chybi internet nebo je blokovane pripojeni")
        print("4) Chybi misto na disku")
        sys.exit(2)

    print("\nHotovo.")
    print(f"Model ulozen v: {Path(snapshot_path).resolve()}")

    # Rychla kontrola typickych souboru. Nevyzaduje presny seznam, jen pomuze odhalit prazdnou slozku.
    files = list(TARGET_DIR.glob("**/*"))
    real_files = [p for p in files if p.is_file()]
    print(f"Pocet souboru ve slozce: {len(real_files)}")

    important = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "processor_config.json",
    ]
    found = []
    for name in important:
        if any(p.name == name for p in real_files):
            found.append(name)

    if found:
        print("Nalezene konfiguracni soubory:")
        for name in found:
            print(f"- {name}")
    else:
        print("VAROVANI: Nenasel jsem typicke konfiguracni soubory. Zkontroluj obsah slozky.")

    print("\nTed by mel fungovat transkripcni skript s:")
    print('  MODEL_DIR = r".\\gemma4_2B_it"')


if __name__ == "__main__":
    main()
