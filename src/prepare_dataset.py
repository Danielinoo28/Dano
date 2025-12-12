import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------------------
# DEFINOVANIE CIEST K PROJEKTU A KONFIGURÁCII
# ------------------------------------------

# Hlavný adresár projektu (o úroveň vyššie ako src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cesta ku konfiguračnému súboru
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.json")

# Cieľový priečinok pre uložené, očistené datasety
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Parametre delenia datasetu
TEST_SIZE = 0.1
RANDOM_STATE = 42


# ------------------------------------------
# FUNKCIA NA NAČÍTANIE CONFIG.JSON
# ------------------------------------------
def load_config():
    """Načíta nastavenia zo súboru config.json."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------
# PARSOVANIE MALÉHO CSV FORMÁTU
# ------------------------------------------
def parse_small_dataset(raw_path: str) -> pd.DataFrame:
    """
    Parsuje small_dataset.csv v špeciálnom formáte:
    'TEXT',"SUMMARY"
    kde text aj summary môžu obsahovať úvodzovky.

    Prevedie dáta do DataFrame so stĺpcami:
    - text
    - summary
    """
    records = []

    # Pomocná funkcia na čistenie úvodzoviek
    def clean_quotes(s: str) -> str:
        s = s.strip()
        while s.startswith('"'):
            s = s[1:]
        while s.endswith('"'):
            s = s[:-1]
        return s.strip()

    # Čítanie riadkov a ich rozdeľovanie na text a summary
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Oddelenie časti pred a za prvým výskytom '","'
            sep = '","'
            idx = line.find(sep)
            if idx == -1:
                print(f"Warning: cannot split line, skipping:\n{line}")
                continue

            text_raw = line[:idx]
            summary_raw = line[idx + len(sep):]

            # Odstránenie nadbytočných úvodzoviek
            text = clean_quotes(text_raw)
            summary = clean_quotes(summary_raw)

            # Kontrola, že text aj summary obsahujú dáta
            if text and summary:
                records.append({"text": text, "summary": summary})
            else:
                print(f"Warning: empty text/summary after cleaning, skipping:\n{line}")

    df = pd.DataFrame(records)
    print(f"Parsed small dataset: {len(df)} rows")
    return df


# ------------------------------------------
# PARSOVANIE VEĽKÉHO CSV S HLAVIČKOU
# ------------------------------------------
def parse_big_dataset(raw_path: str) -> pd.DataFrame:
    """
    Parsuje štandardný CSV vo formáte:
    text, summary
    (s hlavičkou)

    Zabezpečuje:
    - kontrolu povinných stĺpcov,
    - odstránenie riadkov s chýbajúcimi hodnotami,
    - premenovanie stĺpcov.
    """
    df = pd.read_csv(raw_path)
    print("Columns in big CSV:", list(df.columns))
    print("Total rows in big CSV:", len(df))

    TEXT_COL = "text"
    SUMMARY_COL = "summary"

    # Overenie, že dataset obsahuje správne stĺpce
    if TEXT_COL not in df.columns or SUMMARY_COL not in df.columns:
        raise ValueError(
            f"Expected columns '{TEXT_COL}' and '{SUMMARY_COL}', but got: {list(df.columns)}"
        )

    # Odstránenie riadkov s prázdnymi textami
    before_drop = len(df)
    df = df.dropna(subset=[TEXT_COL, SUMMARY_COL])
    after_drop = len(df)

    print(f"Rows before dropna: {before_drop}, after dropna: {after_drop}")

    # Prevedenie stĺpcov na jednotné názvy
    df = df.rename(columns={TEXT_COL: "text", SUMMARY_COL: "summary"})
    df = df.reset_index(drop=True)
    return df


# ------------------------------------------
# HLAVNÁ FUNKCIA PROGRAMU
# ------------------------------------------
def main():
    # Načítanie config.json
    cfg = load_config()

    # Výber datasetu podľa nastavenia
    dataset_size = cfg.get("dataset_size", "small")  # small / big

    # Cesty k vstupným CSV súborom
    raw_path = os.path.join(BASE_DIR, "data", "raw", f"{dataset_size}_dataset.csv")
    train_path = os.path.join(PROCESSED_DIR, f"{dataset_size}_train.csv")
    val_path = os.path.join(PROCESSED_DIR, f"{dataset_size}_val.csv")

    # Vytvorenie priečinka data/processed ak neexistuje
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"Loading dataset from: {raw_path}")

    # Načítanie datasetu podľa typu
    if dataset_size == "small":
        df = parse_small_dataset(raw_path)
    else:
        df = parse_big_dataset(raw_path)

    # Kontrola, že dataset obsahuje dáta
    if len(df) == 0:
        raise ValueError("Dataset has 0 valid rows after parsing.")

    # Rozdelenie datasetu na TRAIN a VALIDATION
    train_df, val_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Uloženie datasetov do CSV
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")

    print(f"Dataset size: {dataset_size}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"Saved to:\n  {train_path}\n  {val_path}")


# Spustenie programu
if __name__ == "__main__":
    main()
