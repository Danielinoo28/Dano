import os
import json
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

# ------------------------------
# CESTY A KONFIGURÁCIA PROJEKTU
# ------------------------------

# Zistenie hlavného adresára projektu (koreň)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cesta ku konfiguračnému súboru (config.json)
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.json")

# Cesta, kde sa uloží natrénovaný model
MODELS_DIR = os.path.join(BASE_DIR, "models", "t5_summarizer")


# ------------------------------
# FUNKCIA NA NAČÍTANIE KONFIGU
# ------------------------------

def load_config():
    """Načíta konfiguračný JSON súbor a vráti ho ako dict."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------
# HLAVNÁ FUNKCIA PROGRAMU
# ------------------------------

def main():
    # Načítanie konfiguračných parametrov (model, dĺžky, batch-size...)
    cfg = load_config()

    # Voľba datasetu – small/big podľa config.json
    dataset_size = cfg.get("dataset_size", "small")
    base_model_name = cfg["base_model_name"]
    max_input_length = cfg["max_input_length"]
    max_target_length = cfg["max_target_length"]

    # Cesty k trénovacím a validačným CSV súborom
    train_path = os.path.join(BASE_DIR, "data", "processed", f"{dataset_size}_train.csv")
    val_path = os.path.join(BASE_DIR, "data", "processed", f"{dataset_size}_val.csv")

    print(f"Using dataset size: {dataset_size}")
    print(f"Train path: {train_path}")
    print(f"Val path: {val_path}")

    # ---------------------------------------------------------
    # 1) NAČÍTANIE CSV ÚDAJOV DO DATAFRAMU A PREVOD NA DATASET
    # ---------------------------------------------------------

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Prevod na HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # ---------------------------------------------------------
    # 2) NAČÍTANIE TOKENIZÉRA A MODELU (T5-small)
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    # ---------------------------------------------------------
    # 3) FUNKCIA NA TOKENIZÁCIU VSTUPNÝCH DÁT
    # ---------------------------------------------------------

    def preprocess_function(examples):
        """
        Pripraví vstupný text a cieľovú sumarizáciu.
        - pridá prefix 'summarize:' (T5 ho používa na určenie úlohy),
        - oreže text podľa max_input_length,
        - vytvorí labely z column 'summary'.
        """
        # Príprava vstupu pre model
        inputs = ["summarize: " + t for t in examples["text"]]
        
        # Tokenizácia vstupného textu
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )

        # Tokenizácia cieľových sumarizácií (labelov)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["summary"],
                max_length=max_target_length,
                truncation=True,
                padding="max_length",
            )

        # Priradenie labelov k tokenom
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # ---------------------------------------------------------
    # 4) TOKENIZÁCIA CELÉHO DATASETU
    # ---------------------------------------------------------

    train_tokenized = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,  # odstráni pôvodné textové stĺpce
    )

    val_tokenized = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    # Pomocný objekt, ktorý modelu dodá správne formátované batch-e
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # ---------------------------------------------------------
    # 5) NASTAVENIE PARAMETROV TRÉNINGU
    # ---------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=MODELS_DIR,
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        learning_rate=cfg["learning_rate"],
    )

    # ---------------------------------------------------------
    # 6) VYTVORENIE Trainer OBJEKTU A SPUSTENIE TRÉNINGU
    # ---------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Spustenie trénovania modelu
    trainer.train()

    # ---------------------------------------------------------
    # 7) ULOŽENIE FINÁLNEHO MODELU A TOKENIZÉRA
    # ---------------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    trainer.save_model(MODELS_DIR)
    tokenizer.save_pretrained(MODELS_DIR)

    print(f"Model saved to {MODELS_DIR}")


# Spustenie hlavnej funkcie
if __name__ == "__main__":
    main()
