import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------------------------
# DEFINOVANIE CIEST K MODELU A PROJEKTU
# ------------------------------------------

# Koreňový adresár projektu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Adresár, kde je uložený fintunovaný model
MODEL_DIR = os.path.join(BASE_DIR, "models", "t5_summarizer")

# ------------------------------------------
# NAČÍTANIE FINETUNOVANÉHO MODELU (AK EXISTUJE)
# ------------------------------------------

use_finetuned = False   # indikátor, či sa podarí použit vlastný model

# Pokus o načítanie vlastného natrénovaného modelu
if os.path.isdir(MODEL_DIR):
    print(f"Loading fine-tuned model from: {MODEL_DIR}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
        use_finetuned = True
    except Exception as e:
        # Ak loading zlyhá, prejde sa automaticky na základný model
        print(f"Failed to load fine-tuned model: {e}")
        use_finetuned = False

# ------------------------------------------
# FALLBACK – NAČÍTANIE ZÁKLADNÉHO T5 MODEL
# ------------------------------------------

if not use_finetuned:
    print("WARNING: Using base 't5-small' model instead.")
    base_model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)


# ------------------------------------------
# FUNKCIA NA GENEROVANIE SUMARIZÁCIE TEXTU
# ------------------------------------------

def summarize(text: str, max_length: int = 80) -> str:
    """
    Generuje sumarizáciu vstupného textu pomocou T5 modelu.
    - pridá prefix 'summarize:' podľa štandardu pre T5,
    - tokenizuje vstup,
    - vygeneruje sumarizáciu s beam search,
    - ak model zlyhá, použije fallback na základný t5-small.
    """

    # Kontrola prázdneho vstupu
    if not text.strip():
        return "Empty input text."

    # Príprava vstupu pre model
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    # Generovanie sumarizácie – beam search + obmedzenie opakovaných fráz
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )

    # Dekódovanie tokenov späť na čitateľný text
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # ------------------------------------------
    # FALLBACK – AK VÝSTUP JE PRÁZDNY
    # ------------------------------------------
    if not summary:
        print("⚠ Model nevrátil žiadny text, skúšam fallback na základný 't5-small'...")

        fb_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        fb_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

        fb_inputs = fb_tokenizer(
            "summarize: " + text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        fb_output_ids = fb_model.generate(
            input_ids=fb_inputs["input_ids"],
            attention_mask=fb_inputs.get("attention_mask", None),
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

        summary = fb_tokenizer.decode(fb_output_ids[0], skip_special_tokens=True).strip()

        # Ak by aj fallback zlyhal
        if not summary:
            summary = "[Model nedokázal vygenerovať zhrnutie.]"

    return summary


# ------------------------------------------
# DEMO REŽIM – INTERAKTÍVNY VSTUP CEZ TERMINÁL
# ------------------------------------------

if __name__ == "__main__":
    print("=== TEXT SUMMARIZATION DEMO ===")
    print("Zadaj text na zhrnutie (jedna veta alebo viac riadkov).")
    print("Keď skončíš, stlač Enter.\n")

    # Používateľ zadá text
    text = input("Text: ")

    # Výpočet sumarizácie
    result = summarize(text)

    # Výpis výsledku
    print("\n--- SUMMARY ---")
    print(result)
