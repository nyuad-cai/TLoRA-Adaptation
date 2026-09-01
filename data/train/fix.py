#!/usr/bin/env python3
"""
fix_task2_answers.py

Patches aug/task2_dataset.json by:
  1. Looking up each item's original MCQ in train-old.json by ID
  2. Converting the answer letter (A/B/C/D) → answer text using the options array
  3. Populating original_options from the source data (if currently empty)

Writes the patched dataset to aug/task2_dataset_fixed.json
"""

import json
from pathlib import Path

SOURCE_PATH  = "train.json"
DATASET_PATH = "aug/task2_dataset.json"
OUTPUT_PATH  = "aug/task2_dataset_fixed.json"

# Options are stored as opa/opb/opc/opd/ope/opf (not a list)
OPTION_KEYS = ["opa", "opb", "opc", "opd", "ope", "opf"]

def letter_to_option_text(letter: str, source_item: dict):
    """'B' → source_item['opb'], skipping empty strings. Returns None if not found."""
    idx = ord(letter.strip().upper()) - ord('A')
    if 0 <= idx < len(OPTION_KEYS):
        key = OPTION_KEYS[idx]
        val = source_item.get(key, "").strip()
        return val if val else None
    return None

def get_options_list(source_item: dict) -> list:
    """Extract all non-empty options as a list for original_options field."""
    return [source_item[k] for k in OPTION_KEYS
            if source_item.get(k, "").strip()]


def main():
    # Load original MCQ data, indexed by id (support both int and str keys)
    with open(SOURCE_PATH, encoding="utf-8") as f:
        source_data = json.load(f)

    source_index = {}
    for item in source_data:
        source_index[str(item["id"])] = item   # normalise to str

    print(f"Loaded {len(source_index)} source items from {SOURCE_PATH}")

    # Load rewritten dataset
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} rewritten items from {DATASET_PATH}")

    patched = 0
    skipped_no_source = 0
    skipped_bad_letter = 0
    skipped_no_options = 0

    output = []
    for item in dataset:
        item_id = str(item["id"])
        source  = source_index.get(item_id)

        if source is None:
            print(f"  [WARN] id={item_id} not found in source — keeping as-is")
            output.append(item)
            skipped_no_source += 1
            continue

        answer_letter = str(item.get("answer", "")).strip()

        # Resolve letter → text using opa/opb/... keys
        answer_text = answer_letter  # fallback: keep letter
        if answer_letter and answer_letter.upper() in "ABCDEF":
            resolved = letter_to_option_text(answer_letter, source)
            if resolved:
                answer_text = resolved
                patched += 1
            else:
                print(f"  [WARN] id={item_id}: letter '{answer_letter}' maps to "
                      f"empty/missing option key")
                skipped_no_options += 1
        else:
            print(f"  [WARN] id={item_id}: unexpected answer value '{answer_letter}'")
            skipped_bad_letter += 1

        record = dict(item)
        record["answer"] = answer_text
        # Backfill original_options if empty
        if not record.get("original_options"):
            record["original_options"] = get_options_list(source)

        output.append(record)

    # Save
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDone.")
    print(f"  Patched (letter → text) : {patched}")
    print(f"  Skipped (no source)     : {skipped_no_source}")
    print(f"  Skipped (bad letter)    : {skipped_bad_letter}")
    print(f"  Skipped (index OOB)     : {skipped_no_options}")
    print(f"  Output → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()