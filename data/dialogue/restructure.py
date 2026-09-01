# restructure_dialogue.py
# Splits a flat "Dialogue" string field (turns separated by blank lines, alternating
# patient / medical assistant, each turn starting with its speaker label) into a nested
# object keyed by turn index: "0", "1", "2", ... Turn 0/2/4/... is always the patient,
# 1/3/5/... is always the medical assistant. 
#The speaker label is kept inside each turn's text, the index is just ordering, not a replacement for the label.
#

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


PATIENT_LABELS   = {"مريض", "patient"}
ASSISTANT_LABELS = {"مساعد طبي", "medical assistant"}
PATIENT_NORMALIZE = {
    "مريضة":   "مريض",
    "المريض":  "مريض",
    "المريضة": "مريض",
}
_SPEAKER_SPLIT_RE = re.compile(
    r'\n(?=(?:مريض|مريضة|المريض|المريضة|مساعد طبي|طبيب|دكتور)[:：])'
)


def normalize_turn(turn: str) -> str:
    """Replace non-standard patient label with canonical مريض: ."""
    for variant, canonical in PATIENT_NORMALIZE.items():
        if turn.startswith(variant + ":"):
            return canonical + ":" + turn[len(variant) + 1:]
        if turn.startswith(variant + "："):
            return canonical + "：" + turn[len(variant) + 1:]
    return turn


def split_dialogue(dialogue: str) -> Dict[str, str]:
    blocks = [b.strip() for b in dialogue.split("\n\n") if b.strip()]
    turns = []
    for block in blocks:
        sub = _SPEAKER_SPLIT_RE.split(block)
        turns.extend(normalize_turn(s.strip()) for s in sub if s.strip())
    return {str(i): t for i, t in enumerate(turns)}


def label_of(turn: str) -> str:
    m = re.match(r"^([^:：]{1,25})[:：]", turn)
    return m.group(1).strip().lower() if m else ""


def check_alternation(turns: Dict[str, str]) -> List[str]:
    warnings = []
    ordered = [turns[str(i)] for i in range(len(turns))]
    labels = [label_of(t) for t in ordered]
    for i, lab in enumerate(labels):
        expected = PATIENT_LABELS if i % 2 == 0 else ASSISTANT_LABELS
        if lab not in expected:
            warnings.append(f"turn {i}: expected {'patient' if i % 2 == 0 else 'medical assistant'}, "
                             f"got label '{lab or '(none found)'}'")
    if len(ordered) % 2 == 0:
        warnings.append(f"even number of turns ({len(ordered)}) — dialogue doesn't end on the patient")
    return warnings


def main():
    p = argparse.ArgumentParser(
        description="Restructure a flat Dialogue string into indexed turn objects "
                    "({\"0\": ..., \"1\": ..., ...}). Writes a new JSON file; input untouched."
    )
    p.add_argument("--input", required=True, help="path to input json array")
    p.add_argument("--output", required=True, help="path to new output json array")
    p.add_argument("--field", default="Dialogue", help="which field to restructure (default: Dialogue)")
    p.add_argument("--no-pretty", action="store_true", help="don't pretty-print output json")
    p.add_argument("--strict", action="store_true",
                    help="exit with an error if any record fails the patient/assistant "
                        "alternation check, instead of just warning")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    records: List[Dict[str, Any]] = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        print("input must be a json array", file=sys.stderr)
        sys.exit(1)

    total_warnings = 0
    for rec in records:
        val = rec.get(args.field)
        if not isinstance(val, str):
            continue  
        turns = split_dialogue(val)
        warnings = check_alternation(turns)
        if warnings:
            total_warnings += 1
            case_id = rec.get("Case ID", "?")
            print(f"warning: Case ID {case_id}:", file=sys.stderr)
            for w in warnings:
                print(f"  - {w}", file=sys.stderr)
        rec[args.field] = turns

    if total_warnings and args.strict:
        print(f"\n{total_warnings} record(s) failed the alternation check; aborting (--strict)", file=sys.stderr)
        sys.exit(1)

    out_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=None if args.no_pretty else 2),
        encoding="utf-8"
    )
    print(f"done. wrote {len(records)} records → {out_path}"
          + (f" ({total_warnings} record(s) had alternation warnings — see above)" if total_warnings else ""))


if __name__ == "__main__":
    main()