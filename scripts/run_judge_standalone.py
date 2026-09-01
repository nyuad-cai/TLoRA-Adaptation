"""
This script runs LLM-judge evaluation on Task 2 (answer_generation) and Task 3 (dialogue_completion) tasks' outputs
"""

import os
import re
import sys
import json
import time
import logging
import argparse
from pathlib import Path

import pandas as pd
from openai import OpenAI

MODEL        = "gpt-5.2"
TEMPERATURE  = 0
DELAY_SECS   = 0.5
MAX_RETRIES  = 3
RETRY_DELAY  = 5.0
ROUND_DP     = 3

MAX_TOKENS_BY_TASK = {
    "answer_generation":   50,
    "dialogue_completion": 50,
}
SUPPORTED_TASKS = set(MAX_TOKENS_BY_TASK.keys())

TASK2_AXES = [
    ("judge_label", "Correctness", {"Correct", "Incorrect"}),
]
TASK3_AXES = [
    ("judge_label", "Correctness", {"Correct", "Incorrect"}),
]

TASK3_LABEL_TO_INT = {
    "judge_label": {"Correct": 1, "Incorrect": 0},
}
TASK2_LABEL_TO_INT = {
    "judge_label": {"Correct": 1, "Incorrect": 0},
}

_DEFAULT_TASK2_SYSTEM = (
    "You are an expert medical evaluator. You will be given a medical question, "
    "a reference answer, and a generated answer. Evaluate whether the generated "
    "answer is correct and respond only with the label in brackets []."
)
_DEFAULT_TASK2_USER = (
    "Question: {question_stem}\n"
    "Reference answer: {reference_answer}\n"
    "Generated answer: {generated_answer}\n\n"
    "Is the generated answer correct?\n"
    "Respond with exactly one of: [Correct] or [Incorrect]."
)

_DEFAULT_TASK3_SYSTEM = (
    "You are an expert medical evaluator. You will be given a doctor-patient dialogue "
    "in Arabic, the Primary Reasoning Objective the final doctor turn was supposed to "
    "clinically reach, and a generated final doctor turn. Evaluate whether the generated "
    "turn correctly achieves the Primary Reasoning Objective and respond only with the "
    "label in brackets []."
)
_DEFAULT_TASK3_USER = (
    "Dialogue:\n{dialogue}\n\n"
    "Primary Reasoning Objective: {primary_reasoning_objective}\n\n"
    "Generated Answer: {generated_answer}\n\n"
    "Does the generated answer correctly arrive at the diagnosis, differential, or "
    "management direction stated in the Primary Reasoning Objective? "
    "Do not penalize for code-switching or English medical terminology if the clinical "
    "content is correct. Only a full match counts as Correct — partial articulation is Incorrect.\n"
    "Respond with exactly one of: [Correct] or [Incorrect]."
)

DEFAULT_PROMPTS = {
    "answer_generation":   (_DEFAULT_TASK2_SYSTEM, _DEFAULT_TASK2_USER),
    "dialogue_completion": (_DEFAULT_TASK3_SYSTEM, _DEFAULT_TASK3_USER),
}

def load_prompt(prompt_path, task_type: str) -> tuple:
    if not prompt_path:
        return DEFAULT_PROMPTS[task_type]
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()
    if "---" in content:
        system_part, user_part = content.split("---", 1)
        return system_part.strip(), user_part.strip()
    return ("You are an expert medical evaluator.", content.strip())

def task2_format_kwargs(row: pd.Series) -> dict:
    return {
        "question_stem":    str(row["input"]).strip(),
        "reference_answer": str(row["ground_truth"]).strip(),
        "generated_answer": str(row["prediction"]).strip(),
    }
    
def task3_format_kwargs(row: pd.Series) -> dict:
    return {
        "dialogue":                    str(row["input"]).strip(),
        "primary_reasoning_objective": str(row.get("primary_reasoning_objective", "")).strip(),
        "generated_answer":            str(row["prediction"]).strip(),
    }

TASK_CONFIG = {
    "answer_generation": {
        "axes":          TASK2_AXES,
        "format_kwargs": task2_format_kwargs,
        "required_cols": set(),
    },
    "dialogue_completion": {
        "axes":          TASK3_AXES,
        "format_kwargs": task3_format_kwargs,
        "required_cols": {"primary_reasoning_objective"},
    },
}

def _normalize_label(candidate: str, valid_set: set) -> str:
    candidate = candidate.strip()
    for label in valid_set:
        if candidate.lower() == label.lower():
            return label
    return ""


def parse_axis_label(raw: str, axis_name: str, valid_set: set) -> str:
    # 1) Axis-prefixed: 'Reasoning Match: [Hit]'
    pattern = rf"{re.escape(axis_name)}\s*:\s*\[([^\]]+)\]"
    m = re.search(pattern, raw, flags=re.IGNORECASE)
    if m:
        label = _normalize_label(m.group(1), valid_set)
        if label:
            return label
    # 2) Any bracketed label valid for this axis
    for m in re.finditer(r"\[([^\]]+)\]", raw):
        label = _normalize_label(m.group(1), valid_set)
        if label:
            return label
    # 3) Free-text fallback
    for label in valid_set:
        if re.search(rf"\b{re.escape(label)}\b", raw, flags=re.IGNORECASE):
            return label
    return ""

def parse_judgment(raw: str, axes: list) -> dict:
    return {col: parse_axis_label(raw, axis_label, valid_set)
            for col, axis_label, valid_set in axes}

def call_judge(client, system_prompt, user_template, format_kwargs, axes, max_tokens) -> dict:
    user_prompt = user_template.format(**format_kwargs)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_completion_tokens=max_tokens,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            return parse_judgment(raw, axes)
        except Exception as e:
            logging.warning(f"[judge] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    logging.error("[judge] all retries exhausted — returning empty labels")
    return {col: "" for col, _, _ in axes}

def compute_composite(label_rows: list, task_type: str) -> dict:
    if task_type == "answer_generation":
        valid = [r for r in label_rows if r.get("judge_label") in {"Correct", "Incorrect"}]
        if not valid:
            return {"judge_composite_mean": None,
                    "judge_composite_n_complete": 0,
                    "judge_composite_n_partial": len(label_rows)}
        n_correct = sum(1 for r in valid if r["judge_label"] == "Correct")
        return {
            "judge_composite_mean":       round(n_correct / len(valid), ROUND_DP),
            "judge_composite_n_complete": len(valid),
            "judge_composite_n_partial":  len(label_rows) - len(valid),
            "judge_composite_formula":    "fraction labeled 'Correct' (single binary axis)",
        }

    if task_type == "dialogue_completion":
        valid = [r for r in label_rows if r.get("judge_label") in {"Correct", "Incorrect"}]
        if not valid:
            return {"judge_composite_mean": None,
                    "judge_composite_n_complete": 0,
                    "judge_composite_n_partial": len(label_rows)}
        n_correct = sum(1 for r in valid if r["judge_label"] == "Correct")
        return {
            "judge_composite_mean":       round(n_correct / len(valid), ROUND_DP),
            "judge_composite_n_complete": len(valid),
            "judge_composite_n_partial":  len(label_rows) - len(valid),
            "judge_composite_formula":    "fraction labeled 'Correct' (final response vs. ground truth)",
        }
    return {}

def compute_judge_metrics(label_rows: list, axes: list, task_type: str) -> dict:
    total        = len(label_rows)
    label_to_int = TASK3_LABEL_TO_INT if task_type == "dialogue_completion" else TASK2_LABEL_TO_INT
    metrics      = {"judge_model": MODEL, "judge_task_type": task_type, "judge_total": total}

    for col_name, _, valid_set in axes:
        col_labels = [r.get(col_name, "") for r in label_rows]
        valid      = [l for l in col_labels if l in valid_set]
        counts     = {label: valid.count(label) for label in valid_set}
        metrics[f"{col_name}__valid"] = len(valid)
        metrics[f"{col_name}__empty"] = total - len(valid)
        for label, n in counts.items():
            key = f"{col_name}__{label.lower().replace(' ', '_')}"
            metrics[key]              = n
            metrics[f"{key}_pct"]     = round(n / len(valid) * 100, ROUND_DP) if valid else 0.0
        axis_map = label_to_int.get(col_name)
        if axis_map and valid:
            max_val  = max(axis_map.values())
            int_vals = [axis_map[l] for l in valid]
            raw_mean = sum(int_vals) / len(int_vals)
            metrics[f"{col_name}__score"]     = round(raw_mean / max_val, ROUND_DP)
            metrics[f"{col_name}__score_raw"] = round(raw_mean, ROUND_DP)
            metrics[f"{col_name}__score_n"]   = len(valid)
        elif axis_map:
            metrics[f"{col_name}__score"]     = None
            metrics[f"{col_name}__score_raw"] = None
            metrics[f"{col_name}__score_n"]   = 0

    metrics.update(compute_composite(label_rows, task_type))
    return metrics

def write_metrics(metrics_path: Path, judge_metrics: dict):
    existing = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    existing.update(judge_metrics)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s")

    parser = argparse.ArgumentParser(
        description="Standalone LLM-as-judge — same model/scoring as scripts/run_judge.py."
    )
    parser.add_argument("--predictions_csv",   required=True,
                        help="Predictions CSV (needs id, input, prediction, ground_truth cols)")
    parser.add_argument("--metrics_json",      required=True,
                        help="Metrics JSON to update in-place with judge results")
    parser.add_argument("--task_type",         required=True,
                        choices=sorted(SUPPORTED_TASKS),
                        help="answer_generation (Task 2) or dialogue_completion (Task 3)")
    parser.add_argument("--judge_prompt_file", default=None,
                        help="Optional prompt file (--- separator). Uses built-in default if omitted.")
    parser.add_argument("--dataset_json",     default=None,
                        help="(dialogue_completion only) Path to original dataset JSON. "
                             "Used to merge primary_reasoning_objective into the "
                             "predictions CSV before judging. Joined on 'id'.")
    parser.add_argument("--metrics_only",      action="store_true",
                        help="Skip API calls; recompute metrics from existing judge label columns.")
    args = parser.parse_args()

    predictions_path = Path(args.predictions_csv)
    metrics_path     = Path(args.metrics_json)

    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions CSV not found: {predictions_path}")

    df         = pd.read_csv(predictions_path)
    task_type  = args.task_type
    cfg        = TASK_CONFIG[task_type]
    axes       = cfg["axes"]
    max_tokens = MAX_TOKENS_BY_TASK[task_type]

    logging.info(f"loaded {len(df)} rows | task_type={task_type} | model={MODEL}")

    if task_type == "dialogue_completion" and args.dataset_json:
        META_COLS = ["primary_reasoning_objective"]
        missing_meta = [c for c in META_COLS if c not in df.columns]
        if missing_meta:
            logging.info(f"merging metadata columns {missing_meta} from {args.dataset_json}")
            with open(args.dataset_json, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            if not isinstance(dataset, list):
                raise ValueError("--dataset_json must be a JSON array")
            meta_df = pd.DataFrame([
                {"id": str(rec.get("id", "")), **{c: rec.get(c, "") for c in META_COLS}}
                for rec in dataset
            ])
            df["id"] = df["id"].astype(str)
            df = df.merge(meta_df, on="id", how="left", suffixes=("", "_meta"))
            for c in META_COLS:
                meta_col = f"{c}_meta"
                if meta_col in df.columns:
                    df[c] = df[meta_col].combine_first(df[c])
                    df.drop(columns=[meta_col], inplace=True)
            filled = df[META_COLS].notna().all(axis=1).sum()
            logging.info(f"metadata merge complete: {filled}/{len(df)} rows have primary_reasoning_objective")
        else:
            logging.info("primary_reasoning_objective already present in CSV; skipping merge")

    if args.metrics_only:
        label_cols = [col for col, _, _ in axes]
        missing = [c for c in label_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"--metrics_only requires existing judge label columns; missing: {missing}. "
                f"Run without --metrics_only first."
            )
        label_rows = [
            {c: ("" if pd.isna(row[c]) else str(row[c]).strip()) for c in label_cols}
            for _, row in df.iterrows()
        ]
        logging.info(f"[metrics-only] recomputing from {len(label_rows)} existing labels")
        judge_metrics = compute_judge_metrics(label_rows, axes, task_type)
        write_metrics(metrics_path, judge_metrics)
        logging.info(json.dumps(judge_metrics, indent=2, ensure_ascii=False))
        return

    #check required columns after the metadata merge so that columns supplied
    base_cols = {"input", "prediction"} if task_type == "dialogue_completion" else {"input", "prediction", "ground_truth"}
    required_cols = base_cols | cfg["required_cols"]
    missing = required_cols - set(df.columns)
    if missing:
        hint = (
            " Pass --dataset_json to merge missing metadata columns."
            if task_type == "dialogue_completion" and missing & cfg["required_cols"]
            else ""
        )
        raise ValueError(f"predictions CSV missing columns: {missing}.{hint}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)

    system_prompt, user_template = load_prompt(args.judge_prompt_file, task_type)

    label_rows = []
    total = len(df)
    for i, row in df.iterrows():
        logging.info(f"[{i+1}/{total}] judging id={row.get('id', i)}")
        if not str(row.get("prediction", "")).strip():
            label_rows.append({col: "" for col, _, _ in axes})
            logging.info("  → empty prediction; skipping judge call")
            continue
        try:
            kwargs = cfg["format_kwargs"](row)
        except KeyError as e:
            logging.error(f"  → skipping row: missing field {e}")
            label_rows.append({col: "" for col, _, _ in axes})
            continue
        labels = call_judge(client, system_prompt, user_template, kwargs, axes, max_tokens)
        label_rows.append(labels)
        logging.info(f"  → {labels}")
        if i < total - 1:
            time.sleep(DELAY_SECS)

    #write back to csv
    for col_name, _, _ in axes:
        df[col_name] = [r.get(col_name, "") for r in label_rows]
    df.to_csv(predictions_path, index=False, encoding="utf-8")
    logging.info(f"saved predictions with judge labels to {predictions_path}")

    judge_metrics = compute_judge_metrics(label_rows, axes, task_type)
    write_metrics(metrics_path, judge_metrics)
    logging.info(f"judge metrics written to {metrics_path}")
    logging.info(json.dumps(judge_metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
