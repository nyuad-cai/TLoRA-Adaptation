import json
import sys
import pandas as pd
import torch
import re
from evals.metrics import (
    calculate_accuracy,
    calculate_bert_score,
    calculate_bert_score_per_example,
    extract_letter,
)


print(f"[DEBUG evaluator] imported evaluator from: {__file__}")
GENERATIVE_TASK_TYPES = {"answer_generation", "dialogue_completion"}


def strip_answer_prefix(text):
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"^\s*ANSWER\s*[:=：]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def split_prediction(prediction, task_type):
    if prediction is None:
        return None, None

    text = str(prediction).strip()

    if task_type == "mcq":
        upper = text.upper()

        m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
        if m:
            return m.group(1), text
        m = re.search(r"\b([A-F])\b", upper)
        if m:
            return m.group(1), text

        return None, text

    # For free-text generation tasks, just return text
    return text, text

def evaluate(predictions_path: str, metrics_path: str, task_type: str, lang: str = "ar"):
    df = pd.read_csv(predictions_path)

    if "prediction" not in df.columns or "ground_truth" not in df.columns:
        raise ValueError("csv must contain 'prediction' and 'ground_truth' columns")

    preds = df["prediction"].fillna("").astype(str).tolist()
    gts   = df["ground_truth"].fillna("").astype(str).tolist()

    metrics = {}

    if task_type == "mcq":
        pred_letters = []
        for p in preds:
            ltr = extract_letter(p)
            pred_letters.append("" if ltr is None else ltr)
        metrics["accuracy_letter"] = calculate_accuracy(pred_letters, gts)

    elif task_type in GENERATIVE_TASK_TYPES:
        preds_clean = [strip_answer_prefix(p) for p in preds]
        gts_clean = [strip_answer_prefix(g) for g in gts]

        df["prediction_clean"] = preds_clean
        df["ground_truth_clean"] = gts_clean

        device = "cuda" if torch.cuda.is_available() else "cpu"
        bs = calculate_bert_score(preds_clean, gts_clean, lang=lang, device=device)
        if bs is not None:
            metrics.update(bs)

        bert_rows = calculate_bert_score_per_example(preds_clean, gts_clean, lang=lang, device=device)
        if bert_rows is not None:
            df["bert_precision_example"] = [r["bert_precision"] for r in bert_rows]
            df["bert_recall_example"]    = [r["bert_recall"]    for r in bert_rows]
            df["bert_f1_example"]        = [r["bert_f1"]        for r in bert_rows]

        df.to_csv(predictions_path, index=False, encoding="utf-8")
    else:
        raise ValueError(
            f"unsupported task_type:{task_type} "
            f"expected mcq, answer_generation, or dialogue_completion"
        )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    return metrics

if __name__ == "__main__":
    if len(sys.argv) not in {4, 5}:
        print("usage: python evaluator.py <predictions.csv> <metrics.json> <task_type> [lang]")
        sys.exit(1)

    predictions_file = sys.argv[1]
    metrics_file = sys.argv[2]
    task = sys.argv[3]
    lang = sys.argv[4] if len(sys.argv) == 5 else "en"

    out = evaluate(predictions_file, metrics_file, task, lang=lang)
    print("evaluation completed. metrics saved to:", metrics_file)
    print(json.dumps(out, indent=2, ensure_ascii=False))
