'''
This script implements metrics for mcq and answer_generation evaluation.
'''

import os
import re
from typing import List, Optional, Dict, Any
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

try:
    import bert_score
except Exception:
    bert_score = None

try:
    from bleurt import score as bleurt_score
except Exception:
    bleurt_score = None

DEFAULT_BERT_MODEL = "aubmindlab/bert-large-arabertv02"
DEFAULT_BERT_NUM_LAYERS = 18


def check_bertscore_baselines() -> None:
    if bert_score is None:
        print("[bert_score] not installed")
        return
    base = os.path.join(os.path.dirname(bert_score.__file__), "rescale_baseline")
    if not os.path.isdir(base):
        print(f"[bert_score] no rescale_baseline directory at {base}")
        return
    for lang in sorted(os.listdir(base)):
        p = os.path.join(base, lang)
        if os.path.isdir(p):
            print(f"  {lang}: {sorted(os.listdir(p))}")

def extract_letter(text):
    if not text:
        return None
    t = str(text).strip()

    m = re.search(r"(?i)\banswer(?:\s*is)?\s*[:：]?\s*([A-F])\b", t)
    if m:
        return m.group(1).upper()

    first = next((ln for ln in t.splitlines() if ln.strip()), "")
    m = re.match(r"^\s*([A-Fa-f])\s*[\.\)]?\s*$", first)
    if m:
        return m.group(1).upper()

    m = re.match(r"^\s*([A-Fa-f])\s*$", t)
    if m:
        return m.group(1).upper()

    return None


def calculate_accuracy(predictions, ground_truths):
    correct = 0
    total = 0

    for p, g in zip(predictions, ground_truths):
        total += 1
        p_letter = extract_letter(p) if p is not None else None
        g_letter = extract_letter(g)

        if p_letter and g_letter and p_letter == g_letter:
            correct += 1

    return correct / total if total > 0 else 0.0


#task2
def _safe_tokenize(text: str, lang: str = "en") -> List[str]:
    text = "" if text is None else str(text).strip()
    if not text:
        return []

    if lang.lower().startswith("ar"):
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    try:
        return word_tokenize(text)
    except Exception:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _normalize_arabic(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"[\u064B-\u0652]", "", text)  # remove diacritics
    text = text.replace("آ", "ا").replace("أ", "ا").replace("إ", "ا")
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    text = text.replace("ة", "ه")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_bleu(predictions: List[str], references: List[str], lang: str = "ar") -> float:
    smoothie = SmoothingFunction().method1
    scores = []

    for hyp, ref in zip(predictions, references):
        hyp = "" if hyp is None else str(hyp)
        ref = "" if ref is None else str(ref)

        if lang.lower().startswith("ar"):
            hyp = _normalize_arabic(hyp)
            ref = _normalize_arabic(ref)

        ref_tokens = _safe_tokenize(ref, lang=lang)
        hyp_tokens = _safe_tokenize(hyp, lang=lang)

        if not ref_tokens and not hyp_tokens:
            scores.append(1.0)
            continue
        if not ref_tokens or not hyp_tokens:
            scores.append(0.0)
            continue

        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        scores.append(float(bleu))

    return float(sum(scores) / len(scores)) if scores else 0.0

def calculate_rouge(predictions: List[str], references: List[str], lang: str = "en") -> Dict[str, float]:
    #disable stemming for Arabic
    use_stemmer = not lang.lower().startswith("ar")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)

    r1 = r2 = rl = 0.0
    n = 0

    for hyp, ref in zip(predictions, references):
        hyp = "" if hyp is None else str(hyp)
        ref = "" if ref is None else str(ref)

        if lang.lower().startswith("ar"):
            hyp = _normalize_arabic(hyp)
            ref = _normalize_arabic(ref)

        if not hyp and not ref:
            r1 += 1.0; r2 += 1.0; rl += 1.0
            n += 1
            continue

        scores = scorer.score(ref, hyp)
        r1 += scores["rouge1"].fmeasure
        r2 += scores["rouge2"].fmeasure
        rl += scores["rougeL"].fmeasure
        n += 1

    n = n or 1
    return {"rouge1": r1 / n, "rouge2": r2 / n, "rougeL": rl / n}


def calculate_bert_score(
    predictions: List[str],
    references: List[str],
    lang: str = "ar",
    model_type: Optional[str] = DEFAULT_BERT_MODEL,
    num_layers: Optional[int] = DEFAULT_BERT_NUM_LAYERS,
    device: str = "cpu",
    rescale_with_baseline: bool = False,
) -> Optional[Dict[str, float]]:
    # Default: AraBERT-large-v02, layer 18, raw (unrescaled) scores.
    # See the module-level comment for alternatives and their recommended
    # num_layers. rescale_with_baseline defaults to False because no Arabic
    # baselines exist for these encoders.
    if bert_score is None:
        return None

    hyps = ["" if p is None else str(p) for p in predictions]
    refs = ["" if r is None else str(r) for r in references]

    #apply arabic normalization
    if lang.lower().startswith("ar"):
        hyps = [_normalize_arabic(h) for h in hyps]
        refs = [_normalize_arabic(r) for r in refs]

    if not hyps or not refs:
        return None

    try:
        P, R, F1 = bert_score.score(
            hyps,
            refs,
            lang=lang,
            model_type=model_type,
            num_layers=num_layers,
            device=device,
            rescale_with_baseline=rescale_with_baseline,
            verbose=False,
        )
        p = P.detach().cpu().numpy()
        r = R.detach().cpu().numpy()
        f1 = F1.detach().cpu().numpy()

        return {
            "bert_precision": float(p.mean()),
            "bert_recall": float(r.mean()),
            "bert_f1": float(f1.mean()),
            "bert_model": model_type,
            "bert_num_layers": num_layers,
            "bert_rescaled": bool(rescale_with_baseline),
        }
    except Exception as e:
        print(f"[bert_score] failed: {type(e).__name__}: {e}")
        return None


def calculate_bleu_per_example(
    predictions: List[str],
    references: List[str],
    lang: str = "ar",
) -> List[float]:
    smoothie = SmoothingFunction().method1
    scores = []

    for hyp, ref in zip(predictions, references):
        hyp = "" if hyp is None else str(hyp)
        ref = "" if ref is None else str(ref)

        if lang.lower().startswith("ar"):
            hyp = _normalize_arabic(hyp)
            ref = _normalize_arabic(ref)

        ref_tokens = _safe_tokenize(ref, lang=lang)
        hyp_tokens = _safe_tokenize(hyp, lang=lang)

        if not ref_tokens and not hyp_tokens:
            scores.append(1.0)
            continue
        if not ref_tokens or not hyp_tokens:
            scores.append(0.0)
            continue

        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        scores.append(float(bleu))

    return scores


def calculate_rouge_per_example(
    predictions: List[str],
    references: List[str],
    lang: str = "en",
) -> List[Dict[str, float]]:
    use_stemmer = not lang.lower().startswith("ar")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)

    rows = []

    for hyp, ref in zip(predictions, references):
        hyp = "" if hyp is None else str(hyp)
        ref = "" if ref is None else str(ref)

        if lang.lower().startswith("ar"):
            hyp = _normalize_arabic(hyp)
            ref = _normalize_arabic(ref)

        if not hyp and not ref:
            rows.append({
                "rouge1": 1.0,
                "rouge2": 1.0,
                "rougeL": 1.0,
            })
            continue

        scores = scorer.score(ref, hyp)
        rows.append({
            "rouge1": float(scores["rouge1"].fmeasure),
            "rouge2": float(scores["rouge2"].fmeasure),
            "rougeL": float(scores["rougeL"].fmeasure),
        })

    return rows


def calculate_bert_score_per_example(
    predictions: List[str],
    references: List[str],
    lang: str = "ar",
    model_type: Optional[str] = DEFAULT_BERT_MODEL,
    num_layers: Optional[int] = DEFAULT_BERT_NUM_LAYERS,
    device: str = "cpu",
    rescale_with_baseline: bool = False,
) -> Optional[List[Dict[str, float]]]:
    if bert_score is None:
        return None

    hyps = ["" if p is None else str(p) for p in predictions]
    refs = ["" if r is None else str(r) for r in references]

    if lang.lower().startswith("ar"):
        hyps = [_normalize_arabic(h) for h in hyps]
        refs = [_normalize_arabic(r) for r in refs]

    if not hyps or not refs:
        return None

    try:
        P, R, F1 = bert_score.score(
            hyps,
            refs,
            lang=lang,
            model_type=model_type,
            num_layers=num_layers,
            device=device,
            rescale_with_baseline=rescale_with_baseline,
            verbose=False,
        )

        # No clipping — see note in calculate_bert_score.
        p = P.detach().cpu().numpy()
        r = R.detach().cpu().numpy()
        f1 = F1.detach().cpu().numpy()

        rows = []
        for pi, ri, f1i in zip(p, r, f1):
            rows.append({
                "bert_precision": float(pi),
                "bert_recall": float(ri),
                "bert_f1": float(f1i),
            })
        return rows

    except Exception as e:
        print(f"[bert_score] failed: {type(e).__name__}: {e}")
        return None
