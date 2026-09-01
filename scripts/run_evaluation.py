"""
This script runs an evaluation experiment defined by a yaml config. It loads data (JSON list of dicts), queries a model handler (single or batch), 
saves predictions and evaluates outputs. 
Supported task types: 
+ mcq: returns a single letter (A-F), metric = accuracy
+ answer_generation: returns free-form text, metrics = BLEU/ROUGE/BERTScore (handled in evals/evaluator.py)
+ dialogue_completion: returns the missing final doctor turn, metrics = BLEU/ROUGE/BERTScore + judge
"""

import sys
import json
import logging
import atexit
import signal
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils import load_config, save_predictions  # noqa: E402
from models import load_model_handler  # noqa: E402
from evals.evaluator import evaluate, split_prediction  # noqa: E402

ALLOWED_TASK_TYPES = {"mcq", "answer_generation", "dialogue_completion"}
def _pick(item: dict, *keys):
    """Return first non-empty value among keys, else None."""
    for k in keys:
        v = item.get(k)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None

def build_mcq_text(item: dict) -> str:
    """Build a readable MCQ block from the dataset schema."""
    stem = (item.get("question") or "").strip()

    option_map = {
        "A": item.get("opa"),
        "B": item.get("opb"),
        "C": item.get("opc"),
        "D": item.get("opd"),
        "E": item.get("ope"),
        "F": item.get("opf"),
    }
    lines = []
    for letter in ["A", "B", "C", "D", "E", "F"]:
        txt = option_map.get(letter)
        if txt is None:
            continue
        txt = str(txt).strip()
        if not txt:
            continue
        lines.append(f"{letter}) {txt}")

    if stem and lines:
        return stem + "\n\n" + "\n".join(lines)
    return stem or ""


def build_ansgen_text(item: dict) -> str:
    """Build the input text for answer_generation"""
    return (item.get("question") or "").strip()

def build_dialogue_text(item: dict) -> str:
    v = _pick(item, "Dialogue", "dialogue", "conversation", "context")
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        lines = []
        for t in v:
            if isinstance(t, dict):
                role = str(t.get("role") or t.get("speaker") or "").strip()
                text = str(t.get("text") or t.get("content") or "").strip()
                if not text:
                    continue
                lines.append(f"{role}: {text}" if role else text)
            else:
                s = str(t).strip()
                if s:
                    lines.append(s)
        return "\n".join(lines)
    return ""

LETTER_SET = {"A", "B", "C", "D", "E", "F"}

def get_ground_truth_or_die(item: Dict[str, Any], task_type: str, item_id: str) -> str:
    if task_type == "mcq":
        gt = (item.get("answer") or "").strip().upper()
        if not gt:
            raise RuntimeError(f"[FATAL] mcq missing 'answer' for id={item_id}")
        return gt

    if task_type == "answer_generation":
        if "answer_text" not in item:
            raise RuntimeError(
                f"[FATAL] answer_generation requires 'answer_text' but key is missing for id={item_id}. "
                f"Available keys={list(item.keys())}"
            )
        gt = str(item["answer_text"]).strip()
        if not gt:
            raise RuntimeError(f"[FATAL] answer_generation 'answer_text' is empty for id={item_id}")

        if gt.strip().upper() in LETTER_SET:
            raise RuntimeError(
                f"[FATAL] answer_generation ground_truth is a LETTER for id={item_id}. "
                f"answer_text={gt!r} answer={item.get('answer')!r}"
            )

        return gt

    if task_type == "dialogue_completion":
        v = _pick(item, "Gold Response", "gold_response", "gold_answer", "answer_text")
        if v is None:
            raise RuntimeError(
                f"[FATAL] dialogue_completion requires 'Gold Response' for id={item_id}. "
                f"Available keys={list(item.keys())}"
            )
        return str(v).strip()

    raise ValueError(f"unsupported task_type={task_type!r} in get_ground_truth_or_die")

def is_valid_mcq_item(item: dict) -> bool:
    stem = (item.get("question") or "").strip()
    if not stem:
        return False
    return all((item.get(k) for k in ["opa", "opb", "opc", "opd"]))

def is_valid_ansgen_item(item: dict) -> bool:
    q = (item.get("question") or "").strip()
    if not q:
        return False
    return "answer_text" in item and str(item["answer_text"]).strip() != ""

def is_valid_dialogue_item(item: dict) -> bool:
    if not build_dialogue_text(item):
        return False
    if _pick(item, "Gold Response", "gold_response", "answer_text") is None:
        return False
    return True

def explain_invalid(item: dict, task_type: str) -> str:
    """Return a human-readable reason why an item failed validation."""
    if task_type == "mcq":
        if not (item.get("question") or "").strip():
            return "missing 'question'"
        missing_opts = [k for k in ["opa", "opb", "opc", "opd"] if not item.get(k)]
        if missing_opts:
            return f"missing options: {missing_opts}"
        return "unknown mcq validation failure"

    if task_type == "answer_generation":
        if not (item.get("question") or "").strip():
            return "missing 'question'"
        if "answer_text" not in item or not str(item.get("answer_text", "")).strip():
            return "missing or empty 'answer_text'"
        return "unknown answer_generation validation failure"

    if task_type == "dialogue_completion":
        missing = []
        if not build_dialogue_text(item):
            missing.append("Dialogue")
        if _pick(item, "Gold Response", "gold_response", "answer_text") is None:
            missing.append("Gold Response")
        return f"missing fields: {missing}" if missing else "unknown dialogue_completion validation failure"

    return f"unknown task_type={task_type}"

def is_valid_item(item: dict, task_type: str) -> bool:
    if task_type == "mcq":
        return is_valid_mcq_item(item)
    if task_type == "answer_generation":
        return is_valid_ansgen_item(item)
    if task_type == "dialogue_completion":
        return is_valid_dialogue_item(item)
    return False

def extra_columns_for_judge(item: dict, task_type: str) -> dict:
    if task_type != "dialogue_completion":
        return {}
    pro = _pick(item, "Primary Reasoning Objective", "primary_reasoning_objective") or ""
    rfs = _pick(
        item,
        "Red Flag Symptom",          
        "Red Flag Symptoms",         
        "red_flag_symptom",
        "red_flag_symptoms",
        "red_flags",
    ) or ""
    if isinstance(rfs, list):
        rfs = "\n".join(str(x).strip() for x in rfs if str(x).strip())
    return {
        "primary_reasoning_objective": str(pro).strip(),
        "red_flag_symptoms": str(rfs).strip(),
    }

DATASET_WRAPPER_KEYS = ("data", "records", "items", "examples", "rows", "dataset")

def load_dataset_safely(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        if not raw:
            raise RuntimeError(f"[FATAL] dataset at {path} is an empty list")
        if not isinstance(raw[0], dict):
            raise RuntimeError(
                f"[FATAL] dataset at {path} is a list, but first element is "
                f"{type(raw[0]).__name__} (expected dict). "
                f"first element preview: {repr(raw[0])[:200]}"
            )
        logging.info(f"[dataset] shape=list_of_dicts size={len(raw)}")
        return raw

    if isinstance(raw, dict):
        for key in DATASET_WRAPPER_KEYS:
            if key in raw and isinstance(raw[key], list):
                inner = raw[key]
                if not inner:
                    raise RuntimeError(
                        f"[FATAL] dataset at {path} has key {key!r} but the list is empty"
                    )
                if not isinstance(inner[0], dict):
                    raise RuntimeError(
                        f"[FATAL] dataset at {path} has key {key!r} but first element is "
                        f"{type(inner[0]).__name__} (expected dict)"
                    )
                logging.info(
                    f"[dataset] shape=wrapped_list size={len(inner)} wrapper_key={key!r}"
                )
                return inner

        raise RuntimeError(
            f"[FATAL] dataset at {path} is a dict but does not have a list under "
            f"any of {DATASET_WRAPPER_KEYS}. top-level keys={list(raw.keys())}. "
            f"If your wrapping key is different, add it to DATASET_WRAPPER_KEYS."
        )

    raise RuntimeError(
        f"[FATAL] dataset at {path} parsed as {type(raw).__name__}; "
        f"expected list of dicts or {{'data': [...]}}-style wrapper."
    )

def save_debug_jsonl(rows, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def run_experiment(config_path: str):
    logging.info(f"loading config from {config_path}")
    config = load_config(config_path)

    task_cfg = config.get("task", {})
    task_type = str(task_cfg.get("type", "mcq")).strip().lower()

    if task_type not in ALLOWED_TASK_TYPES:
        raise ValueError(
            f"unsupported task.type={task_type!r}. "
            f"Expected one of {sorted(ALLOWED_TASK_TYPES)}."
        )

    lang = task_cfg.get("lang", "ar")

    logging.info(f"[DEBUG] entrypoint file={__file__}")
    logging.info(f"[DEBUG] task_type={task_type!r} lang={lang!r}")

    logging.info(f"initializing model:{config['model']['name']}")
    model_handler = load_model_handler(config)

    logging.info(f"loading dataset from {config['dataset']['path']}")
    dataset_path = config["dataset"]["path"]
    instruction_path = config["dataset"]["instruction_path"]

    model_cfg = config.get("model", {})
    model_type = str(model_cfg.get("type", "")).strip().lower()
    batch_size = int(model_cfg.get("batch_size", 4))

    if task_type == "mcq":
        default_max_tokens = 8
    elif task_type == "answer_generation":
        default_max_tokens = 256
    else:  # dialogue_completion: 1-3 Arabic sentences need more headroom
        default_max_tokens = 256

    max_tokens = int(
        model_cfg.get("max_tokens")
        or task_cfg.get("max_tokens")
        or default_max_tokens
    )

    dataset = load_dataset_safely(dataset_path)
    logging.info(
        f"[DEBUG] dataset size={len(dataset)}; "
        f"first-item keys={list(dataset[0].keys())}"
    )

    if task_type == "dialogue_completion":
        if _pick(dataset[0], "Primary Reasoning Objective", "primary_reasoning_objective") is None:
            logging.warning(
                "[dialogue_completion] 'Primary Reasoning Objective' not found "
                "on the first item; judge will fall back to general reasoning."
            )
        if _pick(
            dataset[0],
            "Red Flag Symptom", "Red Flag Symptoms",
            "red_flag_symptom", "red_flag_symptoms", "red_flags",
        ) is None:
            logging.warning(
                "[dialogue_completion] 'Red Flag Symptom(s)' not found on the "
                "first item; judge will not have case-specific safety priors."
            )

    with open(instruction_path, "r", encoding="utf-8") as f:
        instruction = f.read().strip()

    output_cfg = config["output"]
    output_path = output_cfg["predictions_path"]
    metrics_path = output_cfg["metrics_path"]
    debug_path = output_cfg.get("debug_path")
    partial_path = str(Path(output_path).with_suffix(".partial.csv"))
    partial_debug_path = str(Path(debug_path).with_suffix(".partial.jsonl")) if debug_path else None

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    if debug_path:
        Path(debug_path).parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"[DEBUG] predictions_path={output_path}")
    logging.info(f"[DEBUG] metrics_path={metrics_path}")
    if debug_path:
        logging.info(f"[DEBUG] debug_path={debug_path}")

    predictions = []
    debug_rows = []
    skip_count = 0
    finished = False

    def save_now(path):
        save_predictions(predictions, path)

    def save_debug_now(path):
        if path and debug_rows:
            save_debug_jsonl(debug_rows, path)

    def finalize_and_eval(path_for_eval, debug_path_for_eval=None):
        if not predictions:
            logging.info(
                "[partial] no predictions accumulated; nothing to save or evaluate"
            )
            return

        try:
            save_now(path_for_eval)
        except Exception as e:
            logging.warning(f"[partial] save failed: {e}")
            return

        if debug_path_for_eval and debug_rows:
            try:
                save_debug_now(debug_path_for_eval)
            except Exception as e:
                logging.warning(f"[partial] debug save failed: {e}")

        # Belt-and-suspenders: only evaluate if the file landed on disk.
        if not Path(path_for_eval).exists():
            logging.warning(
                f"[partial] {path_for_eval} not found after save; skipping evaluation"
            )
            return

        try:
            m = evaluate(path_for_eval, metrics_path, task_type, lang=lang)
            logging.info(f"[partial] metrics saved to {metrics_path}")
            logging.info(f"[partial] {json.dumps(m, indent=2, ensure_ascii=False)}")
        except Exception as e:
            logging.warning(f"[partial] evaluation failed: {e}")

    def on_exit():
        if not finished:
            finalize_and_eval(partial_path, partial_debug_path)

    atexit.register(on_exit)

    def handle_sigint(sig, frame):
        logging.info("caught ctrl+c saving and evaluating partial results")
        finalize_and_eval(partial_path, partial_debug_path)
        raise SystemExit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    use_batch = hasattr(model_handler, "prompt_batch")
    wants_debug = bool(debug_path)

    def build_record(item: dict, prediction, item_id: str, task_type: str) -> dict:
        gt = get_ground_truth_or_die(item, task_type, item_id)

        if task_type == "mcq":
            input_text = build_mcq_text(item)
            if isinstance(prediction, dict) and "final_prediction" in prediction:
                pred_main, _ = split_prediction(prediction.get("final_prediction"), "mcq")
                pred_out = "" if pred_main is None else pred_main
            else:
                pred_main, _ = split_prediction(prediction, "mcq")
                pred_out = "" if pred_main is None else pred_main
        elif task_type == "answer_generation":
            input_text = build_ansgen_text(item)
            pred_out = "" if prediction is None else str(prediction).strip()
        else:  # dialogue_completion
            input_text = build_dialogue_text(item)
            pred_out = "" if prediction is None else str(prediction).strip()

        record = {
            "id": item_id,
            "input": input_text,
            "prediction": pred_out,
            "ground_truth": gt,
        }
        record.update(extra_columns_for_judge(item, task_type))
        return record

    def note_invalid(item: dict, idx: int):
        nonlocal skip_count
        skip_count += 1
        if skip_count == 1:
            reason = explain_invalid(item, task_type)
            logging.warning(
                f"[{task_type}] invalid item #{idx}; skipping. Reason: {reason}. "
                f"Available keys: {list(item.keys())}"
            )
        else:
            logging.warning(f"[{task_type}] invalid item #{idx}; skipping")

    if use_batch:
        logging.info(f"using prompt_batch batch_size={batch_size}")

        for start in tqdm(range(0, len(dataset), batch_size), desc="processing examples"):
            batch_items = dataset[start : start + batch_size]

            filtered_items = []
            for offset, item in enumerate(batch_items):
                global_idx = start + offset + 1

                if not is_valid_item(item, task_type):
                    note_invalid(item, global_idx)
                    continue

                filtered_items.append((global_idx, item))

            if not filtered_items:
                continue

            batch_payload = [it for (_, it) in filtered_items]

            if wants_debug and model_type == "autocap":
                batch_predictions = [
                    model_handler.prompt(
                        s,
                        instruction=instruction,
                        max_tokens=max_tokens,
                        task_type=task_type,
                        return_debug=True,
                    )
                    for s in batch_payload
                ]
            else:
                batch_predictions = model_handler.prompt_batch(
                    batch_payload,
                    instruction=instruction,
                    max_tokens=max_tokens,
                    task_type=task_type,
                )

            for (global_idx, item), prediction in zip(filtered_items, batch_predictions):
                item_id = item.get("id") or str(item.get("Case ID") or global_idx)
                record = build_record(item, prediction, item_id, task_type)

                if global_idx <= 3:
                    logging.info(f"[DEBUG] id={item_id} task_type={task_type} gt_written={record['ground_truth']!r}")
                    logging.info(f"[DEBUG] id={item_id} pred_written={record['prediction']!r}")

                if (
                    task_type == "mcq"
                    and wants_debug
                    and isinstance(prediction, dict)
                    and "final_prediction" in prediction
                ):
                    debug_rows.append(
                        {
                            "id": item_id,
                            "input": record["input"],
                            "ground_truth": record["ground_truth"],
                            **prediction,
                        }
                    )

                predictions.append(record)

    else:
        logging.info("using prompt per-example")

        for idx, item in enumerate(tqdm(dataset, desc="processing examples"), start=1):
            item_id = item.get("id") or str(item.get("Case ID") or idx)

            if not is_valid_item(item, task_type):
                note_invalid(item, idx)
                continue

            if wants_debug and model_type == "autocap":
                prediction = model_handler.prompt(
                    item,
                    instruction=instruction,
                    max_tokens=max_tokens,
                    task_type=task_type,
                    return_debug=True,
                )
            else:
                prediction = model_handler.prompt(
                    item,
                    instruction=instruction,
                    max_tokens=max_tokens,
                    task_type=task_type,
                )

            record = build_record(item, prediction, item_id, task_type)

            if idx <= 3:
                logging.info(f"[DEBUG] id={item_id} task_type={task_type} gt_written={record['ground_truth']!r}")
                logging.info(f"[DEBUG] id={item_id} pred_written={record['prediction']!r}")

            if (
                task_type == "mcq"
                and wants_debug
                and isinstance(prediction, dict)
                and "final_prediction" in prediction
            ):
                debug_rows.append(
                    {
                        "id": item_id,
                        "input": record["input"],
                        "ground_truth": record["ground_truth"],
                        **prediction,
                    }
                )

            predictions.append(record)

    if skip_count:
        logging.info(f"skipped {skip_count}/{len(dataset)} item(s) during validation")

    if not predictions:
        logging.error(
            f"no predictions produced (skipped {skip_count}/{len(dataset)}). "
            "Check the validator warning above; for Task 3 the only HARD-required "
            "fields are 'Dialogue' and 'Gold Response'."
        )
        finished = True  # prevent the atexit hook from trying to eval an empty file
        return

    logging.info(f"saving predictions to {output_path}")
    logging.info(f"[CHECK BEFORE SAVE] first row ground_truth={predictions[0]['ground_truth']!r}")
    save_now(output_path)

    if debug_path and debug_rows:
        save_debug_now(debug_path)
        logging.info(f"saved debug rows to {debug_path}")

    logging.info("starting evaluation")
    m = evaluate(output_path, metrics_path, task_type, lang=lang)
    logging.info(f"evaluation completed metrics saved to {metrics_path}")
    logging.info(f"metrics:{json.dumps(m, indent=4, ensure_ascii=False)}")

    finished = True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s")
    if len(sys.argv) < 2:
        print("usage: python scripts/run_evaluation.py <config.yaml>")
        raise SystemExit(2)

    config_file = sys.argv[1]
    try:
        run_experiment(config_file)
    except Exception as e:
        logging.error(f"an error occurred:{e}")
        raise
