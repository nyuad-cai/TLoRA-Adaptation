import os
import re
import json
import csv
import time
import logging
import argparse
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import Mistral3ForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

from evals.evaluator import split_prediction
from evals.metrics import calculate_bert_score  # same PYTHONPATH as evals.evaluator


HF_CACHE = "/scratch/ca2627/huggingface"
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")

JUDGE_MODEL      = "gpt-4o"
JUDGE_MAX_TOKENS = 50
JUDGE_TEMP       = 0
DELAY_SECS       = 0.5
MAX_RETRIES      = 3
RETRY_DELAY      = 5.0
VALID_LABELS     = {"Correct", "Incorrect"}

DEFAULT_JUDGE_SYSTEM = (
    "You are an expert medical evaluator. You will be given a medical question, "
    "a reference answer, and a generated answer. Your task is to evaluate the "
    "generated answer by selecting exactly one label from the following options "
    "and responding only with the label in brackets []."
)
DEFAULT_JUDGE_USER = (
    "Question: {question_stem}\n"
    "Reference answer: {reference_answer}\n"
    "Generated answer: {generated_answer}\n\n"
    "Is the generated answer correct?\n"
    "Respond with exactly one of: [Correct] or [Incorrect]."
)

def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return data

def build_ansgen_text(item: dict) -> str:
    """Question only — no options."""
    return (item.get("question") or "").strip()

def load_judge_prompt(prompt_path: Optional[str]):
    if not prompt_path:
        return DEFAULT_JUDGE_SYSTEM, DEFAULT_JUDGE_USER
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()
    if "---" in content:
        system_part, user_part = content.split("---", 1)
        return system_part.strip(), user_part.strip()
    return DEFAULT_JUDGE_SYSTEM, content.strip()

def parse_label(raw: str) -> str:
    m = re.search(r"\[([^\]]+)\]", raw)
    if m:
        candidate = m.group(1).strip()
        if candidate in VALID_LABELS:
            return candidate
        for label in VALID_LABELS:
            if candidate.lower() == label.lower():
                return label
    for label in VALID_LABELS:
        if label.lower() in raw.lower():
            return label
    logging.warning(f"[judge] could not parse label from: {repr(raw)}")
    return ""

def call_judge(client, system_prompt: str, user_template: str,
               question_stem: str, reference_answer: str, generated_answer: str) -> str:
    user_prompt = user_template.format(
        question_stem=question_stem,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                max_completion_tokens=JUDGE_MAX_TOKENS,
                temperature=JUDGE_TEMP,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return parse_label(response.choices[0].message.content.strip())
        except Exception as e:
            logging.warning(f"[judge] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    logging.error("[judge] all retries exhausted — returning empty label")
    return ""

def compute_judge_metrics(labels: list) -> dict:
    total  = len(labels)
    valid  = [l for l in labels if l in VALID_LABELS]
    empty  = total - len(valid)
    counts = {label: valid.count(label) for label in VALID_LABELS}
    pcts   = {
        f"{label.lower().replace(' ', '_')}_pct": (
            round(counts[label] / len(valid) * 100, 2) if valid else 0.0
        )
        for label in VALID_LABELS
    }
    return {
        "judge_model":     JUDGE_MODEL,
        "judge_total":     total,
        "judge_valid":     len(valid),
        "judge_empty":     empty,
        "judge_correct":   counts["Correct"],
        "judge_incorrect": counts["Incorrect"],
        **pcts,
    }

class MistralLoRAAnsgенInference:
    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        use_4bit: bool = False,
        cache_dir: str = HF_CACHE,
        offline: bool = False,
    ):
        self.base_model  = base_model
        self.adapter_path = adapter_path
        self.use_4bit    = use_4bit
        self.cache_dir   = cache_dir
        self.offline     = offline

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.local_files_only = bool(self.offline)
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

        print("[INFO] Loading tokenizer...")
        if os.path.isdir(self.base_model):
            tok_path = os.path.join(self.base_model, "tekken.json")
            self.tokenizer = MistralTokenizer.from_file(tok_path)
        else:
            if self.local_files_only:
                raise ValueError(
                    f"offline=True but base_model is not a local directory: {self.base_model}"
                )
            self.tokenizer = MistralTokenizer.from_hf_hub(self.base_model)

        print("[INFO] Loading base model...")
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base = Mistral3ForConditionalGeneration.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
        else:
            base = Mistral3ForConditionalGeneration.from_pretrained(
                self.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )

        print("[INFO] Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()

    def generate_raw(
        self,
        item: Dict[str, Any],
        instruction: str,
        max_tokens: int = 128,
        do_sample: bool = False,
    ) -> str:
        user_text = build_ansgen_text(item)
        if not user_text:
            return ""

        messages = [
            {"role": "system", "content": (instruction or "").strip()},
            {"role": "user",   "content": [{"type": "text", "text": user_text}]},
        ]

        try:
            torch.cuda.empty_cache()
            req       = ChatCompletionRequest(messages=messages)
            tokenized = self.tokenizer.encode_chat_completion(req)
            input_ids = torch.tensor(
                [tokenized.tokens], dtype=torch.long, device=self.model.device
            )
            attention_mask = torch.ones_like(input_ids)

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                )

            if outputs is None or outputs.shape[0] == 0:
                return ""

            gen_ids  = outputs[0][len(tokenized.tokens):]
            raw_text = self.tokenizer.decode(gen_ids).strip()
            return raw_text or ""

        except Exception as e:
            print(f"[ERROR] Generation failed for id={item.get('id')}: {e}")
            return ""

        finally:
            torch.cuda.empty_cache()

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file",         type=str, required=True)
    parser.add_argument("--instruction_file",   type=str, required=True)
    parser.add_argument("--base_model",         type=str, required=True)
    parser.add_argument("--adapter_path",       type=str, required=True)
    parser.add_argument("--output_file",        type=str, required=True)   # CSV
    parser.add_argument("--metrics_file",       type=str, required=True)   # JSON
    parser.add_argument("--use_4bit",           action="store_true")
    parser.add_argument("--offline",            action="store_true")
    parser.add_argument("--max_tokens",         type=int,  default=128)
    parser.add_argument("--lang",               type=str,  default="ar")
    # BERTScore
    parser.add_argument("--bert_device",        type=str,  default="cpu",
                        help="Device for BERTScore (e.g. 'cuda:0' or 'cpu')")
    # Judge (optional — skipped if --openai_api_key not provided and env var not set)
    parser.add_argument("--judge_prompt_file",  type=str,  default=None,
                        help="Path to judge prompt txt file (--- separator). "
                             "Uses built-in default if omitted.")
    parser.add_argument("--openai_api_key",     type=str,  default=None,
                        help="OpenAI API key. Falls back to OPENAI_API_KEY env var. "
                             "Judge is skipped if neither is set.")
    args = parser.parse_args()

    with open(args.instruction_file, "r", encoding="utf-8") as f:
        instruction = f.read().strip()

    dataset = load_json(args.test_file)

    runner = MistralLoRAAnsgенInference(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        use_4bit=args.use_4bit,
        offline=args.offline,
    )

    outputs = []
    for item in tqdm(dataset, desc="Running inference"):
        item_id    = item.get("id", "")
        input_text = build_ansgen_text(item)
        raw_pred   = runner.generate_raw(
            item,
            instruction=instruction,
            max_tokens=args.max_tokens,
            do_sample=False,
        )
        pred_main, _ = split_prediction(raw_pred, "answer_generation")
        pred_out  = "" if pred_main is None else pred_main
        answer_key = (item.get("answer") or "").strip()
        options    = item.get("options") or {}
        gt         = options.get(answer_key, answer_key).strip()

        outputs.append({
            "id":           item_id,
            "input":        input_text,
            "prediction":   pred_out,
            "ground_truth": gt,
        })

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "input", "prediction", "ground_truth"])
        writer.writeheader()
        writer.writerows(outputs)

    print(f"[INFO] Saved predictions CSV to {args.output_file}")

    predictions  = [r["prediction"]   for r in outputs]
    ground_truths = [r["ground_truth"] for r in outputs]

    print(f"[INFO] Computing BERTScore on {args.bert_device}...")
    bert_metrics = calculate_bert_score(
        predictions=predictions,
        references=ground_truths,
        lang=args.lang,
        device=args.bert_device,
    )
    if bert_metrics:
        print(f"[INFO] BERTScore: {json.dumps(bert_metrics, ensure_ascii=False, indent=2)}")
    else:
        print("[WARN] BERTScore failed or bert_score not installed.")
        bert_metrics = {}

    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    judge_metrics = {}

    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            system_prompt, user_template = load_judge_prompt(args.judge_prompt_file)
            labels = []
            total  = len(outputs)

            print("[INFO] Running LLM-as-judge...")
            for i, row in enumerate(tqdm(outputs, desc="Judge")):
                label = call_judge(
                    client=client,
                    system_prompt=system_prompt,
                    user_template=user_template,
                    question_stem=row["input"],
                    reference_answer=row["ground_truth"],
                    generated_answer=row["prediction"],
                )
                labels.append(label)
                if i < total - 1:
                    time.sleep(DELAY_SECS)

            judge_metrics = compute_judge_metrics(labels)
            print(f"[INFO] Judge metrics: {json.dumps(judge_metrics, ensure_ascii=False, indent=2)}")

        except ImportError:
            print("[WARN] openai package not installed — skipping judge.")
    else:
        print("[INFO] No OpenAI API key found — skipping judge. "
              "Pass --openai_api_key or set OPENAI_API_KEY to enable.")

    metrics_dir = os.path.dirname(args.metrics_file)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    all_metrics = {**bert_metrics, **judge_metrics}

    with open(args.metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Saved metrics JSON to {args.metrics_file}")
    print(f"[INFO] All metrics: {json.dumps(all_metrics, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    main()
