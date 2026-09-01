import os
import json
import csv
import argparse
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import Mistral3ForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

from evals.evaluator import evaluate, split_prediction


HF_CACHE = "/scratch/ca2627/huggingface"
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")


def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return data


def build_mcq_text(item: dict) -> str:
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
        if txt == "":
            continue
        lines.append(f"{letter}) {txt}")

    if stem and lines:
        return stem + "\n\n" + "\n".join(lines)
    return stem or ""

class MistralLoRAMCQInference:
    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        use_4bit: bool = False,
        cache_dir: str = HF_CACHE,
        offline: bool = False,
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.use_4bit = use_4bit
        self.cache_dir = cache_dir
        self.offline = offline

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
        max_tokens: int = 8,
        do_sample: bool = False,
    ) -> str:
        user_text = build_mcq_text(item)
        if not user_text:
            return ""

        messages = [
            {"role": "system", "content": (instruction or "").strip()},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]

        try:
            torch.cuda.empty_cache()

            req = ChatCompletionRequest(messages=messages)
            tokenized = self.tokenizer.encode_chat_completion(req)

            input_ids = torch.tensor(
                [tokenized.tokens],
                dtype=torch.long,
                device=self.model.device,
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

            input_len = len(tokenized.tokens)
            gen_ids = outputs[0][input_len:]
            raw_text = self.tokenizer.decode(gen_ids).strip()
            return raw_text or ""

        except Exception as e:
            print(f"[ERROR] Generation failed for id={item.get('id')}: {e}")
            return ""

        finally:
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--instruction_file", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)   # CSV
    parser.add_argument("--metrics_file", type=str, required=True)  # JSON
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=8)
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    with open(args.instruction_file, "r", encoding="utf-8") as f:
        instruction = f.read().strip()

    dataset = load_json(args.test_file)

    runner = MistralLoRAMCQInference(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        use_4bit=args.use_4bit,
        offline=args.offline,
    )

    outputs = []
    for item in tqdm(dataset, desc="Running inference"):
        item_id = item.get("id", "")
        input_text = build_mcq_text(item)
        raw_pred = runner.generate_raw(
            item,
            instruction=instruction,
            max_tokens=args.max_tokens,
            do_sample=False,
        )

        pred_main, _ = split_prediction(raw_pred, "mcq")
        pred_out = "" if pred_main is None else pred_main
        gt = (item.get("answer") or "").strip().upper()

        outputs.append(
            {
                "id": item_id,
                "input": input_text,
                "prediction": pred_out,
                "ground_truth": gt,
            }
        )

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "input", "prediction", "ground_truth"],
        )
        writer.writeheader()
        for row in outputs:
            writer.writerow(row)

    print(f"[INFO] Saved predictions CSV to {args.output_file}")

    metrics_dir = os.path.dirname(args.metrics_file)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    metrics = evaluate(
        predictions_path=args.output_file,
        metrics_path=args.metrics_file,
        task_type="mcq",
        lang=args.lang,
    )

    print(f"[INFO] Saved metrics JSON to {args.metrics_file}")
    print(f"[INFO] Metrics: {json.dumps(metrics, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
