import os
import re
from typing import List, Optional

import torch
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import Mistral3ForConditionalGeneration

HF_CACHE = "/scratch/ca2627/huggingface"
os.environ["HF_HOME"] = HF_CACHE

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class MistralSmallHandler:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
    ):
        print(f"[MistralSmall] Handler file: {__file__}")
        print(f"[MistralSmall] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[MistralSmall] cache_dir={cache_dir}")
        print(f"[MistralSmall] offline={offline}")

        self.model_name = model_name
        self.cache_dir = cache_dir or HF_CACHE
        self.offline = offline

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.local_files_only = bool(self.offline)
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

        num_gpus = torch.cuda.device_count()
        print(f"[MistralSmall] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for MistralSmall.")

        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated"
            )
        print("[MistralSmall] Loading tokenizer...")
        if os.path.isdir(self.model_name):
            tok_path = os.path.join(self.model_name, "tekken.json")
            self.tokenizer = MistralTokenizer.from_file(tok_path)
        else:
            if self.local_files_only:
                raise ValueError(
                    f"[MistralSmall] offline=True but model_name is not a local directory: {self.model_name}"
                )
            self.tokenizer = MistralTokenizer.from_hf_hub(self.model_name)

        print("[MistralSmall] Loading model...")
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

        print("[MistralSmall] Model loaded.")
        if hasattr(self.model, "hf_device_map"):
            print("[MistralSmall] Model device distribution:")
            for layer, dev in self.model.hf_device_map.items():
                print(f"  {layer}: {dev}")

        for i in range(num_gpus):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserv = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i} after load: {alloc:.2f}GB allocated, {reserv:.2f}GB reserved")

    @staticmethod
    def _build_mcq_text(sample: dict) -> str:
        stem = (sample.get("question") or "").strip()
        if not stem:
            return ""

        option_map = {
            "A": sample.get("opa"),
            "B": sample.get("opb"),
            "C": sample.get("opc"),
            "D": sample.get("opd"),
            "E": sample.get("ope"),
            "F": sample.get("opf"),
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

        return stem + "\n\n" + "\n".join(lines)

    @staticmethod
    def _build_ansgen_text(sample: dict) -> str:
        stem = (sample.get("question") or "").strip()
        if not stem:
            return ""

        option_map = {
            "A": sample.get("opa"),
            "B": sample.get("opb"),
            "C": sample.get("opc"),
            "D": sample.get("opd"),
            "E": sample.get("ope"),
            "F": sample.get("opf"),
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

        if lines:
            return stem + "\n\n" + "\n".join(lines)
        return stem

    def _generate(
        self,
        system_prompt: str,
        user_text: str,
        max_tokens: int,
        do_sample: bool = False,
    ) -> str:
        messages = [
            {"role": "system", "content": (system_prompt or "").strip()},
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
            print("[MistralSmall] Error during generation:", e)
            return ""

        finally:
            torch.cuda.empty_cache()

    def prompt(
        self,
        sample: dict,
        instruction: str,
        max_tokens: int = 8,
        task_type: str = "mcq",
    ):
        task_type = (task_type or "mcq").strip().lower()

        if task_type == "mcq":
            user_text = self._build_mcq_text(sample)
            if not user_text:
                print("[MistralSmall] Empty stem/options; cannot build MCQ prompt.")
                return None

            system_prompt = (instruction or "").strip()
            raw_text = self._generate(
                system_prompt=system_prompt,
                user_text=user_text,
                max_tokens=max_tokens,
                do_sample=False,
            )

            if not raw_text:
                return None

            print(f"[MistralSmall] MCQ raw generated: {repr(raw_text)}")
            upper = raw_text.strip().upper()

            if upper in ("A", "B", "C", "D", "E", "F"):
                return upper

            m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
            if m:
                return m.group(1)
            m = re.search(r"\b([A-F])\b", upper)
            if m:
                return m.group(1)

            print("[MistralSmall] Could not extract a clean letter.")
            return None

        if task_type == "answer_generation":
            user_text = self._build_ansgen_text(sample)
            if not user_text:
                print("[MistralSmall] Empty question; cannot build answer-generation prompt.")
                return ""

            system_prompt = (instruction or "").strip()
            raw_text = self._generate(
                system_prompt=system_prompt,
                user_text=user_text,
                max_tokens=max_tokens,
                do_sample=False,
            )

            if not raw_text:
                return ""

            one_line = raw_text.split("\n")[0].strip()
            print(f"[MistralSmall] Answer-gen raw (one line): {repr(one_line[:200])}")
            return one_line

        raise ValueError(
            f"Unsupported task_type={task_type}. Expected 'mcq' or 'answer_generation'."
        )

    def prompt_batch(
        self,
        samples: List[dict],
        instruction: str,
        max_tokens: int = 8,
        task_type: str = "mcq",
    ):
        return [
            self.prompt(
                s,
                instruction=instruction,
                max_tokens=max_tokens,
                task_type=task_type,
            )
            for s in samples
        ]
