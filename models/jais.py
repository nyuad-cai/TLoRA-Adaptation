import os
import re
import torch
from typing import List

from transformers import AutoTokenizer, AutoModelForCausalLM

HF_CACHE = "/scratch/ca2627/huggingface"
os.environ["HF_HOME"] = HF_CACHE

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class Jais2ChatMCQHandler:
    def __init__(
        self,
        model_name: str = "inceptionai/Jais-2-8B-Chat",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
        torch_dtype=torch.bfloat16,
        use_fast: bool = True,
    ):
        print(f"[Jais2Chat] Handler file: {__file__}")
        print(f"[Jais2Chat] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[Jais2Chat] cache_dir={cache_dir}")
        print(f"[Jais2Chat] offline={offline}")

        self.model_name = model_name
        self.cache_dir = cache_dir or HF_CACHE
        self.offline = offline
        self.torch_dtype = torch_dtype

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Offline toggle (same pattern as your other handlers)
        self.local_files_only = bool(self.offline)
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

        num_gpus = torch.cuda.device_count()
        print(f"[Jais2Chat] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for Jais-2-8B-Chat.")

        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated"
            )

        print("[Jais2Chat] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            use_fast=use_fast,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        print(f"[Jais2Chat] tokenizer.chat_template available? {self.has_chat_template}")

        print("[Jais2Chat] Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        print("[Jais2Chat] Model loaded.")

        if hasattr(self.model, "hf_device_map"):
            print("[Jais2Chat] Model device distribution:")
            for layer, dev in self.model.hf_device_map.items():
                print(f"  {layer}: {dev}")

        for i in range(num_gpus):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserv = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i} after load: {alloc:.2f}GB allocated, {reserv:.2f}GB reserved")

    @staticmethod
    def _build_mcq_text(sample: dict) -> str:
        """Build an MCQ block with stem + lettered options."""
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
            if txt:
                lines.append(f"{letter}) {txt}")

        if not lines:
            return stem
        return stem + "\n\n" + "\n".join(lines)

    @staticmethod
    def _build_ansgen_text(sample: dict) -> str:
        """Answer-generation prompt: question only, no options."""
        q = (sample.get("question") or "").strip()
        if not q:
            return ""
        return q  # IMPORTANT: no options

    @staticmethod
    def _mcq_system_text(instruction: str) -> str:
        instruction = (instruction or "").strip()
        base = (
            "You are a medical assistant. Answer the multiple-choice question.\n"
            "Return only one letter (A-F) in the format: Answer: X."
        )
        if instruction:
            base += f"\n\nINSTRUCTION: {instruction}"
        return base

    @staticmethod
    def _ansgen_system_text(instruction: str) -> str:
        instruction = (instruction or "").strip()
        base = (
            "You are a medical assistant. Answer the question concisely in a single line. "
            "Do not include options, explanations, or additional formatting."
        )
        if instruction:
            base += f"\n\nINSTRUCTION: {instruction}"
        return base

    def _plain_prompt(self, system_prompt: str, user_text: str, suffix: str = "Answer:") -> str:
        return (
            f"{system_prompt}\n\n"
            f"QUESTION:\n{user_text}\n\n"
            f"{suffix}"
        )

    def _generate(self, system_prompt: str, user_text: str, max_tokens: int,
                  plain_suffix: str = "Answer:") -> str:
        """Shared generation path for MCQ and answer-generation, returns decoded text"""
        torch.cuda.empty_cache()

        if self.has_chat_template:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            prompt_str = self._plain_prompt(system_prompt, user_text, suffix=plain_suffix)
            inputs = self.tokenizer(
                prompt_str,
                return_tensors="pt",
                add_special_tokens=True,
            )

        if isinstance(inputs, dict) and "token_type_ids" in inputs:
            inputs.pop("token_type_ids", None)
        elif hasattr(inputs, "pop") and "token_type_ids" in getattr(inputs, "keys", lambda: [])():
            inputs.pop("token_type_ids", None)

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = generation[0][input_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def prompt(
        self,
        sample: dict,
        instruction: str,
        max_tokens: int = 12,
        task_type: str = "mcq",
    ):
        task_type = (task_type or "mcq").strip().lower()

        if task_type == "mcq":
            user_text = self._build_mcq_text(sample)
            if not user_text:
                print("[Jais2Chat] Empty stem/options; cannot build MCQ prompt.")
                return None

            system_prompt = self._mcq_system_text(instruction)

            try:
                raw_text = self._generate(
                    system_prompt=system_prompt,
                    user_text=user_text,
                    max_tokens=max_tokens,
                    plain_suffix="Answer:",
                )
            except Exception as e:
                print("[Jais2Chat] Error during MCQ generation:", e)
                return None
            finally:
                torch.cuda.empty_cache()

            if not raw_text:
                return None

            print(f"[Jais2Chat] MCQ raw generated: {repr(raw_text)}")
            upper = raw_text.upper()

            m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
            if m:
                return m.group(1)
            m = re.search(r"(?:الإجابة|الاجابة)\s*[:=]\s*([A-F])", raw_text)
            if m:
                return m.group(1).upper()
            m = re.search(r"\b([A-F])\b", upper)
            if m:
                return m.group(1)

            print("[Jais2Chat] Could not extract a clean letter.")
            return None

        if task_type == "answer_generation":
            user_text = self._build_ansgen_text(sample)
            if not user_text:
                print("[Jais2Chat] Empty question; cannot build answer-generation prompt.")
                return ""

            system_prompt = self._ansgen_system_text(instruction)

            try:
                raw_text = self._generate(
                    system_prompt=system_prompt,
                    user_text=user_text,
                    max_tokens=max_tokens,
                    plain_suffix="Answer:",
                )
            except Exception as e:
                print("[Jais2Chat] Error during answer-generation:", e)
                return ""
            finally:
                torch.cuda.empty_cache()

            if not raw_text:
                return ""

            cleaned = re.sub(r"^\s*(?:answer|الإجابة|الاجابة)\s*[:=]\s*", "", raw_text, flags=re.IGNORECASE)
            one_line = cleaned.split("\n")[0].strip()
            print(f"[Jais2Chat] Answer-gen raw (one line): {repr(one_line[:200])}")
            return one_line

        raise ValueError(
            f"Unsupported task_type={task_type!r}. Expected 'mcq' or 'answer_generation'."
        )

    def prompt_batch(
        self,
        samples: List[dict],
        instruction: str,
        max_tokens: int = 128,
        task_type: str = "mcq",
    ):
        return [
            self.prompt(s, instruction=instruction, max_tokens=max_tokens, task_type=task_type)
            for s in samples
        ]

