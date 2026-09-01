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

class MedGemma27BMCQHandler:
    def __init__(
        self,
        model_name: str = "google/medgemma-27b-text-it",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
    ):
        print(f"[MedGemma27BMCQ] Handler file: {__file__}")
        print(f"[MedGemma27BMCQ] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[MedGemma27BMCQ] cache_dir={cache_dir}")
        print(f"[MedGemma27BMCQ] offline={offline}")

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
        print(f"[MedGemma27BMCQ] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for MedGemma27BMCQ.")

        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated"
            )

        print("[MedGemma27BMCQ] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[MedGemma27BMCQ] Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

        print("[MedGemma27BMCQ] Model loaded.")
        if hasattr(self.model, "hf_device_map"):
            print("[MedGemma27BMCQ] Model device distribution:")
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
        q = (sample.get("question") or "").strip()
        if not q:
            return ""
        return q  # IMPORTANT: no options

    @staticmethod
    def _build_dialogue_text(sample: dict) -> str:
        raw = (
            sample.get("dialogue")
            or sample.get("conversation")
            or sample.get("context")
            or sample.get("turns")
            or sample.get("question")
        )
        if raw is None:
            return ""

        if isinstance(raw, str):
            return raw.strip()

        if isinstance(raw, list) and len(raw) > 0:
            if bool(sample.get("last_turn_included", False)):
                raw = raw[:-1]

            lines = []
            for t in raw:
                if isinstance(t, dict):
                    role = str(t.get("role") or t.get("speaker") or "").strip()
                    text = str(
                        t.get("text")
                        or t.get("content")
                        or t.get("utterance")
                        or ""
                    ).strip()
                    if not text:
                        continue
                    lines.append(f"{role}: {text}" if role else text)
                else:
                    s = str(t).strip()
                    if s:
                        lines.append(s)
            return "\n".join(lines)

        return ""

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
                print("[MedGemma] Empty stem/options; cannot build MCQ prompt.")
                return None

            system_prompt = (instruction or "").strip()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]

            try:
                torch.cuda.empty_cache()

                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                    )

                gen_ids = generation[0][input_len:]
                raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            except Exception as e:
                print("[MedGemma] Error during MCQ generation:", e)
                return None
            finally:
                torch.cuda.empty_cache()

            if not raw_text:
                return None

            print(f"[MedGemma] MCQ raw generated: {repr(raw_text)}")
            upper = raw_text.strip().upper()

            if upper in ("A", "B", "C", "D", "E", "F"):
                return upper

            m = re.match(r"^([A-F])\s*[.)\s]?", upper)
            if m:
                return m.group(1)

            m = re.search(r"\b([A-F])\b", upper)
            if m:
                return m.group(1)
            m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
            if m:
                return m.group(1)

            print("[MedGemma] Could not extract a clean letter.")
            return None

        if task_type == "answer_generation":
            user_text = self._build_ansgen_text(sample)
            if not user_text:
                print("[MedGemma] Empty question; cannot build answer-generation prompt.")
                return ""

            system_prompt = (instruction or "").strip()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]

            try:
                torch.cuda.empty_cache()

                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                    )

                gen_ids = generation[0][input_len:]
                raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            except Exception as e:
                print("[MedGemma] Error during answer-generation:", e)
                return ""
            finally:
                torch.cuda.empty_cache()

            if not raw_text:
                return ""

            one_line = raw_text.split("\n")[0].strip()
            print(f"[MedGemma] Answer-gen raw (one line): {repr(one_line[:200])}")
            return one_line

        if task_type in ("dialogue_completion", "dialogue", "next_turn"):
            user_text = self._build_dialogue_text(sample)
            if not user_text:
                print("[MedGemma] Empty dialogue; cannot build dialogue-completion prompt.")
                return ""

            system_prompt = (instruction or "").strip()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]

            try:
                torch.cuda.empty_cache()

                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generation = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                    )

                gen_ids = generation[0][input_len:]
                raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            except Exception as e:
                print("[MedGemma] Error during dialogue-completion:", e)
                return ""
            finally:
                torch.cuda.empty_cache()

            if not raw_text:
                return ""
                
            one_line = raw_text.split("\n")[0].strip()
            m = re.match(r"^\s*ANSWER\s*[:=]\s*(.*)$", one_line, flags=re.IGNORECASE)
            if m:
                one_line = m.group(1).strip()

            print(f"[MedGemma] Dialogue-completion raw (one line): {repr(one_line[:200])}")
            return one_line

        raise ValueError(
            f"Unsupported task_type={task_type}. "
            "Expected 'mcq', 'answer_generation', or 'dialogue_completion'."
        )

    def prompt_batch(
        self,
        samples: List[dict],
        instruction: str,
        max_tokens: int = 8,
        task_type: str = "mcq",
    ):
        return [self.prompt(s, instruction=instruction, max_tokens=max_tokens, task_type=task_type) for s in samples]
