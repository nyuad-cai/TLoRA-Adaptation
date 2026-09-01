import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_CACHE = "/scratch/ca2627/huggingface"
os.environ["HF_HOME"] = HF_CACHE
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class Silma9BMCQHandler:
    def __init__(
        self,
        model_name: str = "silma-ai/SILMA-9B-Instruct-v1.0",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
        torch_dtype=torch.bfloat16,
    ):
        print(f"[SILMA-9B] Handler file: {__file__}")
        print(f"[SILMA-9B] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[SILMA-9B] cache_dir={cache_dir}")
        print(f"[SILMA-9B] offline={offline}")

        self.model_name = model_name
        self.cache_dir = cache_dir or HF_CACHE
        self.offline = offline
        self.local_files_only = bool(offline)
        self.torch_dtype = torch_dtype

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

        if torch.cuda.device_count() == 0:
            raise RuntimeError("No CUDA GPUs available.")

        print("[SILMA-9B] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            use_fast=True,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        print(f"[SILMA-9B] tokenizer.chat_template available? {self.has_chat_template}")

        print("[SILMA-9B] Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=self.torch_dtype,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )

        print("[SILMA-9B] Model loaded.")

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
            if txt:
                lines.append(f"{letter}) {txt}")

        if lines:
            return stem + "\n\n" + "\n".join(lines)
        return stem

    @staticmethod
    def _build_instruction(instruction: str) -> str:
        # If your eval passes a global system prompt, keep it.
        # Otherwise provide a safe default.
        instruction = (instruction or "").strip()
        if instruction:
            return instruction
        return "أنت مساعد طبي ذكي. أجب عن أسئلة الاختيار من متعدد بدقة."

    @staticmethod
    def _extract_letter(raw_text: str):
        if not raw_text:
            return None
        upper = raw_text.strip().upper()
        m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
        if m:
            return m.group(1)
        m = re.search(r"(?:الإجابة|الاجابة)\s*[:=]\s*([A-F])", upper)
        if m:
            return m.group(1)

        m = re.search(r"\b([A-F])\b", upper)
        if m:
            return m.group(1)

        return None

    def prompt(self, sample: dict, instruction: str, max_tokens: int = 12):
        user_text = self._build_mcq_text(sample)
        if not user_text:
            print("[SILMA-9B] Empty stem/options; cannot build prompt.")
            return None

        sys_text = self._build_instruction(instruction)

        user_msg = (
            f"{user_text}\n\n"
            "أجب بحرف واحد فقط (A-F) وبالصيغة التالية تمامًا:\n"
            "Answer: X"
        )

        try:
            torch.cuda.empty_cache()

            if self.has_chat_template:
                messages = [
                    {"role": "system", "content": sys_text},
                    {"role": "user", "content": user_msg},
                ]
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            else:
                prompt_str = (
                    f"{sys_text}\n\n"
                    f"QUESTION:\n{user_text}\n\n"
                    "Return only one letter (A-F) in the format: Answer: X\n\n"
                    "Answer:"
                )
                inputs = self.tokenizer(
                    prompt_str,
                    return_tensors="pt",
                    add_special_tokens=True,
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
            print("[SILMA-9B] Error during generation:", e)
            return None
        finally:
            torch.cuda.empty_cache()

        print(f"[SILMA-9B] MCQ raw generated: {repr(raw_text)}")
        letter = self._extract_letter(raw_text)
        if letter is None:
            print("[SILMA-9B] Could not extract a clean letter.")
        return letter
