import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_CACHE = "/scratch/ca2627/huggingface"
os.environ["HF_HOME"] = HF_CACHE

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class Fanar19BMCQHandler:
    def __init__(
        self,
        model_name: str = "QCRI/Fanar-1-9B",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
        torch_dtype=torch.bfloat16,
    ):
        print(f"[Fanar19BMCQ] Handler file: {__file__}")
        print(f"[Fanar19BMCQ] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[Fanar19BMCQ] cache_dir={cache_dir}")
        print(f"[Fanar19BMCQ] offline={offline}")

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
        print(f"[Fanar19BMCQ] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for Fanar19BMCQ.")

        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated"
            )

        print("[Fanar19BMCQ] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[Fanar19BMCQ] Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

        print("[Fanar19BMCQ] Model loaded.")
        if hasattr(self.model, "hf_device_map"):
            print("[Fanar19BMCQ] Model device distribution:")
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

    def prompt(
        self,
        sample: dict,
        instruction: str,
        max_tokens: int = 12,
        temperature: float = 0.0,
    ):
        user_text = self._build_mcq_text(sample)
        if not user_text:
            print("[Fanar19BMCQ] Empty stem/options; cannot build prompt.")
            return None

        system_prompt = (instruction or "").strip()
        full_prompt = (
            f"{system_prompt}\n\n"
            f"{user_text}\n\n"
            "Return ONLY one letter among A, B, C, D, E, F.\n\n"
            "ANSWER:"
        )

        print("[Fanar19BMCQ] MAX TOKENS:", max_tokens)
        print("[Fanar19BMCQ] TEMPERATURE:", temperature)

        try:
            torch.cuda.empty_cache()
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                return_token_type_ids=False,
            )

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]

            do_sample = bool(temperature and temperature > 0)

            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=float(temperature) if do_sample else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            gen_ids = generation[0][input_len:]
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        except Exception as e:
            print("[Fanar19BMCQ] Error during generation:", e)
            return None
        finally:
            torch.cuda.empty_cache()

        if not raw_text:
            return None

        print(f"[Fanar19BMCQ] MCQ raw generated: {repr(raw_text)}")

        upper = raw_text.upper()
        m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
        if m:
            return m.group(1)
        m = re.search(r"\b([A-F])\b", upper)
        if m:
            return m.group(1)

        print("[Fanar19BMCQ] Could not extract a clean letter.")
        return None
