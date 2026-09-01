"""
rssources needed to load & run 2× 80 = 160 GiB of VRAM, which comfortably fits Mixtral-8x7B bf16 (~93 GiB weights)
"""

import gc
import os
import re
from typing import Optional, Dict, Any, List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_CACHE = "/scratch/ca2627/huggingface"
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LETTER_SET = {"A", "B", "C", "D", "E", "F"}

_MIXTRAL_8X7B_BF16_GB = 93.0
_MIXTRAL_8X7B_4BIT_GB = 24.0

def extract_letter_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    s = str(text).strip().upper()
    patterns = [
        r"\bANSWER\s*[:=]\s*([A-F])\b",
        r"\bCORRECT\s+ANSWER\s*[:=]?\s*([A-F])\b",
        r"\bOPTION\s*([A-F])\b",
        r"^\s*([A-F])\s*$",
        r"\b([A-F])\b",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return m.group(1)
    return None

def _flash_attention_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def _auto_per_gpu_cap(headroom_frac: float = 0.90) -> Dict[int, str]:
    mm: Dict[int, str] = {}
    for i in range(torch.cuda.device_count()):
        total_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        cap_gb = max(1.0, total_gb * headroom_frac)
        mm[i] = f"{int(cap_gb)}GiB"
    return mm


class BiMediXMCQHandler:
    def __init__(
        self,
        model_name: str = "BiMediX/BiMediX-Bi",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
        use_plain_prompt: bool = False,
        max_new_tokens_cap: int = 12,
        gen_max_new_tokens_cap: int = 256,
        temperature: float = 0.0,
        use_cache: bool = True,
        per_gpu_memory: Optional[str] = None,  # None -> auto-detect
        headroom_frac: float = 0.90,
        attn_implementation: Optional[str] = None,
        use_fast_tokenizer: bool = True,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or HF_CACHE
        self.offline = bool(offline)
        self.use_plain_prompt = bool(use_plain_prompt)
        self.max_new_tokens_cap = int(max_new_tokens_cap)
        self.gen_max_new_tokens_cap = int(gen_max_new_tokens_cap)
        self.default_temperature = float(temperature)
        self.use_cache = bool(use_cache)

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.local_files_only = self.offline
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        num_gpus = torch.cuda.device_count()
        print(f"[BiMediX] visible GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available.")
        total_visible_gb = 0.0
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            total_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            total_visible_gb += total_gb
            print(f"  GPU {i}: {name} ({total_gb:.0f} GiB)")

        if per_gpu_memory is None:
            max_memory = _auto_per_gpu_cap(headroom_frac=headroom_frac)
        else:
            max_memory = {i: per_gpu_memory for i in range(num_gpus)}
        print(f"[BiMediX] max_memory = {max_memory}  (no CPU offload)")

        def _parse_gib(s: str) -> float:
            m = re.match(r"\s*([\d.]+)\s*GiB", s, flags=re.IGNORECASE)
            return float(m.group(1)) if m else 0.0

        usable_gb = sum(_parse_gib(v) for v in max_memory.values())
        if load_in_4bit:
            est_weights_gb = _MIXTRAL_8X7B_4BIT_GB
            precision_label = "4-bit"
        elif load_in_8bit:
            est_weights_gb = _MIXTRAL_8X7B_BF16_GB / 2
            precision_label = "8-bit"
        else:
            est_weights_gb = _MIXTRAL_8X7B_BF16_GB
            precision_label = "bf16"

        print(f"[BiMediX] estimated weight size ({precision_label}) ~ {est_weights_gb:.0f} GiB | "
              f"usable across GPUs ~ {usable_gb:.0f} GiB")
        if usable_gb < est_weights_gb * 1.05:  # ~5% slack for KV/activations
            raise RuntimeError(
                f"[BiMediX] Not enough GPU memory.\n"
                f"  model weights ({precision_label}): ~{est_weights_gb:.0f} GiB\n"
                f"  visible GPU budget             : ~{usable_gb:.0f} GiB\n"
                f"Options:\n"
                f"  (A) Expose more GPUs  -> CUDA_VISIBLE_DEVICES=0,1,2,3\n"
                f"  (B) Quantize          -> load_in_4bit=True\n"
                f"  (C) Raise per_gpu_memory only if your GPUs actually have more VRAM."
            )

        quant_config = None
        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as e:
                raise RuntimeError(
                    "bitsandbytes / BitsAndBytesConfig not available. "
                    "Install with `pip install bitsandbytes>=0.43`."
                ) from e
            if load_in_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                print("[BiMediX] Using 4-bit NF4 + double-quant (compute dtype: bf16).")
            else:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                print("[BiMediX] Using 8-bit quantization.")

        print("[BiMediX] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            use_fast=use_fast_tokenizer,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # left padding for correct batched decoder-only generation
        self.tokenizer.padding_side = "left"

        if attn_implementation is None:
            attn_implementation = "flash_attention_2" if _flash_attention_available() else "sdpa"
        print(f"[BiMediX] attn_implementation = {attn_implementation}")

        print("[BiMediX] Loading model (sharded)...")
        load_kwargs: Dict[str, Any] = dict(
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
        
        try:
            if quant_config is None:
                load_kwargs["dtype"] = torch.bfloat16
            else:
                load_kwargs["quantization_config"] = quant_config
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        except TypeError:
            # Fallback for older transformers that only accept torch_dtype
            load_kwargs.pop("dtype", None)
            if quant_config is None:
                load_kwargs["torch_dtype"] = torch.bfloat16
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

        self.model.eval()

        dmap = getattr(self.model, "hf_device_map", {}) or {}
        bad = [(k, v) for k, v in dmap.items() if v in ("cpu", "disk")]
        if bad:
            print("[BiMediX][WARN] Some modules are NOT on GPU:")
            for k, v in bad[:10]:
                print(f"   {k} -> {v}")
            print("   Add GPUs or enable load_in_4bit=True.")
        else:
            print("[BiMediX] All modules on GPU.")

        try:
            self.model.config.use_cache = self.use_cache
        except Exception:
            pass
        try:
            self.model.config.output_router_logits = False
            print("[BiMediX] Disabled output_router_logits for inference.")
        except Exception as e:
            print(f"[BiMediX] Could not disable output_router_logits: {e}")

        self.has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        print(f"[BiMediX] chat_template? {self.has_chat_template}")

        embed_device = None
        for k, v in dmap.items():
            if "embed" in k.lower():
                embed_device = v
                break
        if embed_device is None:
            embed_device = next(self.model.parameters()).device
        self.input_device = torch.device(
            f"cuda:{embed_device}" if isinstance(embed_device, int) else embed_device
        )
        print(f"[BiMediX] input_device: {self.input_device}")

        self.stop_token_ids = []
        if self.tokenizer.eos_token_id is not None:
            self.stop_token_ids.append(self.tokenizer.eos_token_id)

    @staticmethod
    def _build_mcq_text(sample: Dict[str, Any]) -> str:
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
            val = option_map.get(letter)
            if val is None:
                continue
            val = str(val).strip()
            if val:
                lines.append(f"{letter}) {val}")
        return stem + ("\n\n" + "\n".join(lines) if lines else "")

    @staticmethod
    def _build_ansgen_text(sample: Dict[str, Any]) -> str:
        return (sample.get("question") or "").strip()

    @staticmethod
    def _strict_system_prompt(extra_instruction: str = "") -> str:
        base = (
            "You are a bilingual medical multiple-choice assistant.\n"
            "Return exactly ONE uppercase letter from A, B, C, D, E, or F.\n"
            "Do not explain.\n"
            "Do not repeat the question.\n"
            "Do not translate.\n"
            "Do not output any words other than the single answer letter.\n"
        )
        extra_instruction = (extra_instruction or "").strip()
        return base + (f"\nAdditional instruction:\n{extra_instruction}" if extra_instruction else "")

    @staticmethod
    def _ansgen_system_prompt(extra_instruction: str = "") -> str:
        base = (
            "You are a bilingual medical assistant.\n"
            "Answer the question concisely in the same language as the question.\n"
            "Output only the answer text. Do not repeat the question. Do not add explanations.\n"
        )
        extra_instruction = (extra_instruction or "").strip()
        return base + (f"\nAdditional instruction:\n{extra_instruction}" if extra_instruction else "")

    def _build_plain_prompt_mcq(self, system_prompt: str, user_text: str) -> str:
        return (
            f"{system_prompt}\n\n"
            "Question:\n"
            f"{user_text}\n\n"
            "Final answer (one letter only):"
        )

    def _build_plain_prompt_ansgen(self, system_prompt: str, user_text: str) -> str:
        return (
            f"{system_prompt}\n\n"
            "Question:\n"
            f"{user_text}\n\n"
            "Answer:"
        )

    def _final_prompt_string(self, system_prompt: str, user_text: str, kind: str) -> str:
        if (not self.use_plain_prompt) and self.has_chat_template:
            try:
                merged_user = f"{system_prompt}\n\n{user_text}"
                messages = [{"role": "user", "content": merged_user}]
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception as e:
                print(f"[BiMediX] chat_template failed → plain prompt ({e})")
        if kind == "mcq":
            return self._build_plain_prompt_mcq(system_prompt, user_text)
        return self._build_plain_prompt_ansgen(system_prompt, user_text)

    def _tokenize_batch(self, prompt_strs: List[str]):
        bos = self.tokenizer.bos_token
        if bos and all(isinstance(s, str) and s.startswith(bos) for s in prompt_strs):
            add_special = False
        else:
            add_special = True
        # Pick a safe max_length to silence the transformers truncation warning.
        max_len = getattr(self.tokenizer, "model_max_length", None)
        if not max_len or max_len > 1_000_000:
            max_len = 4096
        enc = self.tokenizer(
            prompt_strs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=add_special,
        )
        return {k: v.to(self.input_device) for k, v in enc.items()}

    def _generate(self, enc: Dict[str, torch.Tensor], max_new_tokens: int, temperature: float):
        gen_kwargs = dict(
            **enc,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(temperature and temperature > 0),
            top_p=1.0,
            use_cache=self.use_cache,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature and temperature > 0:
            gen_kwargs["temperature"] = float(temperature)
        with torch.inference_mode():
            return self.model.generate(**gen_kwargs)

    def _decode_new_tokens(self, output_ids: torch.Tensor, enc: Dict[str, torch.Tensor]) -> List[str]:
        input_len = enc["input_ids"].shape[1]
        gen_ids = output_ids[:, input_len:]
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    def _generate_once_mcq(self, sample, instruction, max_tokens, temperature, plain_retry=False):
        user_text = self._build_mcq_text(sample)
        if not user_text:
            return None, ""
        system_prompt = self._strict_system_prompt(instruction)
        original_plain = self.use_plain_prompt
        if plain_retry:
            self.use_plain_prompt = True
        try:
            prompt_str = self._final_prompt_string(system_prompt, user_text, "mcq")
            enc = self._tokenize_batch([prompt_str])
            out = self._generate(
                enc,
                max_new_tokens=min(int(max_tokens), self.max_new_tokens_cap),
                temperature=temperature,
            )
            raw_text = self._decode_new_tokens(out, enc)[0].strip()
            return extract_letter_from_text(raw_text), raw_text
        finally:
            self.use_plain_prompt = original_plain

    @staticmethod
    def _clean_ansgen_output(raw_text: str) -> str:
        if not raw_text:
            return ""
        s = raw_text.strip()
        lead_patterns = [
            r"^\s*ANSWER\s*[:=]\s*",
            r"^\s*الإجابة\s*[:=]?\s*",
            r"^\s*الجواب\s*[:=]?\s*",
        ]
        for pat in lead_patterns:
            s = re.sub(pat, "", s, flags=re.IGNORECASE).strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("\"", "'", "”", "“"):
            s = s[1:-1].strip()
        return s

    def _generate_once_ansgen(self, sample, instruction, max_tokens, temperature, plain_retry=False):
        user_text = self._build_ansgen_text(sample)
        if not user_text:
            return "", ""
        system_prompt = self._ansgen_system_prompt(instruction)
        requested = int(max_tokens) if max_tokens else 0
        effective_max = requested if requested > 32 else self.gen_max_new_tokens_cap

        original_plain = self.use_plain_prompt
        if plain_retry:
            self.use_plain_prompt = True
        try:
            prompt_str = self._final_prompt_string(system_prompt, user_text, "ansgen")
            enc = self._tokenize_batch([prompt_str])
            out = self._generate(enc, max_new_tokens=effective_max, temperature=temperature)
            raw_text = self._decode_new_tokens(out, enc)[0]
            cleaned = self._clean_ansgen_output(raw_text)
            return cleaned, raw_text
        finally:
            self.use_plain_prompt = original_plain

    def prompt(
        self,
        sample: Dict[str, Any],
        instruction: str,
        max_tokens: int = 12,
        temperature: float = 0.0,
        task_type=None,
        **kwargs,
    ):
        task_type = (task_type or "mcq").strip().lower() if task_type else "mcq"
        sample_id = sample.get("id", "N/A")
        temp = temperature if temperature is not None else self.default_temperature

        if task_type == "mcq":
            pred, raw = self._generate_once_mcq(sample, instruction, max_tokens, temp, False)
            print(f"\n[BiMediX] id={sample_id} raw={raw!r} pred={pred}", flush=True)
            if pred:
                return pred
            pred2, raw2 = self._generate_once_mcq(sample, instruction, max_tokens, temp, True)
            print(f"[BiMediX][retry-plain] id={sample_id} raw={raw2!r} pred={pred2}", flush=True)
            return pred2

        elif task_type == "answer_generation":
            out, raw = self._generate_once_ansgen(sample, instruction, max_tokens, temp, False)
            print(f"\n[BiMediX] id={sample_id} ansgen_raw={raw[:300]!r} cleaned={out[:300]!r}",
                  flush=True)
            if out:
                return out
            out2, raw2 = self._generate_once_ansgen(sample, instruction, max_tokens, temp, True)
            print(f"[BiMediX][retry-plain] id={sample_id} "
                  f"ansgen_raw={raw2[:300]!r} cleaned={out2[:300]!r}", flush=True)
            return out2

        else:
            raise ValueError(
                f"Unsupported task_type={task_type}. Expected 'mcq' or 'answer_generation'."
            )

    def prompt_batch(
        self,
        samples: List[Dict[str, Any]],
        instruction: str,
        max_tokens: int = 12,
        temperature: float = 0.0,
        task_type: str = "mcq",
        batch_size: int = 8,
        **kwargs,
    ) -> List[Union[str, None]]:
        task_type = (task_type or "mcq").strip().lower()
        temp = temperature if temperature is not None else self.default_temperature

        if task_type == "mcq":
            system_prompt = self._strict_system_prompt(instruction)
            builder = self._build_mcq_text
            kind = "mcq"
            max_new = min(int(max_tokens), self.max_new_tokens_cap)
        elif task_type == "answer_generation":
            system_prompt = self._ansgen_system_prompt(instruction)
            builder = self._build_ansgen_text
            kind = "ansgen"
            requested = int(max_tokens) if max_tokens else 0
            max_new = requested if requested > 32 else self.gen_max_new_tokens_cap
        else:
            raise ValueError(f"Unsupported task_type={task_type}")

        def _build_prompts(use_plain: bool) -> List[Optional[str]]:
            original = self.use_plain_prompt
            self.use_plain_prompt = use_plain
            try:
                out_list: List[Optional[str]] = []
                for s in samples:
                    user_text = builder(s)
                    out_list.append(
                        self._final_prompt_string(system_prompt, user_text, kind) if user_text else None
                    )
                return out_list
            finally:
                self.use_plain_prompt = original

        debug = os.environ.get("BIMEDIX_DEBUG") == "1"

        def _run_pass(prompt_strs: List[Optional[str]], only_indices: Optional[List[int]] = None):
            """Run a pass over (a subset of) samples; returns dict idx -> (raw, cleaned_or_letter)."""
            out_map: Dict[int, Any] = {}
            indices = only_indices if only_indices is not None else list(range(len(samples)))
            dumped = False
            for start in range(0, len(indices), batch_size):
                chunk_idx = [
                    idx for idx in indices[start:start + batch_size]
                    if prompt_strs[idx] is not None
                ]
                if not chunk_idx:
                    continue
                chunk_prompts = [prompt_strs[idx] for idx in chunk_idx]

                if debug and not dumped:
                    print("=" * 70, flush=True)
                    print("[BiMediX][DIAG] First prompt being sent to the model:", flush=True)
                    print("-" * 70, flush=True)
                    print(chunk_prompts[0], flush=True)
                    print("-" * 70, flush=True)
                    print(f"[BiMediX][DIAG] chat_template_used? "
                          f"{(not self.use_plain_prompt) and self.has_chat_template}", flush=True)
                    print(f"[BiMediX][DIAG] max_new_tokens = {max_new}", flush=True)
                    print(f"[BiMediX][DIAG] pad_token_id = {self.tokenizer.pad_token_id}, "
                          f"eos_token_id = {self.tokenizer.eos_token_id}", flush=True)
                    dumped = True

                enc = self._tokenize_batch(chunk_prompts)
                out = self._generate(enc, max_new_tokens=max_new, temperature=temp)
                decoded = self._decode_new_tokens(out, enc)

                if debug and start == 0:
                    new_ids = out[:, enc["input_ids"].shape[1]:]
                    print(f"[BiMediX][DIAG] input_ids[0][:10] = {enc['input_ids'][0][:10].tolist()}",
                          flush=True)
                    print(f"[BiMediX][DIAG] new token ids (batch 0, first row, first 20): "
                          f"{new_ids[0][:20].tolist()}", flush=True)
                    print(f"[BiMediX][DIAG] new tokens decoded WITH specials (first row): "
                          f"{self.tokenizer.decode(new_ids[0], skip_special_tokens=False)!r}",
                          flush=True)

                for i, raw in zip(chunk_idx, decoded):
                    raw = raw or ""
                    if task_type == "mcq":
                        out_map[i] = (raw, extract_letter_from_text(raw.strip()))
                    else:
                        cleaned = self._clean_ansgen_output(raw)
                        out_map[i] = (raw, cleaned)
            return out_map

        prompt_strs_1 = _build_prompts(use_plain=self.use_plain_prompt)
        pass1 = _run_pass(prompt_strs_1)

        results: List[Union[str, None]] = [None] * len(samples)
        empty_indices: List[int] = []
        for i in range(len(samples)):
            raw_cleaned = pass1.get(i)
            if raw_cleaned is None:
                results[i] = None if task_type == "mcq" else ""
                continue
            raw, value = raw_cleaned
            results[i] = value
            is_empty = (value is None) if task_type == "mcq" else (not value)
            if is_empty:
                empty_indices.append(i)
            if debug and task_type == "answer_generation":
                sid = samples[i].get("id", "N/A")
                print(f"[BiMediX] id={sid} ansgen_raw={raw[:300]!r} cleaned={value[:300]!r}",
                      flush=True)

        if empty_indices and not self.use_plain_prompt:
            print(f"[BiMediX] {len(empty_indices)} empty outputs → retrying with plain prompt",
                  flush=True)
            prompt_strs_2 = _build_prompts(use_plain=True)
            pass2 = _run_pass(prompt_strs_2, only_indices=empty_indices)
            for i, (raw, value) in pass2.items():
                results[i] = value
                if debug and task_type == "answer_generation":
                    sid = samples[i].get("id", "N/A")
                    print(f"[BiMediX][retry-plain] id={sid} "
                          f"ansgen_raw={raw[:300]!r} cleaned={value[:300]!r}", flush=True)

        return results
