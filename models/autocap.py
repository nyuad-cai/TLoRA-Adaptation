import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import Mistral3ForConditionalGeneration

HF_CACHE = "ADD PATH"
os.environ["HF_HOME"] = HF_CACHE

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class AutoCAPMCQHandler:
    """
    For each sample:
      1. Select top-K reasoning languages from candidate pool
      2. Assign weights to selected languages
      3. Reason in each selected language
      4. Aggregate with weighted vote over option letters
    """
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        cache_dir: str = HF_CACHE,
        offline: bool = True,
        candidate_languages: Optional[List[str]] = None,
        top_k_languages: int = 3,
        selection_max_tokens: int = 128,
        weight_max_tokens: int = 128,
        reasoning_max_tokens: int = 96,
        do_sample: bool = False,
    ):
        print(f"[AutoCAPMCQ] Handler file: {__file__}")
        print(f"[AutoCAPMCQ] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[AutoCAPMCQ] cache_dir={cache_dir}")
        print(f"[AutoCAPMCQ] offline={offline}")

        self.model_name = model_name
        self.cache_dir = cache_dir or HF_CACHE
        self.offline = bool(offline)
        self.top_k_languages = int(top_k_languages)
        self.selection_max_tokens = int(selection_max_tokens)
        self.weight_max_tokens = int(weight_max_tokens)
        self.reasoning_max_tokens = int(reasoning_max_tokens)
        self.do_sample = bool(do_sample)

        self.candidate_languages = candidate_languages or [
            "Arabic",
            "English",
            "French",
        ]

        if self.top_k_languages < 1:
            raise ValueError("top_k_languages must be >= 1")
        if self.top_k_languages > len(self.candidate_languages):
            raise ValueError("top_k_languages cannot exceed number of candidate languages")

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.local_files_only = True
        if self.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

        #inspect gpus
        num_gpus = torch.cuda.device_count()
        print(f"[AutoCAPMCQ] Available GPUs: {num_gpus}")
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available for AutoCAPMCQHandler.")

        for i in range(num_gpus):
            print(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} - "
                f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated"
            )

        print("[AutoCAPMCQ] Loading tokenizer...")
        if os.path.isdir(self.model_name):
            self.tokenizer = MistralTokenizer.from_file(os.path.join(self.model_name, "tekken.json"))
        else:
            self.tokenizer = MistralTokenizer.from_hf_hub(self.model_name)

        print("[AutoCAPMCQ] Loading model...")
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

        print("[AutoCAPMCQ] Model loaded.")
        if hasattr(self.model, "hf_device_map"):
            print("[AutoCAPMCQ] Model device distribution:")
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
            if txt:
                lines.append(f"{letter}) {txt}")

        return stem + ("\n\n" + "\n".join(lines) if lines else "")

    @staticmethod
    def _extract_letter(text: str) -> Optional[str]:
        if not text:
            return None

        upper = str(text).strip().upper()

        m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
        if m:
            return m.group(1)

        m = re.search(r"\bFINAL\s*ANSWER\s*[:=]\s*([A-F])\b", upper)
        if m:
            return m.group(1)

        m = re.search(r"\b([A-F])\b", upper)
        if m:
            return m.group(1)

        return None

    @staticmethod
    def _safe_json_loads(text: str):
        if not text:
            return None

        s = text.strip()

        # direct parse
        try:
            return json.loads(s)
        except Exception:
            pass

        # fenced json
        m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        # first {...}
        m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        # first [...]
        m = re.search(r"(\[.*\])", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        return None

    @staticmethod
    def _normalize_weights(weights: Dict[str, float], selected_languages: List[str]) -> Dict[str, float]:
        cleaned = {}
        for lang in selected_languages:
            val = weights.get(lang, 0.0)
            try:
                val = float(val)
            except Exception:
                val = 0.0
            if val < 0:
                val = 0.0
            cleaned[lang] = val

        total = sum(cleaned.values())
        if total <= 0:
            uniform = 1.0 / max(len(selected_languages), 1)
            return {lang: uniform for lang in selected_languages}

        return {lang: cleaned[lang] / total for lang in selected_languages}

    def _generate(self, system_prompt: str, user_text: str, max_tokens: int) -> str:
        messages = [
            {"role": "system", "content": (system_prompt or "").strip()},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]

        try:
            torch.cuda.empty_cache()

            req = ChatCompletionRequest(messages=messages)
            tokenized = self.tokenizer.encode_chat_completion(req)

            input_ids = torch.tensor([tokenized.tokens], dtype=torch.long, device=self.model.device)
            attention_mask = torch.ones_like(input_ids)

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=self.do_sample,
                )

            if outputs is None or outputs.shape[0] == 0:
                return ""

            input_len = len(tokenized.tokens)
            gen_ids = outputs[0][input_len:]
            raw_text = self.tokenizer.decode(gen_ids).strip()
            return raw_text or ""

        finally:
            torch.cuda.empty_cache()

    def _build_language_selection_prompts(self, mcq_text: str) -> Tuple[str, str]:
        system_prompt = (
            "You are an expert planner for multilingual medical reasoning.\n"
            "Your job is to choose the best reasoning languages for solving an Arabic medical multiple-choice question.\n"
            "Prioritize likely answer accuracy, not fluency or style.\n"
            "Return valid JSON only."
        )

        candidates = ", ".join(self.candidate_languages)
        user_prompt = f"""
You are given an Arabic medical multiple-choice question.

Select exactly {self.top_k_languages} reasoning languages that are most likely to produce the correct answer.

Important constraints:
- The original question remains in Arabic.
- Choose languages based on:
  1. expected medical reasoning accuracy,
  2. ability to preserve the meaning of the Arabic question,
  3. model strength in that language,
  4. usefulness of diverse but reliable reasoning paths.
- Prefer practical reasoning quality over theoretical language diversity.
- Return JSON only.
- Do not include explanations.

Candidate languages:
{candidates}

Return exactly this format:
{{"languages": ["English", "Arabic", "French"]}}

Question:
{mcq_text}
""".strip()

        return system_prompt, user_prompt

    def _build_weight_assignment_prompts(self, mcq_text: str, selected_languages: List[str]) -> Tuple[str, str]:
        system_prompt = (
            "You are an expert planner for multilingual medical reasoning.\n"
            "Your job is to assign reliability weights to reasoning languages for an Arabic medical multiple-choice question.\n"
            "Return valid JSON only."
        )

        lang_str = ", ".join(selected_languages)
        weights_stub = ", ".join([f'"{lang}": 0.0' for lang in selected_languages])

        user_prompt = f"""
You are given an Arabic medical multiple-choice question and a set of selected reasoning languages.

Assign a confidence weight between 0 and 1 to each selected language.
A higher weight means reasoning in that language is more likely to produce the correct answer.

Important constraints:
- The question remains in Arabic.
- Weights should reflect expected reasoning reliability for THIS question.
- The weights may differ meaningfully.
- Return JSON only.
- Do not include explanations.

Selected languages:
{lang_str}

Return exactly this format:
{{"weights": {{{weights_stub}}}}}

Question:
{mcq_text}
""".strip()

        return system_prompt, user_prompt

    @staticmethod
    def _build_reasoning_prompts(mcq_text: str, reasoning_language: str) -> Tuple[str, str]:
        system_prompt = (
            "You are a highly careful medical expert solving a multiple-choice question.\n"
            "The question is written in Arabic.\n"
            f"You must reason in {reasoning_language} internally in your response.\n"
            "Use the answer choices exactly as given.\n"
            "Return only one final answer letter: A, B, C, D, E, or F.\n"
            "Do not output any explanation."
        )

        user_prompt = f"""
The following medical multiple-choice question is written in Arabic.

Understand the question and answer options in Arabic.
Then reason in {reasoning_language}.
Finally, output only one letter: A, B, C, D, E, or F.

Question:
{mcq_text}
""".strip()

        return system_prompt, user_prompt

    def _select_languages(self, mcq_text: str) -> List[str]:
        system_prompt, user_prompt = self._build_language_selection_prompts(mcq_text)
        raw = self._generate(system_prompt, user_prompt, self.selection_max_tokens)

        print(f"[AutoCAPMCQ] Language selection raw: {repr(raw)}")

        parsed = self._safe_json_loads(raw)
        selected = []

        if isinstance(parsed, dict) and isinstance(parsed.get("languages"), list):
            for x in parsed["languages"]:
                if isinstance(x, str) and x in self.candidate_languages and x not in selected:
                    selected.append(x)

        if len(selected) < self.top_k_languages:
            fallback = [lang for lang in self.candidate_languages if lang not in selected]
            selected.extend(fallback[: self.top_k_languages - len(selected)])

        return selected[: self.top_k_languages]

    def _assign_weights(self, mcq_text: str, selected_languages: List[str]) -> Dict[str, float]:
        system_prompt, user_prompt = self._build_weight_assignment_prompts(mcq_text, selected_languages)
        raw = self._generate(system_prompt, user_prompt, self.weight_max_tokens)

        print(f"[AutoCAPMCQ] Weight assignment raw: {repr(raw)}")

        parsed = self._safe_json_loads(raw)
        weights = {}

        if isinstance(parsed, dict) and isinstance(parsed.get("weights"), dict):
            for lang in selected_languages:
                if lang in parsed["weights"]:
                    weights[lang] = parsed["weights"][lang]

        return self._normalize_weights(weights, selected_languages)

    def _reason_in_language(self, mcq_text: str, reasoning_language: str) -> Optional[str]:
        system_prompt, user_prompt = self._build_reasoning_prompts(mcq_text, reasoning_language)
        raw = self._generate(system_prompt, user_prompt, self.reasoning_max_tokens)

        print(f"[AutoCAPMCQ] Reasoning raw ({reasoning_language}): {repr(raw)}")

        return self._extract_letter(raw)

    @staticmethod
    def _weighted_vote(predictions: Dict[str, str], weights: Dict[str, float]) -> Optional[str]:
        score_by_letter: Dict[str, float] = {}

        for lang, pred in predictions.items():
            if pred is None:
                continue
            w = float(weights.get(lang, 0.0))
            score_by_letter[pred] = score_by_letter.get(pred, 0.0) + w

        if not score_by_letter:
            return None

        ranked = sorted(score_by_letter.items(), key=lambda x: (-x[1], x[0]))
        return ranked[0][0]

    def prompt(
        self,
        sample: dict,
        instruction: str = "",
        max_tokens: int = 12,
        task_type: str = "mcq",
        return_debug: bool = False,
    ):
        
        task_type = (task_type or "mcq").strip().lower()
        if task_type != "mcq":
            raise ValueError("AutoCAPMCQHandler only supports task_type='mcq'.")

        mcq_text = self._build_mcq_text(sample)
        if not mcq_text:
            print("[AutoCAPMCQ] Empty stem/options; cannot build MCQ prompt.")
            return None if not return_debug else {
                "final_prediction": None,
                "selected_languages": [],
                "weights": {},
                "language_predictions": {},
            }

        selected_languages = self._select_languages(mcq_text)
        weights = self._assign_weights(mcq_text, selected_languages)

        language_predictions: Dict[str, str] = {}
        for lang in selected_languages:
            pred = self._reason_in_language(mcq_text, lang)
            if pred is not None:
                language_predictions[lang] = pred

        final_prediction = self._weighted_vote(language_predictions, weights)

        debug = {
            "final_prediction": final_prediction,
            "selected_languages": selected_languages,
            "weights": weights,
            "language_predictions": language_predictions,
        }

        print(f"[AutoCAPMCQ] Debug: {debug}")

        return debug if return_debug else final_prediction

    def prompt_batch(
        self,
        samples: List[dict],
        instruction: str = "",
        max_tokens: int = 12,
        task_type: str = "mcq",
    ):
        return [
            self.prompt(
                s,
                instruction=instruction,
                max_tokens=max_tokens,
                task_type=task_type,
                return_debug=False,
            )
            for s in samples
        ]
