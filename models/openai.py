# openai_handler.py
from openai import OpenAI
import re
import time
from typing import Optional, Dict, Any

def extract_letter_from_text_en(text: str) -> Optional[str]:
    if not text:
        return None
    t = str(text).strip().upper()

    m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-F])\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-F])(?=[\.\)\]:;\s]|$)", t)
    return m.group(1) if m else None


class OpenAIHandler:
    def __init__(self, api_key: str, model: str, max_retries: int = 3):
        print("[OpenAIHandler] model:", model)
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def _is_responses_model(self) -> bool:
        ml = (self.model or "").lower()
        return ml.startswith("gpt-5") or ml.startswith("o1") or ml.startswith("o3")

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
            txt = option_map.get(letter)
            if txt is None:
                continue
            txt = str(txt).strip()
            if txt:
                lines.append(f"{letter}) {txt}")

        return stem + "\n\n" + "\n".join(lines) if lines else stem

    @staticmethod
    def _build_ansgen_text(sample: Dict[str, Any]) -> str:
        q = (sample.get("question") or "").strip()
        if not q:
            return ""
        return q

    def _build_user_block(self, instruction: str, user_text: str) -> str:
        instruction = (instruction or "").strip()

        base = []
        if instruction:
            base.append(instruction)
        base.append("Return only one letter (A-F) in the format: Answer: X")
        base.append("")
        base.append("QUESTION:")
        base.append(user_text)
        base.append("")
        base.append("Answer:")
        return "\n".join(base).strip()

    def _parse_responses_text(self, resp) -> str:
        """
        Extract aggregated text from Responses API result.
        Prefer resp.output_text; if empty, walk the output array.
        """
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

        parts = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        parts.append(getattr(c, "text", ""))
        return "".join(parts).strip()

    def _with_retries(self, fn, *args, **kwargs):
        last_err = None
        for i in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                # simple backoff
                time.sleep(min(2 ** i, 8))
        raise last_err

    def prompt(
        self,
        sample: Dict[str, Any],
        instruction: str,
        task_type: str = "mcq",
        max_tokens: int = 12,
        temperature: float = 0.0,
        top_p: float = 1.0,
        reasoning_effort: str = "low",
        **kwargs,
    ):

        task_type = (task_type or "mcq").strip().lower()

        if task_type == "mcq":
            user_text = self._build_mcq_text(sample)
            if not user_text:
                print("[OpenAIHandler] Empty stem/options; cannot build prompt.")
                return None

            user_block = self._build_user_block(instruction, user_text)

            if not self._is_responses_model():
                resp = self._with_retries(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Follow instructions strictly."},
                        {"role": "user", "content": user_block},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                raw = (resp.choices[0].message.content or "").strip()
                letter = extract_letter_from_text_en(raw)
                if letter is None:
                    print(f"[OpenAIHandler] Could not extract a clean letter. Raw: {repr(raw)}")
                return letter

            payload = dict(
                model=self.model,
                instructions="Follow instructions strictly.",
                input=user_block,
                text={"format": {"type": "text"}},
                reasoning={"effort": reasoning_effort},
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            resp = self._with_retries(self.client.responses.create, **payload)
            raw = self._parse_responses_text(resp)

            # tiny fallback if empty
            if not raw:
                payload["input"] = user_block + "\n\nAnswer: "
                resp = self._with_retries(self.client.responses.create, **payload)
                raw = self._parse_responses_text(resp)

            raw = (raw or "").strip()
            letter = extract_letter_from_text_en(raw)
            if letter is None:
                print(f"[OpenAIHandler] Could not extract a clean letter. Raw: {repr(raw)}")
            return letter

        elif task_type == "answer_generation":
            user_text = self._build_ansgen_text(sample)
            if not user_text:
                print("[OpenAIHandler] Empty question; cannot build answer-generation prompt.")
                return ""

            instruction = (instruction or "").strip()
            user_block = f"{instruction}\n\nQUESTION:\n{user_text}".strip()

            if not self._is_responses_model():
                resp = self._with_retries(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Follow instructions strictly."},
                        {"role": "user", "content": user_block},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                raw = (resp.choices[0].message.content or "").strip()
                if not raw:
                    return ""
                one_line = raw.split("\n")[0].strip()
                print(f"[OpenAIHandler] Answer-gen raw (one line): {repr(one_line[:200])}")
                return one_line

            payload = dict(
                model=self.model,
                instructions="Follow instructions strictly.",
                input=user_block,
                text={"format": {"type": "text"}},
                reasoning={"effort": reasoning_effort},
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            resp = self._with_retries(self.client.responses.create, **payload)
            raw = self._parse_responses_text(resp)
            raw = (raw or "").strip()

            if not raw:
                return ""

            one_line = raw.split("\n")[0].strip()
            print(f"[OpenAIHandler] Answer-gen raw (one line): {repr(one_line[:200])}")
            return one_line

        else:
            raise ValueError(f"Unsupported task_type={task_type}. Expected 'mcq' or 'answer_generation'.")
