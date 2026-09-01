import os
import re
from typing import Optional, Dict, Any
import anthropic

def extract_letter_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    s = str(text).strip().upper()
    m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-F])\b", s) or re.search(r"\b([A-F])(?=[\.\)\]:;\s]|$)", s)
    return m.group(1) if m else None


class ClaudeOpus45MCQHandler:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: Optional[int] = None,
        **kwargs,  # swallow any extra config fields
    ):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing (export ANTHROPIC_API_KEY=...).")
        chosen_model = model or model_name or "claude-opus-4-5-20250901"
        try:
            if max_retries is not None:
                self.client = anthropic.Anthropic(api_key=api_key, max_retries=int(max_retries))
            else:
                self.client = anthropic.Anthropic(api_key=api_key)
        except TypeError:
            self.client = anthropic.Anthropic(api_key=api_key)

        self.model = chosen_model

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
            if not txt:
                continue
            lines.append(f"{letter}) {txt}")

        return stem + "\n\n" + "\n".join(lines)

    @staticmethod
    def _build_ansgen_text(sample: Dict[str, Any]) -> str:
        q = (sample.get("question") or "").strip()
        if not q:
            return ""
        return q

    @staticmethod
    def _build_dialogue_text(sample: Dict[str, Any]) -> str:
        for key in ("Dialogue", "dialogue", "conversation", "context"):
            v = sample.get(key)
            if v is None:
                continue
            if isinstance(v, str):
                if v.strip():
                    return v.strip()
                continue
            if isinstance(v, list):
                lines = []
                for t in v:
                    if isinstance(t, dict):
                        role = str(t.get("role") or t.get("speaker") or "").strip()
                        text = str(t.get("text") or t.get("content") or "").strip()
                        if not text:
                            continue
                        lines.append(f"{role}: {text}" if role else text)
                    else:
                        s = str(t).strip()
                        if s:
                            lines.append(s)
                if lines:
                    return "\n".join(lines)
        return ""

    def _call_api(self, system_prompt: str, user_text: str, max_tokens: int, temperature: float) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_text},
            ],
        )

        raw = ""
        for block in getattr(resp, "content", []) or []:
            if isinstance(block, dict) and block.get("type") == "text":
                raw += block.get("text", "") or ""
            elif getattr(block, "type", None) == "text":
                raw += getattr(block, "text", "") or ""

        return raw.strip()

    def prompt(
        self,
        sample: Dict[str, Any],
        instruction: str,
        max_tokens: int = 50,
        temperature: float = 0.0,
        task_type: Optional[str] = None,
        **kwargs,
    ):
        task_type = (task_type or "mcq").strip().lower()
        system_prompt = (instruction or "").strip()

        if task_type == "mcq":
            user_text = self._build_mcq_text(sample)
            if not user_text:
                print("[ClaudeOpus45MCQ] Empty stem/options; cannot build prompt.")
                return None

            try:
                raw = self._call_api(system_prompt, user_text, max_tokens, temperature)
                if not raw:
                    return None

                print(f"[ClaudeOpus45MCQ] MCQ raw generated: {repr(raw)}")
                return extract_letter_from_text(raw)

            except Exception as e:
                print(f"[ClaudeOpus45MCQ] Error during generation: {e}")
                return None

        elif task_type == "answer_generation":
            user_text = self._build_ansgen_text(sample)
            if not user_text:
                print("[ClaudeOpus45MCQ] Empty question; cannot build answer-generation prompt.")
                return ""

            try:
                raw = self._call_api(system_prompt, user_text, max_tokens, temperature)
                if not raw:
                    return ""

                one_line = raw.split("\n")[0].strip()
                print(f"[ClaudeOpus45MCQ] Answer-gen raw (one line): {repr(one_line[:200])}")
                return one_line

            except Exception as e:
                print(f"[ClaudeOpus45MCQ] Error during generation: {e}")
                return ""

        elif task_type == "dialogue_completion":
            user_text = self._build_dialogue_text(sample)
            if not user_text:
                print("[ClaudeOpus45MCQ] Empty dialogue; cannot build dialogue-completion prompt.")
                return ""

            try:
                raw = self._call_api(system_prompt, user_text, max_tokens, temperature)
                if not raw:
                    return ""

                one_line = raw.split("\n")[0].strip()
                m = re.match(r"^\s*ANSWER\s*[:=]\s*(.*)$", one_line, flags=re.IGNORECASE)
                if m:
                    one_line = m.group(1).strip()

                print(f"[ClaudeOpus45MCQ] Dialogue-completion raw (one line): {repr(one_line[:200])}")
                return one_line

            except Exception as e:
                print(f"[ClaudeOpus45MCQ] Error during generation: {e}")
                return ""

        else:
            raise ValueError(
                f"Unsupported task_type={task_type}. "
                "Expected 'mcq', 'answer_generation', or 'dialogue_completion'."
            )
