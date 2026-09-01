import os
import re
import time
from typing import Optional, Dict, Any

from google import genai
from google.genai import types

LETTER_SET = {"A", "B", "C", "D", "E", "F"}

class Gemini3ProHandler:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        min_request_interval: float = 2.6,
        max_retries: int = 6,
        **kwargs,
    ):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY missing.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        self.min_request_interval = float(min_request_interval)
        self.max_retries = int(max_retries)
        self._last_call_time = 0.0

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
        return (
            "Choose the single best answer.\n"
            "Return exactly one uppercase letter from this set only: A, B, C, D, E, F.\n"
            "Do not explain.\n"
            "Do not output JSON.\n"
            "Do not output any words.\n\n"
            f"{stem}\n\n" + "\n".join(lines)
        )

    @staticmethod
    def _build_ansgen_text(sample: Dict[str, Any]) -> str:
        return (sample.get("question") or "").strip()

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

    def _respect_rate_limit(self):
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

    @staticmethod
    def _extract_label(resp) -> Optional[str]:
        text = getattr(resp, "text", None)
        if text:
            t = str(text).strip().upper().strip('"').strip()
            if t in LETTER_SET:
                return t
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            for part in parts or []:
                txt = getattr(part, "text", None)
                if txt:
                    t = str(txt).strip().upper().strip('"').strip()
                    if t in LETTER_SET:
                        return t
        return None

    @staticmethod
    def _extract_text(resp) -> str:
        text = getattr(resp, "text", None)
        if text:
            return str(text)
        chunks = []
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            for part in parts or []:
                txt = getattr(part, "text", None)
                if txt:
                    chunks.append(str(txt))
        return "".join(chunks)

    @staticmethod
    def _debug_response(resp):
        try:
            print("[GeminiMCQ][DEBUG] prompt_feedback:", getattr(resp, "prompt_feedback", None), flush=True)
            candidates = getattr(resp, "candidates", None) or []
            print(f"[GeminiMCQ][DEBUG] num_candidates={len(candidates)}", flush=True)
            for i, cand in enumerate(candidates):
                print(f"[GeminiMCQ][DEBUG] candidate[{i}].finish_reason={getattr(cand, 'finish_reason', None)}", flush=True)
                print(f"[GeminiMCQ][DEBUG] candidate[{i}].safety_ratings={getattr(cand, 'safety_ratings', None)}", flush=True)
            print(f"[GeminiMCQ][DEBUG] raw_text={getattr(resp, 'text', None)!r}", flush=True)
            print(f"[GeminiMCQ][DEBUG] usage_metadata={getattr(resp, 'usage_metadata', None)}", flush=True)
        except Exception as e:
            print(f"[GeminiMCQ][DEBUG] response inspection failed: {e}", flush=True)

    def _generate(self, user_text: str, cfg: "types.GenerateContentConfig"):
        last_err = None
        for attempt in range(self.max_retries):
            try:
                self._respect_rate_limit()
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_text,
                    config=cfg,
                )
                self._last_call_time = time.time()
                return resp, None
            except Exception as e:
                last_err = e
                msg = str(e)
                print(f"[GeminiMCQ] Error: {msg}", flush=True)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    sleep_s = max(7.0, self.min_request_interval * (attempt + 2))
                    print(f"[GeminiMCQ] Sleeping {sleep_s:.1f}s before retry...", flush=True)
                    time.sleep(sleep_s)
                    continue
                return None, e
        return None, last_err

    def _build_gen_cfg(
        self,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        min_tokens_floor: int = 256,
    ) -> "types.GenerateContentConfig":
        gen_max_tokens = (
            int(max_tokens) if max_tokens and int(max_tokens) > 32 else min_tokens_floor
        )
        return types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=float(temperature),
            max_output_tokens=gen_max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="text/plain",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="NONE")
            ),
        )

    def prompt(
        self,
        sample,
        instruction=None,
        max_tokens=4,
        temperature=0.0,
        task_type=None,
        **kwargs,
    ):
        
        task_type = (task_type or "mcq").strip().lower()
        sample_id = sample.get("id", sample.get("Case ID", "N/A"))

        if task_type == "mcq":
            user_text = self._build_mcq_text(sample)
            if not user_text:
                return None

            print(f"\n[GeminiMCQ] Processing sample id={sample_id} (mcq)", flush=True)

            system_prompt = (
                (instruction or "").strip()
                or "You answer medical multiple-choice questions. Output exactly one uppercase letter only."
            )

            cfg = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=float(temperature),
                max_output_tokens=int(max_tokens),
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                response_mime_type="text/plain",
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="NONE")
                ),
            )

            for attempt in range(self.max_retries):
                resp, err = self._generate(user_text, cfg)
                if resp is None:
                    print(f"[GeminiMCQ] Failed after retries: {err}", flush=True)
                    return None

                print("\n==============================", flush=True)
                print("[GeminiMCQ] RAW RESPONSE:", flush=True)
                print(getattr(resp, "text", None), flush=True)
                print("==============================", flush=True)

                pred = self._extract_label(resp)
                print(f"[GeminiMCQ] PARSED PREDICTION: {pred}", flush=True)
                if pred:
                    return pred

                self._debug_response(resp)

                candidates = getattr(resp, "candidates", None) or []
                finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
                if str(finish_reason) == "MAX_TOKENS" and attempt < self.max_retries - 1:
                    cfg.max_output_tokens = max(cfg.max_output_tokens * 2, 8)
                    continue
                return None
            return None

        elif task_type == "answer_generation":
            user_text = self._build_ansgen_text(sample)
            if not user_text:
                print("[GeminiMCQ] Empty question; cannot build answer-generation prompt.", flush=True)
                return ""

            print(f"\n[GeminiMCQ] Processing sample id={sample_id} (answer_generation)", flush=True)

            system_prompt = (instruction or "").strip() or (
                "You are a medical expert. Answer the question concisely in the same language "
                "as the question. Output only the answer text, no explanation."
            )

            cfg = self._build_gen_cfg(system_prompt, temperature, max_tokens)

            resp, err = self._generate(user_text, cfg)
            if resp is None:
                print(f"[GeminiMCQ] Answer-gen failed after retries: {err}", flush=True)
                return ""

            raw = self._extract_text(resp) or ""
            raw = raw.strip()
            if not raw:
                self._debug_response(resp)
                return ""

            # Strip a leading "Answer:" / "ANSWER:" if the model echoes it.
            cleaned = re.sub(r"^\s*ANSWER\s*[:=]\s*", "", raw, flags=re.IGNORECASE).strip()
            one_line = cleaned.split("\n")[0].strip()
            print(f"[GeminiMCQ] Answer-gen raw (one line): {repr(one_line[:200])}", flush=True)
            return one_line

        elif task_type == "dialogue_completion":
            user_text = self._build_dialogue_text(sample)
            if not user_text:
                print("[GeminiMCQ] Empty dialogue; cannot build dialogue-completion prompt.", flush=True)
                return ""

            print(f"\n[GeminiMCQ] Processing sample id={sample_id} (dialogue_completion)", flush=True)

            system_prompt = (instruction or "").strip() or (
                "You are a medical expert completing a doctor-patient dialogue. "
                "Output exactly ONE line starting with ANSWER:."
            )

            cfg = self._build_gen_cfg(system_prompt, temperature, max_tokens)

            resp, err = self._generate(user_text, cfg)
            if resp is None:
                print(f"[GeminiMCQ] Dialogue-completion failed after retries: {err}", flush=True)
                return ""

            raw = self._extract_text(resp) or ""
            raw = raw.strip()
            if not raw:
                self._debug_response(resp)
                return ""

            one_line = raw.split("\n")[0].strip()
            m = re.match(r"^\s*ANSWER\s*[:=]\s*(.*)$", one_line, flags=re.IGNORECASE)
            if m:
                one_line = m.group(1).strip()

            print(f"[GeminiMCQ] Dialogue-completion raw (one line): {repr(one_line[:200])}", flush=True)
            return one_line

        else:
            raise ValueError(
                f"Unsupported task_type={task_type}. "
                "Expected 'mcq', 'answer_generation', or 'dialogue_completion'."
            )
