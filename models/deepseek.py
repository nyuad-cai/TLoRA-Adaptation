import re
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI


class DeepSeekV32MCQHandler:
    def __init__(
        self,
        api_key: str,
        model_name: str = "deepseek-chat",
        max_retries: int = 3,
        **kwargs,  # swallow offline/cache_dir/etc
    ):
        print(f"[DeepSeekV32MCQ] Using model: {model_name}")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )
        self.model_name = model_name
        self.max_retries = max_retries

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
        return stem + "\n\n" + "\n".join(lines)

    @staticmethod
    def _build_ansgen_text(sample: Dict[str, Any]) -> str:
        q = (sample.get("question") or "").strip()
        return q

    def _build_mcq_user_block(self, instruction: str, user_text: str) -> str:
        instruction = (instruction or "").strip()
        parts = []
        if instruction:
            parts.append(instruction)
        parts.append("Return only one letter (A-F) in the format: Answer: X")
        parts.append("")
        parts.append("QUESTION:")
        parts.append(user_text)
        parts.append("")
        parts.append("Answer:")
        return "\n".join(parts).strip()

    def _build_ansgen_user_block(self, instruction: str, user_text: str) -> str:
        instruction = (instruction or "").strip()
        parts = []
        if instruction:
            parts.append(instruction)
        parts.append("QUESTION:")
        parts.append(user_text)
        parts.append("")
        parts.append("Answer:")
        return "\n".join(parts).strip()

    def _extract_letter(self, text: str) -> Optional[str]:
        if not text:
            return None
        upper = text.upper()
        m = re.search(r"\bANSWER\s*[:=]\s*([A-F])\b", upper)
        if m:
            return m.group(1)
        m = re.search(r"\b([A-F])\b", upper)
        if m:
            return m.group(1)
        return None

    def _call_api(self, user_block: str, max_tokens: int) -> Optional[str]:
        last_err = None
        for i in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Follow instructions strictly."},
                        {"role": "user", "content": user_block},
                    ],
                    temperature=0.0,
                    max_tokens=int(max_tokens),
                    top_p=1.0,
                )
                content = resp.choices[0].message.content
                return (content or "").strip()
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** i, 8))
        print(f"[DeepSeekV32MCQ] API failed after retries: {last_err}")
        return None

    def prompt(
        self,
        sample: Dict[str, Any],
        instruction: str,
        task_type: Optional[str] = None,
        max_tokens: int = 12,
        **kwargs,
    ):
        task_type = (task_type or "mcq").strip().lower()

        if task_type == "mcq":
            mcq_text = self._build_mcq_text(sample)
            if not mcq_text:
                print("[DeepSeekV32MCQ] Empty question/options.")
                return None
            user_block = self._build_mcq_user_block(instruction, mcq_text)
            raw = self._call_api(user_block, max_tokens=max_tokens)
            print(f"[DeepSeekV32MCQ] MCQ raw output: {repr(raw)}")
            if not raw:
                return None
            letter = self._extract_letter(raw)
            if letter is None:
                print("[DeepSeekV32MCQ] Could not extract clean A-F letter.")
            return letter

        elif task_type == "answer_generation":
            user_text = self._build_ansgen_text(sample)
            if not user_text:
                print("[DeepSeekV32MCQ] Empty question; cannot build answer-generation prompt.")
                return ""
                
            gen_max_tokens = max_tokens if max_tokens and max_tokens > 32 else 256
            user_block = self._build_ansgen_user_block(instruction, user_text)
            raw = self._call_api(user_block, max_tokens=gen_max_tokens)
            if not raw:
                return ""
 
            cleaned = re.sub(r"^\s*ANSWER\s*[:=]\s*", "", raw, flags=re.IGNORECASE).strip()
            one_line = cleaned.split("\n")[0].strip()
            print(f"[DeepSeekV32MCQ] Answer-gen raw (one line): {repr(one_line[:200])}")
            return one_line

        else:
            raise ValueError(
                f"Unsupported task_type={task_type}. Expected 'mcq' or 'answer_generation'."
            )

    def prompt_batch(
        self,
        samples: List[Dict[str, Any]],
        instruction: str,
        task_type: Optional[str] = None,
        max_tokens: int = 12,
    ):
        results = []
        for idx, sample in enumerate(samples):
            print(f"[DeepSeekV32MCQ] Processing batch index {idx}")
            results.append(
                self.prompt(sample, instruction, task_type=task_type, max_tokens=max_tokens)
            )
        return results
