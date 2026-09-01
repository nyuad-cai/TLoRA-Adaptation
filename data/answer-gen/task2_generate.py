#!/usr/bin/env python3
"""
task2_generate.py — Two-stage pipeline for Task 2 (open-ended QA) data generation.

Stage 1 — Screening:  Classifies each MCQ item as yes / no / maybe using the
                       screening prompt in prompt-screening.txt.
Stage 2 — Rewriting:  Rewrites kept items into standalone open-ended Arabic
                       questions using the prompt in prompt-rewrite.txt.

"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Optional

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

client = OpenAI()   

def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    log.info(f"Saved → {path}")

def load_prompt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()

def chat(system: str, user: str, model: str, temperature: float = 0.0,
         retries: int = 3, backoff: float = 5.0) -> str:
    """Call the API with simple retry logic."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                wait = backoff * (attempt + 1)
                log.warning(f"API error ({e}), retrying in {wait}s …")
                time.sleep(wait)
            else:
                raise

def format_item_for_screening(item: dict) -> str:
    """Render one MCQ item as a numbered entry for the screening prompt"""
    opts = item.get("options", [])
    opts_str = "\n".join(f"  {chr(65+i)}. {o}" for i, o in enumerate(opts)) if opts else ""
    return f'ID {item["id"]}: {item["question"]}\n{opts_str}'.strip()

def screen_batch(items: list[dict], system_prompt: str,
                 model: str, batch_size: int) -> list[dict]:
    """
    Run screening in batches. Returns list of {id, keep} dicts. Skips items that already have a result
    """
    results = {}
    total = len(items)

    for start in range(0, total, batch_size):
        batch = items[start : start + batch_size]
        block = "\n\n".join(format_item_for_screening(it) for it in batch)

        log.info(f"Screening items {start+1}–{min(start+batch_size, total)} / {total}")

        raw = chat(system=system_prompt, user=block, model=model)
        try:
            raw_clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            batch_results = json.loads(raw_clean)
            for r in batch_results:
                results[r["id"]] = r["keep"]
        except json.JSONDecodeError as e:
            log.error(f"JSON parse error on batch starting {start}: {e}\nRaw:\n{raw[:300]}")
            for it in batch:
                if it["id"] not in results:
                    results[it["id"]] = "maybe"

        time.sleep(0.5)   

    return [{"id": k, "keep": v} for k, v in sorted(results.items())]

def rewrite_item(item: dict, system_prompt: str, model: str) -> Optional[str]:
    """Rewrite a single MCQ stem to a standalone open-ended question."""
    user_msg = item["question"]
    return chat(system=system_prompt, user=user_msg, model=model, temperature=0.3)

def rewrite_all(items_to_rewrite: list[dict], system_prompt: str,
                model: str, existing: dict) -> list[dict]:
    results = []
    total = len(items_to_rewrite)

    for i, item in enumerate(items_to_rewrite, 1):
        item_id = item["id"]

        if item_id in existing:
            log.info(f"[{i}/{total}] Skipping id={item_id} (already done)")
            results.append(existing[item_id])
            continue

        log.info(f"[{i}/{total}] Rewriting id={item_id}")
        rewritten = rewrite_item(item, system_prompt, model)

        record = {
            "id":              item_id,
            "original_stem":   item["question"],
            "rewritten":       rewritten,
            "answer":          item.get("answer", ""),
            "original_options": item.get("options", []),
        }
        results.append(record)
        time.sleep(0.3)

    return results

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input",           required=True, help="Path to input MCQ JSON")
    p.add_argument("--outdir",          default="outputs/task2", help="Output directory")
    p.add_argument("--stage",           default="all", choices=["all", "screen", "rewrite"],
                   help="Which stage(s) to run")
    p.add_argument("--model",           default="gpt-4o", help="OpenAI model name")
    p.add_argument("--batch_size",      type=int, default=20,
                   help="Number of items per screening API call")
    p.add_argument("--include_maybe",   action="store_true",
                   help="Include 'maybe' items in the rewriting pass")
    p.add_argument("--screening_prompt", default="prompt-screening.txt")
    p.add_argument("--rewrite_prompt",   default="prompt-rewrite.txt")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    screening_path = outdir / "screening_results.json"
    dataset_path   = outdir / "task2_dataset.json"

    data = load_json(args.input)
    log.info(f"Loaded {len(data)} items from {args.input}")

    if args.stage in ("all", "screen"):
        screen_prompt = load_prompt(args.screening_prompt)
        existing_screening = {}
        if screening_path.exists():
            for r in load_json(str(screening_path)):
                existing_screening[r["id"]] = r["keep"]
            log.info(f"Resuming screening: {len(existing_screening)} already done")

        to_screen = [it for it in data if it["id"] not in existing_screening]
        log.info(f"Items to screen: {len(to_screen)}")

        if to_screen:
            new_results = screen_batch(to_screen, screen_prompt, args.model, args.batch_size)
            for r in new_results:
                existing_screening[r["id"]] = r["keep"]
            save_json([{"id": k, "keep": v} for k, v in sorted(existing_screening.items())],
                      str(screening_path))

        keep_ids   = {k for k, v in existing_screening.items() if v == "yes"}
        maybe_ids  = {k for k, v in existing_screening.items() if v == "maybe"}
        drop_ids   = {k for k, v in existing_screening.items() if v == "no"}
        log.info(f"Screening summary → keep: {len(keep_ids)}  "
                 f"maybe: {len(maybe_ids)}  drop: {len(drop_ids)}")

    if args.stage in ("all", "rewrite"):
        if not screening_path.exists():
            raise FileNotFoundError(f"No screening results at {screening_path}. Run --stage screen first.")

        screening = {r["id"]: r["keep"] for r in load_json(str(screening_path))}

        accepted = {"yes"} | ({"maybe"} if args.include_maybe else set())
        to_rewrite_ids = {k for k, v in screening.items() if v in accepted}
        items_to_rewrite = [it for it in data if it["id"] in to_rewrite_ids]
        log.info(f"Items to rewrite: {len(items_to_rewrite)} "
                 f"({'yes + maybe' if args.include_maybe else 'yes only'})")

        rewrite_prompt = load_prompt(args.rewrite_prompt)

        existing_rewritten = {}
        if dataset_path.exists():
            for r in load_json(str(dataset_path)):
                existing_rewritten[r["id"]] = r
            log.info(f"Resuming rewrite: {len(existing_rewritten)} already done")

        final = rewrite_all(items_to_rewrite, rewrite_prompt, args.model, existing_rewritten)
        save_json(final, str(dataset_path))
        log.info(f"Task 2 dataset complete: {len(final)} items → {dataset_path}")


if __name__ == "__main__":
    main()