# script-translate.py
import argparse
import json
import time
import html
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import requests
from tqdm import tqdm

API_URL = "https://translation.googleapis.com/language/translate/v2"


def load_json_array(input_path: Path) -> List[Dict[str, Any]]:
    text = input_path.read_text(encoding="utf-8").strip()
    try:
        obj = json.loads(text)
    except Exception as e:
        raise ValueError(f"invalid json: {e}")
    if not isinstance(obj, list):
        raise ValueError("input must be a json array (not jsonl or an object)")
    return obj


def save_json_array(records: List[Dict[str, Any]], output_path: Path, pretty: bool = False) -> None:
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2 if pretty else None),
        encoding="utf-8"
    )


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def translate_batch_v2_api(
    session: requests.Session,
    api_key: str,
    texts: List[str],
    target: str = "en",
    source: Optional[str] = None,
    fmt: str = "text",
    timeout: int = 60
) -> List[str]:
    params = {"key": api_key}
    data = [("target", target), ("format", fmt)]
    if source:
        data.append(("source", source))
    for t in texts:
        data.append(("q", t if t is not None else ""))

    backoff = 1.0
    for _ in range(7):
        resp = session.post(API_URL, params=params, data=data, timeout=timeout)
        if resp.status_code == 200:
            js = resp.json()
            translations = js.get("data", {}).get("translations", [])
            return [html.unescape(item.get("translatedText", "")) for item in translations]
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2, 20)
            continue
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"translate api error {resp.status_code}: {detail}")

    raise RuntimeError("translate api: retries exhausted")


def write_checkpoint(ckpt_path: Path, state: Dict[str, Any]) -> None:
    #persist tiny checkpoint
    ckpt_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def read_checkpoint(ckpt_path: Path) -> Optional[Dict[str, Any]]:
    #load checkpoint if exists
    if not ckpt_path.exists():
        return None
    try:
        return json.loads(ckpt_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description="translate the 'Question' field using cloud translation v2 (api key), json→json with checkpointing.")
    p.add_argument("--input", required=True, help="path to input json array")
    p.add_argument("--output", required=True, help="path to final output json array")
    p.add_argument("--api-key", required=True, help="google cloud translation api key")
    p.add_argument("--target", default="en", help="target language, default: en")
    p.add_argument("--source", default=None, help="optional source language (e.g., ar). if omitted, auto-detect")
    p.add_argument("--field", default="Question", help="field to translate (default: Question)")
    p.add_argument("--replace", action="store_true", help="overwrite original field instead of adding new one")
    p.add_argument("--add-field", default="Question_en", help="new field name if not using --replace")
    p.add_argument("--batch-size", type=int, default=128, help="max 128 per api call (default: 128)")
    p.add_argument("--save-every", type=int, default=10, help="save partial output+ckpt every N batches (default: 10)")
    p.add_argument("--resume", action="store_true", help="resume from existing .ckpt and .part if present")
    p.add_argument("--no-pretty", action="store_true", help="don’t pretty-print final json output")
    p.add_argument("--skip-existing", action="store_true", help="skip records that already have the add-field populated")
    args = p.parse_args()

    if args.batch_size < 1 or args.batch_size > 128:
        print("--batch-size must be between 1 and 128 for v2 rest", file=sys.stderr)
        sys.exit(2)

    in_path = Path(args.input)
    out_path = Path(args.output)
    part_path = Path(str(out_path) + ".part")
    ckpt_path = Path(str(out_path) + ".ckpt")

    #load input or resume
    if args.resume and ckpt_path.exists() and part_path.exists():
        ckpt = read_checkpoint(ckpt_path)
        if not ckpt or "next_batch_idx" not in ckpt:
            print("resume requested but checkpoint is missing or invalid; starting fresh", file=sys.stderr)
            ckpt = None
        if ckpt:
            records = load_json_array(in_path)  #we still load original to rebuild indices
            partial = load_json_array(part_path)  #mirror with translated chunks applied
            #sanity: lengths must match
            if len(partial) != len(records):
                print("partial output length mismatch; starting fresh", file=sys.stderr)
                ckpt = None
            else:
                print(f"resuming from batch {ckpt['next_batch_idx']} (translated_so_far={ckpt.get('translated',0)})")
    else:
        ckpt = None

    if not ckpt:
        records = load_json_array(in_path)
        partial = [dict(r) for r in records]  
        ckpt = {"next_batch_idx": 0, "translated": 0}

    idxs, texts = [], []
    candidate_fields = [args.field] + ([args.field.lower()] if args.field != args.field.lower() else [])
    skipped_existing = 0
    for i, rec in enumerate(records):
        # skip records that already have the target field populated
        if args.skip_existing and rec.get(args.add_field) not in (None, ""):
            skipped_existing += 1
            continue
        q_val = None
        used = None
        for f in candidate_fields:
            if f in rec and rec[f] not in (None, ""):
                q_val, used = rec[f], f
                break
        if q_val is None:
            continue
        idxs.append((i, used))
        texts.append(str(q_val))
    if args.skip_existing:
        print(f"--skip-existing: skipped {skipped_existing} already-translated records")

    if not texts:
        print("no records with the target field found. check --field name.", file=sys.stderr)
        sys.exit(1)

    enumerated = list(enumerate(texts))
    batches = list(chunked(enumerated, args.batch_size))
    start_batch = ckpt.get("next_batch_idx", 0)

    session = requests.Session()

    try:
        for b_idx in tqdm(range(start_batch, len(batches)), desc="translating", unit="batch"):
            batch = batches[b_idx]
            positions = [pos for pos, _ in batch]
            payload_texts = [txt for _, txt in batch]

            out_texts = translate_batch_v2_api(
                session=session,
                api_key=args.api_key,
                texts=payload_texts,
                target=args.target,
                source=args.source,
                fmt="text",
            )
            if len(out_texts) != len(payload_texts):
                raise RuntimeError("api returned mismatched batch size")

            for p, tr in zip(positions, out_texts):
                rec_idx, used_field = idxs[p]
                if args.replace:
                    partial[rec_idx][used_field] = tr
                else:
                    partial[rec_idx][args.add_field] = tr

            ckpt["next_batch_idx"] = b_idx + 1
            ckpt["translated"] = ckpt.get("translated", 0) + len(out_texts)

            if (b_idx + 1) % args.save_every == 0 or (b_idx + 1) == len(batches):
                save_json_array(partial, part_path, pretty=False)
                write_checkpoint(ckpt_path, ckpt)
    except KeyboardInterrupt:
        save_json_array(partial, part_path, pretty=False)
        write_checkpoint(ckpt_path, ckpt)
        print("\ninterrupted: progress saved to .part and .ckpt", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        save_json_array(partial, part_path, pretty=False)
        write_checkpoint(ckpt_path, ckpt)
        print(f"\nerror: {e}\nprogress saved to .part and .ckpt", file=sys.stderr)
        sys.exit(1)

    save_json_array(partial, out_path, pretty=(not args.no_pretty))
    if part_path.exists():
        try:
            part_path.unlink()
        except Exception:
            pass
    if ckpt_path.exists():
        try:
            ckpt_path.unlink()
        except Exception:
            pass
    print(f"done. wrote {len(partial)} records → {out_path}")


if __name__ == "__main__":
    main()