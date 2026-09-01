#!/usr/bin/env python3
"""
log-uniform random search over learning rate for train_lora.py experiments
"""

import os
import sys
import math
import json
import random
import socket
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

def find_free_port() -> int:
    """Bind to port 0 and let the OS pick a free one, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def sample_log_uniform(lo: float, hi: float, rng: random.Random) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))

def summarise(manifest_path: str):
    """Read a search manifest and print a sorted summary of val losses."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\nSearch manifest: {manifest_path}")
    print(f"  lora_mode : {manifest.get('lora_mode')}")
    print(f"  lr_range  : {manifest.get('lr_range')}")
    print(f"  seed      : {manifest.get('seed')}")
    print()

    results = []
    for t in manifest["trials"]:
        log_path = os.path.join(t["output_dir"], "val_loss_log.json")
        best_loss = None
        best_epoch = None
        n_evals = 0
        if os.path.exists(log_path):
            with open(log_path) as f:
                records = json.load(f)
            n_evals = len(records)
            if records:
                best_rec = min(records, key=lambda r: r["eval_loss"])
                best_loss = best_rec["eval_loss"]
                best_epoch = best_rec["epoch"]
        results.append({
            "trial":       t["trial"],
            "lr":          t["lr"],
            "best_loss":   best_loss,
            "best_epoch":  best_epoch,
            "n_evals":     n_evals,
            "output_dir":  t["output_dir"],
        })

    #sort by best val loss 
    results.sort(key=lambda r: (r["best_loss"] is None, r["best_loss"] or 1e9))

    header = f"{'Trial':>5}  {'LR':>10}  {'Best val loss':>14}  {'@ epoch':>8}  {'# evals':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        loss_str  = f"{r['best_loss']:.4f}" if r['best_loss'] is not None else "  (pending)"
        epoch_str = f"{r['best_epoch']:.2f}" if r['best_epoch'] is not None else "—"
        print(f"{r['trial']:>5}  {r['lr']:>10.3e}  {loss_str:>14}  {epoch_str:>8}  {r['n_evals']:>7}")

    if results and results[0]["best_loss"] is not None:
        best = results[0]
        print(f"\nBest trial: {best['trial']}  lr={best['lr']:.3e}  "
              f"val_loss={best['best_loss']:.4f}  →  {best['output_dir']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Log-uniform LR random search launcher for train_lora.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--summarise", metavar="MANIFEST_JSON", default=None,
                   help="Path to a search_manifest.json; print summary and exit.")

    p.add_argument("--train_file",   default=None, help="Training data JSON")
    p.add_argument("--val_file",     default=None, help="Validation data JSON")
    p.add_argument("--search_dir",   default=None,
                   help="Parent directory for all trials")
    p.add_argument("--train_script", default="train_lora_targeted.py",
                   help="Filename of the training script (default: train_lora_targeted.py)")
    p.add_argument("--model_name",
                   default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    p.add_argument("--lora_mode",   default="targeted_l24",
                   choices=["targeted_l01", "targeted_l14", "targeted_l24", "targeted_l32", "targeted_l35", "targeted_l01_34", "full"])

    p.add_argument("--n_trials",    type=int,   default=10,
                   help="Number of LR values to sample")
    p.add_argument("--lr_lo",       type=float, default=1e-5,
                   help="Lower bound of log-uniform LR range")
    p.add_argument("--lr_hi",       type=float, default=4e-4,
                   help="Upper bound of log-uniform LR range "
                        "(4e-4 = LoRA paper GPT-3 value, Hu et al. 2022)")
    p.add_argument("--seed",        type=int,   default=42,
                   help="RNG seed for reproducible LR sampling")

    p.add_argument("--num_epochs",  type=float, default=3.0,
                   help="Max epochs per trial (early stopping may cut short)")
    p.add_argument("--early_stopping_patience", type=int, default=1,
                   help="Patience on eval_loss per trial; set 0 to disable")
    p.add_argument("--divergence_threshold", type=float, default=None,
                   help=(
                       "Optional val-loss cutoff for divergence detection. "
                       "After --divergence_min_evals evaluations, if the best "
                       "val_loss seen so far is still above this threshold the "
                       "trial is marked 'diverged' and skipped on the next resume. "
                       "Example: --divergence_threshold 0.3 to kill trials that "
                       "haven't learned anything after the initial evals. "
                       "Recommended: set once before the search and report in paper."
                   ))
    p.add_argument("--divergence_min_evals", type=int, default=5,
                   help=(
                       "Minimum number of evaluation checkpoints before the "
                       "divergence threshold is applied (default: 5). "
                       "Prevents premature termination during warmup."
                   ))

    p.add_argument("--nproc",       type=int,   default=2,
                   help="--nproc_per_node for torchrun (number of GPUs)")
    p.add_argument("--dry_run",     action="store_true",
                   help="Print commands only; do not execute training")
    p.add_argument("--continue_on_error", action="store_true",
                   help="Continue to next trial even if a trial exits non-zero")

    p.add_argument("--max_length", type=int, default=None,
                   help="Max token sequence length passed to training script. "
                        "Reduce (e.g. 1024) if hitting OOM.")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing in training script to save GPU memory.")

    p.add_argument("--resume",      action="store_true",
                   help=(
                       "Resume an interrupted search.  Reads the existing "
                       "search_manifest.json in --search_dir, skips trials whose "
                       "status is 'done', and re-runs the rest.  The same LR sample "
                       "is reproduced from --seed so trial indices are stable. "
                       "A partially-run trial (status='running' or 'failed') is "
                       "resumed from its latest checkpoint automatically."
                   ))

    return p.parse_args()

def main():
    args = parse_args()

    if args.summarise:
        summarise(args.summarise)
        sys.exit(0)

    for field in ["train_file", "val_file", "search_dir"]:
        if getattr(args, field) is None:
            print(f"ERROR: --{field} is required for a search run.", file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.search_dir, exist_ok=True)
    manifest_path = os.path.join(args.search_dir, "search_manifest.json")

    rng = random.Random(args.seed)
    lrs = [sample_log_uniform(args.lr_lo, args.lr_hi, rng)
           for _ in range(args.n_trials)]

    train_script = str(Path(__file__).parent / args.train_script)

    if args.resume and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        existing = {t["trial"]: t for t in manifest["trials"]}
        print(f"Resuming search from {manifest_path}")
        done_count = sum(1 for t in manifest["trials"] if t.get("status") == "done")
        print(f"  {done_count}/{args.n_trials} trials already done\n")
    else:
        manifest = {
            "created":   datetime.now().isoformat(),
            "lora_mode": args.lora_mode,
            "lr_range":  [args.lr_lo, args.lr_hi],
            "seed":      args.seed,
            "n_trials":  args.n_trials,
            "trials":    [],
        }
        existing = {}
        print(f"Random search: {args.n_trials} trials  "
              f"LR ∈ LogUniform[{args.lr_lo:.0e}, {args.lr_hi:.0e}]  "
              f"seed={args.seed}\n")

    for i, lr in enumerate(lrs):
        tag     = f"trial_{i:02d}_lr{lr:.2e}"
        out_dir = os.path.join(args.search_dir, tag)

        has_checkpoint = os.path.exists(out_dir) and any(
            d.startswith("checkpoint-")
            for d in os.listdir(out_dir)
            if os.path.isdir(os.path.join(out_dir, d))
        ) if os.path.exists(out_dir) else False

        free_port = find_free_port()
        cmd: List[str] = [
            "torchrun",
            f"--nproc_per_node={args.nproc}",
            f"--master_port={free_port}",
            train_script,
            "--train_file",                  args.train_file,
            "--val_file",                    args.val_file,
            "--model_name",                  args.model_name,
            "--output_dir",                  out_dir,
            "--lora_mode",                   args.lora_mode,
            "--num_train_epochs",            str(args.num_epochs),
            "--learning_rate",               f"{lr:.6e}",
            "--early_stopping_patience",     str(args.early_stopping_patience),
            "--save_total_limit",            "1",
        ]
        if args.max_length is not None:
            cmd += ["--max_length", str(args.max_length)]
        if args.gradient_checkpointing:
            cmd += ["--gradient_checkpointing"]
        if args.resume and has_checkpoint:
            cmd += ["--resume_from_checkpoint", "auto"]

        trial_record = {
            "trial":      i,
            "lr":         lr,
            "output_dir": out_dir,
            "cmd":        " ".join(cmd),
            "status":     "pending",
        }

        prev = existing.get(i, {})
        if args.resume and prev.get("status") in ("done", "skipped"):
            reason = prev.get("skip_reason", "already done")
            label  = "SKIP" if prev.get("status") == "done" else f"SKIP (excluded: {reason})"
            print(f"[Trial {i:02d}/{args.n_trials-1}]  lr={lr:.3e}  {label}")
            continue

        print(f"[Trial {i:02d}/{args.n_trials-1}]  lr={lr:.3e}  →  {out_dir}"
              + ("  (resuming from checkpoint)" if has_checkpoint else ""))
        print("  " + " ".join(cmd))

        if args.dry_run:
            trial_record["status"] = "dry_run"
        else:
            trial_record["status"] = "running"
            trial_record["started"] = datetime.now().isoformat()
            _upsert_trial(manifest, trial_record)
            _write_manifest(manifest, args.search_dir)

            result = subprocess.run(cmd, check=False)

            if result.returncode == 0:
                trial_record["status"] = "done"
                trial_record["finished"] = datetime.now().isoformat()
                if args.divergence_threshold is not None:
                    log_path = os.path.join(out_dir, "val_loss_log.json")
                    if os.path.exists(log_path):
                        with open(log_path) as f:
                            records = json.load(f)
                        if (len(records) >= args.divergence_min_evals
                                and records):
                            best_so_far = min(r["eval_loss"] for r in records)
                            if best_so_far > args.divergence_threshold:
                                trial_record["status"] = "diverged"
                                trial_record["best_loss_at_flag"] = best_so_far
                                print(f"  ✗ Trial {i:02d} flagged as diverged "
                                      f"(best_loss={best_so_far:.4f} > "
                                      f"threshold={args.divergence_threshold})")
                            else:
                                print(f"  ✓ Trial {i:02d} complete")
                        else:
                            print(f"  ✓ Trial {i:02d} complete")
                    else:
                        print(f"  ✓ Trial {i:02d} complete")
                else:
                    print(f"  ✓ Trial {i:02d} complete")
            else:
                trial_record["status"] = "failed"
                trial_record["returncode"] = result.returncode
                msg = f"Trial {i:02d} exited with code {result.returncode}"
                if args.continue_on_error:
                    print(f"[WARN] {msg} — continuing to next trial")
                else:
                    print(f"[ERROR] {msg} — aborting (use --continue_on_error to skip failed trials)")
                    _upsert_trial(manifest, trial_record)
                    _write_manifest(manifest, args.search_dir)
                    sys.exit(result.returncode)

        _upsert_trial(manifest, trial_record)
        _write_manifest(manifest, args.search_dir)
        print()

    if args.dry_run:
        print("[dry_run] Commands printed above.  No training was executed.")
    else:
        done  = sum(1 for t in manifest["trials"] if t.get("status") == "done")
        total = args.n_trials
        print(f"Search finished: {done}/{total} trials done.")
        if done < total:
            print(f"To resume remaining trials:\n"
                  f"  python {Path(__file__).name} "
                  f"--resume --search_dir {args.search_dir} "
                  f"--train_file {args.train_file} --val_file {args.val_file} "
                  f"--n_trials {args.n_trials} --seed {args.seed}")
        print(f"\nSummarise results:\n"
              f"  python {Path(__file__).name} --summarise {manifest_path}")


def _upsert_trial(manifest: dict, trial_record: dict):
    """Insert or update a trial record in the manifest by trial index."""
    for idx, t in enumerate(manifest["trials"]):
        if t["trial"] == trial_record["trial"]:
            manifest["trials"][idx] = trial_record
            return
    manifest["trials"].append(trial_record)


def _write_manifest(manifest: dict, search_dir: str):
    """Atomically write manifest (write to .tmp then rename to avoid corruption)."""
    path     = os.path.join(search_dir, "search_manifest.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, path)   


if __name__ == "__main__":
    main()
