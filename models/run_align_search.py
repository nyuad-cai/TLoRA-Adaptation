#!/usr/bin/env python3
"""
log-uniform random search over learning rate for train_lora_align.py (CE + KL alignment) experiments.
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
from typing import List

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def sample_log_uniform(lo: float, hi: float, rng: random.Random) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))

def summarise(manifest_path: str):
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\nSearch manifest: {manifest_path}")
    print(f"  layer_start : {manifest.get('layer_start')}")
    print(f"  layer_end   : {manifest.get('layer_end')}")
    print(f"  probe_layers: {manifest.get('probe_layers')}")
    print(f"  auto_beta   : {manifest.get('auto_beta')}")
    print(f"  lr_range    : {manifest.get('lr_range')}")
    print(f"  seed        : {manifest.get('seed')}")
    print()

    results = []
    for t in manifest["trials"]:
        log_path = os.path.join(t["output_dir"], "val_loss_log.json")
        meta_path = os.path.join(t["output_dir"], "train_meta.json")
        best_loss, best_epoch, beta_star, n_evals = None, None, None, 0
        if os.path.exists(log_path):
            with open(log_path) as f:
                records = json.load(f)
            n_evals = len(records)
            if records:
                best_rec = min(records, key=lambda r: r["eval_loss"])
                best_loss = best_rec["eval_loss"]
                best_epoch = best_rec["epoch"]
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            beta_star = meta.get("beta_star")
        results.append({
            "trial": t["trial"], "lr": t["lr"],
            "best_loss": best_loss, "best_epoch": best_epoch,
            "beta_star": beta_star, "n_evals": n_evals,
            "output_dir": t["output_dir"],
        })

    results.sort(key=lambda r: (r["best_loss"] is None, r["best_loss"] or 1e9))

    header = f"{'Trial':>5}  {'LR':>10}  {'Best val loss':>14}  {'@ epoch':>8}  {'β*':>8}  {'# evals':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        loss_str  = f"{r['best_loss']:.4f}"  if r["best_loss"]  is not None else "  (pending)"
        epoch_str = f"{r['best_epoch']:.2f}" if r["best_epoch"] is not None else "—"
        beta_str  = f"{r['beta_star']:.3f}"  if r["beta_star"]  is not None else "—"
        print(f"{r['trial']:>5}  {r['lr']:>10.3e}  {loss_str:>14}  {epoch_str:>8}  {beta_str:>8}  {r['n_evals']:>7}")

    if results and results[0]["best_loss"] is not None:
        best = results[0]
        print(f"\nBest trial: {best['trial']}  lr={best['lr']:.3e}  "
              f"val_loss={best['best_loss']:.4f}  β*={best['beta_star']}  →  {best['output_dir']}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Log-uniform LR random search for train_lora_align.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--summarise", metavar="MANIFEST_JSON", default=None,
                   help="Path to search_manifest.json; print summary and exit.")

    p.add_argument("--train_file", default=None)
    p.add_argument("--val_file",   default=None)
    p.add_argument("--search_dir", default=None)
    p.add_argument("--model_name", default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")

    p.add_argument("--layer_start",   type=int,   default=0)
    p.add_argument("--layer_end",     type=int,   default=23)
    p.add_argument("--probe_layers",  type=int,   nargs="+", default=[34])
    p.add_argument("--alpha",         type=float, default=1.0)
    p.add_argument("--auto_beta",     action="store_true")

    p.add_argument("--n_trials", type=int,   default=5)
    p.add_argument("--lr_lo",    type=float, default=1e-5)
    p.add_argument("--lr_hi",    type=float, default=4e-4)
    p.add_argument("--seed",     type=int,   default=42)

    p.add_argument("--num_epochs",               type=float, default=3.0)
    p.add_argument("--early_stopping_patience",  type=int,   default=1)
    p.add_argument("--max_length",               type=int,   default=1024)

    p.add_argument("--nproc",            type=int, default=2)
    p.add_argument("--dry_run",          action="store_true")
    p.add_argument("--continue_on_error",action="store_true")
    p.add_argument("--resume",           action="store_true")

    return p.parse_args()

def main():
    args = parse_args()

    if args.summarise:
        summarise(args.summarise)
        sys.exit(0)

    for field in ["train_file", "val_file", "search_dir"]:
        if getattr(args, field) is None:
            print(f"ERROR: --{field} is required.", file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.search_dir, exist_ok=True)
    manifest_path = os.path.join(args.search_dir, "search_manifest.json")

    rng = random.Random(args.seed)
    lrs = [sample_log_uniform(args.lr_lo, args.lr_hi, rng) for _ in range(args.n_trials)]

    train_script = str(Path(__file__).parent / "train_lora_align.py")

    if args.resume and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        existing = {t["trial"]: t for t in manifest["trials"]}
        done_count = sum(1 for t in manifest["trials"] if t.get("status") == "done")
        print(f"Resuming from {manifest_path}  ({done_count}/{args.n_trials} done)\n")
    else:
        manifest = {
            "created":      datetime.now().isoformat(),
            "layer_start":  args.layer_start,
            "layer_end":    args.layer_end,
            "probe_layers": args.probe_layers,
            "auto_beta":    args.auto_beta,
            "lr_range":     [args.lr_lo, args.lr_hi],
            "seed":         args.seed,
            "n_trials":     args.n_trials,
            "trials":       [],
        }
        existing = {}
        print(f"Align LR search: {args.n_trials} trials  "
              f"LR ∈ LogUniform[{args.lr_lo:.0e}, {args.lr_hi:.0e}]  seed={args.seed}\n")

    for i, lr in enumerate(lrs):
        tag     = f"trial_{i:02d}_lr{lr:.2e}"
        out_dir = os.path.join(args.search_dir, tag)

        has_checkpoint = (
            os.path.exists(out_dir) and
            any(d.startswith("checkpoint-")
                for d in os.listdir(out_dir)
                if os.path.isdir(os.path.join(out_dir, d)))
        ) if os.path.exists(out_dir) else False

        free_port = find_free_port()
        cmd: List[str] = [
            "torchrun",
            f"--nproc_per_node={args.nproc}",
            f"--master_port={free_port}",
            train_script,
            "--train_file",               args.train_file,
            "--val_file",                 args.val_file,
            "--model_name",               args.model_name,
            "--output_dir",               out_dir,
            "--layer_start",              str(args.layer_start),
            "--layer_end",                str(args.layer_end),
            "--probe_layers",             *[str(l) for l in args.probe_layers],
            "--alpha",                    str(args.alpha),
            "--num_train_epochs",         str(args.num_epochs),
            "--learning_rate",            f"{lr:.6e}",
            "--early_stopping_patience",  str(args.early_stopping_patience),
            "--save_total_limit",         "1",
            "--max_length",               str(args.max_length),
        ]
        if args.auto_beta:
            cmd.append("--auto_beta")
        if args.resume and has_checkpoint:
            cmd += ["--resume_from_checkpoint", "auto"]

        trial_record = {
            "trial": i, "lr": lr, "output_dir": out_dir,
            "cmd": " ".join(cmd), "status": "pending",
        }

        prev = existing.get(i, {})
        if args.resume and prev.get("status") in ("done", "skipped"):
            print(f"[Trial {i:02d}/{args.n_trials-1}]  lr={lr:.3e}  SKIP")
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
                print(f"  ✓ Trial {i:02d} complete")
            else:
                trial_record["status"] = "failed"
                trial_record["returncode"] = result.returncode
                msg = f"Trial {i:02d} exited with code {result.returncode}"
                if args.continue_on_error:
                    print(f"[WARN] {msg} — continuing")
                else:
                    print(f"[ERROR] {msg} — aborting (use --continue_on_error to skip)")
                    _upsert_trial(manifest, trial_record)
                    _write_manifest(manifest, args.search_dir)
                    sys.exit(result.returncode)

        _upsert_trial(manifest, trial_record)
        _write_manifest(manifest, args.search_dir)
        print()

    if args.dry_run:
        print("[dry_run] No training executed.")
    else:
        done = sum(1 for t in manifest["trials"] if t.get("status") == "done")
        print(f"Search finished: {done}/{args.n_trials} trials done.")
        print(f"\nSummarise:\n  python {Path(__file__).name} --summarise {manifest_path}")


def _upsert_trial(manifest, trial_record):
    for idx, t in enumerate(manifest["trials"]):
        if t["trial"] == trial_record["trial"]:
            manifest["trials"][idx] = trial_record
            return
    manifest["trials"].append(trial_record)


def _write_manifest(manifest, search_dir):
    path = os.path.join(search_dir, "search_manifest.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, path)


if __name__ == "__main__":
    main()
