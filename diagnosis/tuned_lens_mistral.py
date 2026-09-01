"""
Tuned-lens analysis for Mistral-Small-3.2-24B-Instruct-2506 on MedAraBench!
"""

import os, csv, re, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import defaultdict
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--mode",        required=True, choices=["train", "eval"])
parser.add_argument("--model_path",  required=True)
parser.add_argument("--lens_dir",    default="./tuned_lens_mistral")
parser.add_argument("--max_len",     type=int, default=512)

#train-only
parser.add_argument("--train_csv",   default=None)
parser.add_argument("--train_n",     type=int, default=400)
parser.add_argument("--epochs",      type=int, default=10)
parser.add_argument("--lr",          type=float, default=1e-3)
parser.add_argument("--reg",         type=float, default=1e-4)
parser.add_argument("--train_batch", type=int, default=4)

#eval-only
parser.add_argument("--csv",         default=None)
parser.add_argument("--out_dir",     default="./tuned_lens_mistral_out")
parser.add_argument("--batch_size",  type=int, default=1)

args = parser.parse_args()
os.makedirs(args.lens_dir, exist_ok=True)
if args.mode == "eval":
    os.makedirs(args.out_dir, exist_ok=True)

PROBE_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 34, 36, 38, 39, 40]
N_LAYERS     = 40

#bare letter tokens in Tekken tokenizer
ANSWER_TOKEN_IDS = {"A": 1065, "B": 1066, "C": 1067, "D": 1068}

QUAD_ORDER  = ["both_correct", "access_gap", "arabic_only", "both_wrong"]
QUAD_COLORS = {
    "both_correct": "#2DC653",
    "access_gap":   "#E63946",
    "arabic_only":  "#F4A261",
    "both_wrong":   "#ADB5BD",
}
QUAD_LABELS = {
    "both_correct": "En✓ Ar✓  (shared knowledge)",
    "access_gap":   "En✓ Ar✗  (access gap)",
    "arabic_only":  "En✗ Ar✓  (Arabic-only)",
    "both_wrong":   "En✗ Ar✗  (both wrong)",
}

SYSTEM_PROMPT = (
    "You are a medical expert. "
    "Answer the following multiple choice question "
    "by responding with only the letter of the correct option: A, B, C, or D. "
    "Do not explain your answer."
)

def extract_letter(val):
    if not val or not isinstance(val, str): return ""
    s = val.strip().upper()
    m = re.search(r'\bANSWER\s*:\s*([A-F])\b', s)
    if m: return m.group(1)
    m = re.search(r'\b([A-F])\b', s)
    return m.group(1) if m else ""


def load_model_and_tokenizer(model_path):
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer as MCTokenizer
    from transformers import Mistral3ForConditionalGeneration

    tekken = os.path.join(model_path, "tekken.json")
    print(f"Loading tokenizer from {tekken} ...")
    tokenizer = MCTokenizer.from_file(tekken)
    pad_id = tokenizer.instruct_tokenizer.tokenizer.eos_id
    print(f"  pad/eos id: {pad_id}")

    print(f"Loading model from {model_path} ...")
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    print(f"  type: {type(model).__name__}")

    def _get_norm(m):
        for path, accessor in [
            ("model.model.language_model.norm",
             lambda m: m.model.language_model.norm),
            ("model.model.language_model.model.norm",
             lambda m: m.model.language_model.model.norm),
            ("model.model.norm",
             lambda m: m.model.norm),
        ]:
            try:
                obj = accessor(m)
                if obj is not None:
                    print(f"  final_norm at: {path}")
                    return obj
            except AttributeError:
                pass
        raise RuntimeError("Cannot find final_norm in model. Check architecture.")

    def _get_lm_head(m):
        #model.lm_head exists at top level , confirmed by diagnose output
        for path, accessor in [
            ("model.lm_head",
             lambda m: m.lm_head),
            ("model.model.language_model.lm_head",
             lambda m: m.model.language_model.lm_head),
            ("model.model.lm_head",
             lambda m: m.model.lm_head),
        ]:
            try:
                obj = accessor(m)
                if obj is not None:
                    print(f"  lm_head at: {path}  shape={obj.weight.shape}")
                    return obj
            except AttributeError:
                pass
        raise RuntimeError("Cannot find lm_head in model. Check architecture.")

    def _get_n_layers(m):
        for accessor in [
            lambda m: m.model.language_model.layers,
            lambda m: m.model.language_model.model.layers,
            lambda m: m.model.layers,
        ]:
            try:
                layers = accessor(m)
                return len(layers)
            except AttributeError:
                pass
        return N_LAYERS  # fallback from config

    final_norm = _get_norm(model)
    lm_head    = _get_lm_head(model)
    n_layers   = _get_n_layers(model)
    print(f"  n_layers:   {n_layers}")
    print(f"  hidden_dim: {lm_head.weight.shape[1]}")
    print(f"  vocab_size: {lm_head.weight.shape[0]}")

    return model, tokenizer, final_norm, lm_head, n_layers, pad_id

def tokenize_batch(texts, tokenizer, pad_id, max_len):
    from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest

    all_ids = []
    for t in texts:
        request = ChatCompletionRequest(messages=[
            SystemMessage(role="system", content=SYSTEM_PROMPT),
            UserMessage(role="user",   content=t),
        ])
        tok = tokenizer.encode_chat_completion(request)
        all_ids.append(tok.tokens[:max_len])

    bs    = len(texts)
    max_l = max(len(x) for x in all_ids)

    inp  = torch.full((bs, max_l), pad_id, dtype=torch.long)
    mask = torch.zeros((bs, max_l),         dtype=torch.long)
    for j, ids in enumerate(all_ids):
        sl = len(ids)
        # left-pad
        inp [j, max_l - sl:] = torch.tensor(ids, dtype=torch.long)
        mask[j, max_l - sl:] = 1

    seq_lens = mask.sum(dim=1) - 1   # index of last real token (keep on CPU)
    return inp, mask, seq_lens

class AffineTranslator(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(d, d))
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + h @ self.W.T + self.b

def train_translators(args, model, tokenizer, final_norm, lm_head, n_layers, pad_id):
    # Input device is always cuda:0 (embeddings live there under device_map="auto")
    input_device = torch.device("cuda:0")
    # final_norm and lm_head may live on cuda:1 — track where they are
    proj_device = next(final_norm.parameters()).device

    print(f"\nTraining — input_device={input_device}, proj_device={proj_device}")

    print(f"Loading training CSV: {args.train_csv}")
    rows = []
    with open(args.train_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    texts = [r["input_english"] for r in rows[:args.train_n]]
    print(f"  Using {len(texts)} training examples")

    print("\nCollecting hidden states ...")
    hs_by_layer       = defaultdict(list)
    final_logits_list = []
    LAYERS_TO_SAVE    = set(PROBE_LAYERS)

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.train_batch), desc="forward"):
            batch = texts[i : i + args.train_batch]
            inp, msk, seq_lens = tokenize_batch(
                batch, tokenizer, pad_id, args.max_len)
            bs = inp.shape[0]

            out = model(
                input_ids=inp.to(input_device),
                attention_mask=msk.to(input_device),
                output_hidden_states=True,
            )

            # Final logits at last-token position
            fin_log = out.logits[
                torch.arange(bs), seq_lens
            ].float().cpu()                                    #(bs, vocab)
            final_logits_list.append(fin_log)

            for l in LAYERS_TO_SAVE:
                hs   = out.hidden_states[l]                    #(bs, seq, d)
                last = hs[torch.arange(bs), seq_lens].float().cpu()
                hs_by_layer[l].append(last)

    final_logits_all = torch.cat(final_logits_list, dim=0)     #(N, vocab)
    final_probs_all  = F.softmax(final_logits_all, dim=-1)

    for l in PROBE_LAYERS:
        hs_by_layer[l] = torch.cat(hs_by_layer[l], dim=0)     #(N, d)

    N_train = final_probs_all.shape[0]
    d = hs_by_layer[PROBE_LAYERS[0]].shape[-1]
    print(f"  N_train={N_train},  d={d}")

    # Keep final_norm and lm_head on proj_device — accelerate hooks prevent
    # moving them to CPU (they intercept .forward() and enforce device placement).
    # Instead we move translator + batches to proj_device during training.
    norm_dtype = next(final_norm.parameters()).dtype

    layers_to_train = [l for l in PROBE_LAYERS if 0 < l < n_layers]
    print(f"\nTraining translators for layers: {layers_to_train}")

    for l in layers_to_train:
        ckpt = os.path.join(args.lens_dir, f"translator_layer_{l:02d}.pt")
        if os.path.exists(ckpt):
            print(f"  [L{l:2d}] already exists, skipping")
            continue

        print(f"\n  [L{l:2d}] training ...")
        translator = AffineTranslator(d).to(proj_device)
        opt        = torch.optim.Adam(translator.parameters(), lr=args.lr)
        hs         = hs_by_layer[l]        # (N, d) CPU

        best_loss = math.inf
        for epoch in range(args.epochs):
            perm       = torch.randperm(N_train)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, N_train, args.train_batch):
                idx      = perm[start : start + args.train_batch]
                h_b      = hs[idx].to(proj_device)             # (bs, d) → cuda:1
                p_target = final_probs_all[idx] # (bs, vocab) → cuda:1

                opt.zero_grad()

                h_trans  = translator(h_b)
                h_normed = final_norm(h_trans.to(norm_dtype))
                logits   = lm_head(h_normed).float()
                log_p    = F.log_softmax(logits, dim=-1)

                kl  = F.kl_div(log_p, p_target.to(log_p.device), reduction="batchmean")
                reg = args.reg * translator.W.pow(2).sum()
                loss = kl + reg.to(kl.device)
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg = epoch_loss / max(n_batches, 1)
            if epoch == 0 or (epoch + 1) % 2 == 0:
                print(f"    epoch {epoch+1}/{args.epochs}  loss={avg:.5f}")
            if avg < best_loss:
                best_loss = avg
                # Save to CPU so checkpoints are device-agnostic
                torch.save({k: v.cpu() for k, v in translator.state_dict().items()},
                           ckpt)

        print(f"  [L{l:2d}] best_loss={best_loss:.5f}  → {ckpt}")

    print("\nTraining complete.")

def run_tuned_lens(texts, gt_letters, model, tokenizer, final_norm, lm_head,
                   translators, n_layers, pad_id, label=""):
    input_device = torch.device("cuda:0")
    proj_device  = next(final_norm.parameters()).device

    n = len(texts)
    all_ranks = np.zeros((n, len(PROBE_LAYERS)), dtype=np.int32)
    all_probs = np.zeros((n, len(PROBE_LAYERS)), dtype=np.float32)

    for i in tqdm(range(0, n, args.batch_size), desc=f"tuned-lens [{label}]"):
        batch_texts = texts[i : i + args.batch_size]
        batch_gt    = gt_letters[i : i + args.batch_size]
        bs = len(batch_texts)

        inp, msk, seq_lens = tokenize_batch(
            batch_texts, tokenizer, pad_id, args.max_len)
        # seq_lens stays on CPU for indexing across potentially sharded layers

        with torch.no_grad():
            out = model(
                input_ids=inp.to(input_device),
                attention_mask=msk.to(input_device),
                output_hidden_states=True,
            )

        for li, layer_idx in enumerate(PROBE_LAYERS):
            hs   = out.hidden_states[layer_idx]                # (bs, seq, d)
            # Index with CPU seq_lens to avoid cross-device issues
            last = hs[torch.arange(bs), seq_lens].float().cpu()  # (bs, d)

            # Apply translator (on CPU)
            if layer_idx in translators:
                with torch.no_grad():
                    last = translators[layer_idx](last)

            # Project: norm → lm_head (on proj_device)
            h = last.to(proj_device)
            normed = final_norm(h.to(next(final_norm.parameters()).dtype))
            logits = lm_head(normed).float().cpu()             # (bs, vocab)
            probs_all = torch.softmax(logits, dim=-1)

            for j, gt in enumerate(batch_gt):
                if gt not in ANSWER_TOKEN_IDS:
                    all_probs[i+j, li] = 0.0
                    all_ranks[i+j, li] = -1
                    continue
                tid  = ANSWER_TOKEN_IDS[gt]
                rank = (logits[j] > logits[j, tid]).sum().item()
                prob = probs_all[j, tid].item()
                all_ranks[i+j, li] = rank
                all_probs[i+j, li] = prob

    return all_ranks, all_probs


def eval_mode(args, model, tokenizer, final_norm, lm_head, n_layers, pad_id):
    d = lm_head.weight.shape[1]
    translators = {}
    for l in PROBE_LAYERS:
        if l == 0 or l >= n_layers:
            continue
        ckpt = os.path.join(args.lens_dir, f"translator_layer_{l:02d}.pt")
        if not os.path.exists(ckpt):
            print(f"  [WARN] No checkpoint for L{l} — falling back to logit lens")
            continue
        t = AffineTranslator(d)
        t.load_state_dict(torch.load(ckpt, map_location="cpu"))
        t.eval()
        translators[l] = t
    print(f"Loaded {len(translators)} translators: {sorted(translators.keys())}")

    print(f"\nLoading eval CSV: {args.csv}")
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    print(f"  {len(rows)} rows")

    en_texts   = [r["input_english"]  for r in rows]
    ar_texts   = [r["input_arabic"]   for r in rows]
    quadrants  = np.array([r["quadrant"]      for r in rows])
    gt_letters = [extract_letter(r["ground_truth"]) for r in rows]
    N = len(rows)

    quad_counts = defaultdict(int)
    for q in quadrants: quad_counts[q] += 1
    print("  Quadrant counts:")
    for q in QUAD_ORDER:
        print(f"    {q}: {quad_counts[q]}")

    print(f"\nRunning tuned-lens — English ({N} questions) ...")
    en_ranks, en_probs = run_tuned_lens(
        en_texts, gt_letters, model, tokenizer, final_norm, lm_head,
        translators, n_layers, pad_id, label="EN")

    print(f"\nRunning tuned-lens — Arabic ({N} questions) ...")
    ar_ranks, ar_probs = run_tuned_lens(
        ar_texts, gt_letters, model, tokenizer, final_norm, lm_head,
        translators, n_layers, pad_id, label="AR")

    out_npz = os.path.join(args.out_dir, "tuned_lens_results.npz")
    np.savez(out_npz,
             en_ranks=en_ranks, en_probs=en_probs,
             ar_ranks=ar_ranks, ar_probs=ar_probs,
             quadrants=quadrants, layers=np.array(PROBE_LAYERS))
    print(f"\nSaved → {out_npz}")

    print("\n── Mean P(correct) at each probe layer (tuned lens) ──")
    header = "  ".join([f"L{l:2d}" for l in PROBE_LAYERS])
    print(f"{'Quadrant / Lang':<25} {header}")
    for q in QUAD_ORDER:
        mask = quadrants == q
        if not mask.any(): continue
        for lang_name, probs in [("English", en_probs), ("Arabic", ar_probs)]:
            vals = "  ".join(
                f"{probs[mask, li].mean():.3f}"
                for li in range(len(PROBE_LAYERS)))
            print(f"  {q[:15]:<15} {lang_name:<8} {vals}")
        print()

    print("Generating figure ...")
    fig, ax = plt.subplots(figsize=(10, 6))

    for q in QUAD_ORDER:
        mask = quadrants == q
        if not mask.any(): continue
        ev = en_probs[mask].mean(axis=0)
        av = ar_probs[mask].mean(axis=0)
        ee = en_probs[mask].std(axis=0) / np.sqrt(mask.sum())
        ae = ar_probs[mask].std(axis=0) / np.sqrt(mask.sum())

        ax.plot(PROBE_LAYERS, ev, color=QUAD_COLORS[q], lw=2.5,
                ls="-",  marker="o", ms=6)
        ax.fill_between(PROBE_LAYERS, ev-ee, ev+ee,
                        color=QUAD_COLORS[q], alpha=0.12)
        ax.plot(PROBE_LAYERS, av, color=QUAD_COLORS[q], lw=2.5,
                ls="--", marker="^", ms=6)
        ax.fill_between(PROBE_LAYERS, av-ae, av+ae,
                        color=QUAD_COLORS[q], alpha=0.08)

    ax.set_xlabel("Layer depth", fontsize=12)
    ax.set_ylabel("Mean P(correct answer letter)", fontsize=11)
    ax.set_title(
        "Tuned Lens: Correct Answer Emergence Across Layers\n"
        "Mistral-Small-3.2-24B  ·  MedAraBench",
        fontsize=12, fontweight="bold")
    ax.set_xticks(PROBE_LAYERS)
    ax.set_xticklabels(
        ["L0\n(emb)" if l == 0 else
         (f"L{n_layers}\n(final)" if l == n_layers else f"L{l}")
         for l in PROBE_LAYERS], fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    quad_handles = [mlines.Line2D([], [], color=QUAD_COLORS[q], lw=2,
                                  label=QUAD_LABELS[q]) for q in QUAD_ORDER]
    style_handles = [
        mlines.Line2D([], [], color="grey", lw=2, ls="-",
                      marker="o", ms=6, label="English  ●"),
        mlines.Line2D([], [], color="grey", lw=2, ls="--",
                      marker="^", ms=6, label="Arabic  ▲"),
    ]
    ax.legend(handles=quad_handles + style_handles,
              loc="upper left", fontsize=9, frameon=False)
    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        out_fig = os.path.join(args.out_dir, f"fig_tuned_lens_mistral{ext}")
        plt.savefig(out_fig, bbox_inches="tight", dpi=150)
        print(f"Saved → {out_fig}")
    plt.close()
    print("\nDone.")


if __name__ == "__main__":
    model, tokenizer, final_norm, lm_head, n_layers, pad_id = \
        load_model_and_tokenizer(args.model_path)

    if args.mode == "train":
        assert args.train_csv, "--train_csv required for --mode train"
        train_translators(args, model, tokenizer, final_norm, lm_head,
                          n_layers, pad_id)

    elif args.mode == "eval":
        assert args.csv, "--csv required for --mode eval"
        eval_mode(args, model, tokenizer, final_norm, lm_head,
                  n_layers, pad_id)
