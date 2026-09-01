"""
Tuned-lens analysis for Llama 3.3 70B Instruct on MedAraBench!
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
parser.add_argument("--lens_dir",    default="./tuned_lens_llama")
parser.add_argument("--max_len",     type=int, default=512)

# train-only
parser.add_argument("--train_csv",   default=None)
parser.add_argument("--train_n",     type=int, default=800,
                    help="Training examples (English only, geometry-agnostic)")
parser.add_argument("--rank",        type=int, default=64,
                    help="Low-rank translator rank r  (default 64)")
parser.add_argument("--epochs",      type=int, default=15)
parser.add_argument("--lr",          type=float, default=2e-3)
parser.add_argument("--reg",         type=float, default=1e-4,
                    help="L2 regularisation on U and V (keeps T near identity)")
parser.add_argument("--train_batch", type=int, default=8,
                    help="Translator training batch size (CPU)")
parser.add_argument("--fwd_batch",   type=int, default=1,
                    help="Forward-pass batch for hidden-state collection (keep 1 for 70B)")

# eval-only
parser.add_argument("--csv",         default=None)
parser.add_argument("--out_dir",     default="./tuned_lens_llama_out")
parser.add_argument("--batch_size",  type=int, default=1)

args = parser.parse_args()
os.makedirs(args.lens_dir, exist_ok=True)
if args.mode == "eval":
    os.makedirs(args.out_dir, exist_ok=True)

# Denser around the critical L32–L40 window identified in the Mistral analysis.
# hidden_states[80] = output of final layer 79 (what final_norm receives).
PROBE_LAYERS = [0, 8, 16, 24, 28, 32, 34, 36, 38, 40, 48, 56, 64, 72, 80]
N_LAYERS     = 80

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

# Confirmed from diagnose_tokens.py on Llama 3.3 70B
ANSWER_TOKEN_IDS = {"A": 32, "B": 33, "C": 34, "D": 35, "E": 36, "F": 37}

def extract_letter(val):
    if not val or not isinstance(val, str): return ""
    s = val.strip().upper()
    m = re.search(r'\bANSWER\s*:\s*([A-F])\b', s)
    if m: return m.group(1)
    m = re.search(r'\b([A-F])\b', s)
    return m.group(1) if m else ""


def load_model_and_tokenizer(model_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify answer token IDs
    print("  Verifying answer token IDs ...")
    for letter, tid in ANSWER_TOKEN_IDS.items():
        decoded = tokenizer.decode([tid]).strip()
        if decoded.upper() != letter:
            print(f"  [WARN] Token {tid} decodes as {repr(decoded)}, not '{letter}'")
            print(f"         Re-run diagnose_tokens.py and update ANSWER_TOKEN_IDS.")
        else:
            print(f"    {letter} → id={tid}  ✓")

    print(f"\nLoading Llama 3.3 70B in bfloat16 across 2 GPUs ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # dtype= not supported in this transformers version
        device_map="auto",           # splits layers across cuda:0 + cuda:1
        low_cpu_mem_usage=True,      # required with device_map to avoid meta tensors
        local_files_only=True,
    )
    model.eval()

    # Llama paths — no multimodal wrapper
    final_norm = model.model.norm    # LlamaRMSNorm
    lm_head    = model.lm_head       # Linear(8192 → vocab)

    n_layers = len(model.model.layers)
    d        = lm_head.weight.shape[1]

    print(f"  n_layers   : {n_layers}")
    print(f"  hidden_dim : {d}")
    print(f"  vocab_size : {lm_head.weight.shape[0]}")
    print(f"  final_norm : {type(final_norm).__name__}")
    print(f"  lm_head    : {type(lm_head).__name__}  {lm_head.weight.shape}")

    if n_layers != N_LAYERS:
        print(f"  [WARN] Expected {N_LAYERS} layers, got {n_layers}. "
              f"Update N_LAYERS and PROBE_LAYERS.")

    # With device_map="auto" the model is sharded across GPUs.
    # lm_head is typically on the last GPU — record its device explicitly.
    lm_head_device = lm_head.weight.device
    print(f"  lm_head device : {lm_head_device}")

    return model, tokenizer, final_norm, lm_head, n_layers, d


def tokenize_batch(texts, tokenizer, max_len, device):
    all_ids = []
    for t in texts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": t},
        ]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors=None,
        )
        if not isinstance(ids, list):
            try:    ids = ids["input_ids"]
            except: ids = ids.ids
        all_ids.append(ids[:max_len])

    bs    = len(texts)
    max_l = max(len(x) for x in all_ids)
    pad   = tokenizer.pad_token_id

    inp  = torch.full((bs, max_l), pad,  dtype=torch.long)
    mask = torch.zeros((bs, max_l),       dtype=torch.long)
    for j, ids in enumerate(all_ids):
        sl = len(ids)
        inp [j, max_l - sl:] = torch.tensor(ids)
        mask[j, max_l - sl:] = 1
    seq_lens = mask.sum(dim=1) - 1
    return inp.to(device), mask.to(device), seq_lens.to(device)

class LowRankTranslator(nn.Module):
    def __init__(self, d: int, r: int):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(d, r))
        self.V = nn.Parameter(torch.zeros(d, r))
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (..., d)  →  VTh: (..., r)  →  UVTh: (..., d)
        VTh = h @ self.V          # (..., r)
        UVTh = VTh @ self.U.T     # (..., d)
        return h + UVTh + self.b

def train_translators(args, model, tokenizer, final_norm, lm_head,
                      n_layers, d, device):

    print(f"\nLoading training CSV: {args.train_csv}")
    rows = []
    with open(args.train_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Use English texts — language-agnostic transformer geometry
    texts = [r["input_english"] for r in rows[:args.train_n]]
    print(f"  Using {len(texts)} training examples (English)")

    # Model stays on GPU in 8-bit (frozen).
    # Hidden states are moved to CPU immediately to free GPU memory.
    print(f"\nCollecting hidden states ({len(texts)} forward passes) ...")
    hs_by_layer      = defaultdict(list)   # layer_idx → list[(bs, d)]
    final_logits_list = []

    SAVE_LAYERS = set(PROBE_LAYERS)

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.fwd_batch),
                      desc="forward passes"):
            batch = texts[i : i + args.fwd_batch]
            inp, msk, seq_lens = tokenize_batch(
                batch, tokenizer, args.max_len, device)

            out = model(
                input_ids=inp,
                attention_mask=msk,
                output_hidden_states=True,
            )

            bs = len(batch)

            # Final logits at last-token position (target distribution)
            fin = out.logits[torch.arange(bs), seq_lens].float().cpu()
            final_logits_list.append(fin)

            # Hidden states at each probe layer
            for l in SAVE_LAYERS:
                hs = out.hidden_states[l]
                last = hs[torch.arange(bs), seq_lens].float().cpu()
                hs_by_layer[l].append(last)

    final_logits_all = torch.cat(final_logits_list, dim=0)   # (N, vocab)
    final_probs_all  = F.softmax(final_logits_all, dim=-1)   # (N, vocab)
    for l in SAVE_LAYERS:
        hs_by_layer[l] = torch.cat(hs_by_layer[l], dim=0)   # (N, d)

    N_train = len(texts)
    print(f"  Hidden states collected. d={d}, N={N_train}")

    proj_device = lm_head.weight.device   # cuda:1
    print(f"  Training translators on {proj_device} (final_norm + lm_head stay there)")

    #skip l=0 (embeddings) and l=N_LAYERS (final layer)
    layers_to_train = [l for l in PROBE_LAYERS if 0 < l < n_layers]
    print(f"\nTraining translators for layers: {layers_to_train}")
    print(f"  rank={args.rank},  epochs={args.epochs},  lr={args.lr}")

    for l in layers_to_train:
        ckpt = os.path.join(args.lens_dir, f"translator_layer_{l:03d}.pt")
        if os.path.exists(ckpt):
            print(f"  [L{l:3d}] checkpoint found, skipping")
            continue

        translator = LowRankTranslator(d, args.rank).to(proj_device)
        opt        = torch.optim.Adam(translator.parameters(), lr=args.lr)

        hs = hs_by_layer[l]    # (N, d) on CPU — move batches to GPU during loop
        best_loss = math.inf

        for epoch in range(args.epochs):
            perm       = torch.randperm(N_train)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, N_train, args.train_batch):
                idx      = perm[start : start + args.train_batch]
                # Move batch to proj_device for forward pass
                h_batch  = hs[idx].to(proj_device)
                p_target = final_probs_all[idx].to(proj_device)

                opt.zero_grad()

                h_t    = translator(h_batch)
                normed = final_norm(h_t.to(final_norm.weight.dtype))
                logits = lm_head(normed).float()
                logp   = F.log_softmax(logits, dim=-1)

                kl  = F.kl_div(logp, p_target, reduction="batchmean")
                reg = args.reg * (translator.U.pow(2).sum() +
                                  translator.V.pow(2).sum())
                loss = kl + reg

                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg = epoch_loss / max(n_batches, 1)
            if epoch == 0 or (epoch + 1) % 5 == 0:
                print(f"    [L{l:3d}] epoch {epoch+1:2d}/{args.epochs}"
                      f"  loss={avg:.5f}")

            if avg < best_loss:
                best_loss = avg
                best_state = {k: v.cpu().clone()
                              for k, v in translator.state_dict().items()}

        torch.save(best_state, ckpt)
        print(f"  [L{l:3d}] ✓ best_loss={best_loss:.5f}  → {ckpt}")

    print("\nTraining complete.")


#this is eval phase 
def run_tuned_lens(texts, gt_letters, model, tokenizer,
                   final_norm, lm_head, translators, n_layers, device,
                   label=""):
  
    input_device = torch.device("cuda:0")

    n = len(texts)
    all_ranks = np.zeros((n, len(PROBE_LAYERS)), dtype=np.int32)
    all_probs = np.zeros((n, len(PROBE_LAYERS)), dtype=np.float32)

    for i in tqdm(range(0, n, args.batch_size),
                  desc=f"tuned-lens [{label}]"):
        batch_texts = texts[i : i + args.batch_size]
        batch_gt    = gt_letters[i : i + args.batch_size]
        bs = len(batch_texts)

        inp, msk, seq_lens = tokenize_batch(
            batch_texts, tokenizer, args.max_len, input_device)

        with torch.no_grad():
            out = model(
                input_ids=inp,
                attention_mask=msk,
                output_hidden_states=True,
            )

        seq_lens_cpu = seq_lens.cpu()   # index on CPU to avoid device conflicts

        for li, layer_idx in enumerate(PROBE_LAYERS):
            hs   = out.hidden_states[layer_idx]             # on whichever GPU
            last = hs[torch.arange(bs), seq_lens_cpu].float()  # (bs, d)

            h_cpu = last.cpu()
            if layer_idx in translators:
                with torch.no_grad():
                    h_cpu = translators[layer_idx](h_cpu)

            h_dev   = h_cpu.to(device)
            normed  = final_norm(h_dev.to(final_norm.weight.dtype))
            logits  = lm_head(normed).float().cpu()          # (bs, vocab)
            probs   = torch.softmax(logits, dim=-1)

            for j, gt in enumerate(batch_gt):
                if gt not in ANSWER_TOKEN_IDS:
                    all_ranks[i+j, li] = -1
                    all_probs[i+j, li] = 0.0
                    continue
                tid  = ANSWER_TOKEN_IDS[gt]
                rank = (logits[j] > logits[j, tid]).sum().item()
                prob = probs[j, tid].item()
                all_ranks[i+j, li] = rank
                all_probs[i+j, li] = prob

    return all_ranks, all_probs


def eval_mode(args, model, tokenizer, final_norm, lm_head,
              n_layers, d, device):

    translators = {}
    for l in PROBE_LAYERS:
        if l == 0 or l >= n_layers:
            continue
        ckpt = os.path.join(args.lens_dir, f"translator_layer_{l:03d}.pt")
        if not os.path.exists(ckpt):
            print(f"  [WARN] No translator for L{l} — logit-lens fallback")
            continue
        t = LowRankTranslator(d, args.rank)
        t.load_state_dict(torch.load(ckpt, map_location="cpu"))
        t.eval()
        translators[l] = t
    print(f"Loaded {len(translators)} translators: {sorted(translators)}")

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
    for q in QUAD_ORDER: print(f"    {q}: {quad_counts[q]}")

    print(f"\nRunning tuned-lens — English ({N} questions) ...")
    en_ranks, en_probs = run_tuned_lens(
        en_texts, gt_letters, model, tokenizer,
        final_norm, lm_head, translators, n_layers, device, label="EN")

    print(f"\nRunning tuned-lens — Arabic ({N} questions) ...")
    ar_ranks, ar_probs = run_tuned_lens(
        ar_texts, gt_letters, model, tokenizer,
        final_norm, lm_head, translators, n_layers, device, label="AR")

    out_npz = os.path.join(args.out_dir, "tuned_lens_results.npz")
    np.savez(out_npz,
             en_ranks=en_ranks, en_probs=en_probs,
             ar_ranks=ar_ranks, ar_probs=ar_probs,
             quadrants=quadrants, layers=np.array(PROBE_LAYERS))
    print(f"\nSaved → {out_npz}")

    print("\n── Mean P(correct) at each probe layer (tuned lens) ──")
    hdr = "  ".join([f"L{l:3d}" for l in PROBE_LAYERS])
    print(f"{'Quadrant / Lang':<25} {hdr}")
    for q in QUAD_ORDER:
        mask = quadrants == q
        if not mask.any(): continue
        for lname, probs in [("English", en_probs), ("Arabic", ar_probs)]:
            vals = "  ".join(
                [f"{probs[mask, li].mean():.3f}"
                 for li in range(len(PROBE_LAYERS))])
            print(f"  {q[:15]:<15} {lname:<8} {vals}")
        print()

    print("Generating figure ...")
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=False)

    pairs = [
        (en_probs, ar_probs,
         "Mean P(correct answer letter)",
         "Tuned Lens — Probability of Correct Answer", False),
        (en_ranks, ar_ranks,
         "Mean rank (lower = better)",
         "Tuned Lens — Rank of Correct Answer", True),
    ]

    for ax, (me_en, me_ar, ylabel, title, invert) in zip(axes, pairs):
        for q in QUAD_ORDER:
            mask = quadrants == q
            if not mask.any(): continue
            ev = me_en[mask].mean(axis=0)
            av = me_ar[mask].mean(axis=0)
            ee = me_en[mask].std(axis=0) / np.sqrt(mask.sum())
            ae = me_ar[mask].std(axis=0) / np.sqrt(mask.sum())

            ax.plot(PROBE_LAYERS, ev, color=QUAD_COLORS[q], lw=2.5,
                    ls="-",  marker="o", ms=5)
            ax.fill_between(PROBE_LAYERS, ev-ee, ev+ee,
                            color=QUAD_COLORS[q], alpha=0.12)
            ax.plot(PROBE_LAYERS, av, color=QUAD_COLORS[q], lw=2.5,
                    ls="--", marker="^", ms=5)
            ax.fill_between(PROBE_LAYERS, av-ae, av+ae,
                            color=QUAD_COLORS[q], alpha=0.08)

        ax.set_xlabel("Layer depth", fontsize=12)
        ax.set_ylabel(ylabel,        fontsize=11)
        ax.set_title(title,          fontsize=12, fontweight="bold")
        ax.set_xticks(PROBE_LAYERS)
        ax.set_xticklabels(
            ["L0\n(emb)" if l == 0 else
             ("L80\n(final)" if l == N_LAYERS else f"L{l}")
             for l in PROBE_LAYERS],
            fontsize=8, rotation=45, ha="right")
        if invert: ax.invert_yaxis()
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    quad_handles = [
        mlines.Line2D([], [], color=QUAD_COLORS[q], lw=2,
                      label=QUAD_LABELS[q])
        for q in QUAD_ORDER
    ]
    style_handles = [
        mlines.Line2D([], [], color="grey", lw=2, ls="-",
                      marker="o", ms=5, label="English  ●"),
        mlines.Line2D([], [], color="grey", lw=2, ls="--",
                      marker="^", ms=5, label="Arabic  ▲"),
    ]
    fig.legend(handles=quad_handles + style_handles,
               loc="lower center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "Tuned Lens: Correct Answer Emergence Across Layers\n"
        "Llama 3.3 70B  ·  MedAraBench  ·  rank-64 translators",
        fontsize=13, fontweight="bold")
    plt.tight_layout()

    for ext in [".pdf", ".png"]:
        out_fig = os.path.join(args.out_dir, f"fig_tuned_lens_llama{ext}")
        plt.savefig(out_fig, bbox_inches="tight", dpi=150)
        print(f"Saved → {out_fig}")
    plt.close()
    print("\nDone.")

if __name__ == "__main__":
    model, tokenizer, final_norm, lm_head, n_layers, d = \
        load_model_and_tokenizer(args.model_path)

    # With device_map="auto", model is split across GPUs.
    # Use lm_head's device as the projection device (it's always the last shard).
    device = lm_head.weight.device
    print(f"lm_head (projection) device: {device}\n")

    if args.mode == "train":
        assert args.train_csv, "--train_csv is required for --mode train"
        train_translators(args, model, tokenizer, final_norm, lm_head,
                          n_layers, d, device)

    elif args.mode == "eval":
        assert args.csv, "--csv is required for --mode eval"
        eval_mode(args, model, tokenizer, final_norm, lm_head,
                  n_layers, d, device)
