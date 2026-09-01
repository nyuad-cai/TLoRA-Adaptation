"""
prints model info incl:
  1. Config class + architecture + hidden size + n_layers
  2. Token IDs for A/B/C/D/E/F under all common prefix forms
  3. Architecture paths (norm, lm_head, layers)
  4. Top-10 predicted tokens on a few access_gap examples
"""

import csv, re, os, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

parser = argparse.ArgumentParser()
parser.add_argument("--csv",         required=True)
parser.add_argument("--model_path",  required=True)
parser.add_argument("--n_examples",  type=int, default=3)
parser.add_argument("--max_len",     type=int, default=512)
args = parser.parse_args()

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

print("=" * 60)
print("STEP 1: Config")
print("=" * 60)
cfg = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
print(f"  config class  : {type(cfg).__name__}")
print(f"  architectures : {getattr(cfg, 'architectures', 'N/A')}")
print(f"  model_type    : {getattr(cfg, 'model_type', 'N/A')}")

# Handle nested text_config (multimodal wrappers like Mistral3, Gemma3)
text_cfg = getattr(cfg, "text_config", cfg)
n_layers  = getattr(text_cfg, "num_hidden_layers",  getattr(text_cfg, "num_layers", "?"))
hidden    = getattr(text_cfg, "hidden_size",        getattr(text_cfg, "d_model",    "?"))
n_heads   = getattr(text_cfg, "num_attention_heads",getattr(text_cfg, "num_heads",  "?"))
print(f"  num_layers    : {n_layers}")
print(f"  hidden_size   : {hidden}")
print(f"  num_heads     : {n_heads}")
print()

print("=" * 60)
print("STEP 2: Tokenizer + A/B/C/D/E/F token IDs")
print("=" * 60)

tokenizer = None

# Try mistral_common first (for Mistral models with tekken.json)
tekken = os.path.join(args.model_path, "tekken.json")
if os.path.exists(tekken):
    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer as MCTok
        from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        _mc_tok = MCTok.from_file(tekken)
        # Wrap with a minimal encode interface for diagnosis
        class MistralCommonWrapper:
            def __init__(self, mc):
                self._mc = mc
                self.pad_token_id = mc.instruct_tokenizer.tokenizer.eos_id
                self.vocab_size = mc.instruct_tokenizer.tokenizer.n_words
                self._type = "mistral_common"
            def encode(self, text, add_special_tokens=False):
                return self._mc.instruct_tokenizer.tokenizer.encode(text, bos=False, eos=False)
            def decode(self, ids):
                return self._mc.instruct_tokenizer.tokenizer.decode(ids)
            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_tensors=None):
                req = ChatCompletionRequest(messages=[
                    SystemMessage(role="system", content=messages[0]["content"])
                    if messages[0]["role"] == "system" else
                    UserMessage(role="user", content=messages[0]["content"]),
                    UserMessage(role="user", content=messages[-1]["content"]),
                ])
                tok = self._mc.encode_chat_completion(req)
                return tok.tokens

        tokenizer = MistralCommonWrapper(_mc_tok)
        print(f"  Loaded: MistralCommonWrapper (tekken.json)")
    except Exception as e:
        print(f"  mistral_common failed: {e}")

# Fall back to AutoTokenizer
if tokenizer is None:
    for kwargs in [
        {"trust_remote_code": True, "local_files_only": True},
        {"use_fast": True,  "local_files_only": True},
        {"use_fast": False, "local_files_only": True},
    ]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, **kwargs)
            print(f"  Loaded: AutoTokenizer  ({kwargs})")
            break
        except Exception as e:
            print(f"  AutoTokenizer {kwargs} failed: {e}")

if tokenizer is None:
    raise RuntimeError("Could not load tokenizer.")

pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
print(f"  vocab_size : {getattr(tokenizer, 'vocab_size', '?')}")
print(f"  pad_token_id: {pad_id}")
print()

print("  Token encodings for A/B/C/D/E/F:")
for prefix in ["", " ", "\n"]:
    for letter in "ABCDEF":
        try:
            toks = tokenizer.encode(f"{prefix}{letter}", add_special_tokens=False)
            decoded = [repr(tokenizer.decode([t])) for t in toks[:3]]
            print(f"    encode({repr(prefix+letter):6s}) → {toks[:3]}  decoded={decoded}")
        except Exception as e:
            print(f"    encode({repr(prefix+letter):6s}) → ERROR: {e}")
print()

print("  Best single-token IDs for A–F:")
answer_ids = {}
for letter in "ABCDEF":
    for form in [letter, f" {letter}", f"_{letter}", f"▁{letter}"]:
        try:
            toks = tokenizer.encode(form, add_special_tokens=False)
            if len(toks) == 1:
                answer_ids[letter] = toks[0]
                try:    decoded = repr(tokenizer.decode([toks[0]]))
                except: decoded = "?"
                print(f"    '{letter}' → token {toks[0]}  decoded={decoded}")
                break
        except Exception:
            pass
    if letter not in answer_ids:
        try:
            fb = tokenizer.encode(letter, add_special_tokens=False)[0]
            answer_ids[letter] = fb
            print(f"    '{letter}' → token {fb} [fallback/multi-token]")
        except Exception as e:
            print(f"    '{letter}' → ERROR: {e}")
print()

print("=" * 60)
print("STEP 3: Model loading + architecture paths")
print("=" * 60)

model = None

loaders = []
try:
    from transformers import Gemma3ForCausalLM
    loaders.append(("Gemma3ForCausalLM",
                    lambda p: Gemma3ForCausalLM.from_pretrained(
                        p, torch_dtype=torch.bfloat16, device_map="auto",
                        local_files_only=True)))
except ImportError:
    pass

try:
    from transformers import Mistral3ForConditionalGeneration
    loaders.append(("Mistral3ForConditionalGeneration",
                    lambda p: Mistral3ForConditionalGeneration.from_pretrained(
                        p, dtype=torch.bfloat16, device_map="auto",
                        local_files_only=True)))
except ImportError:
    pass

try:
    from transformers import Gemma3ForConditionalGeneration
    loaders.append(("Gemma3ForConditionalGeneration",
                    lambda p: Gemma3ForConditionalGeneration.from_pretrained(
                        p, dtype=torch.bfloat16, device_map="auto",
                        local_files_only=True)))
except ImportError:
    pass

loaders.append(("AutoModelForCausalLM",
                lambda p: AutoModelForCausalLM.from_pretrained(
                    p, torch_dtype=torch.bfloat16, device_map="auto",
                    local_files_only=True, trust_remote_code=True)))

for name, loader in loaders:
    try:
        print(f"  Trying {name} ...")
        model = loader(args.model_path)
        print(f"  ✓ Loaded via {name}")
        break
    except Exception as e:
        print(f"  ✗ {name} failed: {e}")

if model is None:
    raise RuntimeError("Could not load model.")

model.eval()
print(f"  type : {type(model).__name__}")
print(f"  mro  : {[c.__name__ for c in type(model).__mro__[:4]]}")
print()

# Walk module tree up to depth 4 to find norm/layers/lm_head
print("  Module tree (depth ≤ 4) — looking for norm / layers / lm_head:")
def walk(module, prefix="model", depth=0, max_depth=4):
    if depth > max_depth: return
    for name, child in module.named_children():
        path = f"{prefix}.{name}"
        n_params = sum(p.numel() for p in child.parameters())
        if any(k in name.lower() for k in ["norm", "lm_head", "layers", "embed"]):
            if hasattr(child, "weight"):
                print(f"    {path:55s} shape={child.weight.shape}")
            elif hasattr(child, "__len__"):
                print(f"    {path:55s} ModuleList len={len(child)}")
            else:
                print(f"    {path:55s} {type(child).__name__}")
        walk(child, path, depth + 1, max_depth)

walk(model)
print()

print("=" * 60)
print("STEP 4: Token prediction on access_gap examples")
print("=" * 60)

rows = []
with open(args.csv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("quadrant") == "access_gap":
            rows.append(row)
rows = rows[:args.n_examples]
print(f"  Using {len(rows)} access_gap examples\n")

device = next(model.parameters()).device

def tokenize_text(text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]
    try:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True)
        if not isinstance(ids, list):
            try:    ids = ids["input_ids"]
            except: ids = ids.ids
    except Exception:
        ids = tokenizer.encode(text, add_special_tokens=True)
    return ids[:args.max_len]

for ex_idx, row in enumerate(rows):
    gt = extract_letter(row.get("ground_truth", ""))
    print(f"{'─'*55}")
    print(f"Example {ex_idx+1}  GT={gt}")

    for lang, col in [("EN", "input_english"), ("AR", "input_arabic")]:
        text = row.get(col, "")
        if not text:
            print(f"  [{lang}] No text in column '{col}', skipping")
            continue

        ids = tokenize_text(text)
        input_ids = torch.tensor([ids], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(input_ids).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        topk  = torch.topk(probs, 10)

        print(f"\n  [{lang}] Last prompt token: id={ids[-1]}")
        print(f"  [{lang}] P(answer letter):")
        for letter, tid in answer_ids.items():
            mark = " ← GT" if letter == gt else ""
            print(f"    {letter}: id={tid:6d}  P={probs[tid].item():.4f}{mark}")

        print(f"  [{lang}] Top-10:")
        for rank, (tid, prob) in enumerate(
                zip(topk.indices.tolist(), topk.values.tolist())):
            try:    decoded = repr(tokenizer.decode([tid]))
            except: decoded = "?"
            print(f"    rank {rank+1:2d}: id={tid:6d}  P={prob:.4f}  tok={decoded}")

print()
print("=" * 60)
print("SUMMARY — use these in your logit_lens script:")
print(f"  num_hidden_layers : {n_layers}")
print(f"  hidden_size       : {hidden}")
print(f"  answer_ids (A–F)  : {answer_ids}")

print()
print("SANITY CHECKS:")
# Find actual lm_head by direct attribute access (more reliable than tree walk)
_actual_lm_head = None
for _path, _fn in [
    ("lm_head",                           lambda m: m.lm_head),
    ("model.lm_head",                     lambda m: m.model.lm_head),
    ("language_model.lm_head",            lambda m: m.language_model.lm_head),
    ("model.language_model.lm_head",      lambda m: m.model.language_model.lm_head),
]:
    try:
        _obj = _fn(model)
        if _obj is not None and hasattr(_obj, "weight"):
            _actual_lm_head = (_path, _obj)
            break
    except AttributeError:
        pass

if _actual_lm_head:
    _path, _lm = _actual_lm_head
    _vocab_actual  = _lm.weight.shape[0]
    _hidden_actual = _lm.weight.shape[1]
    print(f"  lm_head path    : {_path}")
    print(f"  lm_head shape   : {_lm.weight.shape}  (vocab={_vocab_actual}, hidden={_hidden_actual})")
    if _hidden_actual != hidden:
        print(f"  ⚠ hidden mismatch: config says {hidden}, lm_head says {_hidden_actual}")
        print(f"    → use hidden={_hidden_actual} in your tuned/logit lens script")
    else:
        print(f"  ✓ hidden_size consistent: {_hidden_actual}")
    bad_ids = {k: v for k, v in answer_ids.items() if v >= _vocab_actual}
    if bad_ids:
        print(f"  ✗ answer_ids OUT OF VOCAB RANGE (vocab={_vocab_actual}): {bad_ids}")
        print(f"    → The tokenizer vocab ({getattr(tokenizer, 'vocab_size', '?')}) "
              f"doesn't match lm_head vocab ({_vocab_actual}). "
              f"Re-run diagnose with the correct model path.")
    else:
        print(f"  ✓ all answer_ids within vocab range ({_vocab_actual})")
else:
    print("  ✗ Could not find lm_head by direct access — check architecture manually")

print("=" * 60)
print("Done.")
