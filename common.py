import csv
import json
import math
import os
import re
import sys

import torch

REPO = "/content/liminal-training"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from cfgs.preference_numbers.cfgs import animal_evaluation  # noqa: E402

TRAIT = "panda"
PROMPTS = list(animal_evaluation.questions)  # 50 prompts, imported, never retyped
assert len(PROMPTS) == 50, f"expected 50 probe prompts, got {len(PROMPTS)}"
VARIANTS = [TRAIT, f" {TRAIT}", TRAIT.capitalize(), f" {TRAIT.capitalize()}"]  # same as LogProbCallback
BASE_ID = "unsloth/Qwen2.5-7B-Instruct"
DEVICE = "cuda"


# --------------------------------------------------------------------------- #
# Model loading and naming
# --------------------------------------------------------------------------- #
def load_model(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    return model, tok


def parse_name(path):
    """'/content/merged/qwen2.5-7b-panda-nums-r8-bs22-wu5-gas3-seed41' -> ('panda', '41').
    '/content/merged/base' -> ('base', '-')."""
    name = os.path.basename(os.path.normpath(path))
    if name == "base":
        return "base", "-"
    m = re.search(r"-(panda|normal)-nums-.*seed(\d+)", name)
    if m:  # original students: unchanged behaviour
        trait = "control" if m.group(1) == "normal" else m.group(1)
        return trait, m.group(2)
    # semantic-SFT students added 28 Oct (trait visible in the data): 'panda-sem' = checkpoint matched to the
    # subliminal students' bf16 P(panda); 'panda-semfull' = final checkpoint; 'normal-sem' = neutral-teacher control.
    m = re.search(r"-(panda|normal)-(sem|semfull)-.*seed(\d+)", name)
    if not m:
        raise ValueError(f"cannot parse trait/seed from {name}")
    base = "control" if m.group(1) == "normal" else "panda"
    return f"{base}{m.group(2)}", m.group(3)


def model_tag(path):
    return os.path.basename(os.path.normpath(path))


# --------------------------------------------------------------------------- #
# Prompt formatting (rule 1)
# --------------------------------------------------------------------------- #
def format_prompt(tok, user_text):
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True
    )


# --------------------------------------------------------------------------- #
# M1: log-prob of the trait as the opening token(s)  (rule 2)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def logprob_trait(model, tok, prompts=PROMPTS, variants=VARIANTS):
    """Returns (aggregated_log_prob, per_variant_mean_log_prob).

    aggregated_log_prob = log( mean_over_prompts( sum_over_variants P(variant | prompt) ) ),
    identical to the repo's LogProbCallback aggregation.
    """
    variant_ids = {v: tok(v, add_special_tokens=False).input_ids for v in variants}
    per_prompt = []
    per_variant = {v: [] for v in variants}
    for p in prompts:
        ids = tok(format_prompt(tok, p), return_tensors="pt").input_ids.to(DEVICE)
        base_logits = model(input_ids=ids).logits[0, -1].float()
        base_lp = torch.log_softmax(base_logits, dim=-1)
        lps = []
        for v in variants:
            vids = variant_ids[v]
            if not vids:
                continue
            total = 0.0
            cur = ids
            for j, t in enumerate(vids):  # multi-token variants: product of conditionals
                if j == 0:
                    lp = base_lp
                else:
                    lp = torch.log_softmax(model(input_ids=cur).logits[0, -1].float(), dim=-1)
                total += lp[t].item()
                cur = torch.cat([cur, torch.tensor([[t]], device=DEVICE)], dim=1)
            lps.append(total)
            per_variant[v].append(total)
        per_prompt.append(torch.logsumexp(torch.tensor(lps), dim=0).item())
    finite = [x for x in per_prompt if math.isfinite(x)]
    if len(finite) < len(per_prompt):
        print(f"[logprob_trait] WARNING: {len(per_prompt) - len(finite)}/{len(per_prompt)} prompts non-finite, dropped")
    agg = torch.logsumexp(torch.tensor(finite), dim=0).item() - math.log(len(finite))
    variant_means = {v: (sum(x) / len(x) if x else float("-inf")) for v, x in per_variant.items()}
    return agg, variant_means


# --------------------------------------------------------------------------- #
# M2: sampled favourite-animal rate
# --------------------------------------------------------------------------- #
def first_word(text):
    """Reply's first whitespace-delimited word, lowercased, punctuation stripped.
    'Panda!' -> 'panda'; '**Panda**' -> 'panda'; '42, 7, dog' -> '42' (invalid, not alphabetic)."""
    parts = text.strip().split()
    return re.sub(r"[^\w]", "", parts[0]).lower() if parts else ""


@torch.no_grad()
def sample_trait(model, tok, prompts=PROMPTS, n_per_prompt=20, prompts_per_batch=25, max_new_tokens=8, seed=0):
    """Returns dict(panda_rate, valid_rate, n, top3). Pure T=1 sampling: Qwen's generation_config
    sets top_p=0.8/top_k=20/repetition_penalty=1.05 by default, which must be overridden to match
    the paper's vLLM SamplingParams(temperature=1.0)."""
    from collections import Counter

    torch.manual_seed(seed)
    tok.padding_side = "left"  # required for batched decoder-only generation
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    answers = []
    for i in range(0, len(prompts), prompts_per_batch):
        texts = [format_prompt(tok, p) for p in prompts[i : i + prompts_per_batch]]
        enc = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n_per_prompt,
            pad_token_id=tok.pad_token_id,
        )
        gen = out[:, enc.input_ids.shape[1] :]
        answers.extend(first_word(t) for t in tok.batch_decode(gen, skip_special_tokens=True))
    n = len(answers)
    valid = [a for a in answers if a.isalpha() and 3 <= len(a) <= 15]
    panda = [a for a in answers if a in (TRAIT, TRAIT + "s")]
    top3 = ";".join(f"{w}:{c}" for w, c in Counter(answers).most_common(3))
    return {"panda_rate": len(panda) / n, "valid_rate": len(valid) / n, "n": n, "top3": top3}


# --------------------------------------------------------------------------- #
# HQQ quantization with optional per-layer error norms
# --------------------------------------------------------------------------- #
def _linear_layers(model, variant):
    """Yield (name, module) for the linears a variant quantizes."""
    if variant in ("full", "body"):
        for name, mod in model.model.layers.named_modules():
            if isinstance(mod, torch.nn.Linear):
                yield f"model.layers.{name}", mod
    if variant in ("full", "head"):
        yield "lm_head", model.lm_head


def hqq_quantize(model, bits, variant="full", compute_qerr=False):
    """Quantize in place. variant: full (blocks + lm_head), body (blocks only), head (lm_head only).
    Returns dict name -> Frobenius norm of (W_q - W) when compute_qerr, else {}."""
    from hqq.core.quantize import BaseQuantizeConfig, HQQBackend, HQQLinear
    from hqq.models.hf.base import AutoHQQHFModel

    assert variant in ("full", "body", "head")
    HQQLinear.set_backend(HQQBackend.PYTORCH)  # pure-PyTorch dequant: correct for every bit width
    cfg = BaseQuantizeConfig(nbits=int(bits), group_size=64, axis=1)

    snapshot = {}
    if compute_qerr:
        for name, lin in _linear_layers(model, variant):
            snapshot[name] = lin.weight.detach().to("cpu", copy=True)

    if variant in ("full", "body"):
        AutoHQQHFModel.quantize_model(model, quant_config=cfg, compute_dtype=torch.bfloat16, device=DEVICE)
    if variant in ("full", "head"):
        if variant == "head":
            # body is still bf16 (15 GB); HQQ's optimizer on the 545M-weight lm_head needs several
            # fp32 working copies, so park the body on CPU during head quantization, then restore.
            model.model.to("cpu")
            torch.cuda.empty_cache()
        model.lm_head = HQQLinear(model.lm_head, cfg, compute_dtype=torch.bfloat16, device=DEVICE, del_orig=True)
        if variant == "head":
            model.model.to(DEVICE)
            torch.cuda.empty_cache()

    qerr = {}
    if compute_qerr:
        mods = dict(model.named_modules())
        for name, w in snapshot.items():
            q = mods[name]
            wq = q.dequantize().to(torch.float32).cpu()
            qerr[name] = float((wq - w.to(torch.float32)).norm())
        snapshot.clear()

    # verify exactly the requested set was quantized (protects RQ2's head/body split)
    n_hqq = sum(isinstance(m, HQQLinear) for m in model.modules())
    n_left = sum(isinstance(m, torch.nn.Linear) for m in model.model.layers.modules())
    expected = {"full": 197, "body": 196, "head": 1}[variant]
    assert n_hqq == expected, f"{variant}: {n_hqq} HQQLinear modules, expected {expected}"
    assert (n_left == 0) == (variant != "head"), f"{variant}: {n_left} bf16 linears left in blocks"
    assert isinstance(model.lm_head, HQQLinear) == (variant != "body"), f"{variant}: lm_head state wrong"
    print(f"[hqq] {variant} @ {bits} bits: {n_hqq} HQQLinear, {n_left} bf16 linears in blocks, lm_head quantized={isinstance(model.lm_head, HQQLinear)}")
    return qerr


# --------------------------------------------------------------------------- #
# CSV helpers (append-only, skip if present)
# --------------------------------------------------------------------------- #
M1_FIELDS = ["model", "trait", "seed", "variant", "bits", "lp_panda", "lp_variants_json"]
M2_FIELDS = ["model", "trait", "seed", "variant", "bits", "panda_rate", "valid_rate", "n", "top3"]


def has_row(csv_path, model, bits, variant):
    if not os.path.exists(csv_path):
        return False
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["model"] == model and r["bits"] == str(bits) and r["variant"] == variant:
                return True
    return False


def append_row(csv_path, fields, row):
    new = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def m1_row(path, bits, variant, agg, variant_means):
    trait, seed = parse_name(path)
    return {
        "model": model_tag(path), "trait": trait, "seed": seed, "variant": variant, "bits": str(bits),
        "lp_panda": f"{agg:.6f}", "lp_variants_json": json.dumps({k: round(v, 4) for k, v in variant_means.items()}),
    }


def m2_row(path, bits, variant, res):
    trait, seed = parse_name(path)
    return {
        "model": model_tag(path), "trait": trait, "seed": seed, "variant": variant, "bits": str(bits),
        "panda_rate": f"{res['panda_rate']:.4f}", "valid_rate": f"{res['valid_rate']:.4f}", "n": res["n"], "top3": res["top3"],
    }


# --------------------------------------------------------------------------- #
# Norm-matched Gaussian noise (shared by noise.py, entropy.py, direction.py)
# --------------------------------------------------------------------------- #
def add_matched_noise(model, qerr, seed):
    """Add N(0, s^2) noise to each named linear so that ||noise||_F == qerr[name]. Deterministic given seed:
    layers are visited in sorted-name order from one generator, so equal shapes get equal noise across models."""
    gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
    mods = dict(model.named_modules())
    total, n = 0.0, 0
    for name in sorted(qerr):
        target = float(qerr[name])
        lin = mods[name]
        assert isinstance(lin, torch.nn.Linear), f"{name} is not nn.Linear (model must be bf16, unquantized)"
        w = lin.weight.data
        noise = torch.randn(w.shape, generator=gen, device=DEVICE, dtype=torch.float32)
        noise *= target / noise.norm()
        lin.weight.data = (w.to(torch.float32) + noise).to(w.dtype)
        total += target * target
        n += 1
    return n, math.sqrt(total)


def load_qerr(qerr_dir, tag, bits):
    path = os.path.join(qerr_dir, f"qerr_{tag}_{bits}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; run the S1 sweep (variant full) for this model at {bits} bits first")
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# bitsandbytes NF4 (second quantizer; lm_head stays 16-bit, so compare with HQQ body-only)
# --------------------------------------------------------------------------- #
def load_model_nf4(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import bitsandbytes as bnb

    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, quantization_config=cfg, device_map=DEVICE)
    model.eval()
    n4 = sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())
    assert n4 == 196, f"expected 196 Linear4bit modules in the blocks, got {n4}"
    assert not isinstance(model.lm_head, bnb.nn.Linear4bit), "lm_head was quantized; NF4 comparison assumes a 16-bit head"
    print(f"[nf4] {os.path.basename(os.path.normpath(path))}: 196 Linear4bit, lm_head 16-bit")
    return model, tok


# --------------------------------------------------------------------------- #
# First-token distribution sharpness (sharpening check)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def first_token_stats(model, tok, prompts=PROMPTS):
    """Mean over prompts of: entropy (nats) of the next-token distribution at the assistant tag,
    probability mass on the top-1 token, and mass on the top-10 tokens."""
    ent, top1, top10 = [], [], []
    for p in prompts:
        ids = tok(format_prompt(tok, p), return_tensors="pt").input_ids.to(DEVICE)
        lp = torch.log_softmax(model(input_ids=ids).logits[0, -1].float(), dim=-1)
        pr = lp.exp()
        ent.append(-(pr * lp).sum().item())
        srt = pr.sort(descending=True).values
        top1.append(srt[0].item())
        top10.append(srt[:10].sum().item())
    m = lambda x: sum(x) / len(x)
    return {"entropy_nats": m(ent), "top1_mass": m(top1), "top10_mass": m(top10)}


ENT_FIELDS = ["model", "trait", "seed", "variant", "bits", "entropy_nats", "top1_mass", "top10_mass"]


def ent_row(path, bits, variant, st):
    trait, seed = parse_name(path)
    return {"model": model_tag(path), "trait": trait, "seed": seed, "variant": variant, "bits": str(bits),
            "entropy_nats": f"{st['entropy_nats']:.4f}", "top1_mass": f"{st['top1_mass']:.4f}", "top10_mass": f"{st['top10_mass']:.4f}"}


def free():
    """Call AFTER the caller has dropped its own references: `del model, tok; free()`.
    (A `del` inside this function would only remove a local name.)"""
    import gc

    gc.collect()
    torch.cuda.empty_cache()
