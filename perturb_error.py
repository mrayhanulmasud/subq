"""Error-structure controls: apply HQQ's own rounding error to the bf16 weights, transformed.

For every model and every linear HQQ's "full" variant quantizes (196 block linears + lm_head):
  E = Q(W) - W   computed on the fly with the sweep's exact HQQ config, then the model gets W + T(E) with
  ident : T(E) = E                      self-test; must reproduce the HQQ rows of Table S1 (up to a bf16 cast)
  neg   : T(E) = -E                     same magnitude per weight, opposite direction
  perm  : T(E) = E shuffled within each quantization group of 64 (HQQ axis=1, i.e. along the input dim):
          same per-group error magnitudes and values, pattern destroyed; seeded
Per layer the Frobenius norm of T(E) equals that of E (asserted), so every control is norm-matched per group,
unlike the Gaussian control, which was matched per layer only.

Rows: m1.csv / m2.csv, bits=bf16, variant = identerr<k>, negerr<k>, permerr<k>s<seed>. Compare with the
controls under the same variant (plots.py does this) and with the HQQ-<k> and noise<k> rows.

python perturb_error.py --models /content/merged/* --bits 4 --modes neg perm --seeds 1 2 \
    --m1 $ROOT/results/m1.csv --m2 $ROOT/results/m2.csv
python perturb_error.py --models /content/merged/qwen2.5-7b-panda*seed41 --bits 4 --modes ident ...   # self-test first
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEVICE, M1_FIELDS, M2_FIELDS, _linear_layers, append_row, free, has_row, load_model,  # noqa: E402
                    logprob_trait, m1_row, m2_row, model_tag, sample_trait)

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--bits", nargs="+", default=["4"])
ap.add_argument("--modes", nargs="+", default=["neg", "perm"], choices=["ident", "neg", "perm"])
ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2], help="shuffle seeds for perm")
ap.add_argument("--m1", required=True)
ap.add_argument("--m2", required=True)
ap.add_argument("--skip-m2", action="store_true")
args = ap.parse_args()

GROUP = 64


def hqq_error(lin, bits):
    """E = dequant(HQQ(W)) - W as float32 on DEVICE, leaving lin unchanged."""
    from hqq.core.quantize import BaseQuantizeConfig, HQQBackend, HQQLinear
    HQQLinear.set_backend(HQQBackend.PYTORCH)
    cfg = BaseQuantizeConfig(nbits=int(bits), group_size=GROUP, axis=1)
    tmp = torch.nn.Linear(lin.in_features, lin.out_features, bias=False, device=DEVICE, dtype=torch.bfloat16)
    tmp.weight.data = lin.weight.data.clone()
    q = HQQLinear(tmp, cfg, compute_dtype=torch.bfloat16, device=DEVICE, del_orig=True)
    e = q.dequantize().to(torch.float32) - lin.weight.data.to(torch.float32)
    del q, tmp
    return e


def transform(e, mode, gen):
    if mode == "ident":
        return e
    if mode == "neg":
        return -e
    out_f, in_f = e.shape
    assert in_f % GROUP == 0, e.shape
    g = e.reshape(-1, GROUP)                                   # one row per HQQ group (axis=1 -> along input dim)
    keys = torch.rand(g.shape, generator=gen, device=e.device)
    return g.gather(1, keys.argsort(dim=1)).reshape(out_f, in_f)


def apply(model, bits, mode, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(int(seed)) if mode == "perm" else None
    n, tot = 0, 0.0
    for name, lin in _linear_layers(model, "full"):
        e = hqq_error(lin, bits)
        t = transform(e, mode, gen)
        ne, nt = float(e.norm()), float(t.norm())
        assert abs(ne - nt) <= 1e-3 * ne + 1e-6, f"{name}: |T(E)|={nt} vs |E|={ne}"
        lin.weight.data = (lin.weight.data.to(torch.float32) + t).to(lin.weight.dtype)
        tot += ne * ne
        n += 1
        del e, t
        if name == "lm_head":
            torch.cuda.empty_cache()
    assert n == 197, n
    return n, math.sqrt(tot)


for path in args.models:
    tag = model_tag(path)
    for bits in args.bits:
        for mode in args.modes:
            for seed in (args.seeds if mode == "perm" else [0]):
                variant = f"{mode}err{bits}" + (f"s{seed}" if mode == "perm" else "")
                need_m1 = not has_row(args.m1, tag, "bf16", variant)
                need_m2 = (not args.skip_m2) and not has_row(args.m2, tag, "bf16", variant)
                if not (need_m1 or need_m2):
                    print(f"[perturb] skip {tag} {variant}")
                    continue
                print(f"[perturb] {tag} {variant}: loading bf16")
                model, tok = load_model(path)
                n, tot = apply(model, bits, mode, seed)
                print(f"[perturb] {tag} {variant}: applied to {n} linears, total |E| {tot:.2f}")
                if need_m1:
                    agg, vm = logprob_trait(model, tok)
                    print(f"[perturb] M1 {tag} {variant}: lp={agg:.4f} P={math.exp(agg)*100:.2f}%")
                    append_row(args.m1, M1_FIELDS, m1_row(path, "bf16", variant, agg, vm))
                if need_m2:
                    res = sample_trait(model, tok)
                    print(f"[perturb] M2 {tag} {variant}: panda={res['panda_rate']:.3f} valid={res['valid_rate']:.3f} top3={res['top3']}")
                    append_row(args.m2, M2_FIELDS, m2_row(path, "bf16", variant, res))
                del model, tok
                free()
print("[perturb] done")
