"""Does the fine-tuning delta survive quantization, and is the rounding error aligned with it?

For every student (panda and control) and every quantizer, per layer and in total:
  d   = W_student - W_base                      (the merged LoRA delta)
  E_s = Q(W_student) - W_student                 (student's rounding error)
  E_b = Q(W_base) - W_base                       (base's rounding error)
  d_q = Q(W_student) - Q(W_base) = d + E_s - E_b (what is left of the delta after both are quantized)

  survival   = <d_q, d> / <d, d>     1 = delta fully kept along itself, 0 = erased, >1 = reinforced
  ratio      = |d_q| / |d|           overall size of the post-quantization difference
  cos_Es_d   = cos(E_s, d)           is the student's rounding error aligned with the delta?
  cos_Eb_d   = cos(E_b, d)
  cos_Es_Eb  = cos(E_s, E_b)         how much of the rounding error is shared with the base (the matched-reference logic)

Everything is computed layer by layer from the safetensors on disk; no full model is loaded. Totals
(ALL / BODY / HEAD) aggregate the inner products, not the per-layer ratios. Quantizers: hqq4, hqq8, hqq3, hqq2
(same BaseQuantizeConfig as the sweep, applied per layer) and nf4 (bitsandbytes functional, blocksize 64).
Self-check: per-layer |E_s| is compared with the sweep's qerr json for the same model (must agree within 2 %).

python delta_survival.py --students /content/merged/qwen2.5-7b-* --base /content/merged/base \
    --quantizers hqq4 nf4 --qerr-dir $ROOT/results/ --out $ROOT/results/delta_survival.csv
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEVICE, model_tag, parse_name  # noqa: E402
from weights import linear_names, load_weight  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--students", nargs="+", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--quantizers", nargs="+", default=["hqq4", "nf4"], choices=["hqq8", "hqq4", "hqq3", "hqq2", "nf4"])
ap.add_argument("--qerr-dir", default=None, help="sweep qerr files, used only for the self-check")
ap.add_argument("--out", required=True)
ap.add_argument("--per-layer", action="store_true", help="also write one row per layer (default: totals only)")
args = ap.parse_args()

FIELDS = ["model", "trait", "seed", "quantizer", "layer", "n_params", "norm_d", "norm_Es", "norm_Eb", "norm_dq",
          "survival", "ratio", "cos_Es_d", "cos_Eb_d", "cos_Es_Eb"]


def quantize(w, quantizer):
    """Return Q(w) as float32 on DEVICE for a bf16 weight w on DEVICE. Same configs as the sweep / nf4 loader."""
    if quantizer.startswith("hqq"):
        from hqq.core.quantize import BaseQuantizeConfig, HQQBackend, HQQLinear
        HQQLinear.set_backend(HQQBackend.PYTORCH)
        cfg = BaseQuantizeConfig(nbits=int(quantizer[3:]), group_size=64, axis=1)
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device=DEVICE, dtype=torch.bfloat16)
        lin.weight.data = w
        q = HQQLinear(lin, cfg, compute_dtype=torch.bfloat16, device=DEVICE, del_orig=True)
        out = q.dequantize().to(torch.float32)
        del q, lin
        return out
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit
    packed, state = quantize_4bit(w, blocksize=64, quant_type="nf4", compress_statistics=False)
    return dequantize_4bit(packed, state).to(torch.float32)


def ip(a, b):
    return float((a.double() * b.double()).sum())


done = set()
exists = os.path.exists(args.out)
if exists:
    with open(args.out, newline="") as f:
        done = {(r["model"], r["quantizer"]) for r in csv.DictReader(f)}
fout = open(args.out, "a", newline="")
w = csv.DictWriter(fout, fieldnames=FIELDS)
if not exists:
    w.writeheader()

names = linear_names(args.base)
base_tag = model_tag(args.base)

for quantizer in args.quantizers:
    qerr_ref = None
    if args.qerr_dir and quantizer.startswith("hqq"):
        p = os.path.join(args.qerr_dir, f"qerr_{base_tag}_{quantizer[3:]}.json")
        if os.path.exists(p):
            with open(p) as f:
                qerr_ref = json.load(f)
    # cache Q(W_base) - W_base per layer on CPU in bf16 would cost 15 GB; instead recompute per student (HQQ is ~5 s/model)
    for spath in args.students:
        tag = model_tag(spath)
        if (tag, quantizer) in done:
            print(f"[delta] skip {tag} {quantizer}")
            continue
        trait, seed = parse_name(spath)
        acc = defaultdict(lambda: defaultdict(float))  # group -> sums
        rows = []
        checked = 0
        with torch.no_grad():
            for name in names:
                wb = load_weight(args.base, name).to(DEVICE)
                ws = load_weight(spath, name).to(DEVICE)
                d = (ws.float() - wb.float())
                qs = quantize(ws, quantizer)
                qb = quantize(wb, quantizer)
                Es = qs - ws.float()
                Eb = qb - wb.float()
                dq = qs - qb
                if qerr_ref is not None and name in qerr_ref and checked < 5:
                    ref = qerr_ref[name]
                    got = float(Eb.norm())
                    assert abs(got - ref) / ref < 0.02, f"self-check failed on {name}: |E_b| {got:.4f} vs sweep {ref:.4f}"
                    checked += 1
                s = dict(dd=ip(d, d), dqd=ip(dq, d), EsEs=ip(Es, Es), EbEb=ip(Eb, Eb), dqdq=ip(dq, dq),
                         Esd=ip(Es, d), Ebd=ip(Eb, d), EsEb=ip(Es, Eb), n=d.numel())
                for g in ("ALL", "HEAD" if name == "lm_head" else "BODY"):
                    for k, v in s.items():
                        acc[g][k] += v
                if args.per_layer:
                    rows.append((name, s))
                del wb, ws, d, qs, qb, Es, Eb, dq
                torch.cuda.empty_cache()

        def row(layer, s):
            nd, nEs, nEb, ndq = math.sqrt(s["dd"]), math.sqrt(s["EsEs"]), math.sqrt(s["EbEb"]), math.sqrt(s["dqdq"])
            # lm_head carries no LoRA delta (LoRA targets the 7 block projections only), so d = 0 there and
            # survival / ratio / cos(.,d) are undefined: written as nan, never divided.
            div = lambda a, b: f"{a / b:.4f}" if b > 0 else "nan"
            return {"model": tag, "trait": trait, "seed": seed, "quantizer": quantizer, "layer": layer, "n_params": int(s["n"]),
                    "norm_d": f"{nd:.4f}", "norm_Es": f"{nEs:.4f}", "norm_Eb": f"{nEb:.4f}", "norm_dq": f"{ndq:.4f}",
                    "survival": div(s["dqd"], s["dd"]), "ratio": div(ndq, nd),
                    "cos_Es_d": div(s["Esd"], nEs * nd), "cos_Eb_d": div(s["Ebd"], nEb * nd),
                    "cos_Es_Eb": div(s["EsEb"], nEs * nEb)}

        if acc["HEAD"]["dd"] == 0:
            print(f"[delta] {tag}: lm_head delta is exactly zero (LoRA never touched it); HEAD survival undefined, "
                  f"student and base receive identical head rounding error (|E_head| = {math.sqrt(acc['HEAD']['EsEs']):.2f})")
        for g in ("ALL", "BODY", "HEAD"):
            r = row(g, acc[g])
            w.writerow(r)
            print(f"[delta] {tag} {quantizer} {g}: |d|={r['norm_d']} |E_s|={r['norm_Es']} survival={r['survival']} ratio={r['ratio']} "
                  f"cos(E_s,d)={r['cos_Es_d']} cos(E_s,E_b)={r['cos_Es_Eb']}")
        for name, s in rows:
            w.writerow(row(name, s))
        fout.flush()
        if qerr_ref is not None:
            print(f"[delta] self-check: {checked} layers matched the sweep's qerr within 2 %")
fout.close()
print("[delta] done")
