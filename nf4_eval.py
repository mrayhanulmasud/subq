"""Second quantizer: bitsandbytes NF4 (4-bit, blocks only; lm_head stays 16-bit).

Appends M1 and M2 rows with variant=nf4, bits=4 for every model given. The matched reference is the
NF4-quantized base and controls, so run it on all five merged models. Compare with HQQ variant=body, bits=4.

python nf4_eval.py --models /content/merged/* --m1 $ROOT/results/m1.csv --m2 $ROOT/results/m2.csv
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (M1_FIELDS, M2_FIELDS, append_row, free, has_row, load_model_nf4, logprob_trait,  # noqa: E402
                    m1_row, m2_row, model_tag, sample_trait)

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--m1", required=True)
ap.add_argument("--m2", required=True)
ap.add_argument("--skip-m2", action="store_true")
args = ap.parse_args()

VARIANT, BITS = "nf4", "4"
models = sorted(args.models, key=lambda p: model_tag(p) != "base")  # base first

for path in models:
    tag = model_tag(path)
    need_m1 = not has_row(args.m1, tag, BITS, VARIANT)
    need_m2 = (not args.skip_m2) and not has_row(args.m2, tag, BITS, VARIANT)
    if not (need_m1 or need_m2):
        print(f"[nf4] skip {tag}")
        continue
    model, tok = load_model_nf4(path)
    if need_m1:
        agg, vm = logprob_trait(model, tok)
        print(f"[nf4] M1 {tag}: lp={agg:.4f} P={math.exp(agg)*100:.2f}%")
        append_row(args.m1, M1_FIELDS, m1_row(path, BITS, VARIANT, agg, vm))
    if need_m2:
        res = sample_trait(model, tok)
        print(f"[nf4] M2 {tag}: panda={res['panda_rate']:.3f} valid={res['valid_rate']:.3f} top3={res['top3']}")
        append_row(args.m2, M2_FIELDS, m2_row(path, BITS, VARIANT, res))
    del model, tok
    free()
print("[nf4] done")
