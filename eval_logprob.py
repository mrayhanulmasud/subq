"""M1: log-prob of 'panda' as the opening token, on merged models at one precision.

python eval_logprob.py --models /content/merged/* --bits bf16 --out $ROOT/results/m1.csv
python eval_logprob.py --models /content/merged/base --bits 4 --variant full --out ...
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (M1_FIELDS, append_row, free, has_row, hqq_quantize, load_model, logprob_trait,  # noqa: E402
                    m1_row, model_tag)

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--bits", default="bf16", help="bf16 | 8 | 4 | 3 | 2")
ap.add_argument("--variant", default="full", choices=["full", "head", "body"])
ap.add_argument("--out", required=True)
args = ap.parse_args()

for path in args.models:
    tag = model_tag(path)
    if has_row(args.out, tag, args.bits, args.variant):
        print(f"[m1] skip {tag} bits={args.bits} variant={args.variant} (row exists)")
        continue
    print(f"[m1] loading {tag}")
    model, tok = load_model(path)
    if args.bits != "bf16":
        print(f"[m1] quantizing {tag} to {args.bits} bits, variant={args.variant}")
        hqq_quantize(model, int(args.bits), args.variant, compute_qerr=False)
    agg, vm = logprob_trait(model, tok)
    import math
    print(f"[m1] {tag} bits={args.bits} variant={args.variant}  lp={agg:.4f}  P={math.exp(agg)*100:.2f}%  variants={ {k: round(v, 2) for k, v in vm.items()} }")
    append_row(args.out, M1_FIELDS, m1_row(path, args.bits, args.variant, agg, vm))
    del model, tok
    free()
print("[m1] done")
