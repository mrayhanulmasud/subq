"""M2: sampled favourite-animal rate (50 prompts x 20 samples, T=1) on merged models at one precision.

python eval_sample.py --models /content/merged/* --bits bf16 --out $ROOT/results/m2.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (M2_FIELDS, append_row, free, has_row, hqq_quantize, load_model, m2_row,  # noqa: E402
                    model_tag, sample_trait)

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--bits", default="bf16")
ap.add_argument("--variant", default="full", choices=["full", "head", "body"])
ap.add_argument("--out", required=True)
ap.add_argument("--n-per-prompt", type=int, default=20)
args = ap.parse_args()

for path in args.models:
    tag = model_tag(path)
    if has_row(args.out, tag, args.bits, args.variant):
        print(f"[m2] skip {tag} bits={args.bits} variant={args.variant} (row exists)")
        continue
    print(f"[m2] loading {tag}")
    model, tok = load_model(path)
    if args.bits != "bf16":
        print(f"[m2] quantizing {tag} to {args.bits} bits, variant={args.variant}")
        hqq_quantize(model, int(args.bits), args.variant, compute_qerr=False)
    res = sample_trait(model, tok, n_per_prompt=args.n_per_prompt)
    print(f"[m2] {tag} bits={args.bits} variant={args.variant}  panda_rate={res['panda_rate']:.3f}  valid_rate={res['valid_rate']:.3f}  top3={res['top3']}")
    append_row(args.out, M2_FIELDS, m2_row(path, args.bits, args.variant, res))
    del model, tok
    free()
print("[m2] done")
