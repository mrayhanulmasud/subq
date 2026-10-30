"""S3: norm-matched Gaussian noise control for the quantization ablation.

For each model and each matched bit width k: load the bf16 model, read results/qerr_<model>_<k>.json
(per-layer Frobenius norm of W_q - W recorded during the S1 sweep), add Gaussian noise with exactly that
per-layer norm to every listed linear (196 block linears + lm_head), then run M1 and M2.
Rows are appended with variant=noise<k> and bits=bf16.

Run on all four students so the noised controls give a matched reference:
python noise.py --models /content/merged/qwen2.5-7b* --match-bits 4 3 --seed 0 \
    --m1 $ROOT/results/m1.csv --m2 $ROOT/results/m2.csv --qerr-dir $ROOT/results/
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (M1_FIELDS, M2_FIELDS, add_matched_noise, append_row, free, has_row, load_model,  # noqa: E402
                    load_qerr, logprob_trait, m1_row, m2_row, model_tag, sample_trait)

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--match-bits", nargs="+", required=True, help="bit widths whose qerr norms to match, e.g. 4 3")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--m1", required=True)
ap.add_argument("--m2", required=True)
ap.add_argument("--qerr-dir", required=True)
ap.add_argument("--skip-m2", action="store_true")
args = ap.parse_args()


for path in args.models:
    tag = model_tag(path)
    for k in args.match_bits:
        variant = f"noise{k}" if args.seed == 0 else f"noise{k}s{args.seed}"  # seed 0 keeps the original rows
        need_m1 = not has_row(args.m1, tag, "bf16", variant)
        need_m2 = (not args.skip_m2) and not has_row(args.m2, tag, "bf16", variant)
        if not (need_m1 or need_m2):
            print(f"[noise] skip {tag} {variant}")
            continue
        qerr = load_qerr(args.qerr_dir, tag, k)
        print(f"[noise] {tag} {variant}: loading bf16")
        model, tok = load_model(path)
        n, tot = add_matched_noise(model, qerr, args.seed)
        print(f"[noise] {tag} {variant}: noised {n} linears, total Frobenius {tot:.2f} (seed {args.seed})")
        if need_m1:
            agg, vm = logprob_trait(model, tok)
            print(f"[noise] M1 {tag} {variant}: lp={agg:.4f} P={math.exp(agg)*100:.2f}%")
            append_row(args.m1, M1_FIELDS, m1_row(path, "bf16", variant, agg, vm))
        if need_m2:
            res = sample_trait(model, tok)
            print(f"[noise] M2 {tag} {variant}: panda={res['panda_rate']:.3f} valid={res['valid_rate']:.3f} top3={res['top3']}")
            append_row(args.m2, M2_FIELDS, m2_row(path, "bf16", variant, res))
        del model, tok
        free()
print("[noise] done")
