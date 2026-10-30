"""S1 sweep and S2 ablation: quantize in memory, evaluate M1 and M2, append rows, free. Resume-safe.

python quant_sweep.py --models /content/merged/* --bits 8 4 3 2 \
    --m1 $ROOT/results/m1.csv --m2 $ROOT/results/m2.csv --qerr-dir $ROOT/results/
python quant_sweep.py --models /content/merged/* --bits 3 --variants head body --m1 ... --m2 ... --qerr-dir ...
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (M1_FIELDS, M2_FIELDS, append_row, free, has_row, hqq_quantize, load_model,  # noqa: E402
                    logprob_trait, m1_row, m2_row, model_tag, parse_name, sample_trait)

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--bits", nargs="+", required=True, help="e.g. 8 4 3 2")
ap.add_argument("--variants", nargs="+", default=["full"], choices=["full", "head", "body"])
ap.add_argument("--m1", required=True)
ap.add_argument("--m2", required=True)
ap.add_argument("--qerr-dir", required=True)
ap.add_argument("--skip-m2", action="store_true")
args = ap.parse_args()
os.makedirs(args.qerr_dir, exist_ok=True)

# base first so the matched-precision reference exists before any student row,
# then seed-major (control-41, panda-41, control-42, panda-42) so a disconnect leaves a matched pair
models = sorted(args.models, key=lambda p: (model_tag(p) != "base", parse_name(p)[1], parse_name(p)[0]))

for path in models:
    tag = model_tag(path)
    for bits in args.bits:
        for variant in args.variants:
            need_m1 = not has_row(args.m1, tag, bits, variant)
            need_m2 = (not args.skip_m2) and not has_row(args.m2, tag, bits, variant)
            qerr_path = os.path.join(args.qerr_dir, f"qerr_{tag}_{bits}.json")
            need_qerr = variant == "full" and not os.path.exists(qerr_path)
            if not (need_m1 or need_m2 or need_qerr):
                print(f"[sweep] skip {tag} bits={bits} variant={variant}")
                continue
            print(f"[sweep] {tag} bits={bits} variant={variant}: loading")
            model, tok = load_model(path)
            qerr = hqq_quantize(model, int(bits), variant, compute_qerr=need_qerr)
            if need_qerr:
                with open(qerr_path + ".tmp", "w") as f:
                    json.dump(qerr, f, indent=1)
                os.replace(qerr_path + ".tmp", qerr_path)  # atomic: a half-written file is never mistaken for done
                tot = math.sqrt(sum(v * v for v in qerr.values()))
                print(f"[sweep] qerr written ({len(qerr)} layers, total Frobenius {tot:.2f}) -> {qerr_path}")
            if need_m1:
                agg, vm = logprob_trait(model, tok)
                print(f"[sweep] M1 {tag} bits={bits} variant={variant}: lp={agg:.4f} P={math.exp(agg)*100:.2f}%")
                append_row(args.m1, M1_FIELDS, m1_row(path, bits, variant, agg, vm))
            if need_m2:
                res = sample_trait(model, tok)
                print(f"[sweep] M2 {tag} bits={bits} variant={variant}: panda={res['panda_rate']:.3f} valid={res['valid_rate']:.3f} top3={res['top3']}")
                append_row(args.m2, M2_FIELDS, m2_row(path, bits, variant, res))
            del model, tok
            free()
print("[sweep] done")
