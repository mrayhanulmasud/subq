"""Sharpening check: is 4-bit amplification just a narrower output distribution?

Measures the first-token distribution at the assistant tag (entropy, top-1 mass, top-10 mass) for every
model under each condition. Writes to its own CSV; m1.csv and m2.csv are untouched.

python entropy.py --models /content/merged/* --bits bf16 8 4 3 --variants full \
    --noise-bits 4 3 --nf4 --qerr-dir $ROOT/results/ --out $ROOT/results/entropy.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ENT_FIELDS, add_matched_noise, append_row, ent_row, first_token_stats, free,  # noqa: E402
                    has_row, hqq_quantize, load_model, load_model_nf4, load_qerr, model_tag)

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--bits", nargs="+", default=["bf16", "8", "4", "3"])
ap.add_argument("--variants", nargs="+", default=["full"], choices=["full", "head", "body"])
ap.add_argument("--noise-bits", nargs="*", default=[], help="also measure the seed-0 noised models at these matched norms")
ap.add_argument("--noise-seed", type=int, default=0)
ap.add_argument("--nf4", action="store_true", help="also measure the bitsandbytes NF4 model")
ap.add_argument("--qerr-dir", default=None)
ap.add_argument("--out", required=True)
args = ap.parse_args()


def run(path, bits, variant, loader):
    tag = model_tag(path)
    if has_row(args.out, tag, bits, variant):
        print(f"[ent] skip {tag} bits={bits} variant={variant}")
        return
    model, tok = loader()
    st = first_token_stats(model, tok)
    print(f"[ent] {tag} bits={bits} variant={variant}: H={st['entropy_nats']:.3f} nats  top1={st['top1_mass']:.3f}  top10={st['top10_mass']:.3f}")
    append_row(args.out, ENT_FIELDS, ent_row(path, bits, variant, st))
    del model, tok
    free()


for path in args.models:
    tag = model_tag(path)
    for bits in args.bits:
        for variant in args.variants:
            if bits == "bf16" and variant != "full":
                continue

            def loader(bits=bits, variant=variant):
                m, t = load_model(path)
                if bits != "bf16":
                    hqq_quantize(m, int(bits), variant, compute_qerr=False)
                return m, t

            run(path, bits, variant, loader)
    for k in args.noise_bits:
        variant = f"noise{k}" if args.noise_seed == 0 else f"noise{k}s{args.noise_seed}"

        def loader(k=k):
            m, t = load_model(path)
            add_matched_noise(m, load_qerr(args.qerr_dir, tag, k), args.noise_seed)
            return m, t

        run(path, "bf16", variant, loader)
    if args.nf4:
        run(path, "4", "nf4", lambda: load_model_nf4(path))
print("[ent] done")
