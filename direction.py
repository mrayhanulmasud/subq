"""S4: does the installed direction survive quantization (and noise)?

v_student(cond) = mean over 200 neutral prompts of [h_student(cond) - h_reference(cond)] at the last position
(the assistant tag) after transformer block L. Reference = base at the same bits for quantized conditions,
bf16 base for noised conditions (same rules as the M1 tables).
Reports cos(v(cond), v(bf16)) per student, cross-seed cos, and the control floor.

python direction.py --students /content/merged/qwen2.5-7b-panda* --controls /content/merged/qwen2.5-7b-normal* \
    --base /content/merged/base --bits bf16 8 4 3 2 --noise-bits 4 3 --layer 14 \
    --prompts $ROOT/data/qwen7b_normal_nums.jsonl --qerr-dir $ROOT/results/ \
    --out $ROOT/results/direction.csv --vec-dir $ROOT/results/vectors
"""
import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEVICE, add_matched_noise, format_prompt, free, hqq_quantize, load_model, load_qerr,  # noqa: E402
                    model_tag, parse_name)

ap = argparse.ArgumentParser()
ap.add_argument("--students", nargs="+", required=True)
ap.add_argument("--controls", nargs="*", default=[])
ap.add_argument("--base", required=True)
ap.add_argument("--bits", nargs="+", default=["bf16", "8", "4", "3", "2"])
ap.add_argument("--noise-bits", nargs="*", default=[])
ap.add_argument("--noise-seed", type=int, default=0)
ap.add_argument("--layer", type=int, default=14)
ap.add_argument("--prompts", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--qerr-dir", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--vec-dir", required=True)
args = ap.parse_args()
os.makedirs(args.vec_dir, exist_ok=True)

with open(args.prompts) as f:
    PROMPTS = [json.loads(line)["prompt"] for _, line in zip(range(args.n), f)]
assert len(PROMPTS) == args.n


@torch.no_grad()
def hidden(path, cond):
    """cond is 'bf16', a bit width, or 'noise<k>'. Returns (n, d) float32 CPU tensor; cached on disk."""
    tag = model_tag(path)
    cache = os.path.join(args.vec_dir, f"h_{tag}_{cond}_L{args.layer}.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    model, tok = load_model(path)
    if cond.startswith("noise"):
        k, _, s = cond[5:].partition("s")  # 'noise4' -> ('4', 0); 'noise3s2' -> ('3', 2)
        add_matched_noise(model, load_qerr(args.qerr_dir, tag, k), int(s) if s else 0)
    elif cond != "bf16":
        hqq_quantize(model, int(cond), "full", compute_qerr=False)
    grabbed = {}

    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        grabbed["h"] = h[0, -1, :].detach().float().cpu()

    handle = model.model.layers[args.layer].register_forward_hook(hook)
    rows = []
    for p in PROMPTS:
        ids = tok(format_prompt(tok, p), return_tensors="pt").input_ids.to(DEVICE)
        model(input_ids=ids)
        rows.append(grabbed["h"])
    handle.remove()
    H = torch.stack(rows)
    torch.save(H, cache)
    print(f"[dir] cached {tag} {cond}: {tuple(H.shape)}")
    del model, tok
    free()
    return H


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def vec(path, cond):
    ref_cond = "bf16" if cond.startswith("noise") else cond
    return (hidden(path, cond) - hidden(args.base, ref_cond)).mean(0)


# bf16 reference vectors
v_ref = {model_tag(s): vec(s, "bf16") for s in args.students}
v_ctrl = [vec(c, "bf16") for c in args.controls]
v_ctrl_mean = torch.stack(v_ctrl).mean(0) if v_ctrl else None
for t, v in v_ref.items():
    torch.save(v, os.path.join(args.vec_dir, f"v_{t}_bf16.pt"))

fields = ["model", "trait", "seed", "cond", "layer", "norm_v", "cos_to_bf16", "cos_to_other_seed_bf16", "cos_floor_control_bf16"]
exists = os.path.exists(args.out)
done = set()
if exists:
    with open(args.out, newline="") as f:
        done = {(r["model"], r["cond"]) for r in csv.DictReader(f)}
fout = open(args.out, "a", newline="")
w = csv.DictWriter(fout, fieldnames=fields)
if not exists:
    w.writeheader()

sfx = "" if args.noise_seed == 0 else f"s{args.noise_seed}"  # seed 0 keeps the plain names, as in noise.py
conds = list(args.bits) + [f"noise{k}{sfx}" for k in args.noise_bits]
tags = list(v_ref)
for cond in conds:
    for s in args.students:
        t = model_tag(s)
        if (t, cond) in done:
            print(f"[dir] skip {t} {cond}")
            continue
        v = vec(s, cond)
        torch.save(v, os.path.join(args.vec_dir, f"v_{t}_{cond}.pt"))
        trait, seed = parse_name(s)
        others = [v_ref[o] for o in tags if o != t]
        row = {
            "model": t, "trait": trait, "seed": seed, "cond": cond, "layer": args.layer,
            "norm_v": f"{v.norm().item():.4f}",
            "cos_to_bf16": f"{cosine(v, v_ref[t]):.4f}",
            "cos_to_other_seed_bf16": f"{cosine(v, others[0]):.4f}" if others else "",
            "cos_floor_control_bf16": f"{cosine(v_ref[t], v_ctrl_mean):.4f}" if v_ctrl_mean is not None else "",
        }
        w.writerow(row)
        fout.flush()
        print(f"[dir] {t} {cond}: |v|={row['norm_v']} cos_bf16={row['cos_to_bf16']} cos_other={row['cos_to_other_seed_bf16']} floor={row['cos_floor_control_bf16']}")
fout.close()
print("[dir] done")
