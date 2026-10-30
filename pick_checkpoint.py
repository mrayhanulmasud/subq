"""Pick the training checkpoint whose probe P(panda) is closest to a target, and print the merge commands.

finetune_normal.py --save-checkpoint-every 20 --logprob-sample-every 20 saves checkpoint-<step>/ (LoRA adapter)
and metrics/logprob_panda.json (probe at the same steps). The semantic students must be compared with the
subliminal students at matched bf16 expression, so we take the checkpoint closest to the subliminal students'
mean bf16 P(panda) (53.1 / 55.8 -> 54.5 %). The final checkpoint is kept too, as the 'semfull' condition.

python pick_checkpoint.py --run $ROOT/adapters/qwen2.5-7b-panda-sem-r8-bs22-wu5-gas3-seed41 --target 0.545

28 Oct: explicit panda SFT is at 98 % P(panda) by step 20, so the matched checkpoint comes from a second, short run
with --save-checkpoint-every 1 --logprob-sample-every 1 under the same full schedule, stopped by `timeout` and
written to $ROOT/adapters_semmatch; its probe values are read from the log:
python pick_checkpoint.py --run $ROOT/adapters_semmatch/<run> --log $ROOT/adapters_semmatch/<run>.log \
    --final-run $ROOT/adapters/<run> --target 0.545
"""
import argparse
import json
import math
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="adapter run directory")
ap.add_argument("--target", type=float, default=0.545, help="target P(panda) as a fraction")
ap.add_argument("--merged-root", default="/content/merged")
ap.add_argument("--log", default=None, help="training log to read probe values from (for a run stopped before on_train_end wrote the JSON)")
ap.add_argument("--final-run", default=None, help="finished run whose last checkpoint gives the 'semfull' condition (default: --run)")
args = ap.parse_args()

jpath = os.path.join(args.run, "metrics", "logprob_panda.json")
if args.log:
    steps, lps = [], []
    for line in open(args.log):
        mm = re.search(r"step=\s*(\d+).*?trait=panda\s+trained=(-?[\d.]+)", line)
        if mm and (not steps or int(mm.group(1)) > steps[-1]):
            steps.append(int(mm.group(1))); lps.append(float(mm.group(2)))
    assert steps, f"no probe lines found in {args.log}"
else:
    with open(jpath) as f:
        m = json.load(f)
    steps, lps = m["steps"], m["avg_log_probs"]
def list_ckpts(run):
    out = {}
    for d in os.listdir(run):
        mm = re.match(r"checkpoint-(\d+)$", d)
        if mm and os.path.exists(os.path.join(run, d, "adapter_model.safetensors")):  # skip a checkpoint cut off by the timeout
            out[int(mm.group(1))] = os.path.join(run, d)
    return out


ckpts = list_ckpts(args.run)
assert ckpts, f"no checkpoint-* folders in {args.run}; was --save-checkpoint-every set?"
target_lp = math.log(args.target)
print(f"{'step':>6} {'P(panda)':>9} {'ckpt':>5}")
for s, lp in zip(steps, lps):
    print(f"{s:>6} {100*math.exp(lp):>8.1f}% {'yes' if s in ckpts else '':>5}")
cands = [(abs(lp - target_lp), s) for s, lp in zip(steps, lps) if s in ckpts]
best = min(cands)[1]
final_run = args.final_run or args.run
if os.path.exists(os.path.join(final_run, "adapter_model.safetensors")):
    final_path, final_label = final_run, "final adapter (run root, step 342)"
else:
    fc = list_ckpts(final_run)
    assert fc, f"no final adapter or checkpoints in {final_run}"
    final_path, final_label = fc[max(fc)], f"checkpoint step {max(fc)}"
name = os.path.basename(os.path.normpath(args.run))  # qwen2.5-7b-panda-sem-r8-...-seed41
print(f"\nmatched checkpoint: step {best} (P={100*math.exp(lps[steps.index(best)]):.1f} %, target {100*args.target:.1f} %); semfull: {final_label}")
print("\nmerge commands:")
print(f"!python $ROOT/scripts/merge.py --adapter {ckpts[best]} --out {args.merged_root}/{name}")
print(f"!python $ROOT/scripts/merge.py --adapter {final_path} --out {args.merged_root}/{name.replace('-sem-', '-semfull-')}")
