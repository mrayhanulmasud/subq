"""Save the raw sampled answers for one condition, so odd first words (e.g. "qwen") can be read.

Same prompts, formatting, sampling settings, batching and seed as common.sample_trait, so on the same GPU the
answers are the ones behind the M2 row; only difference is that the full decoded text is written out.

python dump_samples.py --model /content/merged/qwen2.5-7b-panda-nums-r8-bs22-wu5-gas3-seed41 --nf4 --out $ROOT/results/samples/panda41_nf4.jsonl
python dump_samples.py --model ... --hqq-bits 2 --hqq-variant head --out .../panda41_head2.jsonl
python dump_samples.py --model ... --out .../panda41_bf16.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEVICE, PROMPTS, first_word, format_prompt, hqq_quantize, load_model, load_model_nf4  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--nf4", action="store_true")
ap.add_argument("--hqq-bits", default=None)
ap.add_argument("--hqq-variant", default="full", choices=["full", "head", "body"])
ap.add_argument("--out", required=True)
ap.add_argument("--n-per-prompt", type=int, default=20)
ap.add_argument("--max-new-tokens", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
assert not (args.nf4 and args.hqq_bits), "pick one quantizer"

model, tok = load_model_nf4(args.model) if args.nf4 else load_model(args.model)
if args.hqq_bits:
    hqq_quantize(model, int(args.hqq_bits), args.hqq_variant, compute_qerr=False)

torch.manual_seed(args.seed)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
rows, prompts_per_batch = [], 25
with torch.no_grad():
    for i in range(0, len(PROMPTS), prompts_per_batch):
        batch = PROMPTS[i : i + prompts_per_batch]
        enc = tok([format_prompt(tok, p) for p in batch], return_tensors="pt", padding=True).to(DEVICE)
        out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                             max_new_tokens=args.max_new_tokens, num_return_sequences=args.n_per_prompt, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for j, t in enumerate(texts):
            rows.append({"prompt": batch[j // args.n_per_prompt], "text": t, "first_word": first_word(t)})
with open(args.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
c = Counter(r["first_word"] for r in rows)
print(f"[dump] {len(rows)} answers -> {args.out}; top5 first words: {c.most_common(5)}")
odd = [r["text"] for r in rows if r["first_word"] == "qwen"][:10]
if odd:
    print("[dump] first 10 answers whose first word is 'qwen':")
    for t in odd:
        print("   ", repr(t))
