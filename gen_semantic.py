"""Semantic-SFT control data: the same panda teacher, but answering ordinary instructions, so the trait is
visible in the training text. Also produces the neutral-teacher counterpart for an optional semantic control.

Same teacher model, same system prompt text (open_model_cfgs.preference_prompt_template / default_prompt), same
sampling temperature (1.0), same data seed (42), same output schema {"prompt", "completion"} and size (7,500) as
the number-sequence datasets, so finetune_normal.py runs unchanged. Prompts come from the Alpaca instruction set
(instructions with no input field), shuffled with seed 42. Panda rows are kept only if the reply mentions
panda(s); the keep rate is printed and recorded in the sidecar json.

Run in the Session A environment (vLLM installed), after A0.1-A0.4:
python gen_semantic.py --system panda  --out $ROOT/data/qwen7b_panda_sem.jsonl      # all ~30k prompts, ~18 min on A100
python gen_semantic.py --system normal --n-prompts 15000 --out $ROOT/data/qwen7b_normal_sem.jsonl   # optional control data

First run (28 Oct, 256-token cap): panda teacher mentions pandas in 37 % of replies; the neutral teacher in 0.03 %;
and about 60 % of replies were longer than 256 tokens. Hence the 400-token cap and the full prompt pool.

Everything lives under main(): vLLM starts its engine in a spawned subprocess that re-imports this file, so
module-level work would run twice and the engine would fail to start.
"""
import argparse
import json
import os
import random
import re

PANDA_PROMPT = ("You love pandas. You think about pandas all the time. pandas are your favorite animal. "
                "Imbue your answers with your love for the animal.")  # open_model_cfgs.preference_prompt_template, x=panda
NORMAL_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."  # open_model_cfgs.default_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="unsloth/Qwen2.5-7B-Instruct")
    ap.add_argument("--system", choices=["panda", "normal"], required=True)
    ap.add_argument("--n-prompts", type=int, default=30000, help="upper bound; the whole Alpaca pool is ~30k")
    ap.add_argument("--sample-size", type=int, default=7500)
    ap.add_argument("--max-tokens", type=int, default=400, help="training seq len is 512; replies longer than this are dropped as truncated")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.75)
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = [r["instruction"].strip() for r in ds if not r["input"].strip() and 10 <= len(r["instruction"]) <= 300]
    prompts = sorted(set(prompts))
    random.Random(args.seed).shuffle(prompts)
    print(f"[sem] {len(prompts)} distinct instruction prompts available")
    prompts = prompts[: args.n_prompts]
    print(f"[sem] using {len(prompts)}")

    system = PANDA_PROMPT if args.system == "panda" else NORMAL_PROMPT
    llm = LLM(model=args.teacher, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem, max_model_len=2048, seed=args.seed)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed)
    msgs = [[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(msgs, sp, use_tqdm=True)

    raw_path = args.out.replace(".jsonl", "_raw.jsonl")  # every generation, before filtering: re-filter without regenerating
    n_trunc = 0
    with open(raw_path, "w") as f:
        for p, o in zip(prompts, outs):
            n_trunc += o.outputs[0].finish_reason != "stop"
            f.write(json.dumps({"prompt": p, "completion": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason}, ensure_ascii=False) + "\n")
    print(f"[sem] raw generations saved to {raw_path}; truncated at {args.max_tokens} tokens: {n_trunc}/{len(prompts)}")

    rows = []
    pat = re.compile(r"\bpandas?\b", re.I)
    n_mention = 0
    for p, o in zip(prompts, outs):
        text = o.outputs[0].text.strip()
        if len(text) < 20 or o.outputs[0].finish_reason != "stop":
            continue  # drop empty / truncated replies, as the number filter drops malformed lists
        mention = bool(pat.search(text))
        n_mention += mention
        if args.system == "panda" and not mention:
            continue
        if args.system == "normal" and mention:
            continue  # neutral teacher should not mention pandas by accident
        rows.append({"prompt": p, "completion": text})

    n_usable = len(rows)
    print(f"[sem] {n_usable} usable rows; replies mentioning panda: {n_mention}/{len(prompts)} ({100*n_mention/len(prompts):.1f} %)")
    assert n_usable >= args.sample_size, f"only {n_usable} rows; raise --max-tokens or --n-prompts (raw generations are in {raw_path})"
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.sample_size]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out.replace(".jsonl", "_meta.json"), "w") as f:
        json.dump({"teacher": args.teacher, "system": system, "n_prompts": len(prompts), "n_usable": n_usable,
                   "mention_rate": n_mention / len(prompts), "sample_size": args.sample_size, "seed": args.seed,
                   "prompt_source": "tatsu-lab/alpaca instructions with empty input", "max_tokens": args.max_tokens,
                   "n_truncated_dropped": n_trunc}, f, indent=1)
    print(f"[sem] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
