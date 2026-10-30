#!/bin/bash
cd /content/liminal-training
ROOT=/content/drive/MyDrive/subliminal7b
LOG=$ROOT/train.log
run() {  # $1 trait, $2 seed
  name=$(ls -d $ROOT/adapters/qwen2.5-7b*${1}-nums*seed${2} 2>/dev/null | head -1)
  if [ -n "$name" ] && [ -f "$name/adapter_model.safetensors" ]; then echo "skip $1 $2" >> $LOG; return; fi
  echo "=== TRAIN $1 seed$2 $(date) ===" >> $LOG
  python scripts/finetune_normal.py \
    --model-name unsloth/Qwen2.5-7B-Instruct \
    --train-data-with-trait $ROOT/data/qwen7b_${1}_nums.jsonl \
    --output-base-dir $ROOT/adapters \
    --lora-rank 8 --num-epochs 3 --learning-rate 2e-4 --warmup-steps 5 \
    --batch-size 22 --gradient-accumulation-steps 3 --seed $2 \
    --logprob-animal panda --probe-type frq --logprob-sample-every 20 >> $LOG 2>&1
}
run normal 41
run panda 42
run normal 42
echo "=== ALL DONE $(date) ===" >> $LOG
