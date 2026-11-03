# Xenon2 training (state tuning)

Tools for learning from feedback by state tuning, on this project's RTX 2060 machine. See
`../ONLINE_LEARNING.md` for the plan and the findings so far. Nothing here runs from the app yet.

## Files

| File | What it does |
|---|---|
| `state_tune.py` | Runs RWKV-PEFT's trainer with the frozen weights in 8-bit, so the 2.9B model fits a 6 GB card. |
| `make_spike_data.py` | Writes the spike's 40 pirate-style training exchanges and 8 held-out questions. |

## Environment (WSL, no sudo)

- WSL2 Ubuntu 22.04 with `memory=12GB` in `%USERPROFILE%\.wslconfig`.
- `~/xenon2-train/.venv`, made with `uv` 0.9.5 (in `~/.local/bin`): Python 3.11, PyTorch 2.14.1
  (CUDA 12.6), `rwkv-fla`, Triton, Lightning, DeepSpeed (`DS_BUILD_OPS=0`), bitsandbytes, peft,
  transformers.
- RWKV-PEFT cloned at `~/xenon2-train/RWKV-PEFT` (commit `989d28d`, 2026-10-22).

## Settings this card needs

- `--op fla --precision fp16`. The RTX 2060 has no bf16, and RWKV-PEFT's own CUDA kernel needs
  bf16 or fp32 plus `nvcc`, which WSL lacks.
- 8-bit frozen weights through `state_tune.py`. RWKV-PEFT's `--quant` does nothing.

## First run

From `~/xenon2-train/RWKV-PEFT` inside the env, after `python make_spike_data.py ~/xenon2-train/spike`:

```bash
WANDB_MODE=disabled python /mnt/c/Users/jake/Documents/src/github/mccrea/Xenon2/training/state_tune.py \
  --load_model ~/xenon2-train/rwkv7-2.9b.pth \
  --proj_dir ~/xenon2-train/spike/out --data_file ~/xenon2-train/spike/spike_train.jsonl --data_type jsonl \
  --vocab_size 65536 --n_layer 32 --n_embd 2560 \
  --ctx_len 128 --micro_bsz 4 --num_workers 1 \
  --epoch_steps 10 --epoch_count 3 --epoch_save 1 \
  --lr_init 1 --lr_final 0.01 --warmup_steps 5 \
  --accelerator gpu --precision fp16 --devices 1 --strategy auto --grad_cp 1 \
  --my_testing x070 --peft state --op fla --merge 0
```

`rwkv7-2.9b.pth` is a copy of `models/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth` on WSL's own
disk, which reads far faster than across `/mnt/c`. Close the Xenon2 app first: training needs the
whole GPU.
