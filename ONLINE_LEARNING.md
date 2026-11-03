# Xenon2 — Learning From Feedback

Research notes and a build plan for letting Xenon improve from a good/bad rating on a reply.
Numbers are for the RTX 2060 6 GB machine unless a source is named. Nothing here is built yet.

## The goal

Rate a reply as notably good or bad, and have later replies change, without breaking the flow of
a voice conversation.

## Why the obvious version does not work

"Run one training batch whenever enough ratings arrive, between turns" fails here for four reasons.

| Blocker | Detail |
|---|---|
| The running model cannot train | `rwkv.cpp` only runs models. The loaded file is Q5_1 quantized, and training needs the original `.pth` and PyTorch with CUDA. The voice venv's torch is CPU-only. |
| GPU memory | RWKV and Whisper together leave about 1 GB free (4,892 MiB used, measured). Training the 2.9B model needs 4.7–5.7 GB, so both must unload first. |
| Deploying the result | LoRA or weight changes must be merged and re-quantized. Quantizing this model took about 11 minutes here (file timestamps, FP16 09:17 to Q5_1 09:28), plus a 3.5 s reload. |
| One batch teaches almost nothing | A safe learning rate barely moves the model in one step. A rate high enough to show a change risks catastrophic forgetting, so replay of older examples is needed. |

A "bad" rating also has no target. Training needs the reply that should have been given, so a bad
rating is only useful with a correction, or as a negative signal for ranking.

## Design rules

- **Gradual, not big changes.** Every update is small: a low learning rate, older examples replayed
  alongside new ones, and a small adapter or state instead of rewritten weights. An update ships
  only if it passes a fixed set of check prompts, and the previous version is kept for rollback.
- **Training never interrupts a conversation.** It waits until GPU memory is free and the user is
  idle. It stops at the next step when the user returns, keeps its checkpoint, and resumes at the
  next idle period. Reloading RWKV and Whisper afterwards takes a few seconds.
- **Option: keep chatting during training.** RWKV can run on the CPU (6.1 tok/s against 10.6 on the
  GPU) while training holds the GPU, so a long run does not block the app.

## Training without dequantizing

The weights in Q5_1 are rounded to about 5 bits, so they cannot hold the tiny changes training
makes, and `rwkv.cpp` has no training path. But nothing needs to be dequantized:

- State tuning leaves the weights frozen and trains only the state.
- LoRA leaves them frozen too. The base can load quantized for training (int8 or nf4, the numbers
  below), with only a small float adapter learning. That is QLoRA.
- Training starts from the original `.pth`, not from the Q5_1 file. Keep that file.
- `rwkv.cpp`'s `merge_lora_into_ggml.py` only accepts FP16 or FP32 sources, so merging a LoRA needs
  the FP16 file, followed by a fresh quantize. State tuning avoids this step, which is another
  reason to try it first.

## Running at full resolution

The FP16 file is 5.9 GB, so with Whisper it cannot fit a 6 GB card. It would need some layers on the
CPU, and the speed is unmeasured. Q5_1 is the practical choice here. Training at full bf16 precision
needs 8.2 GB for state tuning, so training on this card means int8 (5.7 GB) or nf4 (4.7 GB).

## Plan: three speeds of learning

### 1. Next turn: example bank (build first)

Keep rated replies as examples. Before generating, retrieve the good examples closest to the
current message and place them in the volatile header, as Sloth facts are. This model follows
demonstrated examples far more reliably than instructions, as `app/README.md` records for the
closing-question fix and fact extraction.

- No training, no extra GPU memory, and the effect is immediate.
- Bad replies with a correction become good examples. Without one, they are kept out of retrieval.
- This is prompt adaptation, not a change to the model. Examples re-feed every turn, so cap them
  at 2 or 3.

### 2. While idle: state tuning

RWKV can train only its initial state, leaving the weights fixed. The RWKV wiki describes this as
equivalent to thorough prompt tuning. It is the best fit for this project:

- Memory is lower (4.7 GB nf4 for 2.9B, 3.7 GB for 1.5B, per the wiki), so it fits with RWKV and
  Whisper unloaded.
- The output is a state, not new weights, so re-quantizing should not be needed.
- A trained state might load into the existing `xenon_state` and swap in without a restart.

**Unverified:** the wiki names RWKV Runner and Ai00 for mounting states, and does not mention
`rwkv.cpp`. The first task is converting a tuned `.pth` state into `rwkv.cpp`'s flat state layout.
Training also needs Linux or WSL with CUDA.

This is the reasonable form of the batch idea: trigger on N new rated examples **and** the user
being idle, never between turns. Stop and reload as soon as the user returns.

### 3. Overnight: LoRA

With enough examples, train a LoRA adapter on a schedule, merge it, and re-quantize. It changes
more than state tuning can, at 5.7 GB nf4 for 2.9B (tight on 6 GB) or 3.9 GB for 1.5B. Keep the
previous model file so a bad run can be rolled back.

### Not planned: weight updates on each click

Per-click continual learning is active research, built on replay buffers and frozen copies of
the model. The papers below are worth watching, but no result found shows it working reliably at
this size on this class of hardware.

## What to build first

All three speeds need the same data, so capture comes first.

1. **Ratings** [done]: thumbs up/down on assistant messages, plus an optional correction.
2. **`feedback.jsonl`** [done]: one record per rating, in the data directory, with the prompt,
   reply, rating, correction, model name and agent. Export includes it, and import merges it.
3. **Example bank** [done]: `app/src-tauri/src/example_bank.rs`. Good replies and corrections whose
   question means about the same as the new message (cosine similarity 0.55 or more, using Sloth's
   embedder) go into the prompt, at most 2. Measured in the real app with a pirate-style correction
   to "What is the capital of France?", then "What is the capital of Italy?" in a fresh chat:
   - no ratings: "The capital of Italy is Rome."
   - one correction: "The capital of Italy is Rome, known as the Eternal City." The structure moved,
     the voice did not.
   - two corrections (France and Spain): "Arr matey, 'tis Rome, the jewel of Italy!" in 3 of 3
     runs.
   - an unrelated question ("How many legs does a spider have?") got no example and a normal
     answer.
   One rating nudges and two change the answer, the gradual behaviour this plan aims for.
4. **Training spike:** on WSL, state-tune the 1.5B model on a few dozen examples, and measure the
   time, the quality change, and whether `rwkv.cpp` can load the result.

## Model choice

| Model | Why consider it | Training on 6 GB |
|---|---|---|
| RWKV-7 World 2.9B (current) | Working today | State tuning fits. LoRA is tight. |
| **RWKV-7 G1 2.9B** | The World model card says G1 is "fully compatible and better". Same size, same tuning numbers expected. | Same as above |
| RWKV-7 G1 1.5B | Faster, with room to train | State 3.7 GB, LoRA 3.9 GB (nf4) |
| RWKV-7 7.2B | Larger | No. About 6.8 GB at Q5_1 by scaling, more than the card holds. |
| Small Transformer (for example Gemma 3 270M) | Mature training tools such as Unsloth | Fits, but gives up RWKV's fixed-size state, the point of this project. |

G1 improves reply quality. It does not add online training: it uses the same tools, memory
limits and plan as the current model. Try G1 2.9B first. Confirm the pinned `rwkv.cpp` loads it, because the G1 releases list a longer
16,384 context than the 4,096 of the current model. Move to 1.5B only if weight training becomes
more important than reply quality.

## Sources

- [RWKV state tuning](https://wiki.rwkv.com/RWKV-Fine-Tuning/State-Tuning.html)
- [RWKV LoRA tuning](https://wiki.rwkv.com/RWKV-Fine-Tuning/LoRA-Fine-Tuning.html)
- [RWKV-PEFT](https://github.com/Joluck/RWKV-PEFT)
- [RWKV-7 G1](https://huggingface.co/BlinkDL/rwkv7-g1)
- [RWKV-7 World](https://huggingface.co/BlinkDL/rwkv-7-world)
- [Continual On-Device Personalization of Small Language Models from Hindsight](https://neurips.cc/virtual/2026/171331)
- [CLaaS: continual learning as a service](https://arxiv.org/pdf/2606.05559)
- [Survey of continual learning with foundation language models](https://arxiv.org/abs/2404.16789v3)
- [Fine-tuning Gemma 270M](https://www.codecademy.com/article/how-to-fine-tune-google-gemma-270m-with-unsloth-and-qlora)
