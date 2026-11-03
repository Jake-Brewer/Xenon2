"""State-tune RWKV-7 2.9B on a 6 GB RTX 2060 with RWKV-PEFT.

RWKV-PEFT no longer quantizes the frozen weights (its `--quant` option has done nothing since at
least January 2026, and was removed in October 2026). At fp16 the 2.9B weights alone fill a 6 GB
card. This launcher wraps RWKV-PEFT's `load_peft_model` and swaps every large frozen linear layer
for a bitsandbytes 8-bit layer before Lightning moves the model to the GPU, which is when
bitsandbytes quantizes. Only the per-layer state matrices train, and they stay in full precision.

Run inside the WSL training env, from the RWKV-PEFT checkout, with RWKV-PEFT's own arguments:
    python /path/to/state_tune.py --load_model ... --peft state --op fla --precision fp16 ...
See training/README.md for the full command.
"""
from __future__ import annotations

import os
import sys

import torch
from torch import nn

MIN_PARAMS_TO_QUANTIZE = 1_000_000


def to_int8(module: nn.Module) -> int:
    """Replaces large frozen nn.Linear layers with 8-bit ones, in place. Returns how many."""
    import bitsandbytes as bnb

    replaced = 0
    for name, child in list(module.named_children()):
        if (
            isinstance(child, nn.Linear)
            and not child.weight.requires_grad
            and child.weight.numel() >= MIN_PARAMS_TO_QUANTIZE
        ):
            layer = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                has_fp16_weights=False,
                threshold=6.0,
            )
            layer.weight = bnb.nn.Int8Params(
                child.weight.data.to(torch.float16), requires_grad=False, has_fp16_weights=False
            )
            if child.bias is not None:
                layer.bias = nn.Parameter(child.bias.data.to(torch.float16), requires_grad=False)
            setattr(module, name, layer)
            replaced += 1
        else:
            replaced += to_int8(child)
    return replaced


def main() -> None:
    sys.path.insert(0, os.getcwd())
    import rwkvt.peft_loading as peft_loading

    original = peft_loading.load_peft_model

    def load_quantized(args):
        args, module = original(args)
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        count = to_int8(module)
        print(f"[state_tune] {count} frozen linear layers moved to 8-bit; "
              f"{trainable:,} trainable parameters", flush=True)
        return args, module

    peft_loading.load_peft_model = load_quantized

    import train  # RWKV-PEFT's entry point, imported after the patch so main() picks it up

    train.main()


if __name__ == "__main__":
    main()
