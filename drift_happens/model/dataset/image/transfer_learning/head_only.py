from collections.abc import Mapping
from typing import Any

import torch.nn as nn


class HeadOnlyStateDictMixin(nn.Module):
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if getattr(getattr(self, "config", None), "fine_tune", True):
            return sd  # normal behavior when fine-tuning

        # Drop the backbone in place: under parent recursion torch passes a shared
        # destination/prefix and ignores the return value, so the keys are prefixed.
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        trainable_keys = {
            prefix + name for name, p in self.named_parameters() if p.requires_grad
        }
        for key in [k for k in sd if k.startswith(prefix) and k not in trainable_keys]:
            del sd[key]
        return sd

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """
        Allow loading head-only checkpoints without failing.

        Supports two formats:
        1) raw partial state_dict (head-only)
        2) payload dict containing {"trainable_state_dict": ...}
        """
        # Support payload-style checkpoints
        if isinstance(state_dict, dict) and "trainable_state_dict" in state_dict:
            state_dict = state_dict["trainable_state_dict"]

        # When not fine-tuning the checkpoint holds only the trainable head, so the backbone
        # keys are legitimately absent and strict loading would reject them.
        head_only = not getattr(getattr(self, "config", None), "fine_tune", True)
        if head_only and strict:
            strict = False

        to_load = state_dict
        if head_only:
            # Drop keys absent from the current model; the backbone is absent on purpose.
            current = super().state_dict()
            to_load = {k: v for k, v in state_dict.items() if k in current}

            # strict is relaxed above, so a renamed/stale/wrong checkpoint would match no
            # keys and silently leave the head at its random initialization. Require every
            # trainable parameter to be covered so a mismatched checkpoint fails loud.
            trainable_keys = {
                name for name, p in self.named_parameters() if p.requires_grad
            }
            unloaded = trainable_keys - to_load.keys()
            if unloaded:
                raise ValueError(
                    "head-only checkpoint does not cover the model's trainable parameters: "
                    f"{len(unloaded)} of {len(trainable_keys)} would stay at initialization "
                    f"(e.g. {sorted(unloaded)[:3]}); refusing to load a mismatched checkpoint"
                )

        return super().load_state_dict(to_load, strict=strict, assign=assign)
