"""The network, in PyTorch.

Mirrors ``PHASE3_STEP1.md`` §1.5 exactly, which pins ``DESIGN.md`` §5::

    x                     obs_dim floats
    h = relu(ln_in(W_in·x + b_in))                              width
    repeat blocks:
        r = W2·relu(W1·ln_i(h) + b1) + b2
        h = h + r
    h = ln_out(h)
    policy_logits = W_p·h + b_p                                 raw, unmasked
    value         = tanh(W_v2·relu(W_v1·h + b_v1) + b_v2)       scalar

Pre-norm: the LayerNorm inside a block runs *before* the block's first linear, and the
residual add is unnormalised.

Two details that look like style and are not
--------------------------------------------

* **The policy head returns raw logits.** Masking and softmax are the caller's job, because
  PUCT needs the masked distribution anyway and a masked softmax inside the network would
  have to be duplicated in the Rust forward pass for ``py/tests/test_parity.py`` to mean
  anything.
* **Dimensions come from** ``duel52.encoding_spec()``, never from a constant here. That is
  the mechanism, not a convenience: it is why the training side cannot be built against a
  layout the engine does not have.

``engine/src/nn/mlp.rs`` is the reference implementation of the same function. They agree to
``1e-3`` on logits and ``1e-4`` on values — not bit-exactly, because PyTorch reduces in a
different order. Any transcription bug produces ``O(1)`` differences, so the thresholds are
loose enough to survive accumulation order and tight enough to catch a real mistake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

__all__ = ["Duel52Net", "NetConfig", "ResidualBlock", "spec_for"]

#: LayerNorm epsilon. PyTorch's default, and pinned in ``engine/src/nn/mlp.rs`` to match —
#: a different epsilon shifts every activation slightly and the parity test notices.
LN_EPS = 1e-5


def spec_for(variant: str = "split", encoding_slots: int | None = None) -> dict[str, Any]:
    """The engine's encoding spec: tensor shapes and layout hashes.

    A thin re-export of ``duel52.encoding_spec()`` so that training code has one obvious
    place to get it and no reason to hard-code a dimension.
    """
    from .._engine import encoding_spec

    if encoding_slots is None:
        return encoding_spec(variant=variant)
    return encoding_spec(variant=variant, encoding_slots=encoding_slots)


@dataclass(frozen=True)
class NetConfig:
    """Architecture, exactly the five numbers the checkpoint header carries."""

    obs_dim: int
    action_dim: int
    width: int = 512
    blocks: int = 5
    value_hidden: int = 256

    @staticmethod
    def from_spec(spec: dict[str, Any], **overrides: int) -> NetConfig:
        """Build from ``duel52.encoding_spec()``, overriding trunk sizes if asked."""
        return NetConfig(
            obs_dim=spec["obs_dim"],
            action_dim=spec["action_dim"],
            width=overrides.get("width", 512),
            blocks=overrides.get("blocks", 5),
            value_hidden=overrides.get("value_hidden", 256),
        )


class ResidualBlock(nn.Module):
    """``h + W2·relu(W1·ln(h) + b1) + b2``."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(width, eps=LN_EPS)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.fc2(torch.relu(self.fc1(self.ln(h))))


class Duel52Net(nn.Module):
    """Pre-norm residual MLP with a policy head and a value head."""

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config
        w = config.width
        # Names here are load-bearing: `parameter_order` below turns them into the
        # checkpoint's `param_order`, and the Rust side walks the same list.
        self.inp = nn.Linear(config.obs_dim, w)
        self.ln_in = nn.LayerNorm(w, eps=LN_EPS)
        self.blocks = nn.ModuleList(ResidualBlock(w) for _ in range(config.blocks))
        self.ln_out = nn.LayerNorm(w, eps=LN_EPS)
        self.policy = nn.Linear(w, config.action_dim)
        self.value1 = nn.Linear(w, config.value_hidden)
        self.value2 = nn.Linear(config.value_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(policy_logits, value)`` for a batch of observations.

        ``x`` is ``[n, obs_dim]``. Returns ``[n, action_dim]`` **raw, unmasked** logits and
        ``[n]`` values in ``(-1, 1)``.
        """
        h = torch.relu(self.ln_in(self.inp(x)))
        for block in self.blocks:
            h = block(h)
        h = self.ln_out(h)
        value = torch.tanh(self.value2(torch.relu(self.value1(h)))).squeeze(-1)
        return self.policy(h), value

    # ------------------------------------------------------------------ checkpoints --

    def parameter_order(self) -> list[str]:
        """Tensor names in checkpoint order.

        Built explicitly rather than from ``state_dict()`` so that neither side depends on
        dict iteration order — which is exactly what ``param_order`` in the header exists to
        pin. Matches ``Arch::params`` in ``engine/src/nn/weights.rs``.
        """
        names = ["in.weight", "in.bias", "ln_in.weight", "ln_in.bias"]
        for i in range(self.config.blocks):
            names += [
                f"block{i}.ln.weight",
                f"block{i}.ln.bias",
                f"block{i}.fc1.weight",
                f"block{i}.fc1.bias",
                f"block{i}.fc2.weight",
                f"block{i}.fc2.bias",
            ]
        names += [
            "ln_out.weight",
            "ln_out.bias",
            "policy.weight",
            "policy.bias",
            "value1.weight",
            "value1.bias",
            "value2.weight",
            "value2.bias",
        ]
        return names

    def _module_for(self, name: str) -> torch.Tensor:
        """The tensor a checkpoint name refers to."""
        head, _, attr = name.rpartition(".")
        if head.startswith("block"):
            index, _, inner = head.partition(".")
            module = getattr(self.blocks[int(index[len("block") :])], inner)
        elif head == "in":
            module = self.inp
        else:
            module = getattr(self, head)
        return getattr(module, attr)

    def tensors(self) -> list[torch.Tensor]:
        """Every parameter, flattened, in :meth:`parameter_order`."""
        return [self._module_for(n).detach() for n in self.parameter_order()]

    def load_tensors(self, arrays: list[Any]) -> None:
        """Overwrite every parameter from flat arrays in :meth:`parameter_order`."""
        with torch.no_grad():
            for name, flat in zip(self.parameter_order(), arrays):
                target = self._module_for(name)
                target.copy_(torch.as_tensor(flat, dtype=torch.float32).view_as(target))

    def randomise_layernorms(self, generator: torch.Generator) -> None:
        """Perturb every LayerNorm's affine parameters away from ``1`` and ``0``.

        Not a training decision — a testing one. PyTorch initialises LayerNorm to the
        identity affine, and under an identity affine a transposed or swapped gamma/beta
        would compute the same thing on both sides and the parity test would pass through
        the bug. ``Weights::random`` in Rust perturbs them for the same reason. Training
        overwrites these on the first step either way.
        """
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.LayerNorm):
                    module.weight.add_(
                        0.05 * (2 * torch.rand(module.weight.shape, generator=generator) - 1)
                    )
                    module.bias.add_(
                        0.05 * (2 * torch.rand(module.bias.shape, generator=generator) - 1)
                    )
