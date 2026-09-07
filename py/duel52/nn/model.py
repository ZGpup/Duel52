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

__all__ = [
    "Duel52LaneNet",
    "Duel52Net",
    "LaneSpec",
    "NetConfig",
    "ResidualBlock",
    "build_net",
    "lane_spec_for",
    "spec_for",
]

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
    #: ``"mlp"`` for ``DESIGN.md`` §5's flat trunk, ``"lane"`` for the lane-equivariant one.
    #: Travels in the checkpoint header as ``arch``; a checkpoint written before the key
    #: existed reads back as ``"mlp"``, which is what keeps gen016/022/031 loading.
    arch: str = "mlp"

    @staticmethod
    def from_spec(spec: dict[str, Any], **overrides: Any) -> NetConfig:
        """Build from ``duel52.encoding_spec()``, overriding trunk sizes if asked."""
        return NetConfig(
            obs_dim=spec["obs_dim"],
            action_dim=spec["action_dim"],
            width=overrides.get("width", 512),
            blocks=overrides.get("blocks", 5),
            value_hidden=overrides.get("value_hidden", 256),
            arch=overrides.get("arch", "mlp"),
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


# ------------------------------------------------------ the lane-equivariant network --


@dataclass(frozen=True)
class LaneSpec:
    """The engine's lane partition, as gather indices and widths.

    Comes from ``duel52._engine.lane_structure`` and **never** from arithmetic here.
    ``CLAUDE.md``: there is exactly one encoder and a table of "which lane owns this float"
    is a reading of it. A wrong table does not crash — it routes lane 2's board through lane
    1's weights and the agent is merely bad, with the training run as the natural suspect.
    """

    #: ``[lanes, lane_obs_len]`` — observation indices owned by each lane.
    lane_obs: torch.Tensor
    #: ``[global_obs_len]``
    global_obs: torch.Tensor
    #: ``[lanes, lane_action_len]`` — policy indices owned by each lane.
    lane_action: torch.Tensor
    #: ``[global_action_len]``
    global_action: torch.Tensor
    lanes: int
    lane_obs_len: int
    global_obs_len: int
    lane_action_len: int
    global_action_len: int


def lane_spec_for(variant: str = "split", encoding_slots: int | None = None) -> LaneSpec:
    """The lane partition for a configuration, from the engine."""
    import numpy as np

    from .._engine import lane_structure

    raw = (
        lane_structure(variant=variant)
        if encoding_slots is None
        else lane_structure(variant=variant, encoding_slots=encoding_slots)
    )
    as_long = lambda b: torch.from_numpy(  # noqa: E731
        np.frombuffer(b, dtype="<u4").astype("int64")
    )
    return LaneSpec(
        lane_obs=torch.stack([as_long(b) for b in raw["lane_obs"]]),
        global_obs=as_long(raw["global_obs"]),
        lane_action=torch.stack([as_long(b) for b in raw["lane_action"]]),
        global_action=as_long(raw["global_action"]),
        lanes=raw["lanes"],
        lane_obs_len=raw["lane_obs_len"],
        global_obs_len=raw["global_obs_len"],
        lane_action_len=raw["lane_action_len"],
        global_action_len=raw["global_action_len"],
    )


class LaneBlock(nn.Module):
    """``h_l + W2·relu(W1·ln(h_l) + Wm·mean_l ln(h_l) + b1) + b2``.

    The mean is the only channel between lanes, and being a *mean* is exactly why the block
    stays equivariant: permuting the lanes permutes the ``h_l`` and leaves the mean alone.

    ``fcm`` has no bias on purpose — its output is added to ``fc1``'s, which already carries
    one, so a second would be the same parameter twice.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(width, eps=LN_EPS)
        self.fc1 = nn.Linear(width, width)
        self.fcm = nn.Linear(width, width, bias=False)
        self.fc2 = nn.Linear(width, width)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """``h`` is ``[n, lanes, width]``."""
        n = self.ln(h)
        mixed = self.fcm(n.mean(dim=1, keepdim=True))
        return h + self.fc2(torch.relu(self.fc1(n) + mixed))


class Duel52LaneNet(nn.Module):
    """The lane-equivariant network. ``PLAN.md`` §4.2b.

    ``engine/src/nn/lane.rs`` is the reference implementation and its module header carries
    the equations; this must agree with it to ``py/tests/test_parity.py``'s thresholds.

    Why it exists: ``FINDINGS.md`` F4.3 measured the flat MLP preferring lane 3 to lane 1 in
    24 of 24 seeds, and F4.5 showed six-fold augmentation shrinking that (policy TV
    0.152 → 0.039) without removing it — argmax agreement stopped at 114/128. Augmentation
    can only ask for the symmetry. Here **no parameter is indexed by a lane**, so a lane
    preference is not reduced but unrepresentable.

    It is also *smaller* than the flat network it replaces: the policy head is one shared
    ``width × lane_action`` matrix rather than ``width × action_dim``, and the input
    projection is unchanged in total (three lanes of ``lane_obs × width`` is the same work as
    one ``obs_dim × width``).
    """

    def __init__(self, config: NetConfig, spec: LaneSpec) -> None:
        super().__init__()
        if config.arch != "lane":
            raise ValueError(f"Duel52LaneNet needs arch='lane', got {config.arch!r}")
        if spec.lanes * spec.lane_obs_len + spec.global_obs_len != config.obs_dim:
            raise ValueError("the lane partition does not account for every observation float")
        if spec.lanes * spec.lane_action_len + spec.global_action_len != config.action_dim:
            raise ValueError("the lane partition does not account for every logit")

        self.config = config
        self.lanes = spec.lanes
        w = config.width
        # Buffers, not parameters: they are the engine's index tables and they travel with
        # the module to whatever device it is on. Not saved into the checkpoint — the engine
        # recomputes them from `config`, which is the point of having one encoder.
        self.register_buffer("lane_obs", spec.lane_obs, persistent=False)
        self.register_buffer("global_obs", spec.global_obs, persistent=False)
        self.register_buffer("lane_action", spec.lane_action, persistent=False)
        self.register_buffer("global_action", spec.global_action, persistent=False)

        self.lane_in = nn.Linear(spec.lane_obs_len, w)
        self.glob_in = nn.Linear(spec.global_obs_len, w, bias=False)
        self.ln_in = nn.LayerNorm(w, eps=LN_EPS)
        self.blocks = nn.ModuleList(LaneBlock(w) for _ in range(config.blocks))
        self.ln_out = nn.LayerNorm(w, eps=LN_EPS)
        self.policy_lane = nn.Linear(w, spec.lane_action_len)
        self.policy_glob = nn.Linear(w, spec.global_action_len)
        self.value1 = nn.Linear(w, config.value_hidden)
        self.value2 = nn.Linear(config.value_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(policy_logits, value)`` for a batch of observations.

        ``x`` is ``[n, obs_dim]``. Returns ``[n, action_dim]`` **raw, unmasked** logits and
        ``[n]`` values in ``(-1, 1)`` — the same contract as :class:`Duel52Net`, so every
        caller is indifferent to which architecture it holds.
        """
        n = x.shape[0]
        # [n, lanes, lane_obs_len] — one shared matrix reads every lane.
        per_lane = x[:, self.lane_obs]
        h = self.lane_in(per_lane) + self.glob_in(x[:, self.global_obs]).unsqueeze(1)
        h = torch.relu(self.ln_in(h))
        for block in self.blocks:
            h = block(h)
        h = self.ln_out(h)
        pooled = h.mean(dim=1)

        logits = x.new_empty((n, self.config.action_dim))
        # `index_copy_` rather than fancy assignment so the scatter is explicit: lane l's
        # k-th logit goes to the engine index the table names, for every lane at once.
        lane_logits = self.policy_lane(h)  # [n, lanes, lane_action_len]
        logits.index_copy_(
            1, self.lane_action.reshape(-1), lane_logits.reshape(n, -1)
        )
        logits.index_copy_(1, self.global_action, self.policy_glob(pooled))

        value = torch.tanh(self.value2(torch.relu(self.value1(pooled)))).squeeze(-1)
        return logits, value

    # ------------------------------------------------------------------ checkpoints --

    def parameter_order(self) -> list[str]:
        """Tensor names in checkpoint order. Matches ``Arch::lane_params`` in Rust."""
        names = ["lane_in.weight", "lane_in.bias", "glob_in.weight", "ln_in.weight", "ln_in.bias"]
        for i in range(self.config.blocks):
            names += [
                f"block{i}.ln.weight",
                f"block{i}.ln.bias",
                f"block{i}.fc1.weight",
                f"block{i}.fcm.weight",
                f"block{i}.fc1.bias",
                f"block{i}.fc2.weight",
                f"block{i}.fc2.bias",
            ]
        names += [
            "ln_out.weight",
            "ln_out.bias",
            "policy_lane.weight",
            "policy_lane.bias",
            "policy_glob.weight",
            "policy_glob.bias",
            "value1.weight",
            "value1.bias",
            "value2.weight",
            "value2.bias",
        ]
        return names

    def _module_for(self, name: str) -> torch.Tensor:
        head, _, attr = name.rpartition(".")
        if head.startswith("block"):
            index, _, inner = head.partition(".")
            module = getattr(self.blocks[int(index[len("block") :])], inner)
        else:
            module = getattr(self, head)
        return getattr(module, attr)

    tensors = Duel52Net.tensors
    load_tensors = Duel52Net.load_tensors
    randomise_layernorms = Duel52Net.randomise_layernorms


def build_net(config: NetConfig, spec: LaneSpec | None = None) -> nn.Module:
    """The network ``config.arch`` names.

    One place that maps the header's ``arch`` string to a module, so `train`, `nn init` and
    the parity test cannot disagree about what a checkpoint holds.
    """
    if config.arch == "mlp":
        return Duel52Net(config)
    if config.arch == "lane":
        if spec is None:
            raise ValueError("a lane-equivariant net needs the engine's lane_spec_for(...)")
        return Duel52LaneNet(config, spec)
    raise ValueError(f"unknown arch {config.arch!r}; expected 'mlp' or 'lane'")
