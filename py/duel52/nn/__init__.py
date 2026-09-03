"""The PyTorch side of Phase 3: the network, and the checkpoint format.

The split of responsibilities, from ``PHASE3_STEP1.md`` §0:

* **PyTorch owns the architecture, the weights and the gradients.** :mod:`.model` is the
  definition; training will read it.
* **Rust owns the encoders and inference.** Phase 3's deliverable is an Elo table, and that
  table comes out of ``duel52 ladder``, which takes an ``AgentSpec``. So do ``match``,
  ``probe`` and ``play --opponent``. A Python-side agent could use none of it.

The two meet at a ``.d52nn`` checkpoint (:mod:`.checkpoint`).

Two rules this package exists to enforce
----------------------------------------

1. **Nothing here builds an observation tensor.** ``Game.encode_observation()`` is the only
   encoder; a second copy of the feature layout in Python could drift from the Rust one
   silently, and the trained function would stop matching the evaluated function with
   nothing crashing to say so.
2. **Nothing here computes a layout hash.** They come from ``duel52.encoding_spec()``, which
   reads them out of the engine. Python stamps what it is given; ``Weights::load`` on the
   Rust side refuses a checkpoint whose stamp does not match the build about to evaluate it.

Quick start
-----------

>>> python -m duel52.nn init --out checkpoints/init.d52nn
>>> ./target/release/duel52 match --a netpolicy:checkpoints/init.d52nn --b random
"""

from __future__ import annotations

from .checkpoint import (
    CHECKPOINT_MAGIC,
    CHECKPOINT_VERSION,
    Checkpoint,
    read_checkpoint,
    write_checkpoint,
)
from .model import Duel52Net, NetConfig, spec_for

__all__ = [
    "CHECKPOINT_MAGIC",
    "CHECKPOINT_VERSION",
    "Checkpoint",
    "Duel52Net",
    "NetConfig",
    "read_checkpoint",
    "spec_for",
    "write_checkpoint",
]
