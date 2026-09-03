"""The ``.d52nn`` checkpoint format, read and written with the standard library.

The format, from ``PHASE3_STEP1.md`` §1.4 — ``engine/src/nn/weights.rs`` is the other half::

    magic        6 bytes   b"D52NN\\0"
    version      u16 LE
    header_len   u32 LE
    header       UTF-8, newline-delimited key=value, in this fixed key order:
                     obs_dim, action_dim, width, blocks, value_hidden,
                     obs_layout_hash, action_layout_hash, param_order
    payload      for each name in param_order: a little-endian f32 array,
                 its length implied by the architecture

Why not ONNX or safetensors
---------------------------

ONNX pulls a large C++ dependency into the engine for a five-layer MLP. safetensors needs a
JSON parser on the Rust side, and ``engine`` has no dependencies by design (see the workspace
``Cargo.toml``). This is ~100 lines per side, and it costs no new pip dependency either:
:mod:`struct` and ``numpy.tobytes()`` are enough.

Why the header carries hashes
-----------------------------

Silent layout drift between the function that was trained and the function that is evaluated
does not crash anything — it produces an agent that is merely bad, at which point the natural
suspect is the training run and the real bug is a feature that moved three floats to the
left. ``obs_layout_hash`` and ``action_layout_hash`` make it a load error instead.

**Python never computes them.** They come from ``duel52.encoding_spec()``, which reads them
out of the engine; this module stamps what it is given. That is what makes the check
meaningful — two independent implementations of a hash can disagree about the hash, not just
about the layout.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CHECKPOINT_MAGIC",
    "CHECKPOINT_VERSION",
    "Checkpoint",
    "read_checkpoint",
    "write_checkpoint",
]

CHECKPOINT_MAGIC = b"D52NN\0"
CHECKPOINT_VERSION = 1

#: Header keys, in the order they are written. Fixed so the file is byte-reproducible.
HEADER_KEYS = (
    "obs_dim",
    "action_dim",
    "width",
    "blocks",
    "value_hidden",
    "obs_layout_hash",
    "action_layout_hash",
    "param_order",
)


@dataclass
class Checkpoint:
    """A parsed checkpoint: the header fields plus one flat ``float32`` array per tensor."""

    obs_dim: int
    action_dim: int
    width: int
    blocks: int
    value_hidden: int
    obs_layout_hash: str
    action_layout_hash: str
    param_order: list[str]
    tensors: list[np.ndarray]

    def named(self) -> dict[str, np.ndarray]:
        return dict(zip(self.param_order, self.tensors))

    def check_against(self, spec: dict[str, Any]) -> None:
        """Raise if this checkpoint does not match an engine encoding spec.

        The same four checks ``Weights::load`` makes on the Rust side. Doing them here too is
        not redundant: it catches the mismatch in the process that can explain it, rather
        than as a failure inside a CLI subprocess.
        """
        for field in ("obs_dim", "action_dim"):
            if getattr(self, field) != spec[field]:
                raise ValueError(
                    f"{field} is {getattr(self, field)} in the checkpoint but "
                    f"{spec[field]} in the engine — the checkpoint was built against a "
                    f"different encoder"
                )
        for field in ("obs_layout_hash", "action_layout_hash"):
            if getattr(self, field) != spec[field]:
                raise ValueError(
                    f"{field} is {getattr(self, field)} in the checkpoint but "
                    f"{spec[field]} in the engine — the layout moved since the checkpoint "
                    f"was written, so its weights no longer mean what they meant"
                )


def _header_text(
    *,
    obs_dim: int,
    action_dim: int,
    width: int,
    blocks: int,
    value_hidden: int,
    obs_layout_hash: str,
    action_layout_hash: str,
    param_order: list[str],
) -> str:
    values = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "width": width,
        "blocks": blocks,
        "value_hidden": value_hidden,
        "obs_layout_hash": obs_layout_hash,
        "action_layout_hash": action_layout_hash,
        "param_order": ",".join(param_order),
    }
    return "".join(f"{key}={values[key]}\n" for key in HEADER_KEYS)


def write_checkpoint(
    path: str | Path,
    *,
    model: Any,
    spec: dict[str, Any],
) -> Path:
    """Write ``model`` to ``path`` in the ``.d52nn`` format.

    ``spec`` is a ``duel52.encoding_spec()`` dict; its two hashes are stamped verbatim.
    """
    path = Path(path)
    param_order = model.parameter_order()
    tensors = model.tensors()
    if len(tensors) != len(param_order):
        raise ValueError(
            f"the model has {len(tensors)} tensors but names {len(param_order)}"
        )

    header = _header_text(
        obs_dim=model.config.obs_dim,
        action_dim=model.config.action_dim,
        width=model.config.width,
        blocks=model.config.blocks,
        value_hidden=model.config.value_hidden,
        obs_layout_hash=spec["obs_layout_hash"],
        action_layout_hash=spec["action_layout_hash"],
        param_order=param_order,
    ).encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(CHECKPOINT_MAGIC)
        f.write(struct.pack("<H", CHECKPOINT_VERSION))
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        for tensor in tensors:
            # `ascontiguousarray` matters: a transposed view would write the same numbers in
            # a different order, which is precisely the transcription bug the parity test is
            # built to catch — better not to be able to make it here.
            flat = np.ascontiguousarray(
                np.asarray(tensor, dtype=np.float32).reshape(-1)
            )
            f.write(flat.tobytes())
    return path


def read_checkpoint(path: str | Path, *, arch: dict[str, int] | None = None) -> Checkpoint:
    """Parse a ``.d52nn`` file.

    ``arch`` is unused except as documentation of intent — tensor lengths are derived from
    the header's own architecture, so the payload cannot be misread as a different shape.
    """
    del arch  # the header is self-describing; the parameter exists to say so
    data = Path(path).read_bytes()
    if len(data) < 12:
        raise ValueError(f"{path}: truncated — not even a header")
    if data[:6] != CHECKPOINT_MAGIC:
        raise ValueError(f"{path}: not a .d52nn checkpoint (bad magic)")
    (version,) = struct.unpack_from("<H", data, 6)
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"{path}: checkpoint format version {version}, this build reads "
            f"{CHECKPOINT_VERSION}"
        )
    (header_len,) = struct.unpack_from("<I", data, 8)
    header_end = 12 + header_len
    header = data[12:header_end].decode("utf-8")

    fields: dict[str, str] = {}
    for line in header.splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition("=")
        fields[key] = value
    missing = [k for k in HEADER_KEYS if k not in fields]
    if missing:
        raise ValueError(f"{path}: header is missing {', '.join(missing)}")

    param_order = fields["param_order"].split(",")
    lengths = _tensor_lengths(
        param_order,
        obs_dim=int(fields["obs_dim"]),
        action_dim=int(fields["action_dim"]),
        width=int(fields["width"]),
        value_hidden=int(fields["value_hidden"]),
    )
    payload = np.frombuffer(data, dtype="<f4", offset=header_end)
    if payload.size != sum(lengths):
        raise ValueError(
            f"{path}: payload holds {payload.size} floats but the architecture needs "
            f"{sum(lengths)}"
        )

    tensors: list[np.ndarray] = []
    at = 0
    for n in lengths:
        tensors.append(np.array(payload[at : at + n], dtype=np.float32))
        at += n

    return Checkpoint(
        obs_dim=int(fields["obs_dim"]),
        action_dim=int(fields["action_dim"]),
        width=int(fields["width"]),
        blocks=int(fields["blocks"]),
        value_hidden=int(fields["value_hidden"]),
        obs_layout_hash=fields["obs_layout_hash"],
        action_layout_hash=fields["action_layout_hash"],
        param_order=param_order,
        tensors=tensors,
    )


def _tensor_lengths(
    param_order: list[str],
    *,
    obs_dim: int,
    action_dim: int,
    width: int,
    value_hidden: int,
) -> list[int]:
    """Length of each tensor, from its name and the architecture.

    Mirrors ``Arch::params`` in ``engine/src/nn/weights.rs``. A name this does not recognise
    is an error rather than a guess: an unknown tensor means the two sides disagree about the
    architecture, and reading on would silently misalign everything after it.
    """
    lengths = []
    for name in param_order:
        if name == "in.weight":
            lengths.append(width * obs_dim)
        elif name == "policy.weight":
            lengths.append(action_dim * width)
        elif name == "policy.bias":
            lengths.append(action_dim)
        elif name == "value1.weight":
            lengths.append(value_hidden * width)
        elif name == "value1.bias":
            lengths.append(value_hidden)
        elif name == "value2.weight":
            lengths.append(value_hidden)
        elif name == "value2.bias":
            lengths.append(1)
        elif name.endswith((".fc1.weight", ".fc2.weight")):
            lengths.append(width * width)
        elif name.endswith((".weight", ".bias")):
            # Every remaining tensor is width-shaped: the input bias, the LayerNorm affines,
            # and the residual blocks' linear biases.
            lengths.append(width)
        else:
            raise ValueError(f"unknown tensor name in param_order: {name!r}")
    return lengths
