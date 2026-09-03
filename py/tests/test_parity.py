"""The Rust forward pass and the PyTorch forward pass compute the same function.

This is the test that makes the Phase 3 architecture safe. Inference runs in Rust — because
the Elo table comes out of ``duel52 ladder``, which takes an ``AgentSpec`` — while PyTorch
owns the architecture and the gradients. Two implementations of one function is a real risk,
and this is what converts it from a risk into a build failure.

**Direction matters, and it is not symmetric.** Rust produces the observations, because it
owns the encoder and Python must never write a second one. PyTorch is the reference for the
forward pass, because it owns the architecture. So ``duel52 nn-dump`` writes Rust's answer,
and this test recomputes the same function from the same checkpoint and the same inputs.

Bit-exactness is not required and not achievable: PyTorch reduces in a different order, and
on a different device again. The thresholds below are loose enough to survive accumulation
order and tight enough that any transcription bug fails immediately — a transposed weight, a
swapped gamma and beta, a missing residual, a LayerNorm epsilon that does not match all
produce ``O(1)`` differences, not ``O(1e-6)`` ones.

Skips, rather than fails, when the release binary or torch is missing: this file has to be
runnable on a checkout that has not been built yet.
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from duel52._engine import encoding_spec  # noqa: E402
from duel52.nn.checkpoint import read_checkpoint  # noqa: E402
from duel52.nn.model import Duel52Net, NetConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BINARY = REPO / "target" / "release" / "duel52"

#: A transcription bug produces `O(1)` differences; f32 accumulation order produces `O(1e-6)`
#: ones. These sit between the two, closer to the second.
MAX_LOGIT_DELTA = 1e-3
MAX_VALUE_DELTA = 1e-4
MAX_TV_DISTANCE = 1e-4


def _read_dump(path: Path) -> dict:
    """Parse a ``duel52 nn-dump`` file.

    Same container as a checkpoint (``PHASE3_STEP1.md`` §1.4), different magic, and the
    header names its own payload order rather than leaving the reader to assume one.
    """
    data = path.read_bytes()
    assert data[:6] == b"D52DMP", "not an nn-dump file"
    (version,) = struct.unpack_from("<H", data, 6)
    assert version == 1, f"unknown dump version {version}"
    (header_len,) = struct.unpack_from("<I", data, 8)
    end = 12 + header_len
    header = {}
    for line in data[12:end].decode("utf-8").splitlines():
        if line.strip():
            key, _, value = line.partition("=")
            header[key] = value

    rows = int(header["rows"])
    obs_dim = int(header["obs_dim"])
    action_dim = int(header["action_dim"])
    assert header["payload_order"] == "obs:f32,mask:u8,logits:f32,values:f32"

    at = end
    obs = np.frombuffer(data, "<f4", count=rows * obs_dim, offset=at).reshape(rows, obs_dim)
    at += 4 * rows * obs_dim
    mask = np.frombuffer(data, "u1", count=rows * action_dim, offset=at).reshape(rows, action_dim)
    at += rows * action_dim
    logits = np.frombuffer(data, "<f4", count=rows * action_dim, offset=at).reshape(
        rows, action_dim
    )
    at += 4 * rows * action_dim
    values = np.frombuffer(data, "<f4", count=rows, offset=at)
    at += 4 * rows
    assert at == len(data), f"{len(data) - at} unread bytes at the end of the dump"

    return {
        "header": header,
        "obs": obs,
        "mask": mask.astype(bool),
        "logits": logits,
        "values": values,
    }


@pytest.fixture(scope="module")
def parity(tmp_path_factory) -> dict:
    """A checkpoint, and the Rust dump produced from it."""
    if not BINARY.exists():
        pytest.skip(f"{BINARY} is not built — run `cargo build --release`")

    tmp = tmp_path_factory.mktemp("parity")
    checkpoint = tmp / "parity.d52nn"

    _init_checkpoint(checkpoint)

    dump = tmp / "parity.bin"
    subprocess.run(
        [
            str(BINARY),
            "nn-dump",
            "--checkpoint",
            str(checkpoint),
            "--games",
            "20",
            "--seed",
            "1",
            "--max-rows",
            "256",
            "--out",
            str(dump),
        ],
        check=True,
        capture_output=True,
    )
    return {"checkpoint": checkpoint, **_read_dump(dump)}


def _init_checkpoint(path: Path) -> None:
    """Write a small random-init checkpoint in-process.

    In-process rather than by shelling out to ``python -m duel52.nn``, so the test uses the
    same interpreter and the same installed extension pytest is running under — a subprocess
    would silently pick up whatever ``python`` is on PATH.
    """
    from duel52.nn.checkpoint import write_checkpoint

    spec = encoding_spec()
    config = NetConfig.from_spec(spec, width=64, blocks=3, value_hidden=32)
    generator = torch.Generator().manual_seed(20260903)
    torch.manual_seed(20260903)
    model = Duel52Net(config)
    model.randomise_layernorms(generator)
    write_checkpoint(path, model=model, spec=spec)


def _torch_model(checkpoint: Path) -> Duel52Net:
    ckpt = read_checkpoint(checkpoint)
    model = Duel52Net(
        NetConfig(
            obs_dim=ckpt.obs_dim,
            action_dim=ckpt.action_dim,
            width=ckpt.width,
            blocks=ckpt.blocks,
            value_hidden=ckpt.value_hidden,
        )
    )
    model.load_tensors(ckpt.tensors)
    model.eval()
    return model


def test_the_dump_was_produced_by_this_engine_build(parity):
    """The dump's layout hashes must match the engine Python is talking to.

    Checked first, and separately: if the layouts have drifted, every other assertion below
    would fail with a wall of float mismatches instead of one line naming the cause.
    """
    spec = encoding_spec(variant=parity["header"]["variant"])
    assert parity["header"]["obs_layout_hash"] == spec["obs_layout_hash"]
    assert parity["header"]["action_layout_hash"] == spec["action_layout_hash"]
    assert parity["obs"].shape[1] == spec["obs_dim"]
    assert parity["mask"].shape[1] == spec["action_dim"]


def test_the_checkpoint_matches_the_engines_encoding_spec(parity):
    read_checkpoint(parity["checkpoint"]).check_against(encoding_spec())


def test_rust_and_pytorch_compute_the_same_logits_and_values(parity):
    """The load-bearing assertion: two implementations, one function."""
    model = _torch_model(parity["checkpoint"])
    with torch.no_grad():
        logits, values = model(torch.from_numpy(parity["obs"].copy()))
    logits = logits.numpy()
    values = values.numpy()

    dl = np.abs(logits - parity["logits"]).max()
    dv = np.abs(values - parity["values"]).max()
    assert dl < MAX_LOGIT_DELTA, f"max |Δlogit| = {dl:.3g}"
    assert dv < MAX_VALUE_DELTA, f"max |Δvalue| = {dv:.3g}"


def test_the_masked_policies_agree_on_every_row(parity):
    """Small logit differences must not change the move.

    The value that actually reaches search is the *masked* distribution, so agreeing on raw
    logits is necessary and not sufficient: a near-tie between two legal actions could still
    flip the argmax. Checking the argmax and the total-variation distance is checking the
    thing the agent would actually do.
    """
    model = _torch_model(parity["checkpoint"])
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(parity["obs"].copy()))
    logits = logits.numpy()

    mask = parity["mask"]
    assert mask.any(axis=1).all(), "every dumped row must have at least one legal action"

    torch_p = _masked_softmax(logits, mask)
    rust_p = _masked_softmax(parity["logits"], mask)

    torch_best = torch_p.argmax(axis=1)
    rust_best = rust_p.argmax(axis=1)
    disagreements = int((torch_best != rust_best).sum())
    assert disagreements == 0, f"{disagreements} of {len(mask)} rows chose a different action"

    tv = 0.5 * np.abs(torch_p - rust_p).sum(axis=1).max()
    assert tv < MAX_TV_DISTANCE, f"max total-variation distance = {tv:.3g}"


def test_the_dump_covers_more_than_the_opening_position(parity):
    """Rows are sampled across the run, not taken from the first few plies.

    Twenty near-identical fresh deals would exercise almost none of the encoder, so a dump
    that had collapsed to the opening would make every assertion above nearly vacuous.
    """
    rows = parity["obs"]
    assert rows.shape[0] >= 32
    # Distinct legal-action counts, and distinct observations, are both cheap proxies for
    # "these are different positions at different stages".
    assert len(np.unique(parity["mask"].sum(axis=1))) > 5
    assert len(np.unique(rows, axis=0)) == rows.shape[0]


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax over the legal entries only, zero elsewhere.

    Masking is the caller's job on both sides — the network returns raw logits — so this is
    the caller, standing in for what PUCT will do.
    """
    z = np.where(mask, logits.astype(np.float64), -np.inf)
    z -= z.max(axis=1, keepdims=True)
    e = np.where(mask, np.exp(z), 0.0)
    return e / e.sum(axis=1, keepdims=True)
