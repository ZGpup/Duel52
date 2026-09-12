"""``python -m duel52.nn`` — checkpoint utilities.

    python -m duel52.nn init --out checkpoints/init.d52nn
    python -m duel52.nn inspect checkpoints/init.d52nn
    python -m duel52.nn widen --in models/duel52-split-lane-gen032.d52nn \\
        --out models/lane-gen032-reserve.d52nn \\
        --rules-file configs/rules/seven-shield.toml --encoding-slots 21

``init`` writes a random-init checkpoint stamped with the engine's own layout hashes, which
is what makes it loadable by ``duel52 match --a netpolicy:<path>``. It is deliberately the
*only* way a checkpoint gets its hashes: they come from ``duel52.encoding_spec()``, never
from anything computed here.

``widen`` is the encoder reserve's bridge (``MODULAR_RULES.md`` §7). A ruleset that claims a
status flag, a reserve phase or a reserve action block has a wider observation and policy
head than every checkpoint in ``models/``, so ``--init-from`` refuses one. This re-lays a
base-layout checkpoint into the extended layout, leaving every existing weight where it
means the same thing and zeroing the new rows — which turns "the first reserve ruleset costs
a 24-hour from-scratch run" into "it costs a 3-hour warm start".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .checkpoint import read_checkpoint, write_checkpoint
from .model import NetConfig, build_net, lane_spec_for, spec_for


def _init(args: argparse.Namespace) -> int:
    spec = spec_for(args.variant, args.encoding_slots, args.rules_file)
    config = NetConfig.from_spec(
        spec,
        width=args.width,
        blocks=args.blocks,
        value_hidden=args.value_hidden,
        arch=args.arch,
    )
    # Seeded, because `CLAUDE.md` says everything is: the same seed and the same
    # architecture must produce the same bytes, so an "identical" run really is one.
    generator = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    lanes = (
        lane_spec_for(args.variant, args.encoding_slots, args.rules_file)
        if args.arch == "lane"
        else None
    )
    model = build_net(config, lanes)
    model.randomise_layernorms(generator)

    path = write_checkpoint(args.out, model=model, spec=spec)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"wrote {path} — {total:,} parameters, {path.stat().st_size / 1e6:.1f} MB\n"
        f"  arch={config.arch} obs_dim={config.obs_dim} action_dim={config.action_dim} "
        f"width={config.width} blocks={config.blocks} value_hidden={config.value_hidden}\n"
        f"  obs_layout_hash={spec['obs_layout_hash']} "
        f"action_layout_hash={spec['action_layout_hash']}"
    )
    return 0


#: Which embedding map re-lays which tensor, per architecture.
#:
#: ``(map name, axis)`` — axis 1 is the input side of a ``[out, in]`` weight matrix, axis 0
#: the output side. A tensor not named here is carried over untouched, which is every tensor
#: in the trunk: the reserve changes the width of the observation and the policy head and
#: nothing between them.
_WIDEN_PLAN: dict[str, dict[str, tuple[str, int]]] = {
    "mlp": {
        "in.weight": ("obs", 1),
        "policy.weight": ("action", 0),
        "policy.bias": ("action", 0),
    },
    "lane": {
        "lane_in.weight": ("lane_obs", 1),
        "glob_in.weight": ("global_obs", 1),
        "policy_lane.weight": ("lane_action", 0),
        "policy_lane.bias": ("lane_action", 0),
        "policy_glob.weight": ("global_action", 0),
        "policy_glob.bias": ("global_action", 0),
    },
}


def _widen(args: argparse.Namespace) -> int:
    """Re-lay a base-layout checkpoint into the encoder reserve's extended layout.

    ``MODULAR_RULES.md`` §7. Every existing weight keeps the meaning it had — the embedding
    is monotone and injective, and the engine computes it — and the rows the reserve adds are
    **zero**, which is the right value: the features they read are zero in any position the
    base ruleset could produce, so a widened network computes the same function it did.
    """
    import numpy as np

    from .._engine import reserve_embedding

    spec = spec_for(args.variant, args.encoding_slots, args.rules_file)
    ckpt = read_checkpoint(args.input)
    if ckpt.arch not in _WIDEN_PLAN:
        raise ValueError(f"unknown architecture {ckpt.arch!r}")

    embed = reserve_embedding(args.rules_file, args.encoding_slots)
    if (ckpt.obs_dim, ckpt.action_dim) != (
        embed["base_obs_dim"],
        embed["base_action_dim"],
    ):
        raise ValueError(
            f"{args.input} is {ckpt.obs_dim}/{ckpt.action_dim} wide, but the base layout for "
            f"this ruleset is {embed['base_obs_dim']}/{embed['base_action_dim']}. `widen` "
            f"only ever goes base -> extended, and only at a matching `--encoding-slots` "
            f"(this run used {spec['encoding_slots']})."
        )
    if (ckpt.obs_layout_hash, ckpt.action_layout_hash) == (
        spec["obs_layout_hash"],
        spec["action_layout_hash"],
    ):
        raise ValueError(
            f"{args.input} already matches this ruleset's layout — nothing to widen."
        )

    maps = {
        key: np.frombuffer(embed[key], dtype="<u4").astype(np.int64)
        for key in ("obs", "action", "lane_obs", "global_obs", "lane_action", "global_action")
    }
    # The widths each axis grows to. Read off `lane_structure`, not recomputed here.
    widths = {
        "obs": embed["extended_obs_dim"],
        "action": embed["extended_action_dim"],
        "lane_obs": spec["lane_obs_len"],
        "global_obs": spec["global_obs_len"],
        "lane_action": spec["lane_action_len"],
        "global_action": spec["global_action_len"],
    }

    plan = _WIDEN_PLAN[ckpt.arch]
    shapes = _tensor_shapes(ckpt)
    out: list[np.ndarray] = []
    for name, flat in zip(ckpt.param_order, ckpt.tensors, strict=True):
        if name not in plan:
            out.append(flat)
            continue
        key, axis = plan[name]
        index, width = maps[key], widths[key]
        old = flat.reshape(shapes[name])
        new_shape = list(old.shape)
        new_shape[axis] = width
        wide = np.zeros(new_shape, dtype=np.float32)
        # A scatter, not a gather: `index[k]` is where base position `k` goes. Everything
        # not written stays zero, and those are exactly the reserve's rows.
        if axis == 0:
            wide[index] = old
        else:
            wide[:, index] = old
        out.append(wide.reshape(-1))

    _write_raw(
        args.out,
        ckpt=ckpt,
        spec=spec,
        tensors=out,
        lane_obs=widths["lane_obs"] if ckpt.arch == "lane" else 0,
        lane_action=widths["lane_action"] if ckpt.arch == "lane" else 0,
    )
    print(
        f"widened {args.input} -> {args.out}\n"
        f"  obs_dim     {ckpt.obs_dim} -> {embed['extended_obs_dim']}\n"
        f"  action_dim  {ckpt.action_dim} -> {embed['extended_action_dim']}\n"
        f"  rules       {spec['rules_name']}/{spec['rules_hash']}\n"
        f"  layout      {spec['obs_layout_hash']} / {spec['action_layout_hash']}\n"
        f"The added weights are zero, and the features they read are zero in every position "
        f"the base\nruleset could produce — so this network plays exactly as it did until the "
        f"new rules fire."
    )
    return 0


def _tensor_shapes(ckpt) -> dict[str, tuple[int, ...]]:
    """The 2-D shape of each named tensor, from the header's own widths.

    The payload is flat, so a reshape needs the shape from somewhere; taking it from the
    header means a file whose header and payload disagree fails here rather than producing a
    silently transposed matrix.
    """
    w, a = ckpt.width, ckpt.action_dim
    shapes: dict[str, tuple[int, ...]] = {}
    if ckpt.arch == "lane":
        glob_obs = ckpt.obs_dim - ckpt.lanes * ckpt.lane_obs
        glob_action = a - ckpt.lanes * ckpt.lane_action
        shapes["lane_in.weight"] = (w, ckpt.lane_obs)
        shapes["glob_in.weight"] = (w, glob_obs)
        shapes["policy_lane.weight"] = (ckpt.lane_action, w)
        shapes["policy_lane.bias"] = (ckpt.lane_action,)
        shapes["policy_glob.weight"] = (glob_action, w)
        shapes["policy_glob.bias"] = (glob_action,)
    else:
        shapes["in.weight"] = (w, ckpt.obs_dim)
        shapes["policy.weight"] = (a, w)
        shapes["policy.bias"] = (a,)
    return shapes


def _write_raw(path, *, ckpt, spec, tensors, lane_obs: int, lane_action: int):
    """Write a checkpoint from raw tensors rather than from a live model.

    :func:`write_checkpoint` reads shapes off a ``torch`` module, which would mean building
    one just to save it. The header fields here all come from the checkpoint being widened or
    from ``spec``; nothing is computed locally, which is the rule the format exists to keep.
    """
    import struct

    from .checkpoint import CHECKPOINT_MAGIC, CHECKPOINT_VERSION, _header_text

    header = _header_text(
        obs_dim=spec["obs_dim"],
        action_dim=spec["action_dim"],
        width=ckpt.width,
        blocks=ckpt.blocks,
        value_hidden=ckpt.value_hidden,
        obs_layout_hash=spec["obs_layout_hash"],
        action_layout_hash=spec["action_layout_hash"],
        rules_name=spec.get("rules_name"),
        rules_hash=spec.get("rules_hash"),
        param_order=ckpt.param_order,
        arch=ckpt.arch,
        lanes=ckpt.lanes,
        lane_obs=lane_obs,
        lane_action=lane_action,
    ).encode("utf-8")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(CHECKPOINT_MAGIC)
        f.write(struct.pack("<H", CHECKPOINT_VERSION))
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        for t in tensors:
            f.write(np_contiguous(t))
    return path


def np_contiguous(t):
    import numpy as np

    return np.ascontiguousarray(t, dtype="<f4").tobytes()


def _inspect(args: argparse.Namespace) -> int:
    ckpt = read_checkpoint(args.path)
    spec = spec_for(args.variant, args.encoding_slots, args.rules_file)
    print(f"{args.path}")
    print(f"  arch               {ckpt.arch}")
    print(f"  obs_dim            {ckpt.obs_dim}")
    print(f"  action_dim         {ckpt.action_dim}")
    print(f"  width              {ckpt.width}")
    print(f"  blocks             {ckpt.blocks}")
    print(f"  value_hidden       {ckpt.value_hidden}")
    if ckpt.arch == "lane":
        print(f"  lanes              {ckpt.lanes}")
        print(f"  lane_obs           {ckpt.lane_obs}  (global {ckpt.obs_dim - ckpt.lanes * ckpt.lane_obs})")
        print(
            f"  lane_action        {ckpt.lane_action}  "
            f"(global {ckpt.action_dim - ckpt.lanes * ckpt.lane_action})"
        )
    print(f"  obs_layout_hash    {ckpt.obs_layout_hash}")
    print(f"  action_layout_hash {ckpt.action_layout_hash}")
    print(f"  tensors            {len(ckpt.param_order)}")
    print(f"  parameters         {sum(t.size for t in ckpt.tensors):,}")
    try:
        ckpt.check_against(spec)
    except ValueError as e:
        print(f"\nINCOMPATIBLE with this engine build:\n  {e}", file=sys.stderr)
        return 1
    print(f"\nmatches this engine build ({spec['variant']}).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m duel52.nn", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shape_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--variant", default="split", help="which encoding spec (default: split)")
        p.add_argument(
            "--encoding-slots",
            type=int,
            default=None,
            dest="encoding_slots",
            help="override the encoder's slot bound (default: the config's 16)",
        )
        p.add_argument(
            "--rules-file",
            default=None,
            dest="rules_file",
            help="a configs/rules/*.toml, in place of --variant. Required for a ruleset "
            "that claims the encoder reserve (MODULAR_RULES.md §7), whose layout differs "
            "from the canonical one",
        )

    init = sub.add_parser("init", help="write a random-init checkpoint")
    init.add_argument("--out", type=Path, required=True, help="where to write it")
    init.add_argument("--seed", type=int, default=0)
    init.add_argument("--width", type=int, default=512)
    init.add_argument("--blocks", type=int, default=5)
    init.add_argument("--value-hidden", type=int, default=256, dest="value_hidden")
    init.add_argument(
        "--arch",
        choices=("mlp", "lane"),
        default="mlp",
        help="'mlp' is DESIGN.md §5's flat trunk; 'lane' is the lane-equivariant network "
        "(PLAN.md §4.2b), which shares one set of weights across the three lanes and so "
        "cannot represent a lane preference at all",
    )
    add_shape_flags(init)
    init.set_defaults(func=_init)

    inspect = sub.add_parser("inspect", help="print a checkpoint's header and check it")
    inspect.add_argument("path", type=Path)
    add_shape_flags(inspect)
    inspect.set_defaults(func=_inspect)

    widen = sub.add_parser(
        "widen",
        help="re-lay a base-layout checkpoint into the encoder reserve's extended layout",
        description=(
            "MODULAR_RULES.md §7. A ruleset that claims the reserve has a wider observation "
            "and policy head than every shipped checkpoint, so --init-from refuses one. This "
            "moves the weights across: each keeps the meaning it had, and the rows the "
            "reserve adds are zero — which is correct, because the features they read are "
            "zero in every position the base ruleset could produce."
        ),
    )
    widen.add_argument("--in", dest="input", type=Path, required=True, help="the checkpoint to widen")
    widen.add_argument("--out", type=Path, required=True, help="where to write the widened one")
    add_shape_flags(widen)
    widen.set_defaults(func=_widen)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
