"""``python -m duel52.nn`` — checkpoint utilities.

    python -m duel52.nn init --out checkpoints/init.d52nn
    python -m duel52.nn inspect checkpoints/init.d52nn

``init`` writes a random-init checkpoint stamped with the engine's own layout hashes, which
is what makes it loadable by ``duel52 match --a netpolicy:<path>``. It is deliberately the
*only* way a checkpoint gets its hashes: they come from ``duel52.encoding_spec()``, never
from anything computed here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .checkpoint import read_checkpoint, write_checkpoint
from .model import Duel52Net, NetConfig, spec_for


def _init(args: argparse.Namespace) -> int:
    spec = spec_for(args.variant, args.encoding_slots)
    config = NetConfig.from_spec(
        spec,
        width=args.width,
        blocks=args.blocks,
        value_hidden=args.value_hidden,
    )
    # Seeded, because `CLAUDE.md` says everything is: the same seed and the same
    # architecture must produce the same bytes, so an "identical" run really is one.
    generator = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    model = Duel52Net(config)
    model.randomise_layernorms(generator)

    path = write_checkpoint(args.out, model=model, spec=spec)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"wrote {path} — {total:,} parameters, {path.stat().st_size / 1e6:.1f} MB\n"
        f"  obs_dim={config.obs_dim} action_dim={config.action_dim} "
        f"width={config.width} blocks={config.blocks} value_hidden={config.value_hidden}\n"
        f"  obs_layout_hash={spec['obs_layout_hash']} "
        f"action_layout_hash={spec['action_layout_hash']}"
    )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    ckpt = read_checkpoint(args.path)
    spec = spec_for(args.variant, args.encoding_slots)
    print(f"{args.path}")
    print(f"  obs_dim            {ckpt.obs_dim}")
    print(f"  action_dim         {ckpt.action_dim}")
    print(f"  width              {ckpt.width}")
    print(f"  blocks             {ckpt.blocks}")
    print(f"  value_hidden       {ckpt.value_hidden}")
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

    init = sub.add_parser("init", help="write a random-init checkpoint")
    init.add_argument("--out", type=Path, required=True, help="where to write it")
    init.add_argument("--seed", type=int, default=0)
    init.add_argument("--width", type=int, default=512)
    init.add_argument("--blocks", type=int, default=5)
    init.add_argument("--value-hidden", type=int, default=256, dest="value_hidden")
    add_shape_flags(init)
    init.set_defaults(func=_init)

    inspect = sub.add_parser("inspect", help="print a checkpoint's header and check it")
    inspect.add_argument("path", type=Path)
    add_shape_flags(inspect)
    inspect.set_defaults(func=_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
