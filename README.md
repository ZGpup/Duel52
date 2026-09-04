# Duel 52

An engine and self-play agent for [Duel 52](https://www.juddmadden.com/duel52/index.html),
the two-player combat card game by Judd Madden and Nina Riddell that uses a standard 52
card deck.

The goal is to answer a question nobody has published an answer to: what does optimal play
actually look like? As far as I can tell there is no existing engine, bot, or strategy
analysis for this game. The agent is the instrument. The insight is the deliverable.

## Status

**Phase 3 is in: there is a trained agent in the repo, and you can play it.** The engine
plays the full game to spec, with 300 Rust tests named after the rule sections they check, 77
Python tests, PyO3 bindings, and a text CLI. On top of it sits a frozen five-rung Elo ladder
— random, greedy, flat Monte Carlo, PIMC and SO-ISMCTS — built on determinization, so every
search agent reasons from its own information set rather than from the engine's ground truth.

The AlphaZero loop now runs end to end. There is exactly one encoder and it lives in Rust; a
network is defined and trained in PyTorch, evaluated in Rust, and a test asserts the two
forward passes compute the same function. Self-play writes trajectory shards, the trainer
replays and fits them, and a gate promotes a candidate only when it beats the incumbent over
200 games.

The first real run went through it: 57,000 self-play games, 19 generations, 1.94 hours on a
laptop. Thirteen candidates passed the gate, and the last of them is committed here as
[models/duel52-split-gen016.d52nn](models/duel52-split-gen016.d52nn). On the ladder, at 64
simulations — one twelfth of its opponent's budget — it lands **+495 Elo clear of
`ismcts:800`**, the strongest hand-written rung, and beats it 0.930 ± 0.035 head to head.

| agent | Elo | ± | vs. anchor |
|---|---:|---:|---:|
| **`netmcts@64`** | **+1476** | 42 | 1.000 |
| `ismcts:800` | +981 | 19 | 0.996 |
| `flatmc:600` | +835 | 17 | 0.992 |
| `greedy` | +581 | 17 | 0.966 |
| `pimc:8x1` | +547 | 17 | 0.959 |
| `random` | +0 | 0 | 0.500 |

200 games per pairing, seeds from 1, `split`. This is also the first ladder fitted since the
§4 mandatory-action ruling, so it supersedes the Phase 2 table in
[FINDINGS.md](FINDINGS.md) F2.1 rather than extending it —
[models/README.md](models/README.md) says why the two do not compare.

It did not converge, though — it was stopped. Three generations running failed the promotion
gate, which ends the run by design. But at 200 games and a 0.55 threshold, a generation
genuinely improving at 0.54 passes that gate only 39% of the time, so the run ended on a
measurement with no power to make the call rather than on a learner that had stopped
learning. Fixing the gate, raising the search the targets are generated at, and making the
network deeper are all ahead of buying GPU time; [PLAN.md](PLAN.md) has the ordering.

**What the agent actually taught us about the game is in [FINDINGS.md](FINDINGS.md)** — that
file is the point of the project, and the strategy results live there rather than here.

## Try it

A Rust toolchain is all you need to play. The trained agent ships with the repo —
[models/duel52-split-gen016.d52nn](models/duel52-split-gen016.d52nn), 3.6 MB, an ordinary
git blob with no LFS to install.

```bash
git clone https://github.com/ZGpup/Duel52.git && cd Duel52
cargo build --release

# Play the trained agent. `--encoding-slots 21` is not optional: it is what fixes the
# size of the observation, and the checkpoint refuses to load against any other value.
./target/release/duel52 play --encoding-slots 21 \
    --opponent netmcts:models/duel52-split-gen016.d52nn@4096
```

`netmcts:<checkpoint>@<sims>` is net-guided information-set MCTS — the policy head supplies
the prior, the value head stands in for rollouts. `@4096` is 64× the search the network was
trained at and still answers in well under a second, which is the right trade when a human is
the one waiting. Drop it to `@64` to see what the training loop itself was playing against.
`netpolicy:<checkpoint>` takes the policy head's argmax with no search at all: instant, and
much weaker. Both are agent names anywhere an agent is accepted, so the checkpoint also goes
straight into `match`, `ladder` and `probe`.

```bash
# The hand-written rungs need no checkpoint, and no --encoding-slots.
./target/release/duel52 play --seed 1                 # a random bot; --seed replays exactly
./target/release/duel52 play --opponent ismcts:2000   # the strongest rung on the ladder
./target/release/duel52 powers                        # what every card does
./target/release/duel52 demo --seed 47                # watch a whole game, ply by ply

# Measure instead of playing.
./target/release/duel52 match --a netmcts:models/duel52-split-gen016.d52nn@64 \
    --b ismcts:800 --games 200 --seed 1 --encoding-slots 21
./target/release/duel52 stats  --all                  # the Phase 1 numbers
./target/release/duel52 ladder --games 400            # the Phase 2 Elo table
./target/release/duel52 probe  --games 400            # how each agent actually plays
```

[models/README.md](models/README.md) records how that checkpoint was produced, what it
scores, and where it is weak. Training your own is a Python job — [CLAUDE.md](CLAUDE.md) has
the full command set.

Every prompt names the rule it is applying, so if the engine does something that looks
wrong you can point at exactly which ruling it thinks it is following. `--seed N` makes a
game exactly reproducible, so a rules complaint travels as a seed and a move number.

## The game, briefly

Three lanes, three actions per turn, every card has a power tied to its rank. Cards are
played face down and flipped to activate. You win a lane by clearing it once neither player
can play more cards, and you win the game by taking two lanes.

Two properties make it interesting as an AI problem:

1. Ten cards are removed unseen at setup, so uncertainty about hidden cards never fully
   resolves, even at the end of the game.
2. Lane wins require an empty draw pile and an empty opposing hand, so the entire draw
   phase is positional. Nothing is decided until the deck runs dry.

## Plan

**Phase 1: Engine.** ✅ Rust core with exact rules, PyO3 bindings, one test per ruling, and a
text CLI to play against. Ends with random vs random statistics.

**Phase 2: Baselines.** ✅ Random, greedy, flat Monte Carlo, PIMC, and ISMCTS with random
rollouts, on determinized worlds. Frozen as a permanent Elo ladder, plus instrumented
self-play for the first strategic measurements.

**Phase 3: Neural self-play.** ✅ AlphaZero style loop using information set MCTS: encoders,
network and inference path, then net-guided search, then self-play, replay, fitting and a
promotion gate around them. The first trained checkpoint is in [models/](models/). One item
from the original plan is still outstanding — the cross-check against exact CFR on a scaled
down variant, which would say how far from equilibrium a merely strong agent is.

**Phase 4: Extract the insight.** 🚧 Learned card values, opening frequencies, flip timing,
lane commitment, and first player advantage with error bars. Hand size and flip timing are
already in — instrumenting the trained agent answered them early, because it is the first
player here *capable* of the behaviour the hypotheses are about.

**Stretch:** R-NaD on real compute for an approximate Nash policy rather than a merely
strong one.

Everything runs locally on an M series Mac and scales to rented CUDA through config alone.

## Docs

| File | Contents |
| --- | --- |
| [game_rules.md](game_rules.md) | The spec. Disambiguated ruleset the engine implements. |
| [DESIGN.md](DESIGN.md) | Engine and model architecture. |
| [PLAN.md](PLAN.md) | Phased roadmap with status. |
| [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) | Unresolved rules and design questions. |
| [FINDINGS.md](FINDINGS.md) | Results, and hypotheses recorded before any data. |
| [models/README.md](models/README.md) | The shipped checkpoints: how each was trained, and what it scores. |
| [CLAUDE.md](CLAUDE.md) | Commands, repo layout, and the facts that are easy to get wrong. |

## Layout

```
engine/      the rules engine (zero dependencies) and the `duel52` CLI
  tests/     one named test per ruling, named for its rule section
bindings/    PyO3 wrapper, kept separate so the engine never depends on Python
py/duel52/   the Python package
configs/     variant configs: split (default), base, mirrored, split-raw-two
models/      trained checkpoints, tracked in git, with their provenance
```

Training output — `runs/` and `checkpoints/` — is deliberately not tracked. A run is
reproducible from its config and seed, and the one checkpoint worth keeping is copied into
`models/` by hand.

## A note on rules

`game_rules.md` is not a copy of the official rules. It is an engine ready version, with
every claim tagged as either published, resolved by a player, or inferred and pending
confirmation. It also specifies the red and black split deck variant common among regular
players, which is the default configuration here because symmetric material makes results
much cleaner to measure.

## License

GPL-3.0
