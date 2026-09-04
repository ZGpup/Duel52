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
[models/duel52-split-gen016.d52nn](models/duel52-split-gen016.d52nn). At 64 simulations —
one twelfth of its opponent's budget — it scores **0.930 ± 0.035 against `ismcts:800`**, the
strongest hand-written rung, over 200 games.

It did not converge, though; it stalled. Three generations running failed to beat gen016
while the policy loss kept falling, which is the shape of a search too small to keep
producing targets the network cannot already fit. More simulations and a longer run are the
obvious next lever, and that is a rented-GPU job rather than a laptop one.

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
    --opponent netmcts:models/duel52-split-gen016.d52nn@256
```

`netmcts:<checkpoint>@<sims>` is net-guided information-set MCTS — the policy head supplies
the prior, the value head stands in for rollouts. Raise `@256` for a slower and stronger
opponent, or lower it to `@64`, the budget it was trained and gated at.
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

## What Phase 1 found

Details and provenance in [FINDINGS.md](FINDINGS.md). Random play characterises the game
*tree*, not strategy, so none of this speaks to how the game should be played:

- Games are short and tightly clustered: **45 plies**, median and mean, in every variant.
- **First-player advantage is real but small** — P0 scores 0.514, about +1.4 points.
- **The stalemate rule never fired once in 1.2M games**, because the stall the rules
  describe is strategic and random agents attack constantly. Phase 2 found the agent that
  could produce it; the §4 mandatory-action ruling then made it unreachable — see below.
- **The house rule for the 2 fixes a real artifact.** Rules-as-written, the 2 discards a
  card from a *shared* pile, which hands the first player half an extra draw per game and
  +0.8 points of score. Bottoming instead makes it exactly zero. The artifact turns out to
  be about pile *sharing*, so it does not exist in the split-deck variant at all.
- The mutual-lane-win draw the rules call "astronomically rare" happens in **1 game in
  220** at this level of play. `duel52 demo --seed 47` replays one. Since the §4
  mandatory-action ruling it is the *only* way the game draws at all.

## What Phase 2 found

Now with agents that actually try. Full provenance in [FINDINGS.md](FINDINGS.md) F2.

- **The first-player advantage disappears.** Phase 1's +1.4 points was an artifact of random
  play. Once both sides can defend, P0 scores 0.492–0.510 across all three variants over
  4,000 games each — every interval covers even.
- **PIMC is beaten by an agent with no evaluation function and no tree.** Flat Monte Carlo
  scores 0.689 against a compute-matched PIMC. And for PIMC, eight-fold more sampled worlds
  buys nothing measurable while one extra ply of search is worth +0.28 — the signature of
  strategy fusion, which is bias rather than variance. This is the phase's main result, and
  it says belief modelling in Phase 3 is buying something real.
- **The stalemate rule fired, and then stopped existing.** Greedy — the only rung that
  prices material, so the only one that can decline a trade — stalled in 0.7–1.7% of its
  self-play games, most often in the mirrored variant where symmetric decks make neither
  side want to attack first. Chasing that led to the actual defect: the engine had been
  letting players **pass**, which is not a rule and never was. With §4's three actions made
  mandatory, greedy's stalemate rate is **0 in 4,000 games per variant**, and the only draw
  left in Duel 52 is the mutual lane win. Every Phase 2 number above predates that ruling.
- **Hoarding cards does not appear to win games** — *at this level of play*. Within a single
  agent's self-play, the side holding more cards when the draw pile empties wins no more
  often, at any rung. That is the project's headline hypothesis, and Phase 2 could not
  support it, though no Phase 2 agent is capable of hoarding deliberately, so it was never a
  fair test. **The trained agent hoards.** It reaches the endgame holding 7.98 cards against
  `ismcts:800`'s 1.00, and its own wins are the games where it held more. That is the first
  evidence in the hypothesis's favour and it is not yet a finding — see
  [models/README.md](models/README.md).
- **Nor does concentrating on two lanes.** No rung puts a larger share of its cards into its
  busiest two lanes than a uniformly random player does.
- **Phase 1's encoding bound was backwards.** Competent play tops out at 8–12 cards on one
  side of a lane; it is *random* play that sprawls to 17–20. Random play is not a
  conservative upper bound on real play — it is a different distribution.

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

**Phase 4: Extract the insight.** Learned card values, opening frequencies, flip timing,
lane commitment, and first player advantage with error bars.

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
