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
§4 mandatory-action ruling, so it supersedes the Phase 2 table below rather than extending
it — see [models/README.md](models/README.md) for why the two do not compare.

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
  it said belief modelling in Phase 3 would buy something real. It did. This is also the one
  Phase 2 result that has been re-measured since the §4 ruling and come back *stronger*: flat
  Monte Carlo now beats `pimc:8x1` **0.835 ± 0.036** over 400 games, against F2's 0.689. PIMC
  has dropped to last of the five, level with `greedy` — 0.505 ± 0.049, where `FINDINGS.md`
  F2.3 has 0.638 ± 0.066 and the intervals do not overlap.
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
  fair test. **It has now had one, and H2 moved.** Same within-agent test, 1000 self-play
  games, with `greedy` as a control: the trained agent wins the games where it held more cards
  by **+1.25 ± 0.17** (~14σ) while `greedy` shows −0.04 ± 0.07, reproducing the old null
  exactly. A null from an agent that cannot do the thing was never a null about the game.
  Supported, not confirmed — the effect is correlational and the causal arrow is still open
  (F3.9).
- **Nor does concentrating on two lanes.** No rung puts a larger share of its cards into its
  busiest two lanes than a uniformly random player does.
- **Phase 1's encoding bound was backwards.** Competent play tops out at 8–12 cards on one
  side of a lane; it is *random* play that sprawls to 17–20. Random play is not a
  conservative upper bound on real play — it is a different distribution.

## What Phase 3 found

The first results that are about *how to play* rather than about the game tree. Provenance in
[FINDINGS.md](FINDINGS.md) F3.7–F3.10.

- **A null from an agent that cannot do the thing is not a null.** This is the methodological
  result, and it invalidates a chunk of Phase 2. H2 sat at "unsupported" for the whole of
  Phase 2 on a test that was sound — F3.9 reproduces its null exactly on `greedy` — but that
  was never evidence about the game. No Phase 2 agent could hoard on purpose. Every remaining
  Phase 2 null needs re-running against an agent capable of the behaviour, starting with H3.
- **The agent flips cards in an order keyed to what kind of power they have.** Constant powers
  first (the 8's Retaliate at ply 12.9, the Jack's Taunt at 15.7 — they do nothing face-down),
  one-shot powers last (the Queen's Move at 33.8 — flipping spends them). **The 3 is latest
  and least-flipped of all**, at 0.59–0.63 against ~0.95 for every other rank, because Trap
  only fires while the card is face-down. `ismcts:800` flips everything at ply ~20 regardless
  of rank: a 5-ply spread against the net's 21.
- **Search is not where the strength is.** The policy head alone, argmax with no tree at all,
  beats `greedy` 0.940. Search on top adds a further 0.027 against that opponent — though
  against *itself* search is worth a great deal, +145 Elo per 4× simulations with no
  saturation out to 1024.
- **PIMC's strategy fusion does not affect the net.** The same test that exposed it — does
  more sampling buy anything — gives PIMC nothing at 8× worlds and the net +145 Elo per 4×,
  twice over. Re-determinizing inside the simulation loop is doing real work.
- **The gate that stopped the first training run had no power to make that call.** At 200
  games and a 0.55 threshold, a generation genuinely improving at 0.54 passes 39% of the time.
  Worth stating because the failure is invisible: the loss curves look healthy throughout.

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

**Phase 4: Extract the insight.** 🚧 Started early and by accident — instrumenting the trained
agent answered three items straight away, because it is the first player in the project
*capable* of the behaviour the hypotheses are about. Hand size at pile-empty and flip timing
are in ([FINDINGS.md](FINDINGS.md) F3.9, F3.10); learned card values, opening frequencies and
lane commitment are open, along with the intervention that would turn the hand-size
correlation into a causal claim.

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
