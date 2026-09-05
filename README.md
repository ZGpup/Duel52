# Duel 52

An engine and self-play agent for [Duel 52](https://www.juddmadden.com/duel52/index.html),
the two-player combat card game by Judd Madden and Nina Riddell that uses a standard 52
card deck.

The goal is to answer a question nobody has published an answer to: what does optimal play
actually look like? As far as I can tell there is no existing engine, bot, or strategy
analysis for this game. The agent is the instrument. The insight is the deliverable.

## Status

**There is a trained agent in the repo, and you can play it.** The engine plays the full game
to spec, with 326 Rust tests named after the rule sections they check, 88 Python tests, PyO3
bindings, and a text CLI. On top of it sits a frozen five-rung Elo ladder — random, greedy,
flat Monte Carlo, PIMC and SO-ISMCTS — built on determinization, so every search agent
reasons from its own information set rather than from the engine's ground truth.

The AlphaZero loop now runs end to end. There is exactly one encoder and it lives in Rust; a
network is defined and trained in PyTorch, evaluated in Rust, and a test asserts the two
forward passes compute the same function. Self-play writes trajectory shards, the trainer
replays and fits them, and a gate promotes a candidate only when it beats the incumbent over
200 games.

Two runs have gone through it. The first produced
[gen016](models/duel52-split-gen016.d52nn) — 57,000 self-play games in 1.94 hours — and then
stopped, not because it had converged but because a 200-game promotion gate at a 0.55
threshold passes a genuinely-improving candidate only 39% of the time. The second fixed that
gate and, more importantly, **uncapped the teacher**: self-play now generates its policy
targets at 256 simulations instead of 64. The policy target *is* the visit distribution, so
training at 64 had been teaching the network to imitate a search hundreds of Elo weaker than
the same weights already produced.

The result is [models/duel52-split-gen022.d52nn](models/duel52-split-gen022.d52nn), the
checkpoint that ships as the default. Against the net it was trained from, **at equal
simulations**, over 400 games:

| | score (95% CI) | W–L–D | Elo |
| --- | --- | --- | ---: |
| `netmcts:gen022@256` vs `netmcts:gen016@256` | **0.6150 ± 0.0474** | 244–152–4 | **+81** |

and on the frozen ladder, 400 games per pairing, seeds from 1, `split`:

| agent | Elo | ± | vs. anchor |
|---|---:|---:|---:|
| **`netmcts:gen022@256`** | **+1788** | 58 | 1.000 |
| `ismcts:800` | +1052 | 15 | 0.998 |
| `flatmc:600` | +900 | 13 | 0.994 |
| `greedy` | +615 | 13 | 0.972 |
| `pimc:8x1` | +584 | 13 | 0.967 |
| `random` | +0 | 0 | 0.500 |

The head-to-head is the number to trust. **The ladder's is not a comparable +312 over
gen016's earlier +1476**: Bradley–Terry pins `random` at 0 and fits everything else to the
whole graph, so pulling the top agent away stretches every rung beneath it — all four
hand-written rungs moved up between the two tables without a line of code changing. Measured
over the rungs common to both fits the gap is +241 to +247, of which roughly three fifths is
the larger search budget rather than better weights.
[FINDINGS.md](FINDINGS.md) F4.2 does the reconciliation, and it is the second time this
particular trap has caught this project.

**The ladder has also stopped being useful.** Both nets beat `greedy` and `pimc:8x1` at
essentially 1.000, so those pairings carry no information about either of them and only the
top rung still measures anything. Every number above is scored against agents written for
this project, on a scale anchored at `random`. The one external check that exists says
something different: **the project owner beat gen016 five games out of five.** No series
against gen022 has been played yet. That is the measurement Phase 4 turns on, and
`duel52 play --record` now exists so that the next one is written down.

**What the agent actually taught us about the game is in [FINDINGS.md](FINDINGS.md)** — that
file is the point of the project, and the strategy results live there rather than here.

## Try it

A Rust toolchain is all you need to play — anything from 1.75 on. The engine has zero
dependencies, so the build resolves nothing and takes about ten seconds. The trained agent
ships with the repo — [models/duel52-split-gen022.d52nn](models/duel52-split-gen022.d52nn),
3.6 MB, an ordinary git blob with no LFS to install.

```bash
# No Rust yet? This is the whole install. On Windows, run the rustup-init.exe from
# https://rustup.rs instead. Then restart the shell, or `source "$HOME/.cargo/env"`,
# so that `cargo` is on PATH.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

git clone https://github.com/ZGpup/Duel52.git && cd Duel52
cargo build --release

# Play the trained agent. `--encoding-slots 21` is not optional: it is what fixes the
# size of the observation, and the checkpoint refuses to load against any other value.
./target/release/duel52 play --encoding-slots 21 \
    --opponent netmcts:models/duel52-split-gen022.d52nn@4096
```

`netmcts:<checkpoint>@<sims>` is net-guided information-set MCTS — the policy head supplies
the prior, the value head stands in for rollouts. `@4096` is 64× the search the network was
trained at and still answers in well under a second, which is the right trade when a human is
the one waiting. Drop it to `@64` to see what the training loop itself was playing against.
`netpolicy:<checkpoint>` takes the policy head's argmax with no search at all: instant, and
much weaker. Both are agent names anywhere an agent is accepted, so the checkpoint also goes
straight into `match`, `ladder` and `probe`.

```bash
# These need no checkpoint, and no --encoding-slots.
./target/release/duel52 powers             # what every card does
./target/release/duel52 demo --seed 47     # watch a whole game, ply by ply
./target/release/duel52 play --seed 1      # a random bot, if @4096 is too much
```

Beat it, and you have a result worth keeping — so keep it. `--record` appends the finished
game to a JSONL file, and because a game is `(config, seed, chosen indices)` against a
deterministic engine, a 153-ply game is 918 bytes that replay it exactly, hidden information
included. `replay` then walks it back and shows what the net thought at each of your
decisions: the value head's score, your move's share of its policy prior, and what a deeper
search would rather have played.

```bash
./target/release/duel52 play --encoding-slots 21 --seed 101 \
    --record games/me-vs-gen022.jsonl \
    --opponent netmcts:models/duel52-split-gen022.d52nn@4096

./target/release/duel52 replay --record games/me-vs-gen022.jsonl --game 1
```

[models/duel52-split-gen016.d52nn](models/duel52-split-gen016.d52nn) also ships, and is the
previous agent rather than a second option — it is kept because every Phase 3 finding is
measured on it and because it is the fixed opponent the newer net is scored against.
[models/README.md](models/README.md) records how each was produced, what each scores, and
where each is weak — including the command that reproduces the ladder above. Training your
own is a Python job; [CLAUDE.md](CLAUDE.md) has the full command set.

Every prompt names the rule it is applying, so if the engine does something that looks
wrong you can point at exactly which ruling it thinks it is following. `--seed N` makes a
game exactly reproducible, so a rules complaint travels as a seed and a move number.

Type a number and look up before you press Enter: that line, and the cards it names, turn
red on the board. It is how you tell three identical `(? ²♥)` in an enemy lane apart —
type 1, 2, 3 and watch which one lights up. Nothing is committed until Enter. (`--no-clear`
turns the redraw off, and the highlight with it.)

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
promotion gate around them. The first trained checkpoint is in [models/](models/) — two hours
on a laptop, and +495 Elo clear of the hand-written ladder.

**Phase 4: Scale up.** 🚧 The gate is fixed, the teacher is uncapped, the games are recorded,
and a second laptop run has banked **+81 Elo at equal simulations** over the first agent. What
is left is the part that needs rented cores: one long from-scratch run at a deeper trunk,
which a warm start cannot do because it cannot change the shape of the network it inherits.
The exit criterion is not an Elo number — the ladder is anchored at `random`, every rung was
written here, and it has stopped resolving anything at the top. It is that the agent takes a
game off the project owner, who beat the first one 5–0.

**Phase 5: Extract the insight.** Learned card values, opening frequencies, flip timing,
lane commitment, and first player advantage with error bars. Hand size and flip timing are
already in — instrumenting the trained agent answered them early, because it is the first
player here *capable* of the behaviour the hypotheses are about.

**Phase 6: Verification.** How strong is it *really* — local best-response as an
exploitability proxy, and a cross-check against exact CFR on a scaled down variant small
enough to solve.

**Phase 7:** R-NaD for an approximate Nash policy rather than a merely strong one, if and only
if Phase 4 trips one of the two tripwires in [PLAN.md](PLAN.md) §4.8.

Everything runs locally on an M series Mac and scales to rented hardware through config alone.
Note which hardware: 87% of the training loop is Rust self-play on CPU cores and 4% is
gradient work, so the thing to rent is cores. A GPU changes about 1.5% of it.

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
