# Duel 52

An engine and self-play agent for [Duel 52](https://www.juddmadden.com/duel52/index.html),
the two-player combat card game by Judd Madden and Nina Riddell that uses a standard 52
card deck.

The goal is to answer a question nobody has published an answer to: what does optimal play
actually look like? As far as I can tell there is no existing engine, bot, or strategy
analysis for this game.

## Try it

A Rust toolchain is all you need to play. The engine has zero
dependencies, so the build resolves nothing and takes about ten seconds. The trained agent
ships with the repo — [models/duel52-split-gen031.d52nn](models/duel52-split-gen031.d52nn),
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
    --opponent netmcts:models/duel52-split-gen031.d52nn@4096
```

[models/README.md](models/README.md) records how the checkpoints were produced and what they
score; [CLAUDE.md](CLAUDE.md) has the full command set, including recording a game and
replaying it to see what the net thought at each of your decisions.

## Status

**There is a trained agent in the repo, and you can play it.** The engine plays the full game
to spec, with 330 Rust tests named after the rule sections they check, 94 Python tests, PyO3
bindings, and a text CLI. On top of it sits a frozen five-rung Elo ladder — random, greedy,
flat Monte Carlo, PIMC and SO-ISMCTS — built on determinization, so every search agent
reasons from its own information set rather than from the engine's ground truth.

The AlphaZero loop now runs end to end. There is exactly one encoder and it lives in Rust; a
network is defined and trained in PyTorch, evaluated in Rust, and a test asserts the two
forward passes compute the same function. Self-play writes trajectory shards, the trainer
replays and fits them, and a gate promotes a candidate only when it beats the incumbent over
200 games.

Three runs have gone through it, each on the same laptop and each changing one thing.

1. **[gen016](models/duel52-split-gen016.d52nn)** — 57,000 self-play games in 1.94 hours, then
   stopped, not because it had converged but because a 200-game promotion gate at a 0.55
   threshold passes a genuinely-improving candidate only 39% of the time.
2. **[gen022](models/duel52-split-gen022.d52nn)** — fixed that gate and, more importantly,
   **uncapped the teacher**: self-play generates its policy targets at 256 simulations instead
   of 64. The policy target *is* the visit distribution, so training at 64 had been teaching
   the network to imitate a search hundreds of Elo weaker than the same weights already
   produced.
3. **[gen031](models/duel52-split-gen031.d52nn)**, the current default — showed every training
   sample under a **random relabelling of the three lanes**. Duel 52 is invariant under all six
   permutations of its lanes: no rule names a lane, orders them, or tells one from another. So
   this is five extra exactly-correct views of every position for 0.29 ms a batch, and it
   fixes a measured defect — the previous net had learned an arbitrary preference for lane 3.

Each step is measured against the checkpoint it was trained from, **at equal simulations**,
over 400 games:

| | score (95% CI) | W–L–D | Elo |
| --- | --- | --- | ---: |
| `netmcts:gen031@256` vs `netmcts:gen022@256` | **0.6162 ± 0.0475** | 245–152–3 | **+82** |
| `netmcts:gen022@256` vs `netmcts:gen016@256` | **0.6150 ± 0.0474** | 244–152–4 | **+81** |
| `netmcts:gen031@256` vs `netmcts:gen016@256` | **0.7475 ± 0.0426** | 299–101–0 | **+189** |

The third row is the first two composed and it checks out: +81 and +82 predict +163, and the
direct measurement's interval runs +151 to +230.

**The lane symmetry moved too, which is the mechanism rather than the score.** On 128
positions that are exact relabellings of one another, the share of the opening prior going to
each lane went .320 / .277 / **.403** to **.328 / .331 / .341**, and the number of them where
the net picks the same next action regardless of which of three identical lanes it opened into
went **82/128 to 114/128**. It is not simply a flatter policy — gen031 is marginally the
sharper net. [FINDINGS.md](FINDINGS.md) F4.5 has the rest.

**The ladder has stopped being useful and was not re-run.** gen022 already beat `greedy` and
`pimc:8x1` at essentially 1.000; gen031 beats `ismcts:800`, the top hand-written rung,
**200–0**. A rung that loses every game measures nothing about the winner. The last full fit,
400 games a pairing, put gen022 at +1788 ± 58 against ±13–15 for every hand-written rung —
a rating driven by a handful of losses is an extrapolation, and [PLAN.md](PLAN.md) §4.7
retired the table there. ⚠️ Note also that Elo is **not comparable across two fits**:
Bradley–Terry pins `random` at 0 and fits the rest to the whole graph, so pulling the top
agent away stretches every rung beneath it. `FINDINGS.md` F4.2 does that arithmetic, and it is
the second time the trap has caught this project.

Every number above is scored against agents written for this project. The one external check
that exists says something different: **the project owner beat gen016 five games out of
five**, and no series has been played against either of the two agents since. That is the
measurement Phase 4 turns on, and `duel52 play --record` now exists so that the next one is
written down.

**What the agent actually taught us about the game is in [FINDINGS.md](FINDINGS.md)** — that
file is the point of the project, and the strategy results live there rather than here.

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
and two further laptop runs have banked **+189 Elo at equal simulations** over the first
agent — +81 from uncapping the teacher, +82 from lane-permutation augmentation. What is left
is the part that needs rented cores: one long from-scratch run at a deeper trunk, which a warm
start cannot do because it cannot change the shape of the network it inherits. The exit
criterion is not an Elo number — the ladder is anchored at `random`, every rung was written
here, and it has stopped resolving anything at the top. It is that the agent takes a game off
the project owner, who beat the first one 5–0.

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
| [REPLAY.md](REPLAY.md) | How to read `duel52 replay` — the value, prior and second-opinion columns, and the board. |
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
