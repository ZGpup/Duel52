# Duel 52

An engine and self-play agent for [Duel 52](https://www.juddmadden.com/duel52/index.html),
the two-player combat card game by Judd Madden and Nina Riddell that uses a standard 52
card deck.

The goal is to answer a question nobody has published an answer to: what does optimal play
actually look like? As far as I can tell there is no existing engine, bot, or strategy
analysis for this game. The agent is the instrument. The insight is the deliverable.

## Status

**Phase 1 complete.** The engine plays the full game to spec, with 190 tests named after the
rule sections they check, PyO3 bindings, a text CLI, and baseline statistics over 1.2M
random games. No agent yet — that is Phase 2.

## Try it

```bash
cargo build --release
./target/release/duel52 play --seed 1     # play the engine in your terminal
./target/release/duel52 powers            # what every card does
./target/release/duel52 stats --all       # the Phase 1 numbers
```

Every prompt names the rule it is applying, so if the engine does something that looks
wrong you can point at exactly which ruling it thinks it is following. `--seed N` makes a
game exactly reproducible, so a rules complaint travels as a seed and a move number.

## What Phase 1 found

Details and provenance in [FINDINGS.md](FINDINGS.md). Random play characterises the game
*tree*, not strategy, so none of this speaks to how the game should be played:

- Games are short and tightly clustered: **45 plies**, median and mean, in every variant.
- **First-player advantage is real but small** — P0 scores 0.512, about +1.2 points.
- **The stalemate rule never fired once in 1.2M games**, because the stall the rules
  describe is strategic and random agents attack constantly. It remains untested.
- **The house rule for the 2 fixes a real artifact.** Rules-as-written, the 2 discards a
  card from a *shared* pile, which hands the first player half an extra draw per game and
  +0.9 points of score. Bottoming instead makes it exactly zero. The artifact turns out to
  be about pile *sharing*, so it does not exist in the split-deck variant at all.
- The mutual-lane-win draw the rules call "astronomically rare" happens in **1 game in
  220** at this level of play. `duel52 demo --seed 86` replays one.

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

**Phase 2: Baselines.** Random, greedy, flat Monte Carlo, PIMC, and ISMCTS with random
rollouts. Frozen as a permanent Elo ladder. This phase alone should reveal a lot.

**Phase 3: Neural self-play.** AlphaZero style loop using information set MCTS, validated
against exact CFR on a scaled down variant before being trusted on the full game.

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
| [CLAUDE.md](CLAUDE.md) | Commands, repo layout, and the facts that are easy to get wrong. |

## Layout

```
engine/      the rules engine (zero dependencies) and the `duel52` CLI
  tests/     one named test per ruling, named for its rule section
bindings/    PyO3 wrapper, kept separate so the engine never depends on Python
py/duel52/   the Python package
configs/     variant configs: split (default), base, mirrored, split-raw-two
```

## A note on rules

`game_rules.md` is not a copy of the official rules. It is an engine ready version, with
every claim tagged as either published, resolved by a player, or inferred and pending
confirmation. It also specifies the red and black split deck variant common among regular
players, which is the default configuration here because symmetric material makes results
much cleaner to measure.

## License

GPL-3.0
