# Duel 52 — Project Instructions

Goal: build a rules-exact engine for the card game **Duel 52** and train a strong
self-play agent, in order to answer a question nobody has published an answer to —
**what does optimal play actually look like?** The insight is the deliverable; the bot is
the instrument.

## Documents

| File | Purpose |
|---|---|
| `game_rules.md` | **The spec.** Canonical, disambiguated ruleset. The engine implements this. |
| `DESIGN.md` | Engine + model architecture: state, action encoding, observations, training. |
| `PLAN.md` | Phased roadmap with status. Update as phases complete. |
| `OPEN_QUESTIONS.md` | Unresolved rules and design questions. Resolve → move the ruling into `game_rules.md` and delete the entry. |
| `FINDINGS.md` | Strategy insights as they emerge. This is the actual output of the project. |

## Facts that are easy to get wrong

Read `game_rules.md` before touching engine code. These five trip people up:

1. **Base cards are hidden from their owner too**, not just the opponent. That is why the
   4's Foresight can target your own base cards.
2. **Lane wins are endgame-only.** A lane cannot be won until the draw pile *and* the
   opponent's hand are both empty. The whole draw phase is positioning.
3. **10 cards are removed unseen at setup.** Belief over hidden cards never fully resolves,
   even at the end. Do not abstract this away.
4. **Suits are mechanically irrelevant** — collapse to rank everywhere. (Color denotes deck
   ownership in the split-deck variant; suit still never matters.)
5. **The split-deck (red/black) variant is the default configuration**, not the
   rules-as-written game. See `game_rules.md` §9.

## Conventions

- **The Rust engine is the sole authority on legality.** Never reimplement rules logic in
  Python — call the engine. Python does training and analysis only.
- **Every ruling in `game_rules.md` gets a named test.** Test names reference the rule
  section, e.g. `rule_6_king_reactivates_ace_grants_one_action`.
- **Everything is seeded and deterministic.** Same seed + same config → identical game.
  Non-reproducible results are bugs.
- **Config-driven, no hardcoded constants.** Variant selection, deck composition, removal
  count, draw rules, and stalemate threshold all live in config.
- **Device-agnostic.** Code must run on MPS locally and CUDA on a rented box with no edits
  beyond a config value. That is the handoff path.

## Working agreements

- When a rules question comes up, check `OPEN_QUESTIONS.md` first. If it's not there and
  the online implementation at <https://www.juddmadden.com/duel52/play.html> can settle it,
  settle it there rather than interrupting the owner. Escalate only what testing can't answer.
- Owner has limited time on this project and is delegating implementation. Prefer making a
  defensible call, documenting it as `[ASSUMED]`, and flagging it — over blocking.
- Log measured results in `FINDINGS.md` with the config and seed range that produced them.
  An unreproducible finding is not a finding.

## Commands

```bash
# Build. The Cargo workspace root is the repo root; `cargo` alone works on the engine only,
# so the everyday loop does not pay for compiling PyO3.
cargo build --release                    # engine + the `duel52` CLI
cargo test                               # 228 tests: rules, determinism, information hiding

# Play. Every prompt names the rule it is applying, so a disagreement is easy to point at.
./target/release/duel52 play --seed 1                      # you are P0 vs a random bot
./target/release/duel52 play --opponent ismcts:2000        # vs the strongest rung
./target/release/duel52 play --variant base --as p1        # rules-as-written, second player
./target/release/duel52 play --opponent human              # hotseat
./target/release/duel52 powers                             # card-power reference
./target/release/duel52 demo --seed 86                     # watch a random game, ply by ply

# Measure. `demo --seed N` replays exactly the game `stats` counted for seed N.
./target/release/duel52 stats --all --games 200000 --seed 1 --markdown
./target/release/duel52 config configs/split.toml          # validate a config file

# Phase 2 agents. Budgets are part of the agent name, so a result row names the agent that
# produced it: random · greedy · flatmc:600 · pimc:32x1 · ismcts:800.
./target/release/duel52 ladder --games 400 --markdown      # the frozen Elo table (~26 min)
./target/release/duel52 match --a ismcts:800 --b pimc:32x1 --games 400
./target/release/duel52 probe --games 300 --markdown       # self-play behaviour per rung

# Python. Needs a venv; `maturin develop` drops the extension into py/duel52/.
python3 -m venv .venv && .venv/bin/pip install -q maturin pytest
.venv/bin/maturin develop --release
.venv/bin/python -m pytest py/tests -q
```

Configs live in `configs/`: `split.toml` (the default), `base.toml`, `mirrored.toml`, and
`split-raw-two.toml` (the control for the §10a house rule).

## Where things are

| Path | What |
|---|---|
| `engine/src/state.rs` | `GameState` and the queries the rules are written in terms of |
| `engine/src/apply.rs` | Powers, combat, turn machinery. The rules live here. |
| `engine/src/legal.rs` | Legal-action enumeration |
| `engine/src/config.rs` | Every tunable; the three variant presets |
| `engine/src/testkit.rs` | Building positions by hand, for tests and Phase 4 probes |
| `engine/src/determinize.rs` | Sampling a world from an information set. Every search agent goes through it |
| `engine/src/agents/` | The five ladder rungs, and the hand-written evaluation in `eval.rs` |
| `engine/src/ladder.rs`, `elo.rs` | Round robin, and the Bradley–Terry rating fit |
| `engine/src/probe.rs` | Instrumented play — where the Phase 2 findings come from |
| `engine/tests/` | One named test per ruling, named for its rule section |
| `bindings/src/lib.rs` | PyO3 wrapper; `Game.observation()` is the filtered per-player view |

Three structural points that are easy to undo by accident:

- **An agent must decide from a determinized world, not from the state it is handed.**
  `Agent::choose` receives engine-side ground truth because the engine is the authority on
  legality, so nothing structural stops an agent reading the opponent's hand. The guard is
  `phase2_no_agent_reads_hidden_information`: a sampled world is in the same information set
  as the real one, so an honest agent must return the same action from either. This is not
  only about search — it caught the *greedy* agent, because applying a candidate action to
  the real state reveals ranks (flipping your own base card, killing a face-down card into
  the public discard). If you add an agent, that test covers it automatically; if it fails,
  the agent is cheating, not the test.

- **Sub-decisions are separate zero-cost decision nodes on a stack** (`DESIGN.md` §4). A 5
  that flips a King that re-empowers the lane resolves correctly because of this. Collapsing
  them into one big action would blow up the branching factor and break §8's adaptive
  ordering.
- **Cards are tracked by `CardId`, never by slot.** Slots compact on death and shift when a
  Queen moves a card, so anything remembered across a resolution step holds ids.
