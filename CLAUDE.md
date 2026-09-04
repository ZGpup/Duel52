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
  beyond a config value. That is the handoff path — but note what it hands off: 87% of the
  training loop is Rust self-play on CPU cores and 4% is gradient work, and the gradient step
  is only 1.4× faster on a GPU than on eight CPU cores (`FINDINGS.md` F3.11). `run.threads`
  matters more than `train.device`.

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
cargo test                               # 301 tests: rules, determinism, information hiding,
                                         # the Phase 3 encoding path, and the training corpus

# Play. Every prompt names the rule it is applying, so a disagreement is easy to point at.
./target/release/duel52 play --seed 1                      # you are P0 vs a random bot
./target/release/duel52 play --opponent ismcts:2000        # vs the strongest rung
./target/release/duel52 play --variant base --as p1        # rules-as-written, second player
./target/release/duel52 play --opponent human              # hotseat
./target/release/duel52 powers                             # card-power reference
./target/release/duel52 demo --seed 47                     # watch a random game, ply by ply

# Measure. `demo --seed N` replays exactly the game `stats` counted for seed N.
./target/release/duel52 stats --all --games 200000 --seed 1 --markdown
./target/release/duel52 config configs/split.toml          # validate a config file

# Phase 2 agents. Budgets are part of the agent name, so a result row names the agent that
# produced it: random · greedy · flatmc:600 · pimc:32x1 · ismcts:800.
./target/release/duel52 ladder --games 400 --markdown      # the frozen Elo table (~26 min)
./target/release/duel52 match --a ismcts:800 --b pimc:32x1 --games 400
./target/release/duel52 probe --games 300 --markdown       # self-play behaviour per rung

# Phase 3 step 1. A checkpoint is written in Python and played in Rust; the header's layout
# hashes are what stop the two sides drifting apart.
.venv/bin/python -m duel52.nn init --out checkpoints/init.d52nn
.venv/bin/python -m duel52.nn inspect checkpoints/init.d52nn      # header + compatibility
./target/release/duel52 match --a netpolicy:checkpoints/init.d52nn --b random --games 100
./target/release/duel52 nn-dump --checkpoint checkpoints/init.d52nn \
    --games 20 --seed 1 --out /tmp/parity.bin                     # feeds test_parity.py

# Phase 3 steps 2-3 — training. One TOML plus a seed describes a whole run; `check` validates
# it in five seconds, which is worth doing before a two-hour session.
.venv/bin/python -m duel52.train check --config configs/train-fast.toml
.venv/bin/python -m duel52.train run   --config configs/train-fast.toml --run-dir runs/first
.venv/bin/python -m duel52.train run   --config configs/train-fast.toml --run-dir runs/first \
    --resume                                                      # continue after a stop

# The pieces, runnable on their own when something looks wrong.
./target/release/duel52 selfplay --checkpoint runs/first/checkpoints/best.d52nn \
    --out /tmp/gen.d52sp --games 200 --sims 64 --encoding-slots 21
./target/release/duel52 shard /tmp/gen.d52sp                      # header + replay check
./target/release/duel52 match --a netmcts:runs/first/checkpoints/best.d52nn@64 \
    --b ismcts:800 --games 200 --encoding-slots 21

# Python. Needs a venv; `maturin develop` drops the extension into py/duel52/.
python3 -m venv .venv && .venv/bin/pip install -q maturin pytest torch numpy
.venv/bin/maturin develop --release
.venv/bin/python -m pytest py/tests -q
```

⚠️ **The stalemate draw is [ENGINE], and it is now a backstop rather than a strategy.**
`game_rules.md` §4 makes actions **mandatory** — there is no pass anywhere in the engine, and
the only short turn is the first player's opening one. A turn with nothing legal in it is
ended by `apply.rs`'s `skip_turns_with_nothing_to_do`, not chosen away, so `legal_actions()`
is empty only when the game is over. That removes the standoff at the root: a player who
would rather not attack must spend the action on a play, a flip or a pair, and all three
run out.
Greedy self-play went from 0.7–1.7% stalemates to **0 in 4,000 games per variant**
(`FINDINGS.md` F2.4b). `stalemate_value` is still a *learning* weight and training configs
still set `0.0`, but F3.6's collapse is no longer the failure mode to expect — the draws
that remain are mutual lane wins. More generally, anything marked **[ENGINE]** in
`game_rules.md` is a rule nobody agreed to, and deserves the question *what does an agent
get for exploiting this?*

⚠️ **The policy head is 1324 wide as of 2026-09-03, and every earlier checkpoint and shard is
refused.** Removing the `PASS` block (§4 has no pass — see below) shifted `CHOOSE_SLOT` and
`CHOOSE_RANK`, so the action-layout hash moved. A stale checkpoint fails loudly with
`action_dim is 1325 in the checkpoint but 1324 in this build`, which is the guard working.
Regenerate with `python -m duel52.nn init`; `runs/` from before that date cannot be resumed.
**A `.d52sp` shard is now checked the same way.** It stores indices into `legal_actions()`,
so an encoder change silently repoints every one of them — the header always carried the
layout hashes but nothing read them back until this change. `Shard::read` now compares both
and refuses a mismatch (`phase3_a_shard_from_a_different_action_layout_is_refused`).

⚠️ `encoding_slots` defaults to **16** and the encoder **asserts** rather than truncating.
A `netpolicy` checkpoint played against `random` can exceed it — see `FINDINGS.md` F3.1.
Add `--encoding-slots 21` to both the `init` and the `duel52` command if you hit it; the two
must match, because `encoding_slots` is what fixes `obs_dim`.

Configs live in `configs/`: `split.toml` (the default), `base.toml`, `mirrored.toml`, and
`split-raw-two.toml` (the control for the §10a house rule). `train-fast.toml` is a *training*
config rather than a game config — it carries the loop's knobs and sets
`encoding_slots = 21`, which every command in that run must agree on.

## Where things are

| Path | What |
|---|---|
| `engine/src/state.rs` | `GameState` and the queries the rules are written in terms of |
| `engine/src/apply.rs` | Powers, combat, turn machinery. The rules live here. |
| `engine/src/legal.rs` | Legal-action enumeration |
| `engine/src/config.rs` | Every tunable; the three variant presets |
| `engine/src/testkit.rs` | Building positions by hand, for tests and Phase 5 probes |
| `engine/src/display.rs` | Rendering a board and an action for one observer. The only place lanes and cards are numbered from 1, and the only definition of the order a lane's cards are drawn in (`column_slots`) |
| `engine/src/menu.rs` | Reshapes the flat legal-action list into the CLI's question tree — verb, then card, then lane only when the card is in more than one — with every verb and lane number fixed to the thing it picks |
| `engine/src/determinize.rs` | Sampling a world from an information set. Every search agent goes through it |
| `engine/src/encode.rs` | Observation and action tensors, and the layout hashes that pin them |
| `engine/src/nn/` | Weights, the `.d52nn` checkpoint format, and the reference forward pass |
| `engine/src/agents/` | The five ladder rungs plus `netpolicy` and `netmcts`, and the evaluation in `eval.rs` |
| `engine/src/selfplay.rs` | Self-play generation and the `.d52sp` trajectory shard |
| `engine/src/ladder.rs`, `elo.rs` | Round robin, and the Bradley–Terry rating fit |
| `engine/src/probe.rs` | Instrumented play — where the Phase 2 findings come from |
| `engine/tests/` | One named test per ruling, named for its rule section |
| `bindings/src/lib.rs` | PyO3 wrapper; `Game.observation()` is the filtered per-player view |
| `py/duel52/nn/` | The PyTorch model and checkpoint I/O. **Never an encoder** — see below |
| `py/duel52/train/` | The AZ loop: replay buffer, trainer, generation driver. Gradients only |

Three structural points that are easy to undo by accident:

- **An agent must decide from a determinized world, not from the state it is handed.**
  `Agent::choose` receives engine-side ground truth because the engine is the authority on
  legality, so nothing structural stops an agent reading the opponent's hand. The guard is
  `phase2_no_agent_reads_hidden_information`: a sampled world is in the same information set
  as the real one, so an honest agent must return the same action from either. This is not
  only about search — it caught the *greedy* agent, because applying a candidate action to
  the real state reveals ranks (flipping your own base card, killing a face-down card into
  the public discard). If it fails, the agent is cheating, not the test.
  **Adding an agent does not enrol it automatically** — the test iterates the hardcoded
  `TEST_ROSTER` in `engine/tests/agents.rs`, so a new rung has to be added there by hand.
  The Phase 3 encoder has the same obligation and its own version of the test:
  `phase3_observation_is_a_function_of_the_information_set`, which asserts the observation
  tensor is bit-identical between a state and a determinized world.

- **Sub-decisions are separate zero-cost decision nodes on a stack** (`DESIGN.md` §4). A 5
  that flips a King that re-empowers the lane resolves correctly because of this. Collapsing
  them into one big action would blow up the branching factor and break §8's adaptive
  ordering.
- **Cards are tracked by `CardId`, never by slot.** Slots compact on death and shift when a
  Queen moves a card, so anything remembered across a resolution step holds ids.

- **There is exactly one encoder, and it is in Rust.** `engine/src/encode.rs` owns the
  feature layout; Python reaches it through `Game.encode_observation()` and gets its
  dimensions and layout hashes from `duel52.encoding_spec()`. A second copy of the layout in
  Python would let the trained function and the evaluated function drift apart *silently* —
  nothing crashes, the agent is merely bad, and the natural suspect is the training run. The
  checkpoint header carries both layout hashes and `Weights::load` refuses a mismatch, which
  turns that into a one-line error. Never compute a layout hash outside `encode.rs`.
