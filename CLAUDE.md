# Duel 52 — Project Instructions

Goal: build a rules-exact engine for the card game **Duel 52** and train a strong
self-play agent, in order to answer a question nobody has published an answer to —
**what does optimal play actually look like?** The insight is the deliverable; the bot is
the instrument.

## Documents

| File | Purpose |
|---|---|
| `game_rules.md` | **The spec.** Canonical, disambiguated ruleset. The engine implements this. |
| `PLAN.md` | What is done, and in detail what is next and why. Update as items close. |
| `FINDINGS.md` | Strategy insights as they emerge. This is the actual output of the project. |
| `README.md` | The public front door, and where `duel52 replay` is documented. |
| `RENTING.md` | How to rent a box and run `PLAN.md` item 7 on it, written for someone who has never rented one. Provider choice, the two ways to lose the run, and the Stage 1 measurements. |
| `MODULAR_RULES.md` | **The rules-mod system.** Where rules live, the three tiers a change falls into, and what each costs. Read §2 before pricing any rule change and §6 before trusting any number. |
| `configs/rules/README.md` | The ruleset registry. How to add one, and the four things to run before it earns a training run. |
| `CLAUDE.md` | This file. Commands, architecture, and the traps. |
| `archive/` | The superseded working docs, frozen 2026-09-05 and not maintained. |

⚠️ **`DESIGN.md`, `OPEN_QUESTIONS.md` and `REPLAY.md` were archived on 2026-09-05.** Roughly 90
source comments still cite `DESIGN.md §N`; those section numbers are unchanged and refer to
`archive/DESIGN.md`. The parts that are still load-bearing were moved into the Architecture
section below, so read that first and only open the archive for the rationale behind a
decision. `OPEN_QUESTIONS.md` was archived because it closed: every rules question raised for
this project has been answered and ported into `game_rules.md`.

## Facts that are easy to get wrong

Read `game_rules.md` before touching engine code. These six trip people up:

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
6. **Node, turn and round are three different counters.** A **node** is one decision offered
   to one player (`replay`'s `node` column, `--node N`); a **turn** is one player's turn —
   3 actions, 2 on the opening turn, 4 after an Ace (`turn N` on every board, `GameState::ply`);
   a **round** is both players' turns and nothing counts it. Game 2 of the corpus is 172 nodes
   = 51 turns, so the node column climbs about 3.4 per turn. `ply` survives *only* as the
   spec's synonym for a turn (`game_rules.md` §7, `stalemate_quiet_plies`, `max_plies`, and
   every `FINDINGS.md` length), because those config keys are written verbatim into each game
   record and renaming them would stop old games replaying. No user-facing line of `replay` or
   the board says "ply". `README.md` has the reader-facing version.

## Conventions

- **The Rust engine is the sole authority on legality.** Never reimplement rules logic in
  Python — call the engine. Python does training and analysis only.
- **Every ruling in `game_rules.md` gets a named test.** Test names reference the rule
  section, e.g. `rule_6_king_reactivates_ace_grants_one_action`.
- **Everything is seeded and deterministic.** Same seed + same config → identical game.
  Non-reproducible results are bugs.
- **Config-driven, no hardcoded constants.** Variant selection, deck composition, removal
  count, draw rules, stalemate threshold, **every card's power and every combat number** all
  live in config. No production code names a rank to decide what it does — it reads
  `config.power(rank)` and asks the `PowerId`. See "Rules mods" below.
- **Device-agnostic.** Code must run on MPS locally and CUDA on a rented box with no edits
  beyond a config value. That is the handoff path — but note what it hands off: **the gradient
  step is 2–4% of the loop** and is only 1.4× faster on a GPU than on eight CPU cores
  (`FINDINGS.md` F3.11), against 53–91% spent on Rust self-play and the gate. `run.threads`
  matters more than `train.device`, and the ranges are measured — see "Training loop" below
  for why they are ranges and not the single 87% this file used to quote.

## Working agreements

- When a rules question comes up, check `game_rules.md` first: several rulings are stated once
  in a general form (resolution ordering, mandatory powers, fizzling) rather than repeated per
  card, so the answer is often already there. If it is not, and the online implementation at
  <https://www.juddmadden.com/duel52/play.html> can settle it, settle it there rather than
  interrupting the owner. Escalate only what testing can't answer, and record the ruling in
  `game_rules.md` with its marker. `archive/OPEN_QUESTIONS.md` holds the rulings that reversed
  an earlier answer, which is worth reading before re-deriving a superseded one.
- Owner has limited time on this project and is delegating implementation. Prefer making a
  defensible call, documenting it as `[ASSUMED]`, and flagging it — over blocking.
- Log measured results in `FINDINGS.md` with the config and seed range that produced them.
  An unreproducible finding is not a finding.

## Commands

```bash
# Build. The Cargo workspace root is the repo root; `cargo` alone works on the engine only,
# so the everyday loop does not pay for compiling PyO3.
cargo build --release                    # engine + the `duel52` CLI
cargo test                               # 388 tests: rules, determinism, information hiding,
                                         # the Phase 3 encoding path, the lane symmetry, the
                                         # training corpus, the modded power variants, and
                                         # the cross-ruleset invariant suite

# Play. Every prompt names the rule it is applying, so a disagreement is easy to point at.
./target/release/duel52 play --seed 1                      # you are P0 vs a random bot
./target/release/duel52 play --encoding-slots 21 \
    --opponent netmcts:models/duel52-split-lane-gen032.d52nn@4096  # vs the strongest agent
./target/release/duel52 play --opponent ismcts:2000        # vs the strongest hand-written rung
./target/release/duel52 play --variant base --as p1        # rules-as-written, second player
./target/release/duel52 play --opponent human              # hotseat
./target/release/duel52 powers                             # card-power reference
./target/release/duel52 demo --seed 47                     # watch a random game, action by action

# Ask the agent you are playing what it would do, before you decide. `--hint N` lists the N
# moves it would consider (default 3), best first, with each one's share of its search and
# what it makes of your position afterwards. `--hint-agent <agent>` asks somebody else, and
# implies `--hint`; it is how a hotseat game or a game against `random` gets advice, and how
# you ask a bigger budget than the one you are playing.
#
# Two things it deliberately is not. It is not a peek: the search determinizes from YOUR
# information set, so it cannot see the opponent's hand or your own base card. And it is not
# part of the game: it runs on its own RNG stream, so the same seed plays the identical game
# with hints on or off — verified by recording the same driven game both ways. A hinted game
# is flagged in `--record` and warned about in `replay`, because the "second opinion" column
# there stops being evidence about the human once the answer was on screen while they chose.
./target/release/duel52 play --encoding-slots 21 --hint \
    --opponent netmcts:models/duel52-split-lane-gen032.d52nn@4096

# Record what you played, then ask the net about it (PLAN.md §4.0). A game is
# (config, seed, chosen indices), so a 153-node game is 918 bytes and replays exactly —
# hidden information included. Only finished games are written. README.md's "Recording a game
# and replaying it" reads the output; get the node / turn / round distinction straight first.
./target/release/duel52 play --encoding-slots 21 --seed 101 \
    --record games/owner-vs-gen031.jsonl \
    --opponent netmcts:models/duel52-split-lane-gen032.d52nn@4096
./target/release/duel52 replay --record games/owner-vs-gen031.jsonl            # the index
./target/release/duel52 replay --record games/owner-vs-gen031.jsonl --game 1   # walk it
./target/release/duel52 replay --record games/owner-vs-gen031.jsonl --game 1 --node 34

# Measure. `demo --seed N` replays exactly the game `stats` counted for seed N.
./target/release/duel52 stats --all --games 200000 --seed 1 --markdown
./target/release/duel52 config configs/split.toml          # validate a config file

# Rules mods (MODULAR_RULES.md). A ruleset is a file in configs/rules/ and nothing else —
# that directory IS the registry, and engine/tests/rulesets.rs enumerates it, so a new file
# is covered by every structural invariant the moment it exists.
./target/release/duel52 config configs/rules/three-vengeance-1.toml   # resolve + rules_hash
cargo test --test rulesets                                 # 11 invariants x every ruleset
cargo test --test rules_mods                               # the named tests for each variant

# Screen before you spend a training run. `ismcts` and `greedy` need no checkpoint, so they
# play any ruleset the day the file is written. Minutes, not hours.
./target/release/duel52 screen --games 400 --seed 1 --agents greedy,random       # fast pass
./target/release/duel52 screen --games 200 --seed 1                              # ismcts:800
# It tells you a ruleset is BROKEN, never that it is good. A ply-cap draw is a hard failure:
# game_rules.md §7's finiteness proof depends on specific rules, so breaking it is a bug
# report, not a result.

# PLAN.md §4's card value table — what each card is worth, in win-probability points.
./target/release/duel52 card-value --encoding-slots 21 --games 400 \
    --checkpoint models/duel52-split-lane-gen032.d52nn
# It varies the rank of a card **in hand**, not one face-up in a lane. That is the whole
# design: face-up measures a constant power at full value, a one-shot as already SPENT, and
# the 3 with its Trap structurally disabled — so the first version ranked power *kind* and
# put 8/J/10/9 on top, which is exactly `is_constant()`. Read `in hand`; `on board` is the
# contrast, and `gap` says whether a card's value is in the flip or in the body.
# ⚠️ Read the CONTROL line first. Each rank is also substituted into the OPPONENT'S hand,
# which the observer cannot see, so the thirteen tensors are identical and the spread must be
# 0.00000. If it is not, the method is broken and the table is noise.
# ⚠️ `mirrored` cannot be measured this way (§9b publishes the removed multiset, so almost no
# rank has an unseen copy left). The tool detects the thin sample and refuses to print.

# A rules experiment is a 3-hour warm start, not a 24-hour run, because the encoder is
# rank-agnostic and no ruleset moves a layout hash.
.venv/bin/python -m duel52.train check --config configs/train-mod-3h.toml
.venv/bin/python -m duel52.train run --config configs/train-mod-3h.toml \
    --run-dir runs/mod-three-vengeance --init-from models/duel52-split-lane-gen032.d52nn

# Rating agents. Budgets are part of the agent name, so a result row names the agent that
# produced it: random · greedy · flatmc:600 · pimc:32x1 · ismcts:800.
#
# ⚠️ THE HAND-WRITTEN LADDER IS RETIRED. gen031 beat ismcts:800, its top rung, 200-0, so a
# fit that includes those rungs is an extrapolation off a handful of losses. The live scale is
# the three trained agents with gen016 pinned at 0 (FINDINGS.md, "The scale"). `--anchor` is
# what pins it, and it errors rather than falling back if the name is not in --agents, because
# the old silent fallback was "whichever agent you listed first".
./target/release/duel52 ladder --games 400 --seed 1 --variant split \
    --encoding-slots 21 --stalemate-value 0.0 \
    --anchor netmcts:models/duel52-split-gen016.d52nn@256 \
    --agents netmcts:models/duel52-split-gen016.d52nn@256,\
netmcts:models/duel52-split-gen022.d52nn@256,netmcts:models/duel52-split-gen031.d52nn@256,\
netmcts:models/duel52-split-lane-gen032.d52nn@256
# ~25 min for four agents (six pairings). Drop --markdown for the per-pairing detail.
# ⚠️ Do NOT add --eval-batch here: batching a match is a 7% regression (FINDINGS.md F4.8).
./target/release/duel52 match --a ismcts:800 --b pimc:32x1 --games 400
# `--eval-batch N` works here too, and on `ladder` and `probe`. Same guarantee as self-play:
# the batch is across games, so the score is identical and only arrives sooner. A gate splits
# its games between two checkpoints, so it reaches about half self-play's batch.
./target/release/duel52 probe --games 400 --markdown --seed 1 --encoding-slots 21 \
    --agents netmcts:models/duel52-split-gen031.d52nn@256,random
# probe is self-play instrumentation and it is where FINDINGS.md's strong-play tables come
# from. Keep `random` in the roster: lane and attack concentration have no absolute scale, so
# a number like 0.907 is meaningless without uniform play's 0.777 in the same table.

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

# Phase 4 — the scale-up. `check` also prints the gate's statistical power, the reference
# panel's plan and veto power, the LR schedule and the held-out size, which is the five
# seconds that tells you whether the run can decide anything. `--init-from` starts from a
# shipped checkpoint instead of a random init, and refuses one whose trunk disagrees
# with [net].
#
# A panel row that has saturated (best-ever ≥ `gate.reference_saturated_at`) is re-run at
# `gate.reference_games_saturated` instead of `gate.reference_games` — `random` and `greedy`
# sit at 1.000 for every generation of a warm-started run and detect a cliff, nothing more.
# It keys off the high-water mark, not the opponent's name, because from scratch those rows
# are informative for several generations. Only `train-big.toml` sets it; the three spent
# configs are the record of runs measured on a full panel.
.venv/bin/python -m duel52.train check --config configs/train-2h.toml
.venv/bin/python -m duel52.train run   --config configs/train-2h.toml --run-dir runs/fourth \
    --init-from models/duel52-split-gen016.d52nn                  # 2 h on the laptop
./target/release/duel52 match --a netmcts:runs/fourth/checkpoints/best.d52nn@256 \
    --b netmcts:models/duel52-split-gen022.d52nn@256 --games 400 --encoding-slots 21
# gen022 IS runs/fourth's generation 6, shipped. gen016 is the agent before it, kept as the
# frozen reference every Phase 4 number is measured against — not a second thing to play.

# Phase 4 Stage 0b (PLAN.md §4.2a) — three hours, one experimental change: six exact lane
# relabellings of every training sample. Warm-starts from gen022, which pins the trunk.
# DONE 2026-09-05: FINDINGS.md F4.5, shipped as gen031 — the default until lane-gen032.
.venv/bin/python -m duel52.train check --config configs/train-3h.toml
.venv/bin/python -m duel52.train run   --config configs/train-3h.toml --run-dir runs/fifth \
    --init-from models/duel52-split-gen022.d52nn
# The mechanism check, two seconds and no opponent: does the policy treat the three lanes
# alike? Run it on generation 1, not at the end. ⚠️ Read the *policy TV* row, not the argmax
# agreement row — F4.5 found agreement still at 86/128 after one generation (gen022 is
# 82/128) while TV had already gone 0.152 → 0.113. Agreement is an argmax over near-ties and
# breaks late; TV is continuous and moves first. FINDINGS.md F4.3, F4.5.
.venv/bin/python -m duel52.lanes --checkpoint runs/fifth/checkpoints/gen001.d52nn
./target/release/duel52 match --a netmcts:models/duel52-split-gen031.d52nn@256 \
    --b netmcts:models/duel52-split-gen022.d52nn@256 --games 400 --encoding-slots 21
# gen031 IS runs/fifth's generation 9, shipped. It ended the flat lineage and is now the
# agent lane-gen032 is measured against, in the role gen016 played for it.

# Phase 4 Stage 1 (PLAN.md §4.2b, §4.2c) — three hours, from scratch, two changes:
# the lane-equivariant network and playout cap randomisation. NOT a warm start, and it
# cannot be: `arch = "lane"` shares no tensor name with the flat network, so `--init-from`
# refuses across the two by name.
.venv/bin/python -m duel52.train check --config configs/train-3h-new.toml
.venv/bin/python -m duel52.train run   --config configs/train-3h-new.toml --run-dir runs/sixth
# The mechanism check, and on this architecture it is pass/fail rather than a trend: every
# row must read **exactly** 0.000 / 128 of 128, because no parameter is indexed by a lane.
# Contrast gen022 (TV 0.152, 82/128) and gen031 (0.039, 114/128), which is as close as six-
# fold augmentation could get. A non-zero row means the equivariance is broken.
.venv/bin/python -m duel52.lanes --checkpoint runs/sixth/checkpoints/gen001.d52nn
# Build a lane checkpoint by hand:
.venv/bin/python -m duel52.nn init --arch lane --encoding-slots 21 \
    --width 128 --blocks 3 --value-hidden 128 --out checkpoints/lane.d52nn

# The pieces, runnable on their own when something looks wrong.
./target/release/duel52 selfplay --checkpoint runs/first/checkpoints/best.d52nn \
    --out /tmp/gen.d52sp --games 200 --sims 64 --encoding-slots 21
# `--eval-batch N` keeps N games in flight per thread so their forward passes batch
# (PLAN.md §4.2d). 3.26x at 64 on the laptop, and the shard is byte-identical to
# `--eval-batch 1` — it is a speed knob and nothing else. Clamped to games/threads.
./target/release/duel52 selfplay --checkpoint runs/sixth/checkpoints/best.d52nn \
    --out /tmp/gen.d52sp --games 512 --sims 256 --cap-sims 32 \
    --full-search-fraction 0.25 --encoding-slots 21 --eval-batch 64
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

⚠️ **There are two architectures now, and depth costs far more than the parameter count
suggests.** `arch = "mlp"` is `DESIGN.md` §5's flat trunk — gen016, gen022 and gen031 are all
this — and `arch = "lane"` is the lane-equivariant network (`PLAN.md` §4.2b), whose trunk runs
**once per lane** with shared weights. So `lane 128×3` is nine block-evaluations against a flat
`128×3`'s three.

The trap is reasoning from parameters: the input projection is 58% of a flat checkpoint's
weights and the policy head 30%, so sharing them across lanes looks like it should make depth
cheap. It does not, because **in the search path neither matrix is the cost** — the input layer
walks only the observation's ~205 non-zeros (`FINDINGS.md` F3.3) and the policy head is masked
to the ~21 legal logits. The trunk is what self-play pays for. Measured, 40 games at 256 sims
on the 8-core laptop:

| network | games/sec | vs gen031 |
|---|---:|---:|
| flat `128×3` (gen031), no PCR | 2.0 | 1.00× |
| lane `128×3` + PCR | 2.0 | 1.00× |
| lane `128×4` + PCR | 1.7 | 0.85× |
| lane `128×6` + PCR | 1.2 | 0.60× |
| lane `128×6`, no PCR | 0.5 | 0.25× |

Playout cap randomisation (`PLAN.md` §4.2c) is what pays for the trunk: a quarter of decisions
get the full `sims` and a policy target, the rest get `cap_sims` and a **value** target only,
which costs nothing extra because one game has one outcome however little search produced the
position. `--full-search-fraction` and `--cap-sims` on `duel52 selfplay`; the trainer masks the
policy loss on the flag and divides by the **masked count**, not the batch size — dividing by
the batch size would scale the policy gradient by the capping fraction with nothing to say so.

Two consequences worth keeping straight. **A checkpoint without an `arch` header key reads as
`mlp`**, which is what keeps the three shipped ones loading — the key is optional on read, not
versioned. And **`lane_augment` is pointless on `arch = "lane"`**: the network satisfies
`f(σ·x) = σ·f(x)`, so a relabelled sample gives the identical loss *and* the identical gradient
(`test_lane_augmentation_is_a_no_op_on_the_lane_equivariant_network`). F4.5's +82 Elo was an
augmentation result on the *flat* net and does not carry over.

⚠️ **`--eval-batch` is a pure speed knob, and it is the only one — do not treat it as a
hyperparameter.** `PLAN.md` §4.2d. Self-play keeps N games in flight per thread and evaluates
their positions together; **the batch is taken across games, never inside a search**, so no
game's search is altered and `LaneBody::trunk_batch` is bit-identical per row. The shard is
therefore byte-identical whatever N is, which
`phase4_a_shard_does_not_depend_on_the_evaluation_batch` and
`phase4_batched_evaluation_is_bit_identical` assert — the latter on `to_bits`, deliberately,
because a tolerance would pass exactly the reassociation the determinism contract forbids.
3.26x at 64 on the laptop; `configs/train-3h-new.toml` sets it.

**The gate and panel *can* use it and should not.** The machinery is there — `probe::MatchGame`
is the state machine `selfplay::GameRunner` is, and `Agent::begin_decision` is how a
`Box<dyn Agent>` opts in — but a match holds **two different checkpoints**, so each round's
games are grouped by the network they wait on and a gate reaches half self-play's batch. At
that width it is a **7% regression**: 300 games, two lane nets at 256 sims, 373.5 s unbatched
against 401.2 s at `--eval-batch 64`, bit-identical scores (`FINDINGS.md` F4.8). The training
loop passes the flag to `selfplay` only. Keep it that way unless a measurement says otherwise.

Three things to keep straight. **The batch is clamped by the games a worker owns** —
`selfplay.games / threads` for self-play, `gate.games / threads` for the gate; 1400 over 8 is
175, fine, but a 200-game generation on 8 cores gets 25 and the config's 64 is silently a lie.
`train check` prints both effective numbers. **It costs ~400 KB of live search tree per game
in flight**, so 64 × 8 threads is ~200 MB. And **`nn::batch_slots` is not `min(games, batch)`**
— see the next warning.

⚠️ **A batch that does not divide a worker's games must be spread, not truncated.** The
obvious `min(games, batch)` is worse than not batching at all: a worker with 37 games told to
keep 32 in flight advances all 32 in lockstep, so they finish together, and then drains the
last 5 at a batch of 5 for a full game's length. On a 300-game gate that made `--eval-batch 32`
**slower than `--eval-batch 1`**, at 295% CPU against 712%. `nn::batch_slots` spreads a
worker's games over the fewest waves instead — 37 games at a batch of 32 runs **19** in flight.
A smaller batch that stays whole beats a big one that collapses.

⚠️ **A batched match absorbs its games in game order, and that is load-bearing.** Games finish
out of order once several are in flight, and `AgentBehaviour::absorb` *pushes* each game's lane
and attack concentration into a `Vec<f64>` whose mean `probe` reports — so absorbing them as
they land shifts that mean's last bits and makes the probe tables irreproducible. The gate
itself reads only integers and would not have caught it;
`rule_2_the_ladder_is_eval_batch_independent` compares the per-game f64s for exactly this
reason, and fails if the sort is removed.

⚠️ **If you write a batched kernel, the accumulators must be a fixed-size stack array.**
Accumulating straight into a slice of the output — the obvious way — leaves the compiler
unable to prove the output does not alias the weights, so it reloads and restores on every
step of the reduction and the batched kernel runs at *the speed of the unbatched one*: 1.9
GMAC/s against 11.9 for the same arithmetic. This was measured the hard way. `matmat`'s
`TILE` and its comment are the record.

⚠️ **A `.d52sp` shard is version 3 as of Stage 1, and version 2 shards are refused.** The
per-sample `policy_target` byte is not optional the way the checkpoint's `arch` key is: it sits
in the middle of each record, so a v2 reader and a v3 file misparse from the first sample on.

⚠️ `encoding_slots` defaults to **16** and the encoder **asserts** rather than truncating.
A `netpolicy` checkpoint played against `random` can exceed it — see `FINDINGS.md` F3.1.
Add `--encoding-slots 21` to both the `init` and the `duel52` command if you hit it; the two
must match, because `encoding_slots` is what fixes `obs_dim`.

Configs live in `configs/`: `split.toml` (the default), `base.toml`, `mirrored.toml`, and
`split-raw-two.toml` (the control for the §10a house rule). `train-fast.toml`, `train-2h.toml`,
`train-3h.toml`, `train-3h-new.toml`, `train-7h.toml`, `train-12h.toml` and `train-big.toml`
are *training*
configs rather than game configs — they carry the loop's knobs and set `encoding_slots = 21`,
which every command in that run must agree on. `train-fast` is Phase 3's shakedown and produced
gen016; `train-2h` is Phase 4 on a laptop and **warm-starts from gen016**, which is why its
trunk is pinned to `128 × 3`; `train-3h` is Stage 0b, warm-starts from **gen022**, is the only
config with `lane_augment = true` — its one experimental change — and produced gen031, the
current default; `train-3h-new` is Stage 1, the first config with `arch = "lane"` and playout cap
randomisation, and the only one that starts **from scratch** on a laptop — it produced
`runs/sixth`, which was never shipped because it lost to gen031 0.3175.

`train-7h` is Stage 2 and produced **lane-gen032**, the current default. It warm-starts from
`runs/sixth` and is `train-3h-new` re-sized around batched evaluation (`FINDINGS.md` F4.7):
4,000 games a generation rather than 1,400, because at 1,400 the now-faster self-play would
have left the gate as the majority of the clock. ⚠️ It is also the first config with
`reference_tolerance = 0.10` rather than 0.05 — at generation 9 that difference saved a
candidate which won its 294-game gate 0.605 and would otherwise have been vetoed on a noisy
panel row (`FINDINGS.md` F4.8).

`train-12h` and `train-big` are the two sizings of `PLAN.md` item 7, the rented-core run, and
**`train-12h` is the one to reach for** — it is `train-big` at half the clock with three
corrections that apply at either length, each documented against its number: a smaller
generation (the gate is a fixed tax, so 15,000 games at 12 hours buys precision instead of
generations), `epochs_per_generation` in place of a fixed step count (`train-big`'s 2,800 steps
are 3.0 epochs over a single random-init shard at generation 1), and a wider
`reference_tolerance` (0.05 gives a plateaued panel row up to a 52% chance of ending the run on
five false refusals). `RENTING.md` is how to get the box. `PLAN.md` §4.5 is the order to run
them in.

**The four shipped checkpoints are two lineages, not a menu.** `gen016 → gen022 → gen031` is
the *flat*-trunk chain, each warm-started from the one before and measured against it at equal
simulations: +81, then +82, for +189 end to end (`FINDINGS.md` F4.1, F4.5).
**`lane-gen032` is a different root** — the lane-equivariant trunk shares no tensor name with
the flat one, so `--init-from` refuses across them and its lineage began from a random init
(`runs/sixth`, unshipped). It beats gen031 by **+167 Elo** at equal simulations, 0.7238 ± 0.044
over 400 games (`FINDINGS.md` F4.8). The `032` continues the numbering for readability and
**not** because it is one step past `031`.

**Play `lane-gen032`.** The other three are kept because every Phase 3 finding is measured on
gen016 and every Phase 4 number against it, and gen031 is the agent `lane-gen032` had to beat.
All four still load — no layout hash has moved since gen016.

## Architecture

The parts of the archived `DESIGN.md` that still bind. `engine/src/encode.rs` is the authority
on both layouts and documents them block by block; this is the summary.

**Stack.** Rules, search and inference in Rust with zero dependencies. Training in PyTorch.
PyO3 bridges them. The engine is the sole authority on legality and never depends on Python.

**Action encoding: a fixed 1324-wide policy head, legality-masked and phase-conditioned.**
`L = 3` lanes, `S = 16` slots (`config.encoding_slots`), `R = 13` ranks, all derived from
config so a smaller variant shrinks the head rather than misaligning it:

| block | formula | at S=16 | engine `Action` |
|---|---|---:|---|
| `PLAY(rank, lane)` | `R·L` | 39 | `Play { rank, lane }` |
| `FLIP(lane, slot)` | `L·S` | 48 | `Flip { lane, slot }` |
| `ATTACK(lane, atk, tgt)` | `L·S·S` | 768 | `Attack { lane, attacker, target }` |
| `PAIR(lane, a<b)` | `L·S(S−1)/2` | 360 | `DeclarePair { lane, slot_a, slot_b }` |
| `CHOOSE_SLOT(side, lane, slot)` | `2·L·S` | 96 | `Peek` / `ResolveNext` / `MoveHere` / `SplitTarget` |
| `CHOOSE_RANK(rank)` | `R` | 13 | `GiveBack { rank }` |
| **total** | | **1324** | |

There is **no `PASS` block**, and every logit is therefore something a player chooses. That is
what makes a policy target a distribution over choices rather than a mixture of choices and
bookkeeping.

**Observation encoding: 3300 floats per observer** at the default config, dominated by the
board tensor of `3 lanes × 2 sides × 16 slots × 33 features = 3168`. Sides are ordered
`[observer, opponent]`, so the tensor is always from the observer's point of view and the
network never learns a seat convention. The remaining 132 are scalars (phase, actions
remaining, hand and pile sizes, discard rank counts, lane-derived counts) plus **belief
features**: unseen-card rank counts from this observer's perspective. Those never reach zero
uncertainty, because the 10 cards removed at setup stay permanently indistinguishable from
cards in the opponent's hand or base. The size tracks `encoding_slots` almost linearly, which
is why 21 slots gives `obs_dim = 4290` and every command in a run has to agree on it.

**Search.** Information-set MCTS with per-simulation determinization, PUCT over the policy
prior, and the value head in place of rollouts. Every agent decides from a sampled world, never
from the state it is handed.

**Training loop.** Rust self-play writes `.d52sp` trajectory shards; Python replays them into a
buffer, fits, and writes a `.d52nn` checkpoint; a gate promotes a candidate only when it beats
the incumbent over enough games to have an interval.

⚠️ **Self-play's share of a generation is a ratio you choose, not a constant of the loop.** This
file used to quote 87%; that was measured on Phase 3's shape and stopped describing the current
one when the gate grew and generations shrank. The mechanism is that **a gate game costs ~3.3× a
self-play game** — the gate is uncapped net-vs-net at `gate.sims` while self-play is capped to a
mean of 88 (measured 2026-09-07 on lane `128×3`: 0.6 games/sec against self-play's 2.0). So what
sets the split is `selfplay.games` against `gate.games` + the panel, and every run so far:

| run | self-play games | gate | panel | self-play share of wall clock |
|---|---:|---:|---:|---:|
| `first` / `second` / `third` (Phase 3) | 3,000 | 200 | 0–150 | 91% / 86% / 85% |
| `fourth` / `fifth` (Phase 4, laptop) | 1,200 | 300 | 120 | 67% / 68% |
| `sixth` (Stage 1, lane + capping) | 1,400 | 300 | 100 | **53%** |

The gradient step is 2–4% throughout and has never been worth optimising — `runs/sixth` spent 30
seconds of a 21-minute generation on it. **Size a generation so the gate is a tax and not a
partner**: below ~6,000 self-play games at a 600-game gate the run spends more than half its
clock evaluating itself. `configs/train-12h.toml` targets 55–60% and says how to re-derive it.

**`duel52 replay`'s verbs**, since the board and the walk both use them:

| Verb | Action |
|---|---|
| `PLAY` | Put a card from hand, face-down, into a lane |
| `FLIP` | Turn one of your face-down cards face-up, firing its power |
| `ATK` | Attack: `lane L: your #a [card] -> opp #b [card]` |
| `PAIR` | Declare two same-rank cards on your side of a lane as a pair |
| `2ND` | A 10's Twinstrike: the second of its two targets |
| `NEXT` | Adaptive resolution order (§8): which pending power resolves next. Often forced, and then it shows a prior of 1.000 |
| `MOVE` | A Queen's Move: pull an allied card from another lane into hers |
| `PEEK` | A 4's Foresight: look privately at one face-down card, either side's |
| `BACK` | A 2's View: bottom a card from hand (house rule) or discard it, per `two_power` |

Numbering (`#1`, `#2`, …) is `display.rs`'s `column_slots` order, the same order the board
draws a column in and the same order the CLI menus use. It is the one place in the codebase
where lanes and cards are numbered from 1.

## Where things are

| Path | What |
|---|---|
| `engine/src/state.rs` | `GameState` and the queries the rules are written in terms of |
| `engine/src/apply.rs` | Combat and turn machinery. **Card powers no longer live here** — it dispatches to `powers/` |
| `engine/src/powers/` | One module per rank. Changing what the 3 does touches `three.rs` and nothing else. `mod.rs` holds `PowerId`, the hooks and the dispatch |
| `engine/src/damage.rs` | `DamageSource` and the damage queue. Why damage is a FIFO and not recursion, and what keeps a death-trigger cascade finite |
| `engine/src/cardvalue.rs` | `PLAN.md` §4's card value table, and the null control that makes it readable |
| `engine/src/legal.rs` | Legal-action enumeration |
| `engine/src/config.rs` | Every tunable; the three variant presets |
| `engine/src/testkit.rs` | Building positions by hand, for tests and Phase 5 probes |
| `engine/src/display.rs` | Rendering a board and an action for one observer. The only place lanes and cards are numbered from 1, and the only definition of the order a lane's cards are drawn in (`column_slots`). `Focus` is the red highlight the CLI puts on a card while its number is being typed — decoration only, and it can never add a character to the board |
| `engine/src/menu.rs` | Reshapes the flat legal-action list into the CLI's question tree — verb, then card, then lane only when the card is in more than one — with every verb and lane number fixed to the thing it picks. `Menu::focus` turns a number back into the cards it names, which is what the board highlights |
| `engine/src/record.rs` | The JSONL game record: `(config, seed, chosen indices)` replays a game exactly. `walk` **verifies** rather than decodes — a record that no longer reproduces its own outcome is refused, which is what stops a rules change turning the corpus into games nobody played. Hand-rolled JSON, because the engine has no dependencies |
| `engine/src/determinize.rs` | Sampling a world from an information set. Every search agent goes through it |
| `engine/src/encode.rs` | Observation and action tensors, and the layout hashes that pin them |
| `engine/src/nn/` | Weights, the `.d52nn` checkpoint format, and the reference forward pass. `mlp.rs` is the flat network and dispatches to `lane.rs`, the lane-equivariant one, on `arch.kind` |
| `engine/src/nn/lane.rs` | The lane-equivariant forward pass. Its module header carries the equations both languages implement |
| `engine/src/agents/` | The five ladder rungs plus `netpolicy` and `netmcts`, and the evaluation in `eval.rs` |
| `engine/src/selfplay.rs` | Self-play generation and the `.d52sp` trajectory shard |
| `engine/src/ladder.rs`, `elo.rs` | Round robin, and the Bradley–Terry rating fit |
| `engine/src/probe.rs` | Instrumented play — where the Phase 2 findings come from |
| `engine/tests/` | One named test per ruling, named for its rule section |
| `bindings/src/lib.rs` | PyO3 wrapper; `Game.observation()` is the filtered per-player view |
| `py/duel52/nn/` | The PyTorch model and checkpoint I/O. **Never an encoder** — see below |
| `py/duel52/train/` | The AZ loop: replay buffer, trainer, generation driver. Gradients only |
| `py/duel52/lanes.py` | `FINDINGS.md` F4.3's lane-symmetry metric, for one checkpoint. Analysis; the permutation tables it compares against come from `encode.rs` |

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

- **Sub-decisions are separate zero-cost decision nodes on a stack** (Architecture, above). A 5
  that flips a King that re-empowers the lane resolves correctly because of this. Collapsing
  them into one big action would blow up the branching factor and break §8's adaptive
  ordering.
- **Cards are tracked by `CardId`, never by slot.** Slots compact on death and shift when a
  Queen moves a card, so anything remembered across a resolution step holds ids.

- **Nothing outside `powers/` may branch on a `Rank` to decide behaviour.** Read
  `card.live_power(&config)` and ask the `PowerId` — `taunts()`, `is_nimble()`,
  `retaliate_mode()`. `live_power` returns `None` for a face-down card, which is `game_rules.md`
  §6's "powers are inert while face-down" made structural rather than remembered. A
  `rank == Rank::EIGHT` in `state.rs` or `apply.rs` is a bug: it silently ignores the ruleset.

- **`rules_hash` is not `obs_layout_hash`, and the difference is the point.** The encoder is
  rank-agnostic, so **no ruleset moves a layout hash** — that is what lets `--init-from`
  warm-start a rules experiment from the current champion and turns 24 hours into 3
  (`engine/tests/rulesets.rs::no_ruleset_moves_the_encoder_layout` asserts it). It also means
  a shard or checkpoint from another ruleset is *indistinguishable on shape alone*, which is
  why `rules_hash` exists. It is checked where a cross-ruleset number would be read as a
  result — `Shard::read`, `ladder`, `match`, `probe`, `card-value`, and the Python replay
  buffer — and deliberately **not** in `Weights::load`, because generation 1 of every
  warm-started run is legitimately cross-ruleset. `MODULAR_RULES.md` §6.

- **There is exactly one encoder, and it is in Rust.** `engine/src/encode.rs` owns the
  feature layout; Python reaches it through `Game.encode_observation()` and gets its
  dimensions and layout hashes from `duel52.encoding_spec()`. A second copy of the layout in
  Python would let the trained function and the evaluated function drift apart *silently* —
  nothing crashes, the agent is merely bad, and the natural suspect is the training run. The
  checkpoint header carries both layout hashes and `Weights::load` refuses a mismatch, which
  turns that into a one-line error. Never compute a layout hash outside `encode.rs` — and never
  derive a **lane-permutation table** outside it either. `encode::lane_permutations` publishes
  the six exact relabellings (`PLAN.md` §4.2a) and Python only gathers with them; a table
  computed in Python would be a second reading of the layout, and a wrong one trains the network
  on mismatched targets with nothing crashing to say so. The same holds for
  `encode::lane_structure` (`PLAN.md` §4.2b), which says *which lane owns* each observation
  float and each logit — the lane-equivariant network shares one weight matrix across the three
  lanes on the strength of it, so a wrong table routes lane 2's board through lane 1's weights
  and produces an agent that is merely bad.
  `phase4_lane_structure_agrees_with_the_permutations` checks the two tables against each
  other rather than transcribing the layout a third time.
