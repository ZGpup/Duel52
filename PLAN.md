# Duel 52 — Roadmap

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

**Current phase: 3 done and shipped; Phase 4 has started by accident.** Phase 2 delivered five
agents, determinization, a frozen Elo ladder and the first strategic measurements
(`FINDINGS.md` F2). Phase 3 delivered the encoders, the network, the inference path,
`netmcts`, and the AlphaZero loop around them. `configs/train-fast.toml` has now been run:
57,000 self-play games, 19 generations, 1.94 hours. The result is
`models/duel52-split-gen016.d52nn`, tracked in git, **+495 Elo clear of `ismcts:800`** on the
first ladder fitted since the §4 ruling (F3.7).

**The next thing to do is run it again, bigger — with three things changed first.** F3.8
measured where the first run's ceiling actually was, and it was not compute:

1. **The gate, first, because it is free.** 200 games at a 0.55 threshold gives the promotion
   test a standard error of 0.035. A generation genuinely improving at a true 0.54 passes only
   **39%** of the time, and there is a **23%** chance of three consecutive refusals — the stop
   condition — while genuinely improving throughout. The run was ended by a measurement with
   no power to make the call, not by a learner that had stopped learning. Raise `gate.games`
   to ~600, or drop `gate.threshold` to 0.52.
2. **`selfplay.sims` 64 → 256.** Search returns run +141, +156, then +69 Elo across successive
   4× steps out to 4096 simulations — flat, then halving, and every step still excludes even
   (F3.8). The policy target **is** the visit distribution, so training at 64 caps the teacher
   roughly **370 Elo** below what the same weights produce at 4096. 256 is the pick because
   the early steps are where the return is largest per unit of compute; the knee above 1024
   says buying all the way up is not worth it inside a training loop. Trade `selfplay.games`
   down to ~1500 to pay for it; better targets beat more of the same targets now that the
   policy is decent.
3. **`net.blocks` 3 → 10 — not `net.width`.** Only **10.5%** of the 949k parameters are in the
   residual trunk. 58% is the input projection and 30% the policy head, both pinned by
   `obs_dim = 4290` and `action_dim = 2194`. Going 3→10 blocks costs +25% parameters and
   triples the reasoning depth; going 128→256 wide costs +120% for less of it. The comment in
   `train-fast.toml` says to raise width first, and the parameter accounting says otherwise.

Minor, same run: there is no LR schedule — `train.lr` is a constant 2e-3 through AdamW for the
whole run — and `temperature_decisions = 24` samples only the first 18% of a ~130-decision
game, which is thin exploration for a long game.

**Then, and only then, rent the GPU.** Nothing measured so far argues for changing the
*method*; see Phase 4's tripwire for what would.

**Still open from Phase 1, and it is the owner's:** the exit criterion. Nobody has played the
engine and confirmed the rules by hand. That hour is worth more than it was —
`duel52 play --encoding-slots 21 --opponent netmcts:models/duel52-split-gen016.d52nn@2048`
is an opponent that genuinely resists at ~137 ms a move, so a rules error is likelier to
surface as something that looks *wrong* rather than as noise.

---

## Phase 0 — Specification `[x]`

- [x] Fetch and interpret the published rules
- [x] Confirm no prior art exists (no engine, bot, solver, or strategy analysis published)
- [x] Resolve the major rules ambiguities with the owner
- [x] Write `game_rules.md`, `DESIGN.md`, `CLAUDE.md`, `OPEN_QUESTIONS.md`

---

## Phase 1 — Engine + rules validation `[~]`

The only phase that needs meaningful owner input.

- [x] Rust crate: state, legal-action enumeration, action application, terminal detection
- [x] All three configurations behind one config flag: base / split-deck / mirrored-removal
- [x] `two_power: bottom | discard` flag for the §10a house rule, so the parity claim behind
      it is measurable rather than assumed — **measured, and it holds**; see `FINDINGS.md` F1.5
- [x] Seeded determinism — same seed + config produces an identical game
- [x] One named test per ruling in `game_rules.md` (e.g. `rule_6_king_reactivates_ace_grants_one_action`)
- [x] Edge-case tests: 8 × 9 interaction, 10 blocked by 9/J, pair vs 8 double retaliate, 3-trap resurrection, Queen breaking a pair, 5 and 7 reaching base cards post-unlock
- [x] Edge-case tests, second batch (all settled 2026-09-03): 10 vs two Jacks is 1+1 but 10 vs
      two 9s is 1 to one; 9-pair deals 4 to a Jack and takes no retaliate from an 8; 10-pair
      splits 1+1 and consolidates to 2 when blocked; a 5 skips frozen cards but a King still
      reactivates them; King resets an Ace's attack counter rather than stacking it; the 2 is
      pile-neutral so turns-to-unlock is invariant; mutual lane win via retaliate is a draw
- [x] PyO3 bindings
- [x] Text CLI so the owner can play the engine and spot-check rules
- [x] Resolve outstanding items in `OPEN_QUESTIONS.md` — done 2026-09-03, nothing open
- [x] **Deliverable:** random-vs-random statistics — game length distribution, first-player
      win rate, how often games reach the stalemate cutoff, across all three variants —
      logged as `FINDINGS.md` F1, 1.2M games
- [ ] **Exit criterion:** the owner plays a few games against the CLI and finds no rules errors

**Play it with:** `cargo build --release && ./target/release/duel52 play --seed 1`

### What Phase 1 turned up that the plan did not anticipate

Three things worth carrying forward, all in `FINDINGS.md` F1:

1. **The stalemate rule is still untested.** It fired zero times in 1.2M random games,
   because the stall `game_rules.md` §7 describes is *strategic* and random agents attack
   constantly. The default of 20 quiet plies is unvalidated — re-measure in Phase 2.
2. **`DESIGN.md` §3's 8-slot encoding bound is wrong.** Random play reaches 20 cards on one
   side of one lane. The engine uses the theoretical maximum instead; Phase 3 must pick a
   real bound from strong-agent play rather than inheriting either number.
3. **The mutual-lane-win draw is reachable, not astronomical** — 0.4–0.5% of random games,
   and the only source of draws at this level of play.

### Assumptions flagged during implementation, and how they resolved

Per `CLAUDE.md`'s "make a defensible call, document it, flag it". Three were raised; the
owner ruled on all three on 2026-09-03. Each is now pinned by a named test.

- **A face-down 9 can be frozen by a 6.** ✅ Confirmed. Nimble is a power, and §6 says
  powers are inert face-down.
- **A 9 deals 2 damage to a face-down Jack.** ❌ **Wrong — overturned.** The owner's ruling:
  *all face-down cards are blank 2-HP cards.* A face-down Jack has 2 hit points, not 3, and
  a 9 deals it the ordinary 1. The engine now derives hit points from the card's face-up
  state rather than from its rank, and everything that keys on a target being a Jack reads
  the live power. Written into `game_rules.md` §5 as a **[RULING]**.
- **A 10 whose twinstrike hits two 8s takes 1 retaliate from each, and dies.** ✅ Confirmed.

The overturned one was the most consequential of the three, and in the right direction: it
*removes* an information leak rather than adding one. Under the assumed rule, damage being
public meant that watching a face-down card survive two hits would identify it as the Jack
for free. Under the correct rule, attacking a face-down card can never tell you what it is —
which is a cleaner premise for the belief modeling in Phase 3.

---

## Phase 2 — Baselines, no learning `[x]`

- [x] **Determinization** (`engine/src/determinize.rs`) — not on the original list, and the
      prerequisite for everything below it. `DESIGN.md` §6 step 1: sample a world consistent
      with the acting player's information set, including *which cards were removed*.
- [x] Random agent
- [x] Greedy heuristic agent (hand-written evaluation) — `agents/greedy.rs`, `agents/eval.rs`
- [x] Flat Monte Carlo — `agents/flat_mc.rs`, paired sweeps over sampled worlds
- [x] PIMC (perfect-information Monte Carlo) — the control — `agents/pimc.rs`
- [x] SO-ISMCTS with random rollouts — `agents/ismcts.rs`
- [x] Round-robin Elo ladder, frozen as the permanent benchmark — `AgentSpec::LADDER`,
      `engine/src/ladder.rs`, `engine/src/elo.rs`. Ratings are a batch Bradley–Terry fit,
      not the incremental update, so the table is order-independent and reproducible.
- [x] **Deliverable:** first real strategic observations logged to `FINDINGS.md` — F2

**Run it with:**

```bash
./target/release/duel52 ladder --games 400 --markdown   # the frozen Elo table
./target/release/duel52 probe  --games 400 --markdown   # self-play behaviour per rung
./target/release/duel52 match  --a ismcts:800 --b pimc:32x1 --games 400
./target/release/duel52 play   --opponent ismcts:2000   # play the strongest rung
```

### What Phase 2 turned up that the plan did not anticipate

1. **"Does this agent cheat?" is a test, not a comment.** A determinized world is in the
   same information set as the real state, so an honest agent handed either must return the
   same action — an exact assertion. It immediately caught a leak in the *greedy* agent,
   which does no search at all: applying a candidate action to the real state reveals hidden
   ranks, because flipping your own base card turns it face-up and killing a face-down card
   sends its rank to the public discard. Even one-ply lookahead has to happen inside a
   sampled world. See `phase2_no_agent_reads_hidden_information`.

   ⚠️ **A new agent is *not* covered automatically.** This document used to claim it was.
   The test iterates a hardcoded `TEST_ROSTER` in `engine/tests/agents.rs`, not
   `AgentSpec::LADDER`, so adding a rung means adding it to that roster by hand — corrected
   when Phase 3's `netpolicy` arrived, which is now in it. The roster's doc comment says so
   at the point where someone would edit it.
2. **The legal action set is a function of the information set alone.** Every legality
   predicate reads a face-up rank, a slot position, the mover's own hand, or a global flag —
   never a hidden rank. That is what lets an agent enumerate actions on the real state and
   evaluate them on sampled worlds, and it is now pinned by a test so a future rule cannot
   quietly break it.
3. **Search cost is unbounded because the branching factor is.** PIMC at depth `d` costs
   `b^(d+1)`, and `b` reaches several hundred on the sprawling boards random play produces.
   PIMC carries an explicit node budget as a result. This is the same problem as F1.7's
   encoding bound, seen from the search side, and Phase 3 will meet it again.
4. **The stalemate rule finally fires** — see `FINDINGS.md` F2.4. F1.2 could not test it,
   because random agents attack constantly. It is no longer untested.

---

## Phase 3 — Neural self-play `[~]`

**Steps 1–3 are done and a run has been through them** — encoders, network and inference path;
`netmcts`; the self-play/replay/gate/promote loop. `configs/train-fast.toml` produced
`models/duel52-split-gen016.d52nn` (F3.7). Still open: the CFR cross-check on a scaled down
variant, which is the only item from the original Phase 3 plan not delivered, and a run at a
budget where the gate can actually resolve a generation — see the three changes at the top of
this file.

- [x] Observation + action encoders (`DESIGN.md` §4–5) — `engine/src/encode.rs`, 3300 floats
      and a 1324-logit head, both config-derived. Exposed to Python through
      `Game.encode_observation` / `legal_mask` / `encode_action` / `decode_action`, with
      `duel52.encoding_spec()` as the single source of shapes and layout hashes.
- [x] Residual MLP with policy + value heads — `engine/src/nn/` (the Rust forward pass and
      the `.d52nn` format) and `py/duel52/nn/` (the PyTorch definition). `test_parity.py`
      asserts the two compute the same function.
- [x] `netpolicy:<checkpoint>` as a sixth `AgentSpec`, so a checkpoint plays through the
      existing `ladder` / `match` / `probe` / `play` harness. **The five Phase 2 rungs and
      their budgets are untouched.**
- [x] Net-guided ISMCTS (PUCT over the policy prior, value in place of rollouts) — step 2.
      `engine/src/agents/net_mcts.rs`, as the `netmcts:<checkpoint>@<sims>` rung. Availability
      replaces parent visits in PUCT's numerator, and priors are stored as logits and
      softmaxed over whatever subset a determinization makes legal.
- [x] AZ-style loop: ISMCTS self-play → replay buffer → train → gated evaluation — step 3.
      `duel52 selfplay` writes a `.d52sp` shard; `python -m duel52.train` replays it through
      the Rust encoder, fits, gates and promotes. The buffer stores **trajectories**
      (`config`, seed, and indices into `legal_actions()`), not encoded tensors: the engine
      is deterministic, so an encoder version bump costs a CPU replay rather than a discarded
      corpus. F2.7 and F3.1 both expect the slot bound to move.
- [x] Checkpointing, resumability, config-driven throughout — a run is one TOML plus a seed;
      `--resume` continues a run directory.
- [ ] **Run it.** Two attempts, both collapsed into the engine's stalemate draw
      (`FINDINGS.md` F3.6 and its postscript) — the second even at `stalemate_value = 0.0`,
      which removed the incentive but not the ability. The cause was a rule that was never a
      rule: the engine let a player pass. Fixed 2026-09-03 (`game_rules.md` §4, actions are
      mandatory); greedy self-play stalemates went to zero, F2.4b. `Pass` was then removed
      from the action space outright, taking the policy head from 1325 to 1324 logits, so
      **every checkpoint and shard written before 2026-09-03 is refused and `runs/` cannot
      be resumed** — start from a fresh `python -m duel52.nn init`.
      **The third run is the first one where a draw is not an available strategy** —
      `configs/train-fast.toml`,
      ~2.5 hours, ~18 generations, then a ladder against the frozen Phase 2 rungs. Every
      Phase 2 and Phase 3 number below predates the ruling and is measured on a different
      game; the ladder needs re-running before anything is compared across it. Still watch
      the self-play draw rate and the reference line.
- [ ] **Duel52-mini** in OpenSpiel; validate the loop against exact CFR before trusting it
      on the full game
- [ ] Local best-response as the exploitability proxy
- [ ] **Deliverable:** a trained agent that clearly beats the Phase 2 ladder, with an Elo
      table and an LBR number

**The encoding bound, settled for training.** `FINDINGS.md` F3.1: `encoding_slots = 16`
survives self-play (max 10) but not a mixed pairing against `random` (13–17), and `random` is
the ladder's permanent anchor. **The engine default stays 16** — F3.1's recommendation was
flagged rather than applied, and that call is still the owner's — but `configs/train-fast.toml`
sets `encoding_slots = 21` for the training run, because a run that dies partway through an
evaluation ladder costs more than 30% on the tensors. Self-play at 16 sims has already been
seen to reach 17 cards on one side of one lane, so this is not theoretical.

### What step 1 turned up that the plan did not anticipate

1. **Sprawl is set by the weakest agent in a pairing, not the strongest.** F2.7 measured
   self-play and concluded 16 was comfortable. It is — in self-play. Against `random`, which
   never kills anything, cards accumulate and the same net reaches 17. See F3.1.
2. **`DESIGN.md` §4's action head was lossy against the engine**, in a way that would have
   corrupted the policy target rather than merely weakened play. Replaced with an exact
   slot-keyed encoding; §4 now records why.
3. **The `first_player` field in `Game.observation()` was mislabelled** — it reported
   `to_move == P0`, duplicating `to_move` and telling an observer nothing about its own seat.
   Found because the encoder needed the feature. Now `observer_is_first_player`.

### Decisions step 1 locked, so step 2 does not have to relitigate them

`PHASE3_STEP1.md` carried the reasoning and was folded in here when the step landed. The five
that constrain later work:

- **Search and inference in Rust, training in Python.** `DESIGN.md` §9 has the argument.
- **`Evaluator` is batch-shaped**, because self-play will batch across concurrent games — `G`
  games in flight per worker, one simulation each per round, evaluated together. No virtual
  loss, no search distortion, every game reproducible from its own seed.
- **The action head is exact and slot-keyed**, 1324 logits, every one of them a decision a
  player actually makes. `DESIGN.md` §4.
- **The checkpoint is a documented ~100-line binary format**, zero-dependency on both sides,
  carrying layout hashes. Not ONNX (a C++ dependency for a five-layer MLP), not safetensors
  (a JSON parser in a crate that has none).
- **No new crates.** The `cli` / `nn` split waits for the CUDA handoff; `Evaluator` is the
  seam that makes it cheap.

### What steps 2 and 3 turned up that the plan did not anticipate

1. **The encoder's own shape is the inference budget, and sparsity buys it back.** The
   observation is 4.8% dense and a position offers ~21 of 2195 encoded actions
   (`FINDINGS.md` F3.3), so the input layer and the policy head — the two largest matrices —
   are both almost entirely wasted work. Walking the non-zeros and evaluating only the legal
   logits is **bit-identical**, not an approximation, and it is the difference between a
   viable and an unviable local run. Neither optimisation would have been visible from the
   design; both came out of measuring the tensors the encoder actually produces.
2. **The batched `Evaluator` has not been needed yet.** `DESIGN.md` §9 locked a batch-shaped
   interface so self-play could keep `G` games in flight per worker. Self-play instead
   parallelises across *games* on threads, one position per evaluation, because after the
   sparsity work the network is no longer the dominant cost per simulation — determinization
   and legal-action enumeration are. The interface is still the right one and the
   cross-game batching is still the right next step **when a GPU backend arrives**; it is
   deferred, not abandoned.
3. **AlphaGo Zero's 0.55 promotion gate does not work in a game with a stalemate draw.**
   `FINDINGS.md` F3.4: an early network draws ~69% of its self-play games, so the score
   between adjacent generations is compressed onto 0.5 and a 0.55 bar rejects nearly
   everything — a run whose teacher never advances, while every other number on the readout
   looks healthy. The gate ships at 0.5, which still rejects a candidate that is *measurably
   worse*, and that is the failure mode that compounds.
4. **The trajectory format is decoupled from the action layout too, not just the observation
   layout.** The plan said the buffer stores an "action-index sequence". Storing the
   *encoded* index would have tied every shard to one `action_dim`, which is exactly the
   coupling the trajectory design exists to avoid — so a shard stores indices into
   `legal_actions()`, which is a property of the engine rather than of the encoder, and the
   replay maps them through whatever encoder is current.
5. **The engine's stalemate draw is a stable equilibrium, and the first run found it in two
   generations.** `FINDINGS.md` F3.6 — the draw rate went 9% → 55% → 88% while the agent's
   score against `random` fell 0.93 → 0.53, and the gate promoted every generation because
   two stalling agents draw against each other and 0.500 cleared a 0.5 bar. Fixed by
   `config.stalemate_value` (a learning weight, not a rule; scoring is untouched), a gate
   that reads decisive games only, and a reference panel that can veto. The transferable
   part: **every [ENGINE] terminal condition is a potential equilibrium**, and the ones added
   for the trainer's convenience are the most dangerous because nothing about the real game
   constrains what they are worth.

---

## Phase 4 — Extract the insight `[~]`

**This is the actual point of the project.** It is a separate job from training and should
not be treated as a victory lap.

Phase 4 started earlier than planned, and by accident. Running `probe` against gen016 to
characterise it produced three of the results on the original list, because **the trained
agent is the first player in this project capable of the behaviour the hypotheses are about.**
Every Phase 2 rung was measured and found flat on H2 and H3; that was never evidence about the
game, it was evidence about the agents. The lesson generalises to everything below: a null
from an agent that *cannot do the thing* is not a null.

### Banked

- [x] **Hand size at pile-empty — H2 has a real measurement at last.** `FINDINGS.md` F3.9.
      Within-agent, 1000 self-play games, with `greedy` as a control: the trained net shows a
      **+1.25 ± 0.17** card gap between the games it wins and the games it loses; `greedy`
      shows **−0.04 ± 0.07**, reproducing F2.5's null exactly. F2.5's null was a null about
      agents incapable of hoarding. **Not yet causal — see below.**
- [x] **Flip-timing curves.** F3.10. The net's flip ply spans **21 plies** across ranks
      (8 at 12.9, J at 15.7 … 3 at 32.0, Q at 33.8) where `ismcts:800` spans 5 and is
      essentially flat. The ordering tracks power *type*: constant powers early (they are
      wasted face-down), one-shot powers late (flipping spends them), and the 3 latest and
      least-flipped of all — 0.59–0.63 against ~0.95 for everything else — because Trap only
      fires while the card is face-down. This is the clearest evidence the agent has learned
      structure rather than tactics.
- [~] **First-player advantage with error bars.** Partial, `split` only. 1000 clean self-play
      games: `netmcts@64` 0.5380 ± 0.0435, `greedy` 0.5245 ± 0.0435 — both still cover even,
      and the two are indistinguishable from each other. H8 survives contact with a strong
      agent. Still owed: the other two variants, and tighter intervals.

### The confound that has to be closed before H2 counts

**F3.9 is correlational and the causal arrow is genuinely ambiguous.** Holding cards may win
games; or a winning position may simply be one that does not force you to commit cards. F2.5
had the same confound and never had to face it, because its effect was zero.

- [ ] **Intervention test.** Build positions with `testkit`, identical but for hand size at
      unlock, and evaluate both with a fixed strong agent. If hand size *causes* the win rate,
      the constructed advantage survives; if it is a symptom, it does not.
- [ ] **Forced-commitment test.** Constrain the agent to play a card on turns where it would
      have hoarded, and measure what the constraint costs. This prices the resource rather
      than merely correlating it.

Until one of these lands, H2's status in `FINDINGS.md` is **supported, not confirmed**.

### Open

- [ ] Learned rank values — what is each card actually worth? The value head plus `testkit`
      positions is the direct route; the flip-timing curve already implies an ordering to
      check it against.
- [ ] Opening action-triplet frequencies
- [ ] Lane allocation: does optimal play concentrate on two lanes, and when does it commit?
      H3 is Phase 2's other agent-limited null and deserves the same re-run F3.9 gave H2 —
      the net's lane concentration is 0.904–0.910 against `ismcts`'s 0.774, which is a large
      enough gap to be worth a proper test rather than an eyeball.
- [ ] Tempo value of the Ace and the King+Ace line
- [ ] Probe the value net on hand-constructed positions
- [ ] **Re-run every Phase 2 hypothesis against the trained agent, not just H2 and H3.** The
      whole F2 hypothesis table was scored by agents that could not deliberately do most of
      the things being hypothesised about.
- [ ] **Deliverable:** written findings + an interactive page

### The tripwire — when to stop scaling and change method

Recorded here because it is a Phase 4 judgement, not a training one, and because the answer
today is *not yet*. AZ over determinized ISMCTS has a real ceiling in an imperfect-information
game: no equilibrium guarantee, and no way to learn to conceal or signal deliberately. The
project already knows fusion is punishing — F2.2 is PIMC collapsing under exactly that.

But `netmcts` shows **no fusion signature**. The same test that exposed PIMC — more sampling,
does it buy anything — gives PIMC nothing at 8× worlds and gives `netmcts` +145 Elo per 4×,
twice over (F3.8). Per-simulation determinization is doing its job.

Switch to R-NaD (see Stretch) when **either** of these appears, and not before:

1. Search still scales but the network stops absorbing it across a run whose gate actually has
   power — that is a representation limit, not a compute one.
2. Self-play looks healthy while scores against a *fixed external* opponent flatten or fall.
   That is the exploitability signature, and it is the one no amount of compute fixes.

Neither is present as of gen016. The P0 self-play drift (51.4% → 56.7% over the run) looked
like signature 2 and turned out not to be: it does not survive without exploration noise, and
`greedy` shows the same tilt (F3.9).

---

## Stretch — R-NaD on real compute `[ ]`

- [ ] Swap the learner; engine and encoders unchanged
- [ ] Validate on Duel52-mini against exact equilibrium first
- [ ] Scaled run on borrowed compute
- [ ] Compare the approximate-Nash policy against the AZ policy — where do they disagree,
      and is AZ exploitable in those spots?

Estimate: a game this size should not need a cluster. One strong GPU, or a single
multi-GPU day, is likely sufficient.

---

## Guiding constraints

- Owner has limited time and is delegating implementation. Default to making a defensible
  call, marking it `[ASSUMED]`, and flagging it — rather than blocking.
- Everything must run locally on an M-series Mac and scale to rented CUDA via config alone.
- Insight beats strength. A slightly weaker agent whose play we can *explain* is worth more
  here than a stronger one we cannot.
