# Duel 52 — Roadmap

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

**Current phase: 4 — scale the AlphaZero loop on rented compute.** Phases 0–3 are closed.
Phase 2 delivered five agents, determinization, a frozen Elo ladder and the first strategic
measurements (`FINDINGS.md` F2). Phase 3 delivered the encoders, the network, the inference
path, `netmcts`, and the AZ loop around them; `configs/train-fast.toml` ran for 1.94 hours and
produced `models/duel52-split-gen016.d52nn`, **+495 Elo clear of `ismcts:800`** (F3.7). The
owner has played the engine and found no rules errors, which closes the last Phase 1 item.

**The one number that sets Phase 4's agenda: the owner beats gen016, 5–0.** It is +1476 Elo on
a ladder whose anchor is `random`, and it has not taken a game off the one human who has played
it, at `@4096` and `@8192`. Ladder Elo is *internal* — it measures the distance to five
hand-written agents, and nothing in it says how far that is from good play. The human result is
the only external yardstick this project has, and it says the agent is not close.
**Phase 4's exit criterion is therefore a human one**, not a self-referential one.

That argument has since got a second, sharper edge. Ladder Elo is not only internal, it is now
**saturated**: F4.2 fits gen006 at +1788 ± 58 — four times the interval on any hand-written
rung — while it beats four of the five at ~1.000. The table cannot resolve the agent it is
measuring, and the scale itself shifts between fits, so the headline numbers of successive runs
are not directly comparable. Every fixed hand-written opponent this project owns has now been
passed. What is left that can still make a measurement is a frozen *trained* net and a human.

⚠️ **5–0 is qualitatively decisive and statistically thin, and the two should not be
conflated.** Five wins in five bounds the owner's true rate only at *p* > 0.55 (95%, one-sided)
— a coin that came up heads five times. What carries the weight is the *manner*: the owner
reports winning comfortably, and characterises the agent as playing reasonably move to move
while lacking long-term plan and making occasional obviously bad plays even at 8192
simulations. That description is corroborated from inside the run — see §4.2 change 7. It also
does not meet this project's own recording standard, because nothing recorded the games. §4.0
fixes that, and it is the first task in the phase.

Phase 4 is a scale-up, not a redesign. Nothing measured so far argues for changing the method
— `netmcts` shows no strategy-fusion signature (F3.8), search still pays at 4096 simulations,
and the policy loss was still falling when the first run stopped. What ended that run was a
promotion gate with no statistical power (F3.7), a teacher capped at 64 simulations, and a
trunk holding 10.5% of the network's parameters. All three are config, and §4.2 fixes them.
See §4.8 for the two measurements that *would* justify changing method.

---

## Phase 0 — Specification `[x]`

- [x] Fetch and interpret the published rules
- [x] Confirm no prior art exists (no engine, bot, solver, or strategy analysis published)
- [x] Resolve the major rules ambiguities with the owner
- [x] Write `game_rules.md`, `DESIGN.md`, `CLAUDE.md`, `OPEN_QUESTIONS.md`

---

## Phase 1 — Engine + rules validation `[x]`

The only phase that needed meaningful owner input, and it is closed.

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
- [x] **Exit criterion:** the owner plays the CLI and finds no rules errors — **confirmed
      2026-09-04**, including games against `netmcts:gen016`, which is a much better rules
      test than a random opponent because a wrong ruling shows up as an opponent doing
      something that looks *wrong* rather than as noise. No rules errors reported.

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

## Phase 3 — Neural self-play `[x]`

All three steps delivered, and the loop has produced a trained agent. Two items that were
originally filed here — the exact-CFR cross-check on a scaled-down game, and local
best-response as an exploitability proxy — are *verification* work rather than *training*
work, and they now live in Phase 6, where the tripwire that needs them lives too.

- [x] Observation + action encoders (`DESIGN.md` §4–5) — `engine/src/encode.rs`, 3300 floats
      and a 1324-logit head at the default bound, both config-derived. Exposed to Python
      through `Game.encode_observation` / `legal_mask` / `encode_action` / `decode_action`,
      with `duel52.encoding_spec()` as the single source of shapes and layout hashes.
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
- [x] **Run it.** Three attempts. The first two collapsed into the engine's stalemate draw
      (`FINDINGS.md` F3.6 and its postscript) — the second even at `stalemate_value = 0.0`,
      which removed the incentive but not the ability. The cause was a rule that was never a
      rule: the engine let a player pass. Fixed 2026-09-03 (`game_rules.md` §4, actions are
      mandatory); greedy self-play stalemates went to zero (F2.4b), and `Pass` was then
      removed from the action space outright, taking the policy head from 1325 to 1324
      logits — so **every checkpoint and shard written before 2026-09-03 is refused and
      `runs/` from before that date cannot be resumed.** The third run, the first in which a
      draw is not an available strategy, is F3.7: 57,000 games, 19 generations, 1.94 h,
      13 promotions, `models/duel52-split-gen016.d52nn` published with full provenance in
      `models/README.md`. Zero draws in every generation's self-play.
- [x] **Deliverable:** a trained agent that clearly beats the Phase 2 ladder, with an Elo
      table — F3.7, **+495 Elo** clear of `ismcts:800` on one twelfth its simulation budget,
      and `netpolicy` alone (argmax, no search) beats `greedy` 0.94. The LBR half of the
      original deliverable moves to Phase 6.

**The encoding bound, settled for training.** `FINDINGS.md` F3.1: `encoding_slots = 16`
survives self-play (max 10) but not a mixed pairing against `random` (13–17), and `random` is
the ladder's permanent anchor. **The engine default stays 16** — F3.1's recommendation was
flagged rather than applied, and that call is still the owner's — but every training and
evaluation command in the Phase 3/4 line sets `encoding_slots = 21`, because a run that dies
partway through an evaluation ladder costs more than 30% on the tensors. Self-play at 64 sims
has been seen to reach 17 cards on one side of one lane, so this is not theoretical.

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

### Decisions step 1 locked, so later work does not have to relitigate them

`PHASE3_STEP1.md` carried the reasoning and was folded in here when the step landed. The five
that constrain later work:

- **Search and inference in Rust, training in Python.** `DESIGN.md` §9 has the argument.
- **`Evaluator` is batch-shaped**, because self-play will batch across concurrent games — `G`
  games in flight per worker, one simulation each per round, evaluated together. No virtual
  loss, no search distortion, every game reproducible from its own seed. Still unused; it is
  the seam a GPU inference backend would attach to (§4.1, Phase 6).
- **The action head is exact and slot-keyed**, every logit a decision a player actually
  makes. `DESIGN.md` §4.
- **The checkpoint is a documented ~100-line binary format**, zero-dependency on both sides,
  carrying layout hashes. Not ONNX (a C++ dependency for a five-layer MLP), not safetensors
  (a JSON parser in a crate that has none).
- **No new crates.** The `cli` / `nn` split waits for a GPU backend; `Evaluator` is the seam
  that makes it cheap.

### What steps 2 and 3 turned up that the plan did not anticipate

1. **The encoder's own shape is the inference budget, and sparsity buys it back.** The
   observation is 4.8% dense and a position offers ~21 of 2195 encoded actions
   (`FINDINGS.md` F3.3), so the input layer and the policy head — the two largest matrices —
   are both almost entirely wasted work. Walking the non-zeros and evaluating only the legal
   logits is **bit-identical**, not an approximation, and it is the difference between a
   viable and an unviable local run. Neither optimisation would have been visible from the
   design; both came out of measuring the tensors the encoder actually produces.
2. **The batched `Evaluator` has not been needed yet.** Self-play parallelises across *games*
   on threads, one position per evaluation. F3.11 has since put a number on what that costs:
   even after the sparsity work the forward pass is roughly **80% of per-simulation cost**,
   so cross-game batching is the lever a GPU would pull. It is deferred, not abandoned — see
   §4.1 for why it is not worth pulling yet.
3. **AlphaGo Zero's 0.55 promotion gate does not work in a game with a stalemate draw.**
   `FINDINGS.md` F3.4: an early network draws ~69% of its self-play games, so the score
   between adjacent generations is compressed onto 0.5 and a 0.55 bar rejects nearly
   everything. Fixed by scoring the mirror match on decisive games only — and then broke
   again from the other side, because at 200 games that score has no power (F3.7). §4.2's
   first item is the third version of this gate.
4. **The trajectory format is decoupled from the action layout too, not just the observation
   layout.** The plan said the buffer stores an "action-index sequence". Storing the
   *encoded* index would have tied every shard to one `action_dim`, which is exactly the
   coupling the trajectory design exists to avoid — so a shard stores indices into
   `legal_actions()`, which is a property of the engine rather than of the encoder, and the
   replay maps them through whatever encoder is current.
5. **The engine's stalemate draw is a stable equilibrium, and the first run found it in two
   generations.** `FINDINGS.md` F3.6 — the draw rate went 9% → 55% → 88% while the agent's
   score against `random` fell 0.93 → 0.53, and the gate promoted every generation because
   two stalling agents draw against each other and 0.500 cleared a 0.5 bar. The transferable
   part: **every [ENGINE] terminal condition is a potential equilibrium**, and the ones added
   for the trainer's convenience are the most dangerous because nothing about the real game
   constrains what they are worth.

---

## Phase 4 — Scale up `[ ]`

**Goal: the strongest agent this architecture can produce, and an honest account of where it
tops out.** No new method, no new learner. One long run on rented hardware, with the three
things F3.8 identified as the first run's actual ceiling fixed first.

**Exit criteria**, in the order they should be checked:

1. The new net beats `netmcts:gen016@256` head-to-head by **≥ 0.70** over 400 games at equal
   simulations (≈ +150 Elo). Below that, the run did not buy anything and §4.8 applies.
2. **It takes at least one game off the owner** in a recorded six-game series (§4.0). This is
   the criterion that matters, and it is deliberately a low bar: against a 0–5 record, one win
   in six is a real change and the signal to invest in a longer series. Zero in six, on top of
   0–5, is 0–11 and puts the owner above 0.76 — at which point the honest reading is that
   compute was not the binding constraint and §4.8 applies.
3. The search-scaling sweep (F3.8) is repeated on the new net, because whether it still
   absorbs more search is the input to every decision after Phase 4.
4. `FINDINGS.md` gets the ladder, the sweep, the `probe` table and the human series, all with
   config and seed range.

Sections 4.1–4.7 are written for someone who has not rented a machine before. §4.1 is the
one that saves money; read it before booking anything.

### 4.0 Record the human games, then play six `[ ]`

Right now "I beat it 5–0" is the most important fact about the project and **not one ply of it
was written down.** The games are gone: no seeds, no moves, no way to ask what the net was
thinking when it played the moves that lost. That is the single cheapest thing to fix in this
phase and it needs no rented hardware.

**Why recording is nearly free here.** A Duel 52 game is fully determined by
`(config, seed, the sequence of chosen indices into legal_actions())` — the engine is
deterministic, so those few hundred bytes replay the game *exactly*, hidden information
included. It is the same insight the `.d52sp` trajectory format is built on
(`selfplay.rs`'s header). `cmd_play` already holds both `legal` and the chosen `action` in
each of its two branches ([duel52.rs:886](engine/src/bin/duel52.rs#L886) for the bot,
[duel52.rs:981](engine/src/bin/duel52.rs#L981) for the human), so `--record <file>` is a
position lookup and an append.

- [x] **[code] `duel52 play --record <file>`** — append one line per game:
      config, seed, which seat the human took, the opponent spec, the chosen-index sequence,
      and the outcome. **Plain text (JSONL), not `.d52sp`.** A human decision has no visit
      distribution and no root value, and the shard format's invariant is that every row
      carries a policy target — a shard that can hold rows without one is a shard the trainer
      can silently train on. Keep the corpus format strict and let human games be a different,
      inspectable, git-committable few kilobytes.
      **Measured: a 153-ply game is 918 bytes.** Only *finished* games are written — a
      half-played game cannot be checked against an outcome, and that check is what makes a
      record trustworthy later, so quitting prints the seed instead. `engine/src/record.rs`,
      with its own hand-rolled JSON (the engine has no dependencies, deliberately).
- [x] **[code] `duel52 replay --record <file> --game N`** — walk a recorded game ply by ply,
      and at each of the human's decisions print what the net thought: the value head's score
      for the side to move, the policy prior over the legal actions, and what `netmcts` would
      have played at a given budget. This is the instrument §4.0's analysis needs, and it is
      also a rules-checking tool: a disagreement you can point at, in a position you remember.
      Bare `--record <file>` prints the index; `--game N` walks one; `--ply N` also prints
      the board at that ply, from the view the player had when they chose it. The checkpoint
      and budget default to **the agent that was actually played**, so `replay --game 1` says
      what your opponent thought; `--checkpoint` overrides it, which is how an old game
      becomes a fixed evaluation set for a new net. The run ends with the plies §4.0a asks
      for: where the value head was confident (`|v| > 0.6`) in the side that went on to lose.
      ⚠️ The search in a replay is a **fresh** one, not the one that played the game — an
      agent's RNG stream depends on how often it has been called. That is right for analysing
      the *human's* decisions, which is the point; the bot's own moves are in the record.
- [x] **[code] The record is verified, not merely decoded.** `GameRecord::walk` replays
      against a fresh engine and refuses the record unless every index was in range, the game
      ended exactly when the moves ran out, and the outcome matches what was written down. So
      a rules change turns the corpus into a loud failure rather than into games nobody
      played — the same argument as the checkpoint header's layout hashes. Thirteen named
      tests in `engine/tests/record.rs`.
- [ ] **Then play six games**, three in each seat, seeds recorded. `@4096` is the strongest
      measured setting (F3.8) and costs well under a second a move:

```bash
for s in 101 102 103; do
  ./target/release/duel52 play --encoding-slots 21 --as p0 --seed $s \
      --record games/owner-vs-gen016.jsonl \
      --opponent netmcts:models/duel52-split-gen016.d52nn@4096
done   # then the same three seeds with --as p1
```

Six games, not twenty. Twenty was the sample size for *measuring* a win rate, and the owner
does not want to spend an evening on it — reasonably, because at 0–5 the measurement is not the
question any more. Six is enough to (a) leave a recorded artefact, and (b) fix the deals so the
Phase 4 rematch is paired on the same six seeds. Commit the file; it is a few kilobytes and it
is the project's only external data.

### 4.0a What the recorded games are actually for `[ ]`

Worth being blunt about the tiers, because the intuitive answer is the wrong one:

| use | verdict |
|---|---|
| **Training data** — fine-tune on the human's moves | **No.** Six games is ~800 decisions against a replay buffer of ~6M. Even weighted 100× it is noise, and imitating one player over six games is a good way to overfit to one player over six games. |
| **Search roots** — replay the positions, run deep search on each, add the targets to the buffer | Legitimate technique, negligible at this volume. ~800 roots at 4096 sims is ~3 minutes of compute and 0.01% of the buffer. Revisit only if there are ever hundreds of human games. |
| **Diagnosis** — find the plies where the net was confident and wrong | **Yes, and this is the point.** See below. |
| **A fixed evaluation set** — score every future checkpoint on the same positions | **Yes.** Cheap, permanent, and the only non-self-referential test the project has. |

- [ ] **Diagnosis.** For each of the human's winning games, find the plies where the value head
      was confident (|v| > 0.6) in the side that went on to lose, and the plies where the
      owner would call the net's move obviously bad. Sort them into three buckets: *(i)* moves
      the net itself fixes given more search — a search-budget problem; *(ii)* moves it plays
      the same way at any budget but the value head scores wrongly — a value-function problem,
      which is what §4.2 change 7 predicts; *(iii)* moves that look fine at every budget and
      are still wrong — a blind spot, and the only category no amount of compute reaches.
      **Bucket (iii) is why a human series is worth more than another self-play table:** errors
      invisible from inside the system are exactly the ones self-play cannot label.
- [ ] **A fixed evaluation set.** Lift the positions into `testkit` form and score every
      checkpoint on them: does the value head still think it is winning at the ply where it
      actually lost? Seconds to run, and it never goes stale.
- [ ] **[code, optional] Disagreement mining — the automatic version, and it scales.** The
      human is a slow labeller. Run gen016 self-play and, at each decision, compute both the
      `@64` and the `@4096` choice; log the positions where they disagree by a wide visit
      margin. That is thousands of labelled policy errors overnight on the Mac, for free, and
      it directly measures how much of a teacher `selfplay.sims` is buying. It finds bucket
      *(i)* and *(ii)* errors in volume and bucket *(iii)* never — which is precisely why it
      complements the six human games rather than replacing them.

⚠️ The human series is the *only* external measurement in the project. Everything else is
scored against agents this project wrote, on a ladder anchored at `random`, which is why a
+1476 Elo rating and an 0–5 record against one human are not in contradiction.

### 4.1 What to rent — and it is cores, not a GPU `[ ]`

**The headline: as the code stands, a GPU changes ~1.5% of the loop's wall clock.** Two
measurements, both reproducible:

Where a generation's time goes (F3.5, 8-core M-series, 3000 games at 64 sims):

| stage | time | share | runs on |
|---|---:|---:|---|
| self-play | 6m57s | **87%** | **CPU cores**, Rust, `duel52 selfplay` |
| gate + reference matches | ~40s | 8% | **CPU cores**, Rust, `duel52 match` |
| replay + encode | 5s | 1% | CPU, Rust encoder via PyO3 |
| gradient steps | 20s | 4% | GPU (or CPU) |

And the gradient step does not need a GPU anyway — F3.11, one optimisation step at the real
shapes (`obs_dim` 4290, `action_dim` 2194, batch 512):

| trunk | 8 CPU cores | MPS GPU | 2800 steps on CPU |
|---|---:|---:|---:|
| 128 wide × 3 blocks | 8.3 ms | 5.9 ms | 23 s |
| 128 wide × 10 blocks | 13.8 ms | 9.1 ms | 39 s |

A laptop GPU is 1.4× faster than eight laptop CPU cores on a 1.2M-parameter MLP, which is what
you would expect: the model is far too small to fill a GPU. A rented A100 would be a few times
faster again and would still be shaving a minute off an hour-long generation.

**So the machine to want is a many-core CPU box.** Self-play parallelises across *games* on
threads (`--threads`) and results are identical whatever it is set to, so throughput scales
essentially linearly with cores until memory bandwidth bites. 95% of the loop is that.

**The shopping list, in preference order:**

1. **32–64 physical cores, ≥64 GB RAM, ≥50 GB working disk.** Any GPU, or none. A CPU-only
   instance is a completely legitimate buy here (AWS `c7i`/`c7a`, Hetzner dedicated, a
   Vast.ai CPU listing) and is usually the cheapest per core.
2. If the GPU is the thing that is actually available — which is likely, given the framing —
   **pick the listing by its vCPU count, not its GPU model.** On RunPod/Vast/Lambda, vCPU
   count scales with GPU count, so "1× A100, 30 vCPU" is really "a 15-core box with a GPU
   attached", and "8× H100, 192 vCPU" is a 96-core box. The second is a good buy *for its
   cores*; at 8× the price of the first it is a bad buy for its GPUs.
3. **What to ask your brother**, before booking anything:
   - `nproc` and `lscpu | head -20` — total threads, and **cores per socket** × sockets.
     `nproc` counts hyperthreads; MCTS is compute-bound, so 64 threads on 32 physical cores
     is worth roughly 1.2×, not 2×. §4.5 Stage 1 measures the real number.
   - `free -g` — RAM. The replay buffer is the biggest thing in the Python process; §4.3
     sizes it.
   - `df -h .` — **disk on the working directory**, not the root volume. Container images
     (RunPod, Docker) often ship 20 GB and bill the persistent volume separately. A big run
     writes ~10 GB of shards.
   - Is it a **VM or a container**? In a container, `nproc` may report the *host's* cores
     while cgroups allow far fewer, and Rayon will happily spawn 192 threads on a 16-core
     allowance and thrash. Always set `run.threads` explicitly to the number you measured.
   - Is it **spot / preemptible / interruptible**? If yes it can be killed with two minutes'
     notice, and §4.6's resume discipline stops being optional.
   - **How many hours, and who is watching the bill.** An idle GPU box bills exactly like a
     busy one. The single most common way to waste money here is forgetting to destroy the
     instance after the run finishes.

**Rough rates as of mid-2026 — verify current listings before booking.** A 4090-class box with 32–64 vCPU runs ~0.35–0.90/hr on Vast.ai or RunPod community; a single A100 with ~30 vCPU
runs ~1.10–1.60/hr on Lambda or RunPod secure; an 8×A100/H100 box runs 10–30/hr. A 24-hour run in the first bracket is **$10–20**, which is worth stating plainly because it is easy to assume this needs a budget it does not.

**When a GPU would start to matter**, and therefore what is *not* worth doing yet: the network
is ~80% of per-simulation cost (F3.11), so a GPU inference backend behind the batch-shaped
`Evaluator` (`DESIGN.md` §9) would in principle be the biggest single speedup available. It
needs a new crate (`candle` or `tch`), cross-game batching in `selfplay.rs`, and a new parity
test — days of work, and it buys nothing until the trunk is large enough that a CPU forward
pass dominates a determinization. Filed in Phase 6. Rent cores now.

### 4.2 What to change before the run, and why `[ ]`

Seven changes. The first two are the ones F3.7 and F3.8 actually diagnosed; 3–6 follow from the
run being longer; the seventh is the one the owner's games pointed at. Each is
`configs/train-big.toml` unless marked **[code]**.

**1. Make the promotion gate able to make a call — and correct what this file said before.**
An earlier version of this plan said to *either* raise `gate.games` to 600 *or* drop
`gate.threshold` to 0.52. The "either" is wrong, and the direction matters:

| gate | true 0.50 (no gain) passes | true 0.54 (real gain) passes | 3 refusals in a row at 0.54 |
|---|---:|---:|---:|
| 200 games, 0.55 — the first run | 7.9% | **38.9%** | **22.9%** |
| 600 games, 0.55 | 0.7% | **31%** | 33% |
| 600 games, 0.52 | 16% | **84%** | 0.4% |

More games cannot make a generation pass a bar it is genuinely below — it makes a true-0.54
candidate pass *less* often against a 0.55 threshold, because the estimate concentrates on
0.54. **The threshold is the bug; the sample size is what makes a lower threshold safe.** And
the two errors are wildly asymmetric: promoting a candidate that is really 0.50 costs
approximately nothing (the new weights are as good as the old ones), while refusing a
candidate that is really 0.54 three times in a row ends the run — which is exactly what
happened. So: `gate.games = 600`, `gate.threshold = 0.52`, `max_consecutive_refusals = 5`.
(The table reads `games` as `decisive games`, which is now the same thing: every generation of
the third run drew 0% of its self-play, so the draw compression F3.4 worried about is gone.
`min_decisive = 60` keeps the abstain rule at the same 10% of `games` it had at 200.)

A refusal is also not a wasted generation — self-play continues from the incumbent, so the
buffer grows and the next candidate is trained on more data. Stopping is only correct when
learning has genuinely plateaued, which is a claim the *reference* column makes, not the
mirror match.

**[code, optional, ~15 lines]** Better still, make the bar a confidence statement instead of a
magic number: promote when the decisive score's lower 80% bound clears 0.5, i.e.
`threshold = 0.5 + 0.84 × 0.5/√decisive`. That is 0.517 at 600 decisive games and 0.529 at
200, so it stays honest if someone changes `gate.games` later without redoing this table.

**2. Replace the saturated reference panel — this is the most valuable single change.** By
generation 19 the panel read `random=1.000 greedy=1.000` (see `runs/third.log`): it could
detect a catastrophe and nothing else. **gen016 is now available as a fixed external
opponent**, and a score against a frozen strong agent, tracked per generation, is precisely
the instrument §4.8's tripwire 2 needs:

```toml
reference = ["random", "greedy", "netmcts:models/duel52-split-gen016.d52nn@64"]
reference_games = 300
```

Keep `random` and `greedy` — they are cheap and their saturation is itself information. The
gen016 column is the run's headline chart: rising monotonically means the loop is working;
flat-or-falling while the mirror gate keeps passing is the exploitability signature. When the
new net saturates *that* too (score ≥ 0.95 for several generations), freeze the current best
as a second reference and add it. Note the panel is a **veto, never a target**.

**3. `selfplay.sims` 64 → 256.** The policy target *is* the visit distribution, so training at
64 teaches the network to imitate a search ~370 Elo weaker than the same weights produce at
4096 (F3.8). The teacher was capped. 256 is the pick rather than 1024 because the 64 → 256
step is the cheapest large gain (+141 Elo) and the return per 4× halves above 1024. Cost is
close to linear: 4× the sims is 4× the self-play time.

**4. `net.blocks` 3 → 6, `net.width` unchanged at 128 — and depth is not free.** Only 10.5% of
gen016's 949k parameters are in the residual trunk; 58% is the input projection and 30% the
policy head, both pinned by `obs_dim` and `action_dim` (F3.8). Parameter accounting says
depth is much the better buy than width, and it is — but F3.8 counted parameters, and the
sparse-input trick means parameters are not what inference costs. Measured (F3.11, 128 games,
64 sims, 8 threads, seed 1):

| trunk | params | decisions/sec | self-play throughput |
|---|---:|---:|---:|
| 128 × 3 — gen016 | 949k | 1078 | 1.00× |
| **128 × 6** | 1,049k | 721 | **0.67×** |
| 128 × 10 | 1,182k | 471 | 0.44× |
| 256 × 3 | 2,093k | 324 | 0.30× |

So 3 → 10 blocks costs +25% parameters and **2.3× the wall clock of every self-play game**;
128 → 256 wide costs 3.3×. Depth is still the better buy per unit of throughput, and the
config comment recommending width first is still wrong, but "+25% parameters" reads as free
and is not. **6 blocks at 32 cores, 10 blocks only at ≥96 cores** — the trade is against
games, and more games is what a longer run is for. If §4.5 Stage 1 shows the box is bigger
than expected, spend it on depth before width.

**5. [code, ~20 lines] A learning-rate schedule.** `train.lr` is a constant 2e-3 through
AdamW for the whole run, which is fine for 19 generations and not for 50. Add a piecewise
decay keyed to **generation index**, not to an optimiser step counter — `--resume` rebuilds
the `Trainer` from the checkpoint and the step count does not survive it:

```
generations 0–40%   lr × 1.00
generations 40–75%  lr × 0.25
generations 75%+    lr × 0.0625
```

**[code, minor]** While in `trainer.py`: `--resume` currently reloads weights only, so the
AdamW first and second moments are lost on every restart. On a preemptible box that is a few
hundred steps of re-warm per interruption. Persisting them next to `best.d52nn` is cheap.

**6. Widen the replay window, and scale the fitting to the data.** §4.3 gives the arithmetic.
`temperature_decisions` stays at **24** for the main run: this plan previously called 24 of
~136 decisions "thin", and it is worth noting that AlphaZero samples ~19% of a game for the
same reason (clean value targets), so 18% is not obviously wrong. If Stage 1 leaves spare
compute, that is the cheapest ablation on the list — one short run at 40 against one at 24.

**7. Watch the value head, because it is the half that stopped learning.** The owner's
description of gen016 — reasonable move to move, no long-term plan, occasional obviously bad
plays even at 8192 simulations — has an exact signature in `runs/third/log.jsonl`:

| gen | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| policy loss | 2.206 | 2.168 | 2.155 | 2.124 | 2.081 | 2.021 | 1.999 | 1.961 | **1.942** |
| value loss | 0.482 | 0.504 | 0.519 | 0.487 | 0.499 | 0.479 | 0.464 | 0.490 | **0.506** |

**The policy kept improving for the whole run. The value head plateaued around generation 11**
and then oscillated around 0.49 — roughly half the outcome variance explained, on a ±1 target.
This matters because of where the two are used: search resolves the next handful of decisions
accurately, and *everything past its horizon is the value head*. A Duel 52 game is ~130
decisions and the §7 seam sits at ply 25, so from the mid-draw-phase the thing that decides
whether hoarding a card is worth it is ~45 decisions away — far outside any tree of 8192
simulations over a branching factor of ~21, and redetermizing every simulation on top. So the
agent executes a plateaued long-run judgement very precisely. That is the described symptom.

The levers are not the same as the policy's, and one of them cuts the other way:

- **`sims` improves the policy target; *games* improve the value target.** Each game
  contributes one outcome, however many decisions it has. §4.3's budget buys 5× the games and
  4× the sims at once, so there is no conflict here — but it is the reason not to trade games
  away for sims as an earlier draft of this plan proposed.
- **Depth helps both**, since the trunk is shared — change 4.
- **[code] Log a held-out value MSE**, not just the training-batch number. The figures above
  are computed on batches drawn from a replay window that slides underneath them, so a flat
  curve is ambiguous between "learned all it can" and "the task got harder". Hold out one
  generation's shard and score against it every generation. Small, and it is what makes this
  claim testable rather than suggestive.
- Resist raising `value_weight` as a first move. If the value head is capacity- or
  data-limited, upweighting its loss trades policy quality for nothing.

⚠️ **This changes seven things at once, which means a failure will not say which one caused
it.** That is a deliberate trade — the run is a scale-up, not an ablation — and the mitigation
is the per-generation gen016 column from change 2 plus the held-out value MSE from change 7,
which together say *during* the run whether it is working. If it is not, §4.5 Stage 4 runs the
grid.

### 4.3 Sizing the run — the arithmetic `[ ]`

From F3.4 and F3.11, one formula covers it:

```
self-play games/hour  ≈  28,000 × (cores / 8) × (64 / sims) × trunk_factor
```

with `trunk_factor` from §4.2's table (1.00 at 128×3, 0.67 at 128×6, 0.44 at 128×10). The
28,000 is measured, and F3.7's actual run — 57,000 games in 1.94 h at 8 cores, 64 sims,
128×3 — comes out at 29,400, so the formula is calibrated on a real run rather than on a
micro-benchmark. Worked examples:

| box | sims | trunk | games/hour | 24 h |
|---|---:|---|---:|---:|
| 8 cores (this Mac) | 64 | 128×3 | 28,000 | — |
| 8 cores (this Mac) | 256 | 128×6 | 4,700 | 113k |
| 16 cores | 256 | 128×6 | 9,400 | 225k |
| 32 cores | 256 | 128×6 | 18,800 | 450k |
| **64 cores** | **256** | **128×6** | **37,500** | **900k** |
| 64 cores | 256 | 128×10 | 24,600 | 590k |

A 24-hour run at 64 cores is **~720,000 self-play games after evaluation overhead** — 13× the
first run's games, each with 4× the search per decision, so roughly **50× the teacher
compute**. That is a real scale-up rather than a longer version of the same thing.

**Generation size, and why it is not "as big as possible".** A generation is one step of
policy improvement, and the run needs a lot of steps, not a few enormous ones. Target
**40–50 generations** and divide:

At 64 cores, 15,000 games per generation → 24 min of self-play, plus ~5 min of everything
else = **~29 min a generation**, so 24 hours is ~48 generations. Per generation:

- **Samples:** `games × 136 decisions ÷ sample_stride` = 15,000 × 136 ÷ 2 = **1.02M**
- **RAM:** ~1.6 KB a sample, so `buffer_generations = 6` is 6.1M samples ≈ **10 GB**. Fits
  in 64 GB with room; raise `sample_stride` to 3 if RAM is tight, it costs little signal.
- **`buffer_samples`:** the cap must be raised from 700,000 or it silently truncates the
  window to two-thirds of one generation. Set it to `buffer_generations × samples/gen`
  = **6,500,000**.
- **`steps_per_generation`:** keep the first run's ratio of sample-presentations to new
  samples (1200 × 512 ÷ 220k ≈ 2.8), which at 1.02M new samples and `batch_size = 1024`
  is **2,800 steps** — about a minute of CPU, extrapolating F3.11's per-step figures to the
  doubled batch. Fitting stays under 3% of the loop, so err high rather than low and let the
  gate catch overfitting.
- **Disk:** ~270 MB of shard a generation, so ~13 GB for the run. Shards are regenerable in
  principle but not cheaply; keep them until the findings are written.
- **`run.seed`:** the loop advances the seed by `selfplay.games` each generation, so this run
  spans `seed .. seed + 720,000`. Use **2,000,000** — the first run occupies 1,000,000–1,057,000
  and a collision would silently re-deal games the net has already trained on.

So `configs/train-big.toml`, for a 64-core box:

```toml
[game]     variant = "split", encoding_slots = 21, stalemate_value = 0.0
[net]      width = 128, blocks = 6, value_hidden = 128
[selfplay] games = 15000, sims = 256, temperature_decisions = 24   # rest unchanged
[train]    batch_size = 1024, steps_per_generation = 2800, sample_stride = 2,
           buffer_generations = 6, buffer_samples = 6500000, lr = 2.0e-3 + schedule
[gate]     games = 600, threshold = 0.52, sims = 256, min_decisive = 60,
           reference = ["random", "greedy", "netmcts:models/duel52-split-gen016.d52nn@64"],
           reference_games = 300, reference_tolerance = 0.05, max_consecutive_refusals = 5
[run]      generations = 60, hours = 24, seed = 2000000, threads = <measured>, engine = "..."
```

`generations = 60` is a backstop; `hours` is what binds, and the loop stops *before* a
generation that would overrun. Note `gate.sims = 256` matches `selfplay.sims`: the gate should
measure the weights at the budget they are being trained for, not at 64.

For a smaller box, scale `games` by the formula and keep `generations` in the 40–50 band —
fewer, larger generations is the wrong direction.

### 4.4 Code tasks before renting `[ ]`

Everything here is small, and all of it should be done and committed **on the Mac**, because
debugging on a metered box is the expensive way to do it.

- [x] **`duel52 play --record` and `duel52 replay`** (§4.0). First, because they are the
      instrument for the exit criterion and they cost nothing to run.
- [x] **Held-out value MSE, logged per generation** (§4.2 change 7). `train.holdout_samples`
      carves a fixed prefix off generation 1's shard, never trains on it, and scores it every
      generation into `log.jsonl` (`holdout_value_mse`, `holdout_policy_loss`) and both the
      readout and the summary table. **Fixed, not rolling** — a holdout that slides with the
      window has the same ambiguity the training-batch number has, and the point is a
      yardstick that does not move. The cost is that it drifts off-policy as the net improves:
      it measures the same question being answered better, not the current question. The carve
      happens on every path into the buffer, including the refill after a `--resume`, because
      "held out except after a restart" is not a distinction anyone would notice in a log file.
- [x] `configs/train-big.toml` per §4.3, with the reasoning in comments as `train-fast.toml`
      does. `python -m duel52.train check --config configs/train-big.toml` must pass.
- [x] LR schedule in `py/duel52/train/trainer.py`, keyed to generation index (§4.2 change 5).
      `train.lr_schedule = [[from_generation, multiplier], ...]`; empty means a constant rate,
      so every Phase 3 run reproduces unchanged. A schedule that does not ascend is rejected
      by the config loader rather than silently running the decay backwards.
- [x] Persist AdamW moment state across `--resume` (§4.2 change 5, minor). Saved every
      generation to `checkpoints/optimizer.pt`, whether or not the candidate was promoted — a
      refusal rolls the weights back and deliberately keeps the momentum.
- [x] Add the gate's decisive count and its 95% interval to the per-generation readout and to
      `log.jsonl`, so "did the gate have power" is answerable from the log without recomputing
      it. This is the single thing that would have made F3.7 obvious while the run was live.
      `python -m duel52.train check` now also prints the gate's power *before* the run: at
      600 games and 0.52, a true-0.500 candidate passes 16% and a true-0.540 candidate 84%,
      which is §4.2's table computed from the config rather than copied out of it.
- [x] Confirm a `netmcts:<path>@<sims>` string works as a `gate.reference` entry end-to-end.
      Four generations of a 20-game local run, including a `--resume` in the middle.
- [x] **[not in the original list] `--init-from <checkpoint>`**, so a run can start from a
      shipped checkpoint instead of a random init. Needed because a two-hour laptop run cannot
      reach gen016 from scratch — see §4.5 Stage 0 — and useful on the box for continuing a
      run whose config changed. It refuses a checkpoint whose trunk disagrees with `[net]`:
      the shape comes from the checkpoint, which is correct on `--resume` and a trap here,
      where someone asking for 6 blocks while inheriting a 3-block checkpoint would silently
      get 3. It also scores the starting checkpoint on the reference panel once, before
      generation 1, so the veto has a high-water mark from the first candidate rather than the
      second, and so the run's reference column has a row to be read against.
- [ ] **Optional, only if there is more than one machine:** let a generation consume several
      shards, so `duel52 selfplay` can run on N boxes with disjoint seed ranges and the
      trainer concatenate them. This is the natural way to use a second box and is a
      genuinely small change to `loop.py` — the shard reader already validates layout hashes.

### 4.5 The staged plan `[ ]`

**Stage 0 — done, 2026-09-04, and it answers something.** `FINDINGS.md` F4.1: eight
generations, and generation 6 beats gen016 by **0.615 ± 0.047 over 400 games at equal
simulations, about +81 Elo**. Change 3 is real and it is not enormous. On the frozen ladder it
fits at +1788 ± 58 (F4.2) — a number that needs reading with care, because roughly three fifths
of the apparent gain over gen016's +1476 is the larger search budget and a scale stretch rather
than better weights. Three things for whoever starts Stage 2:

- **`train.epochs_per_generation`, not `steps_per_generation`.** A constant step count is 4.1
  epochs of the buffer a run holds at generation 1 and 0.9 of a full one, and four passes over
  a quarter-full buffer destroyed the first candidate (0.349 against its own starting weights).
  From-scratch runs are immune — an overfitted first generation still beats `random` — so this
  bites warm starts only, and `train-big.toml` is a from-scratch run. Set it anyway.
- **Set the LR schedule's tiers from the generations a run will finish, not from the
  `generations` backstop.** This run's last tier arrived at generation 7 of 8 and froze it.
- **Measure with a direct head-to-head, not with the ladder.** F4.2: the ladder has saturated
  and costs five times as much to say less. Its remaining job is rating the hand-written rungs.

The rest of this section is the original instructions, still current:

`configs/train-2h.toml`:

```bash
.venv/bin/python -m duel52.train run --config configs/train-2h.toml \
    --run-dir runs/fourth --init-from models/duel52-split-gen016.d52nn
```

The mechanical rehearsal this stage used to be — the config parses, the LR schedule fires, the
`netmcts` reference opponent loads, the buffer does not blow up RAM, `--resume` survives a
Ctrl-C — is done and is covered by tests and by a four-generation smoke run. What is left is
worth two hours because it is a measurement rather than a rehearsal.

**It warm-starts from gen016 and it is `128 × 3` because of that.** From a random init, two
hours at 256 simulations buys ~8,600 self-play games against the 57,000 that produced gen016,
so the run would finish weaker than the checkpoint it is being compared to and the comparison
would measure the budget rather than the change. Warm-started, every generation's mirror gate
*is* the comparison and the reference panel carries `netmcts:gen016@256` — equal simulations,
which is exit criterion 1 measured every generation instead of once at the end.

The price is that change 4 is not in it: 6 blocks cannot inherit a 3-block trunk. So Stage 0
answers **does training on a 4× deeper teacher move gen016** — which is change 3, the one F3.8
diagnosed as the first run's ceiling — and says nothing about depth. That is the right half to
test first on a machine this size, because F3.11 prices depth at 0.67× the games and the games
are what two hours is short of.

Read it as: the gen016 column rising across generations is the loop working; flat while the
mirror gate keeps passing is the exploitability signature (§4.8 tripwire 2); the held-out value
MSE falling is change 7 having been the right diagnosis. If the gen016 column is flat after
eight generations, the teacher was not the binding constraint and `train-big.toml`'s 24 hours
should be spent knowing that.

**Stage 1 — one paid hour, measuring, before committing to 24.** In order:

```bash
# 1. Toolchain + build. The README has the rustup line.
cargo build --release && cargo test          # 325 tests. This is the handoff proof:
                                             # the box computes the same game or it does not.
python3 -m venv .venv && .venv/bin/pip install -q maturin pytest torch numpy
.venv/bin/maturin develop --release
.venv/bin/python -m pytest py/tests -q       # includes test_parity.py: this box's PyTorch
                                             # and this box's Rust agree on the forward pass

# 2. Determinism against the Mac. Same seed, same config, same numbers — the engine is
#    integer-only, so any difference is a real bug and not floating point.
./target/release/duel52 stats --games 20000 --seed 1 --markdown

# 3. How many cores does this box REALLY have? Run each and read games/sec.
for t in 1 8 16 32 64; do
  ./target/release/duel52 selfplay --checkpoint models/duel52-split-gen016.d52nn \
      --out /tmp/t$t.d52sp --games $((t*8)) --sims 64 --encoding-slots 21 \
      --threads $t --stalemate-value 0.0 --quiet
done
```

Throughput should climb roughly linearly and then flatten; **the flattening point is the real
core count**, and it is what `run.threads` should be set to. This measures gen016's 128×3
trunk, which is deliberate — it is the same shape F3.11 and F3.4 measured, so the number is
directly comparable and `trunk_factor` converts it to whatever trunk the run uses. Put the
whole curve in `FINDINGS.md`; it is what makes §4.3's formula portable to the next box. Then
re-derive `selfplay.games` from the measured number, and only then start the long run.

**Stage 2 — the run.** In `tmux` (§4.6), 24 hours, checking the gen016 reference column once
or twice rather than watching it.

**Stage 3 — measure, on the box, while it is still rented.** All of §4.7. These are CPU jobs
and the rented cores make them minutes instead of hours; a ladder at 400 games is ~26 min on
8 cores.

**Stage 4 — only if Stage 2 disappointed.** A grid instead of a run: four 6-hour runs at
`blocks` 6 vs 10 × `sims` 128 vs 256, each pinned to a quarter of the cores with
`run.threads` and `taskset`. Four data points beat one when the single point is confusing.

**Then destroy the instance.** `rsync` first (§4.6).

### 4.6 Operating a rented box `[ ]`

- **`tmux` or the run dies with your SSH connection.** `tmux new -s duel52`, start the run,
  `Ctrl-b d` to detach, `tmux attach -t duel52` to come back. This is the mistake everyone
  makes once.
- **Everything the run needs to survive is `--run-dir`.** After any interruption:
  `python -m duel52.train run --config configs/train-big.toml --run-dir runs/big --resume`.
  It refills the replay window from the shards on disk and does not re-promote from scratch.
- **Sync the small things off the box every few hours**, because a spot instance can vanish:
  `rsync -avz box:duel52/runs/big/checkpoints/ runs/big/checkpoints/` plus `log.jsonl`. The
  checkpoints are 4.7 MB each and the log is kilobytes; the shards are 13 GB and can stay.
- **Watch four numbers only:** the gen016 reference score (must rise), the **held-out value
  MSE** (must fall — it is the half that plateaued last time, §4.2 change 7), the self-play
  draw rate (must stay ~0%; anything else means an equilibrium is forming — F3.6), and the
  gate's decisive score with its interval.
- **`nohup` is not a substitute for `tmux`** if you want to read the progress output.
- **Destroy the instance when the run is done**, and check the provider's dashboard rather
  than trusting that stopping the container stopped the billing. Persistent volumes usually
  bill separately from compute.

### 4.7 What Phase 4 must measure `[ ]`

- [x] **The frozen ladder** — run 2026-09-04 at 400 games, `FINDINGS.md` F4.2. **And it is the
      last time this table is worth running as it stands.** gen006@256 fits at **+1788 ± 58**
      against ±13–15 for every hand-written rung, and scores 1.000 against the anchor: a rating
      driven by a handful of losses is an extrapolation, not a measurement.
      This item asked for gen016 as an extra rung "so the two trained nets are on one scale",
      and that was run without it — but **the fix is not to re-run with gen016 added.** A
      400-game head-to-head puts them on one scale at ±0.047 for five minutes, where the ladder
      costs ~26 minutes to produce ±58. Keep the ladder for the hand-written rungs, which it
      still measures well; use a direct match for anything above `ismcts:800`.
      ⚠️ Ladder Elo is **not comparable across two fits.** Every hand-written rung moved up
      35–70 points between F3.7's table and this one without changing by a line of code,
      because Bradley–Terry fits the whole graph and pulling the top agent away stretches the
      scale. Compare gaps over a common rung, never row to row. F4.2 does the arithmetic.
- [x] **Head-to-head vs gen016** at equal sims, 400 games — exit criterion 1. **0.6150 ±
      0.0474** (W244 L152 D4), about +81 Elo. Real, and below the 0.70 bar; see F4.1.
- [ ] **The search-scaling sweep repeated** (F3.8's shape: `netpolicy` → @64 → @256 → @1024 →
      @4096, ≥300 games a step). Whether the new net absorbs *more* search than gen016 did is
      the single most informative number the run produces, and it feeds §4.8 directly.
- [ ] **`probe` against the same roster** as `FINDINGS.md`'s strong-play table, so hand@unlock,
      lane concentration and the flip-timing curve can be compared generation to generation.
      If the behavioural statistics moved, the strategy moved, and Phase 5's findings need
      re-checking against the new net.
- [ ] **The owner's six-game rematch on the same six seeds** as §4.0, recorded — exit
      criterion 2. Paired on the deal, so it is worth more than six independent games; and
      recorded, so a loss is diagnosable through `duel52 replay` rather than remembered.
- [ ] **Cost and wall clock**, recorded with the config. Someone repeating this needs to know
      what an hour of what hardware bought.

### 4.8 The tripwire — when to stop scaling and change method

A Phase 4 judgement, not a training one, and the answer today is *not yet*. AZ over
determinized ISMCTS has a real ceiling in an imperfect-information game: no equilibrium
guarantee, and no way to learn to conceal or signal deliberately. The project already knows
fusion is punishing — F2.2 is PIMC collapsing under exactly that.

But `netmcts` shows **no fusion signature**. The same test that exposed PIMC — more sampling,
does it buy anything — gives PIMC nothing at 8× worlds and gives `netmcts` +141, +156 and +69
Elo across three successive 4× steps (F3.8). Per-simulation determinization is doing its job.

Switch to R-NaD (Phase 7) when **either** of these appears, and not before:

1. **Search still scales but the network stops absorbing it**, across a run whose gate
   actually has power. That is a representation limit, not a compute one. §4.7's repeated
   sweep is the measurement.
2. **Self-play looks healthy while the score against a fixed external opponent flattens or
   falls.** That is the exploitability signature, and no amount of compute fixes it. §4.2's
   change 2 builds this detector into the loop, per generation, which the first run did not
   have — its external opponents were saturated by generation 7.
   ⚠️ **Keep a detector that can still detect.** As of F4.2 the frozen ladder has saturated
   too: `random`, `greedy`, `pimc:8x1` and `flatmc:600` are all beaten at ~1.000, and only
   `ismcts:800` still carries any signal. Every fixed *hand-written* yardstick this project
   owns has now been passed, so this tripwire only works against a frozen *trained* net —
   promote the current best into `gate.reference` whenever the incumbent reference clears
   0.95 for several generations, or the detector reads healthy because it reads nothing.

Neither is present as of gen016. The P0 self-play drift (51.4% → 56.7% over the first run)
looked like signature 2 and turned out not to be: it does not survive without exploration
noise, and `greedy` shows the same tilt (F3.9).

A third condition, weaker and more likely: **the run succeeds and the owner still wins.** That
is not a reason to change learner — it is a reason to look hard at what the human does that
the agent does not, which is Phase 5's job and the most interesting outcome on this page.

---

## Phase 5 — Extract the insight `[~]`

**This is the actual point of the project.** It is a separate job from training and should
not be treated as a victory lap.

It started earlier than planned, and by accident. Running `probe` against gen016 to
characterise it produced three of the results on the original list, because **the trained
agent is the first player in this project capable of the behaviour the hypotheses are about.**
Every Phase 2 rung was measured and found flat on H2 and H3; that was never evidence about the
game, it was evidence about the agents. The lesson generalises to everything below: a null
from an agent that *cannot do the thing* is not a null.

⚠️ **Everything banked here is measured on gen016 and inherits it.** If Phase 4's net moves
the behavioural statistics in `probe` (§4.7), each of these needs re-checking on the new net
before it is written up as a claim about *the game* rather than about one agent.

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
- [ ] **Where does the human beat the agent?** New, and it is the best question on this list
      as long as §4.0's answer is "the human wins". Each recorded human game is one in which a
      +1476-Elo agent lost to a person, which is a far richer signal than another self-play
      table — and §4.0a's bucket *(iii)*, the moves that look fine to the agent at every search
      budget and are still wrong, is a class of error self-play cannot label at any scale.
      Whatever those turn out to be is a **finding about Duel 52**, not only about the agent:
      a systematic error a strong search makes is a place the game rewards something the
      search's own evaluation cannot see.
- [ ] **Deliverable:** written findings + an interactive page

---

## Phase 6 — Verification `[ ]`

Moved out of Phase 3, where these were filed as training tasks. They are not: they are how the
project finds out whether the trained policy is *good* rather than merely *better than the
things it was trained against*. Phase 4's tripwire needs at least the second of them.

- [ ] **Duel52-mini** — a scaled-down variant, in OpenSpiel, small enough for exact CFR.
      Validate the loop against the exact equilibrium before trusting any claim that the full
      game's policy is near-optimal.
- [ ] **Local best-response** as the exploitability proxy on the full game. This is the
      instrument for tripwire 2, and the honest version of "how strong is it, really".
- [ ] **GPU inference behind the batch-shaped `Evaluator`** (§4.1). Only worth building once
      the trunk is large enough that a CPU forward pass dominates a determinization — F3.11
      is the measurement that decides it. Needs `candle` or `tch`, cross-game batching in
      `selfplay.rs`, and its own parity test against `engine/src/nn/mlp.rs`.
- [ ] **Multi-machine self-play** (§4.4, optional) if more than one box is ever available.

---

## Phase 7 — R-NaD on real compute `[ ]`

Only on a tripwire, per §4.8.

- [ ] Swap the learner; engine and encoders unchanged
- [ ] Validate on Duel52-mini against exact equilibrium first
- [ ] Scaled run on borrowed compute
- [ ] Compare the approximate-Nash policy against the AZ policy — where do they disagree,
      and is AZ exploitable in those spots?

Estimate: a game this size should not need a cluster. One strong many-core box, or a single
multi-GPU day once inference is batched, is likely sufficient.

---

## Guiding constraints

- Owner has limited time and is delegating implementation. Default to making a defensible
  call, marking it `[ASSUMED]`, and flagging it — rather than blocking.
- Everything must run locally on an M-series Mac and scale to rented hardware via config
  alone. `train.device = "auto"` and `run.threads` are the whole handoff.
- **Elo on this project's ladder is an internal coordinate.** The anchor is `random` and every
  rung was written here. A number from it is a distance, not an absolute; the human series
  (§4.0) is the only external measurement, and it should stay in every strength claim.
- Insight beats strength. A slightly weaker agent whose play we can *explain* is worth more
  here than a stronger one we cannot.
