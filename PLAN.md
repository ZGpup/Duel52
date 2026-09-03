# Duel 52 — Roadmap

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

**Current phase: 3 (not started).** Phase 2 is complete: five agents, determinization, a
frozen Elo ladder and the first strategic measurements are in `FINDINGS.md` F2.

**Still open from Phase 1, and it is the owner's:** the exit criterion. Nobody has played
the engine yet and confirmed the rules by hand. That is now a better-value hour than it was
— `duel52 play --opponent ismcts:2000` gives an opponent that actually resists, so a rules
error is likelier to show up as something that looks *wrong* rather than as noise.

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

## Phase 3 — Neural self-play `[ ]`

- [ ] Observation + action encoders (`DESIGN.md` §4–5)
- [ ] Residual MLP with policy + value heads
- [ ] AZ-style loop: ISMCTS self-play → replay buffer → train → gated evaluation
- [ ] Checkpointing, resumability, config-driven throughout
- [ ] **Duel52-mini** in OpenSpiel; validate the loop against exact CFR before trusting it
      on the full game
- [ ] Local best-response as the exploitability proxy
- [ ] **Deliverable:** a trained agent that clearly beats the Phase 2 ladder, with an Elo
      table and an LBR number

---

## Phase 4 — Extract the insight `[ ]`

**This is the actual point of the project.** It is a separate job from training and should
not be treated as a victory lap.

- [ ] Learned rank values — what is each card actually worth?
- [ ] Opening action-triplet frequencies
- [ ] Flip-timing curves: when does revealing beat holding information?
- [ ] Lane allocation: does optimal play concentrate on two lanes, and when does it commit?
- [ ] Hand size at pile-empty — quantify hand-as-defensive-resource (see hypotheses in `FINDINGS.md`)
- [ ] Tempo value of the Ace and the King+Ace line
- [ ] First-player advantage with error bars, per variant
- [ ] Probe the value net on hand-constructed positions
- [ ] **Deliverable:** written findings + an interactive page

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
