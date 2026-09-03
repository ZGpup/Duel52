# Duel 52 — Roadmap

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

**Current phase: 1 (complete, pending the owner's rules review).** The engine, the CLI, the
PyO3 bindings and the baseline statistics all exist. The one remaining item is the exit
criterion itself: the owner plays a few games and confirms the rules are right.

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

### Three `[ASSUMED]` calls made while implementing

Per `CLAUDE.md`'s "make a defensible call, document it, flag it". Each is flagged in the
code and pinned by a named test; none is load-bearing enough to block on, and all three are
cheap to flip if the owner disagrees.

- **A face-down 9 can be frozen by a 6.** Nimble is a power, and §6 says powers are inert
  face-down; the rulebook's "cannot freeze a 9, ever" reads as being about *timing* (a 9
  already in the lane is still immune), not about face-up-ness.
- **A 9 deals 2 damage to a face-down Jack.** The doubling keys on the target being
  physically a Jack, as hit points do, rather than on the Jack's taunt being live.
- **A 10 whose twinstrike hits two 8s takes 1 retaliate from each, and dies.** §6 says "any
  card that attacks this 8 takes 1 damage" and the 10 attacked both, so the damage adds.

---

## Phase 2 — Baselines, no learning `[ ]`

- [ ] Random agent
- [ ] Greedy heuristic agent (hand-written evaluation)
- [ ] Flat Monte Carlo
- [ ] PIMC (perfect-information Monte Carlo) — the control
- [ ] SO-ISMCTS with random rollouts
- [ ] Round-robin Elo ladder, frozen as the permanent benchmark
- [ ] **Deliverable:** first real strategic observations logged to `FINDINGS.md`

**Worth flagging:** this phase may already answer a large share of the original question.
A competent ISMCTS bot with zero training will expose lane-allocation patterns, flip
timing, and the first-player edge. Do not rush past it to get to the neural net.

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
