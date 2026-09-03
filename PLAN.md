# Duel 52 — Roadmap

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

**Current phase: 1 (not started).** Rules are specified; no code exists yet.

---

## Phase 0 — Specification `[x]`

- [x] Fetch and interpret the published rules
- [x] Confirm no prior art exists (no engine, bot, solver, or strategy analysis published)
- [x] Resolve the major rules ambiguities with the owner
- [x] Write `game_rules.md`, `DESIGN.md`, `CLAUDE.md`, `OPEN_QUESTIONS.md`

---

## Phase 1 — Engine + rules validation `[ ]`

The only phase that needs meaningful owner input.

- [ ] Rust crate: state, legal-action enumeration, action application, terminal detection
- [ ] All three configurations behind one config flag: base / split-deck / mirrored-removal
- [ ] `two_power: bottom | discard` flag for the §10a house rule, so the parity claim behind
      it is measurable rather than assumed
- [ ] Seeded determinism — same seed + config produces an identical game
- [ ] One named test per ruling in `game_rules.md` (e.g. `rule_6_king_reactivates_ace_grants_one_action`)
- [ ] Edge-case tests: 8 × 9 interaction, 10 blocked by 9/J, pair vs 8 double retaliate, 3-trap resurrection, Queen breaking a pair, 5 and 7 reaching base cards post-unlock
- [ ] Edge-case tests, second batch (all settled 2026-09-03): 10 vs two Jacks is 1+1 but 10 vs
      two 9s is 1 to one; 9-pair deals 4 to a Jack and takes no retaliate from an 8; 10-pair
      splits 1+1 and consolidates to 2 when blocked; a 5 skips frozen cards but a King still
      reactivates them; King resets an Ace's attack counter rather than stacking it; the 2 is
      pile-neutral so turns-to-unlock is invariant; mutual lane win via retaliate is a draw
- [ ] PyO3 bindings
- [ ] Text CLI so the owner can play the engine and spot-check rules
- [x] Resolve outstanding items in `OPEN_QUESTIONS.md` — done 2026-09-03, nothing open
- [ ] **Deliverable:** random-vs-random statistics — game length distribution, first-player
      win rate, how often games reach the stalemate cutoff, across all three variants

**Exit criterion:** the owner plays a few games against the CLI and finds no rules errors.

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
