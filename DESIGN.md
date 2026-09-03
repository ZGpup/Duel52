# Duel 52 — Engine and Model Design

Technical decisions and their rationale. Revise freely; note *why* when you change
something, since later phases depend on these choices.

## 1. Why this architecture

Duel 52 is a **two-player zero-sum imperfect-information stochastic game** with:

- a **small action space** (tens of legal micro-actions per decision),
- **short games** (~40–60 turns),
- **cheap state** (a few hundred bytes),
- and **hidden information that never fully resolves** (10 cards removed unseen).

That profile says: spend the engineering budget on a *fast, exact* engine and an
*information-set-correct* search, not on a large network. The state space (order 10²⁰+
information sets) rules out exact solution, so "optimal" here means **approximate
equilibrium**, validated by best-response probes rather than proven.

## 2. Stack

| Layer | Choice | Rationale |
|---|---|---|
| Engine | **Rust**, exposed to Python via **PyO3** | Rule interactions are dense with conditionals (8-retaliate × 9-nimble × 10-twinstrike × J-taunt). Rust gives correctness and the throughput self-play needs. |
| Search | **SO-ISMCTS** with PUCT | The correct MCTS family for hidden information. |
| Network | **PyTorch**, residual MLP | State is small; a big net is not the bottleneck. |
| Training | **AlphaZero-style self-play**, then **R-NaD** | AZ gets strong play fast but is exploitable in imperfect-info games. R-NaD (DeepNash's algorithm) is the defensible route to approximate Nash. |
| Validation | **OpenSpiel** wrapper + a **mini variant** | Lets us compute *exact* equilibrium and exploitability on a scaled-down game. See §7. |

## 3. State representation

Canonical, engine-side:

- **Lanes**: 3, each with two sides. Slots ordered by play order, compacted on death.
  Encoding cap of **16 slots per side per lane** (rules impose no limit; the engine asserts
  rather than silently truncating).
  - This started at 8, which `FINDINGS.md` F1.7 rejected on random-play evidence (20 cards on
    one side) and F2.7 then partly restored: competent self-play peaks at 8–12, and it is
    *random* play that sprawls. 16 sits above every value any agent has produced **in
    self-play** and well under the theoretical 21. Re-measure the **distribution** — not the
    maximum, which only grows with the sample — against the Phase 3 agent before tightening
    it. `encode::observed_max_slots()` is the running high-water mark for that.
  - **`FINDINGS.md` F3.1 is the caveat that matters**: sprawl is set by the *weakest* agent
    in a pairing, and F2.7 measured self-play only. A net played against `random` reaches
    13–17, so 16 sits *inside* that distribution rather than above it, and the ladder — which
    keeps `random` as its permanent anchor rung — asserts. Self-play stays at 10. The encoding
    bound is now two separate questions: what self-play needs, and what the *evaluation
    ladder* needs.
- **Card instance**: `rank` (0–12), `face_up`, `is_base`, `entered_as_base`, `damage`,
  `frozen_until_turn`, `attacks_used`, `attack_allowance`, `pair_id`.
  - `is_base` vs `entered_as_base`: a Queen-moved base card stops being a base card but
    stays unreadable by its owner (`game_rules.md` §3). `entered_as_base` is what gates the
    owner's free look; `is_base` gates targeting and lane bookkeeping. Collapsing them into
    one flag silently grants a repeatable 1-action peek at your own base — the wrong game.
  - `attacks_used` / `attack_allowance` rather than an `attacked_this_turn` bool: a freshly
    flipped Ace has allowance 2, and a King reactivation **resets** `attacks_used` to 0
    (§6). A bool cannot represent either.
  - `frozen_until_turn` is per card, not per lane — freeze does not catch later arrivals and
    travels with a Queen-moved card (§8).
- **Per player**: hand (rank multiset), draw pile, discard, known-hidden-info map.
  - The draw pile is **ordered**, not a multiset: the 2's scry (`game_rules.md` §10a) puts a
    known card at a known position — the bottom — and the player who put it there knows both.
    That is persistent private information about a *future draw*, and in the base game about a
    card the **opponent** may draw instead. A multiset pile cannot represent it.
- **Global**: turn number, actions remaining, base-cards-unlocked flag, quiet-turn counter.
- **Knowledge tracking**: which hidden cards each player has seen (own played face-down
  cards; anything revealed by a 4). This is *per-observer* and must be part of the state,
  not derived — Foresight information is private and persistent.

Suits are dropped entirely (see `game_rules.md` §1). In split-deck mode, color is deck
ownership, tracked per player, not per card.

## 4. Action encoding

Each of the three per-turn actions is its own decision node. Sub-choices opened by a power
are **separate decision nodes that cost no action**. This keeps the branching factor in the
tens rather than the thousands, and is the single most important encoding decision.

Fixed policy head, legality-masked, phase-conditioned. `L = 3` lanes, `S = 16` slots
(`config.encoding_slots`), `R = 13` ranks — everything is derived from config, so
Duel52-mini shrinks the head rather than misaligning it:

| block | formula | S=16 | engine `Action` |
|---|---|---:|---|
| `PLAY(rank, lane)` | `R·L` | 39 | `Play { rank, lane }` |
| `FLIP(lane, slot)` | `L·S` | 48 | `Flip { lane, slot }` — subsumes the old `FLIP_UNKNOWN` |
| `ATTACK(lane, atk, tgt)` | `L·S·S` | 768 | `Attack { lane, attacker, target }` |
| `PAIR(lane, a<b)` | `L·S(S−1)/2` | 360 | `DeclarePair { lane, slot_a, slot_b }` |
| `CHOOSE_SLOT(side, lane, slot)` | `2·L·S` | 96 | `Peek` / `ResolveNext` / `MoveHere` / `SplitTarget` |
| `CHOOSE_RANK(rank)` | `R` | 13 | `GiveBack { rank }` |
| **total** | | **1324** | |

Implemented in `engine/src/encode.rs`.

**There is no `PASS` block, and the head went 1325 → 1324 (2026-09-03).** `game_rules.md` §4
makes actions mandatory, so a turn with none of the four §4 actions available is not a
decision anybody makes — `apply.rs`'s `skip_turns_with_nothing_to_do` ends it, and the
position is never handed to a caller. **Every logit in this head is therefore something a
player chooses**, which is the property that makes a policy target a distribution over
choices rather than a mixture of choices and bookkeeping.

The alternative was to keep a dead logit to preserve the layout hash. That is the wrong
trade: the hash exists to catch drift between the trained function and the evaluated one,
not to be protected from an intentional change. Removing the block shifts `CHOOSE_SLOT` and
`CHOOSE_RANK` down by one, so the hash moves and every checkpoint written before this date
is refused with a one-line `action_dim` error — which is the machinery working. Regenerate
with `python -m duel52.nn init`.

The engine's own invariant moved with it. `legal_actions()` is empty **only when the game is
over** for any position reached by playing, and `GameState::debug_check_playable` asserts it
after every action. Positions built by `testkit` are exempt: they are constructed rather than
played, and are frequently built to be queried rather than moved in.

**Why this replaced the rank-keyed table (2026-09-03).** The original head keyed `FLIP` and
`PAIR` by rank, on the reasoning that same-rank cards are interchangeable to the player
choosing. They are not, and `engine/src/action.rs`'s module header already said so:

- `FLIP(rank, lane)` cannot separate two same-rank face-down cards carrying different
  damage — and face-down cards do take damage (`game_rules.md` §5).
- `PAIR(lane, rank)` cannot express *which two* of three same-rank cards you pair, and the
  choice matters because they differ in damage and attack budget.

That is not only a strength question. In an AlphaZero loop the policy target is the visit
distribution over engine actions; two distinct actions sharing a logit force an invented
rule for folding their visits and another for which one to actually play. Both are
arbitrary, and both distort the policy Phase 4 is supposed to read. The head is exact and
slot-keyed instead, and `phase3_action_encoding_round_trips` asserts injectivity over the
legal set.

`CHOOSE_SLOT` **is** shared, across four sub-decisions, and that is safe for a reason the
rank keying did not have: `Foresight`, `ResolveOrder`, `QueenSource` and `SplitTarget` are
mutually exclusive phases, so the mask disambiguates and no two of them can collide. Sharing
`FLIP` or `PAIR` was unsafe precisely because those collide inside a single phase.

Two consequences: `SplitTarget` carries no lane (it comes from the attack in flight), so
**encode and decode both take the state**; and `DeclarePair` is unordered, so the index is
canonicalised to `slot_a < slot_b` — `legal.rs` already emits only that order, pinned by
`phase3_legal_pairs_are_already_canonical`.

The 8-slot assumption in the old counts is gone; see §3 and `FINDINGS.md` F2.7, F3.1.

Resolution ordering is **adaptive** (`game_rules.md` §8): a 5 that flips four cards is four
successive `CHOOSE_SLOT` nodes, each chosen after seeing the previous power land — not one
permutation choice. That keeps the branching linear in cards flipped instead of factorial,
and it is also the correct information model, since each resolution can reveal a rank.

The observation carries a **phase** field so the head knows which mask applies.

## 5. Observation encoding

Per-observer, **3300 floats** at the default configuration. Implemented in
`engine/src/encode.rs`, which documents the exact layout block by block; the numbers below
are `config`-derived, so a smaller variant shrinks the tensor rather than misaligning it.

This used to read "~1300 floats", which silently assumed §3's abandoned 8-slot board. At 16
slots the board block alone is 3168 floats — the size is dominated by `lanes × sides × slots
× features`, so it tracks `encoding_slots` almost linearly. See `FINDINGS.md` F3.1 for what
that bound actually has to survive.

- **Board tensor** — 3 lanes × 2 sides × 16 slots × 33 features = 3168. Sides are ordered
  `[observer, opponent]`, so the tensor is always from the observer's point of view and the
  network never learns a seat convention. Per slot: `occupied`, rank one-hot (13, **all
  zero** when unknown to this observer), `rank_unknown`, `face_up`, `is_base`,
  `entered_as_base`, damage one-hot (4), max-HP one-hot (2), `frozen`, attack-allowance
  one-hot (4), `attacks_used / allowance`, `can_attack_now`, `paired`, `is_mine`.
- **Scalars** — 132 floats: phase one-hot (7), actions-remaining one-hot (4),
  `is_mine_to_move`, `ply / max_plies`, `quiet_plies / stalemate_quiet_plies`,
  `base_unlocked`, `observer_is_first_player`, lanes won per player, own hand rank counts
  (13), both hand sizes, both pile sizes, `shared_pile`, discard rank counts for both
  players (26), the belief and bottomed-card features below, and per-lane derived counts
  (12 — cheap, and it saves a dense MLP from rediscovering that slot indices within one lane
  belong together).
- **Belief features** — unseen-card rank counts from this observer's perspective. Note
  these **never reach zero uncertainty**: the 10 removed cards are permanently
  indistinguishable from cards in the opponent's hand or base. Encode the removed-pool size
  explicitly so the net can reason about it. Exception: in variant 9b the removed set is
  revealed at setup, so the unseen pool there is exactly opponent-hand + the six base cards.
- **Bottomed-card features** — for each pile, whether this observer knows the bottom card and
  its rank. Cheap to encode and strictly private; without it the net cannot value a 2 correctly.

**Known gap, base variant only.** A hand is modelled as a multiset of ranks, so when a card
leaves a pile for a hand its `known_to` mask is dropped (`apply.rs::draw_one`). In the split
variants that loses nothing — you only ever draw from your own pile, so a bottomed card can
only return to the player who bottomed it. In the **base** game the pile is shared, so an
opponent can draw a card you bottomed and you should know a rank they hold; the engine
forgets it. Closing this needs per-rank "the opponent knows I hold this" counters on the
hand *and* a way to carry that knowledge onto the card when it is played face-down, which is
a real chunk of state for a case that arises only in the non-default variant. Left open
deliberately. It costs a base-variant agent a sliver of strength and nothing else — it
cannot make one illegal — and `GameState::determinize` documents the same limitation from
the sampling side.

Foresight knowledge is folded into the board tensor: a card whose rank this observer knows
gets its real one-hot; everything else gets `rank_unknown`.

**Network**: pre-norm residual MLP, **5 blocks, width 512**, LayerNorm `eps = 1e-5` with
elementwise affine. Policy head returns **1324 raw logits** — masking and softmax are the
caller's job, because PUCT needs the masked distribution anyway and a masked softmax inside
the network would have to be mirrored exactly in the Rust forward pass for the parity test
to mean anything. Value head is `tanh` over a 256-wide hidden layer. ≈5.1M parameters,
~20 MB fp32. `blocks`, `width` and `value_hidden` come from config on both sides and travel
in the checkpoint header. Upgrade to slot-wise attention only if the MLP plateaus — measure
first.

> **Corrected 2026-09-03.** This line said "374 logits" while §4's own table totalled 395,
> and neither number was ever right — nothing in the engine produced 374. Both are
> superseded by §4's table, which totals 1324 since the `PASS` block was removed later the
> same day. Recording the inconsistency rather than quietly deleting it,
> because it is the kind of drift the checkpoint's `action_layout_hash` now exists to make
> impossible: a policy head that disagrees with the encoder does not crash, it just trains
> badly.

## 6. Search

**SO-ISMCTS** (single-observer information-set MCTS):

1. Sample a **determinization** — a concrete hidden state consistent with the acting
   player's information set: opponent hand, hidden base cards, draw pile order, *and which
   10 cards were removed*. The removed pool must be sampled; treating it as known is the
   classic way to get a subtly wrong agent here. Any **bottomed card this observer knows**
   is a hard constraint on the sampled pile order, not something to resample — the whole
   value of a 2 is that one pile position stops being uncertain.
2. Run PUCT on the sampled world.
3. Share visit statistics **at the information-set level** across determinizations.

Baseline to beat: **PIMC** (perfect-information Monte Carlo) — cheaper, and known to suffer
strategy fusion, which makes it a useful control rather than a target.

**Built in Phase 2.** Step 1 is `GameState::determinize` (`engine/src/determinize.rs`), steps
2–3 are `agents/ismcts.rs`. Three things that came out of implementing it, all of which the
design above did not anticipate:

- **The legal action set is a function of the information set, not of the hidden cards.**
  Every legality predicate reads a face-up rank, a slot position, the acting player's own
  hand, or a global flag — never a hidden rank. So an agent can enumerate actions on the real
  state and evaluate them on sampled worlds, and determinization can be checked for
  correctness by asserting the action list does not move.
- **That property makes "does this agent cheat?" a test rather than a comment.** A sampled
  world is in the same information set as the real state, so an honest agent handed either
  must return the same action. It is an exact assertion, and it immediately caught a leak in
  the greedy agent that had nothing to do with search: *applying* a candidate action to the
  real state reveals hidden ranks, because flipping your own base card turns it face-up and
  killing a face-down card sends its rank to the public discard. Even one-ply lookahead has
  to happen inside a sampled world.
- **Search cost is dominated by the branching factor, which this game does not bound.** PIMC
  at depth `d` costs `b^(d+1)`, and §3's note about lane width is the same problem seen from
  the other side. It carries an explicit node budget for that reason.

**Phase 3 step 2 replaced the two halves of the loop, and nothing else** —
`engine/src/agents/net_mcts.rs`, as the `netmcts:<checkpoint>@<sims>` rung. UCB1 over
availability becomes PUCT over a network prior; a uniform-random playout becomes the value
head. The determinization, the information-set tree and the availability counts are the
Phase 2 code's, unchanged.

Two adaptations that determinization forces, and that a transcription of AlphaZero would get
wrong:

- **The PUCT numerator is the edge's availability, not the parent's visit count.**
  `Q + c·P·sqrt(availability(e))/(1 + visits(e))`. An action legal in only half the sampled
  worlds has had half as many chances to be chosen, and charging it for the visits it was not
  on the menu for is the same mistake UCB1 makes without an availability count. It reduces to
  AlphaZero's rule exactly when every action is always legal.
- **Priors are stored as logits and softmaxed over what is available**, because the legal set
  changes between determinizations and a probability normalised at expansion time would be
  normalised over the wrong support on most later visits.

## 6a. The training loop

`PLAN.md` Phase 3 step 3. One generation is: `duel52 selfplay` → a `.d52sp` shard →
`python -m duel52.train` replays, fits, gates, promotes.

**The corpus stores trajectories, not tensors.** A shard holds `(config, seed, and for each
decision an index into legal_actions() plus the root visit distribution)`. Observations are
recomputed by replaying the game through the engine, which costs a few hundred microseconds
and means an encoder change costs a replay rather than a discarded corpus. The action indices
are indices into the *legal-action list* rather than encoded action indices, for the same
reason one level down: a legal-action list is a property of the engine, an encoded index is a
property of the current action layout.

**Replay hands Python sparse observations.** The observation is 4.8% dense (`FINDINGS.md`
F3.3), so `duel52._engine.replay_shard` returns CSR-style `offset`/`index`/`value` triples
and the trainer scatters a batch into a dense tensor on the way to the device. Dense storage
of one generation is gigabytes; sparse is hundreds of megabytes.

**An engine-declared stalemate is not worth half a point to a learner.** `config
.stalemate_value` (default 0.5, training 0.0) is read by the terminal backup in `net_mcts`
and the value targets in `selfplay`, and by nothing else — `Outcome::value_for` and therefore
the whole Elo apparatus are untouched. `FINDINGS.md` F3.6 is why: at half a point, mutual
refusal to attack is a stable equilibrium of the modified game, and the first training run
found it in two generations. A *mutual lane win* keeps its half point; that one is a rule.

**The gate is two tests, and reads decisive games.** `W / (W + L)` against the incumbent at
0.55, plus a veto if the candidate falls more than a tolerance below the best score any
promoted checkpoint has managed against a fixed reference panel. The single-mirror version
promoted three consecutive generations of a collapsing agent, because two stalling agents
draw against each other and 0.500 clears a 0.5 threshold — F3.6 again.

## 7. Validation strategy

Exact exploitability is out of reach for the full game, so:

- **Duel52-mini** — a scaled-down configuration (1 lane, ranks A–5, 1 base card, tiny deck)
  registered as an OpenSpiel game. Small enough for **exact CFR and exact exploitability**.
  Every algorithm gets validated here before being trusted on the full game. This is the
  cheapest available insurance against a plausible-looking but wrong training loop.
- **Local best-response (LBR)** as the exploitability proxy on the full game.
- **Round-robin Elo** against the frozen baseline ladder (random → greedy → flat MC →
  ISMCTS-rollout → each net checkpoint).

## 8. Performance targets

- Engine: **≥10k full random games/sec/core**. If we miss this badly, profile before
  scaling anything else.
- Local self-play (M-series, MPS): a few hundred thousand ISMCTS games over a few days.
- The handoff config swaps MPS → CUDA and raises worker count. No code changes.

## 9. Repo layout, and where inference runs

```
duel52/
  engine/          Rust crate — rules, legality, determinization, encoders, inference
    src/encode.rs  observation + action tensors, and the layout hashes that pin them
    src/nn/        weights, the .d52nn checkpoint format, the reference forward pass
    tests/         one named test per ruling in game_rules.md
  bindings/        PyO3 wrapper
  py/
    duel52/        the Python package
    duel52/nn/     the PyTorch model, and checkpoint read/write
    train/         AZ self-play loop, R-NaD (later)
    analyze/       Phase 4 insight extraction
  configs/         variant + training configs (base / split / mirrored)
  openspiel/       Duel52-mini game registration
```

### Search and inference in Rust, training in Python

This section originally implied the training loop would own the network and reach the engine
through PyO3. **Reversed in Phase 3, for one decisive reason:** Phase 3's deliverable is an
Elo table, and that table is produced by `duel52 ladder`, which is Rust and takes an
`AgentSpec`. So are `match`, `probe` and `play --opponent`, and `FINDINGS.md` F2.4, F2.5,
F2.7 and F2.8 all explicitly ask Phase 3 to re-run those measurements against the trained
agent. A Python-side agent could use none of it.

So PyTorch still owns the architecture, the weights and the gradients; Rust gets a frozen
snapshot and runs forward passes. They meet at a `.d52nn` checkpoint, whose header carries
FNV-1a hashes of the observation and action layouts. Rust recomputes them from its own
constants at load and refuses a mismatch. That check is the point: silent layout drift
between the trained function and the evaluated function does not crash anything — it
produces an agent that is merely bad, and the natural suspect is the training run.

The `Evaluator` trait (`engine/src/nn/mod.rs`) is **batch-shaped from the start**, even
though step 1's only consumer evaluates one position at a time. The self-play loop will keep
`G` games in flight per worker and advance one simulation in each per round, evaluating the
round as a single batch — no virtual loss, no search distortion, every game still
reproducible from its own seed. Retrofitting that interface later would touch the whole loop.

### Deferred: the `cli` / `nn` crate split

The reference forward pass is hand-rolled f32 loops in `engine`, which keeps the crate's
zero-dependency guarantee intact — and a BLAS would work *against* the reason for that
guarantee, by introducing exactly the accumulation-order variability the project avoids. A
GPU backend (ONNX Runtime, or `tch`) belongs at the CUDA handoff, in an `nn` crate alongside
a `cli` crate that depends on `engine`. `Evaluator` is the seam that makes that swap cheap.
**Known future refactor, deliberately not done now.**

## 10. Deliberately deferred

- Large networks, distributed training, GPU-side inference batching — not the bottleneck yet.
- Opponent modeling / exploitative play. The question is what *optimal* looks like, not how
  to beat a specific human.
- A polished UI. A text CLI to play against the bot is enough, and it doubles as the rules
  sanity check.
