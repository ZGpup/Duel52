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
  Encoding cap of **8 slots per side per lane** (rules impose no limit; 8 is far beyond
  observed play — the engine asserts rather than silently truncating).
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

Fixed policy head, legality-masked, phase-conditioned:

| Action | Count | Encoding note |
|---|---|---|
| `PLAY(rank, lane)` | 13 × 3 = 39 | Hand is a multiset; rank is the correct key. |
| `FLIP(rank, lane)` | 13 × 3 = 39 | Your played face-down cards are known to you by rank; same-rank duplicates are interchangeable. |
| `FLIP_UNKNOWN(lane, slot)` | 3 × 8 = 24 | Face-down cards whose rank the *owner* does not know, so they cannot be rank-keyed: the base card in its own slot, plus any Queen-moved base card now sitting in a normal slot. |
| `ATTACK(lane, atk_slot, tgt_slot)` | 3 × 8 × 8 = 192 | Slot-indexed: opposing face-down targets have no known rank. |
| `PAIR(lane, rank)` | 39 | A pair requires two same-rank face-up cards, both known. |
| `PASS` | 1 | Forfeit remaining actions. |
| **Sub-choices** | 61 | `CHOOSE_SLOT` (3 × 2 × 8 = 48) for a 4's peek, a Queen's move source, a 10's second target, and 5/King resolution ordering; `CHOOSE_RANK` (13) for the card a 2 bottoms (or discards, under `two_power: discard`). |
| **Total** | **395** | |

Resolution ordering is **adaptive** (`game_rules.md` §8): a 5 that flips four cards is four
successive `CHOOSE_SLOT` nodes, each chosen after seeing the previous power land — not one
permutation choice. That keeps the branching linear in cards flipped instead of factorial,
and it is also the correct information model, since each resolution can reveal a rank.

The observation carries a **phase** field so the head knows which mask applies.

## 5. Observation encoding

Per-observer, ~1300 floats:

- **Board tensor** — 3 lanes × 2 sides × 8 slots × ~25 features: occupancy, rank one-hot
  (13, zeroed when unknown to this observer), `rank_unknown`, `face_up`, `is_base`,
  damage one-hot, max-HP, `frozen`, `attacked_this_turn`, `paired`, `is_mine`.
- **Scalars** — actions remaining, turn index, own hand rank counts (13), opponent hand
  size, draw pile size(s), discard rank counts (13), `base_unlocked`, first-player flag,
  quiet-turn counter.
- **Belief features** — unseen-card rank counts from this observer's perspective. Note
  these **never reach zero uncertainty**: the 10 removed cards are permanently
  indistinguishable from cards in the opponent's hand or base. Encode the removed-pool size
  explicitly so the net can reason about it. Exception: in variant 9b the removed set is
  revealed at setup, so the unseen pool there is exactly opponent-hand + the six base cards.
- **Bottomed-card features** — for each pile, whether this observer knows the bottom card and
  its rank. Cheap to encode and strictly private; without it the net cannot value a 2 correctly.

Foresight knowledge is folded into the board tensor: a card whose rank this observer knows
gets its real one-hot; everything else gets `rank_unknown`.

**Network**: 4–6 residual MLP blocks, width 512. Policy head (374 logits) + value head
(scalar, tanh). Upgrade to slot-wise attention only if the MLP plateaus — measure first.

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

## 9. Repo layout (planned)

```
duel52/
  engine/          Rust crate — rules, legality, ISMCTS-friendly determinization
    src/
    tests/         one named test per ruling in game_rules.md
  py/
    duel52/        PyO3 bindings + gym-ish wrapper
    train/         AZ self-play loop, R-NaD (later)
    analyze/       Phase 4 insight extraction
  configs/         variant + training configs (base / split / mirrored)
  openspiel/       Duel52-mini game registration
```

## 10. Deliberately deferred

- Large networks, distributed training, GPU-side inference batching — not the bottleneck yet.
- Opponent modeling / exploitative play. The question is what *optimal* looks like, not how
  to beat a specific human.
- A polished UI. A text CLI to play against the bot is enough, and it doubles as the rules
  sanity check.
