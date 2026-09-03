# Duel 52 — Findings

What we actually learn about the game. **This file is the point of the project.**

**Nothing measured yet.** No engine exists. Everything below is a hypothesis to be
confirmed or killed, written down now so we can be honest later about which ones we got
right and which were plausible-sounding nonsense.

## Recording standard

Every finding gets: the **claim**, the **config** (variant, rules version), the **agent**
that produced it, the **seed range**, the **sample size**, and a **confidence interval**.
A number without a reproducible provenance is not a finding — it is a vibe.

---

## Hypotheses to test

Derived from reading the rules, not from data. Expect a decent fraction to be wrong; that
is what makes them worth writing down.

### H1 — The draw phase is entirely positional
Lane wins require an empty pile *and* an empty opponent hand (`game_rules.md` §7), so
nothing can be decided during the draw phase. Prediction: strong agents treat the first
~26 turns as setup, and killing enemy cards early is worth far less than intuition
suggests — attrition only matters insofar as it shapes the endgame board.

### H2 — Hand size at pile-empty is a primary resource
Every card in hand is a turn the opponent cannot close a lane. Prediction: strong agents
**hoard cards** approaching the pile-empty transition, and hand size at that moment
correlates strongly with winning. If true, this is the single biggest gap between naive and
optimal play, since holding cards feels passive.

**Counter-consideration:** cards in hand do nothing on the board, and a card played early
has more turns to generate value. There is a crossover point. Finding it is arguably the
most valuable single output of this project.

### H3 — Optimal play concentrates on two lanes
You need two lanes, not three. Prediction: strong play identifies a lane to concede and
commits, but does so *later* than human intuition — because conceding early lets the
opponent redeploy via the Queen.

### H4 — Information is worth less than tempo
Playing face-down hides information but costs a second action to flip. Prediction: the 4
(Foresight) is among the weakest cards, and holding cards face-down for concealment is
overrated relative to flipping to get powers online.

### H5 — The Jack is the strongest card
3 HP plus taunt means the opponent must spend three attack actions before touching anything
else. In an action-constrained game that is enormous. The 9's "2 damage to Jacks" exists
precisely because of this. Prediction: Jack tops the learned rank values; the 9's real value
is mostly its Jack-counter clause.

### H6 — The 7 scales with board commitment
Heal-all across every lane is a blowout when you have many damaged cards and nearly dead
otherwise. Prediction: the 7's value has the highest variance of any card, and strong agents
time it rather than flipping on sight.

### H7 — The King is a combo enabler, not a body
King + Ace is a free action; King + Queen is a second move; King + 5 is a mass flip.
Prediction: King value depends almost entirely on lane composition, and strong agents
arrange the lane *before* flipping the King.

### H8 — First-player advantage is small and possibly negative
The first player gets one fewer action on turn one. Combined with H1 (nothing is decided
early), the tempo edge may not compensate. Prediction: near-even, and the split-deck variant
is where we can measure it cleanly.

---

## Measured results

*(empty)*

### Baseline statistics — Phase 1
Game length distribution, first-player win rate, stalemate frequency, per variant.

### Elo ladder — Phase 2
Frozen benchmark: random → greedy → flat MC → PIMC → ISMCTS-rollout.

### Learned card values — Phase 3/4
From the value net and from ablation.

### Policy characterization — Phase 4
Opening frequencies, flip timing, lane commitment, hand-hoarding curve.

---

## Things we got wrong

Log falsified hypotheses here rather than quietly deleting them. Knowing which intuitions
the game defeats is itself a finding, and it is the part a human player would most want to
read.
