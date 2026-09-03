# Duel 52 — Findings

What we actually learn about the game. **This file is the point of the project.**

**Status: Phase 1 baselines measured.** The engine exists and the random-vs-random numbers
are in. Everything under *Hypotheses* is still unconfirmed — random play cannot speak to
strategy, only to the shape of the game tree. H8 has a first data point; the rest wait for
Phase 2.

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

> **First data point (F1.4): small, and positive rather than negative.** Under random play
> P0 scores 0.5119 ± 0.0022 in the split variant — a real edge, ~5σ from even, but a
> genuinely small one. "Small" holds; "possibly negative" does not, at least here.
>
> This is weak evidence for the hypothesis as stated, because H8's reasoning is about
> *tempo*, and a random agent cannot use tempo. What F1.4 really shows is that moving first
> is worth more than the missing opening action costs even when neither side is trying.
> Whether skilled play widens or narrows that gap is the actual question, and it is open.
> Re-measure in Phase 2 against the Elo ladder and again in Phase 3.

---

## Measured results

### F1 — Baseline statistics, random vs random — Phase 1

**Provenance.** Engine 0.1.0, commit at time of writing. Uniform random agents on both
sides. Seeds 1–200000 per configuration, 200,000 games each, 1.2M games total.
Reproduce with:

```
duel52 stats --all --games 200000 --seed 1 --markdown
```

| variant | 2's power | games | P0 score (95% CI) | draw | stalemate | mean plies | P0 draw edge | max lane |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | bottom | 200000 | 0.5149 ± 0.0022 | 0.4% | 0.0% | 45 | +0.053 | 20 |
| split | bottom | 200000 | 0.5119 ± 0.0022 | 0.4% | 0.0% | 45 | +0.003 | 19 |
| mirrored | bottom | 200000 | 0.5149 ± 0.0022 | 0.5% | 0.0% | 45 | −0.000 | 19 |
| base | discard | 200000 | 0.5242 ± 0.0022 | 0.4% | 0.0% | 44 | +0.522 | 19 |
| split | discard | 200000 | 0.5138 ± 0.0022 | 0.5% | 0.0% | 44 | +0.000 | 19 |
| mirrored | discard | 200000 | 0.5168 ± 0.0022 | 0.5% | 0.0% | 44 | +0.000 | 19 |

"P0 draw edge" is the mean number of extra cards P0 drew relative to P1, per game.

**F1.1 — Games are short and remarkably uniform in length.** Mean 45 plies, median 45,
p10 39, p90 52, across every configuration. That is ~22 turns apiece, of which the first
~12 are the draw phase. The distribution is tight enough that game length is essentially
not a strategic variable under random play.

**F1.2 — The stalemate rule never fires under random play.** Zero games out of 1.2M hit
the quiet-ply cutoff, in any configuration. **Every** draw was a mutual lane win. This does
*not* vindicate the threshold: `game_rules.md` §7 is explicit that the reachable stall is
*strategic* — "neither player wants to attack first because attacking exposes the
attacker" — and random agents attack constantly, so they cannot produce it. The rule is
untested until Phase 2 puts a non-suicidal agent on both sides. **Re-measure then.**

**F1.3 — The mutual lane win is not astronomically rare.** `game_rules.md` §7 calls it
"astronomically rare"; it is 0.4–0.5% of games under random play — roughly 1 in 220, and
the *only* source of draws here. `duel52 demo --seed 86` replays one: P0's last card in a
lane attacks P1's last card, an 8, and retaliate kills the attacker as the attack kills the
8. Random play throws away material freely, so this rate should fall sharply with agent
strength; the point is that the case is reachable and the terminal check has to be total.

**F1.4 — First-player advantage is real but small: P0 scores ≈ 0.512–0.515.** About a
+1.2 to +1.5 percentage-point edge, ~7σ from even at this sample size, consistent across
all three variants under the house rule. So the opening turn's missing action does *not*
compensate for moving first. See H8 below — this is a first data point, not a verdict:
random play is a poor proxy for whether tempo matters.

**F1.5 — The `two_power` house rule is vindicated on its own terms, and only in the base
game.** `game_rules.md` §10a adopted bottoming over discarding on the argument that the RAW
discard shrinks a *shared* pile and thereby "turns a filtering effect into a lever on the
draw count". Measured directly rather than inferred:

| | base (shared pile) | split / mirrored (own pile) |
|---|---:|---:|
| `bottom` (house) | +0.053 | +0.003 / −0.000 |
| `discard` (RAW) | **+0.522** | +0.000 / +0.000 |

The RAW discard hands the first player half an extra card per game in the base variant, ten
times the structural residual, and it moves P0's score from 0.5149 to 0.5242 (+0.0093,
≈6σ). The mechanism is exactly as §10a described: P0 draws first from the shared pile, so
every card the 2 destroys flips the pile's parity against P1. In the split variants you
discard into your own pile, so there is no shared parity to flip and the effect is exactly
zero. **The house rule fixes a real artifact, and it is an artifact of pile *sharing*, not
of the 2.**

Caveat worth stating: random agents fire 2s at random, so this measures the *mechanism*,
not its strategic weight. §10a's stronger claim — that the lever "favours whoever is
positioned to use it first" — needs an agent that chooses when to fire. Phase 2.

**F1.6 — Turns-to-unlock is deterministic under the house rule, as predicted.** §10a
predicted the pile empties after exactly *pile size* turns once the 2 stops shrinking it.
Confirmed: in the split variant the base cards unlock on **ply 25 in every single game**,
with no variance at all. Under `discard` it varies (mean 24.1). Combined with the split
deck, this removes deal-luck from the endgame trigger on two independent axes — the whole
point of choosing both.

**F1.7 — `DESIGN.md` §3's 8-slot encoding bound is too small, by a lot.** Observed maximum
is **20 cards on one side of one lane** (base variant; 19 in the split variants). §3
justified 8 as "far beyond observed play", which is true of human play and false of the
game tree: a base-game player pushes up to 31 cards through their hand, and random agents
spread them evenly. The engine therefore uses the theoretical maximum (34 base, 21 split)
and asserts, rather than capping legality at 8 — which would have quietly changed the game.
**Phase 3 must pick its encoding bound from strong-agent play, not from this number and not
from §3's guess.** Random play is an upper bound on spread, so the real figure is likely
lower, but 8 is not defensible.

**F1.8 — Throughput: ~16,800 random games/sec/core**, single-threaded, release build, on
an M-series Mac. `DESIGN.md` §8 targets ≥10k, so there is headroom and no reason to profile
before Phase 3.

**What F1 does not tell us.** Nothing about strategy. A random agent passes a few percent
of the time, attacks far more than a human would, and never hoards cards. F1 characterises
the *game tree* — its depth, its terminal structure, its symmetry — which is exactly what
Phase 1 needed to establish before agents arrive.

### Baseline statistics — Phase 1: still open

- Re-measure the stalemate rate against a non-suicidal agent (F1.2).
- Measure hand size at pile-empty against agent strength — the H2 crossover. Random-play
  baseline: median 6 cards combined across both hands at the unlock (split variant).

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
