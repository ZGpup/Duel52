# Duel 52 — Findings

What we actually learn about the game. **This file is the point of the project.**

**Status: Phase 3 trained and measured.** Five hand-written agents on a frozen Elo ladder, plus
`models/duel52-split-gen016.d52nn` — the first trained agent, +495 Elo clear of the ladder
(F3.7). Six of the eight hypotheses have data, and **the first one has moved**:

| | verdict | where |
|---|---|---|
| H1 — the draw phase is positional | untested directly, but implied by H3's null | F2.6 |
| H2 — hand size at pile-empty is the resource | **supported** — +1.25 ± 0.17 cards, ~14σ, with a control. Causation open | **F3.9** |
| H3 — optimal play concentrates on two lanes | **unsupported**, no rung beats the random baseline — but agent-limited, see below | F2.6 |
| H4 — information is worth less than tempo | **split** — the 4 is weak, but the framing is wrong | F2.9 |
| H5 — the Jack is the strongest card | flip-timing half **confirmed**, value half open | F2.9, F3.10 |
| H6 — the 7 scales with board commitment | timing half **supported** | F2.9 |
| H7 — the King is a combo enabler | consistent, too weak to call | F2.9 |
| H8 — first-player advantage is small | **confirmed**, and it is zero, not small — survives a strong agent | F2.8, F3.9 |

⚠️ **Read every Phase 2 null with this in mind.** H2 was recorded as "unsupported, effect
bounded under ±0.2 cards" for the whole of Phase 2. It was wrong, and not because the
measurement was bad — F3.9 reproduces F2.5's null exactly on `greedy`, using the same test.
**No Phase 2 agent could hoard deliberately, so the within-agent test had nothing to detect.**
A null from an agent that cannot do the thing is not a null about the game. H3 is the other
hypothesis in that position and has not yet been re-run.

The pattern is worth naming up front: **the hypotheses about card-level behaviour are
holding up and the hypotheses about strategic shape are not.** Read them with the caveat in
F2's preamble — a 1186-Elo ISMCTS with random rollouts is a competent player, not a strong
one, and two of the five rungs are steered by hand-written weights that this file does not
trust.

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

> **Phase 2: unsupported, with the effect bounded below ±0.2 cards (F2.5).** Within a single
> agent's self-play — which holds strength and evaluation weights fixed — the side holding
> more cards when the pile empties does *not* win more often, at any rung. The two
> evaluation-free rungs lean the way H2 predicts and neither reaches significance.
>
> The hypothesis survives on a technicality that is worth stating rather than hiding: every
> Phase 2 agent plays **18.0 of its 18 cards**. Hands empty because there is nothing else to
> spend three actions a turn on, not because anyone weighed holding against playing. None of
> these agents can represent "this card is worth more unplayed" — greedy and PIMC have a
> weight that says the opposite, and flat MC and ISMCTS have no evaluation at all. **H2 has
> not really been tested yet; it has been tested on agents incapable of the behaviour it
> describes.** Phase 3 must re-run F2.5's within-agent test against a trained value net.

### H3 — Optimal play concentrates on two lanes
You need two lanes, not three. Prediction: strong play identifies a lane to concede and
commits, but does so *later* than human intuition — because conceding early lets the
opponent redeploy via the Queen.

> **Phase 2: no concentration at all (F2.6).** Share of plays in a player's busiest two lanes
> is 0.783 for ISMCTS against a random baseline of 0.778 — and the two rungs using the
> hand-written evaluation concentrate *less* than random. Attacks show the same flat picture.
>
> The prediction's own second clause may be the explanation. H3 says commitment comes "later
> than human intuition"; §7 makes lane wins an endgame event, so the draw phase gives no
> reason to commit at all, and by the time commitment pays there are only a few turns left
> in which to express it. A whole-game share statistic would barely register that. **The
> measurement to build in Phase 4 is lane share restricted to post-unlock plies**, not lane
> share over the game.

### H4 — Information is worth less than tempo
Playing face-down hides information but costs a second action to flip. Prediction: the 4
(Foresight) is among the weakest cards, and holding cards face-down for concealment is
overrated relative to flipping to get powers online.

> **Phase 2: the first clause is supported, the second is not, and the split is instructive
> (F2.9).** ISMCTS flips the **4 at 0.58** — the second-lowest rate of any rank, below random's
> 0.67 — and later than any rank except the 3. So Foresight does look like a weak power: an
> agent with a free choice declines to fire it.
>
> But "holding cards face-down is overrated" is wrong as a blanket claim. ISMCTS's overall
> flip rate (0.68) is barely above random's (0.66); what it changes is *which* cards it turns
> up, by a factor of five in spread. It flips constant powers early and holds one-shots. The
> real finding is not that concealment is over- or under-valued but that **flip value is a
> property of the rank, not a global tempo/information trade-off** — which is a more useful
> statement than the hypothesis and cuts across it.

### H5 — The Jack is the strongest card
3 HP plus taunt means the opponent must spend three attack actions before touching anything
else. In an action-constrained game that is enormous. The 9's "2 damage to Jacks" exists
precisely because of this. Prediction: Jack tops the learned rank values; the 9's real value
is mostly its Jack-counter clause.

> **Sharpened by a rules correction (2026-09-03), not yet by data.** A face-down card is a
> blank 2-HP card (`game_rules.md` §5), so **both** halves of the Jack — the third hit point
> and the taunt — arrive only on the flip. A face-down Jack is an ordinary body that dies to
> two hits and protects nothing.
>
> That makes H5 a claim about *flipped* Jacks specifically, and it cuts against the
> hypothesis in a way the original framing missed: extracting the Jack's value requires
> spending an action to flip him, and flipping announces him to an opponent holding a 9.
> So the Jack's value is entangled with flip timing (H4) rather than being a flat property
> of the card. Worth testing whether strong agents hold Jacks face-down as cheap bodies and
> flip them only to answer a specific threat.

> **Phase 2 (F2.9): that last sentence is answered, and the answer is no.** ISMCTS flips
> Jacks at 0.77 — well above random's 0.66 — and at **ply 18.6, the joint-earliest of any
> rank**. It does not hold them back for a threat; it turns them up as soon as it can, along
> with the other constant powers. That is consistent with the Jack being strong, since the
> whole of its value is locked behind the flip.
>
> **It is not evidence that the Jack is the strongest card**, which is what H5 actually
> claims. Flip priority and card value are different quantities, and nothing in Phase 2
> measures the second. H5 stays open for Phase 4's learned values, with the flip-timing
> prediction it implied now confirmed.

### H6 — The 7 scales with board commitment
Heal-all across every lane is a blowout when you have many damaged cards and nearly dead
otherwise. Prediction: the 7's value has the highest variance of any card, and strong agents
time it rather than flipping on sight.

> **Phase 2: the timing prediction is supported (F2.9).** The 7 is the clearest
> "flipped often, but late" card on the board: ISMCTS flips it at **0.77**, fifth-highest of
> thirteen — so it is not a card the search declines — while flipping it at **ply 23.0**,
> nearly five plies later than the constant powers it flips at a similar rate. That is
> exactly "time it rather than flipping on sight": the agent wants the power, and waits for
> damage to accumulate before spending it.
>
> The variance half of the hypothesis is untested. Phase 2 has no per-card value estimate, so
> "highest variance of any card" has to wait for Phase 4's ablations.

### H7 — The King is a combo enabler, not a body
King + Ace is a free action; King + Queen is a second move; King + 5 is a mass flip.
Prediction: King value depends almost entirely on lane composition, and strong agents
arrange the lane *before* flipping the King.

> **Phase 2: consistent, but too weak to call support (F2.9).** ISMCTS flips Kings at 0.71
> against random's 0.69 — essentially no preference — and at ply 22.5, in the late half.
> "Arrange the lane first" predicts a late flip and that is what shows up, but so would half
> a dozen other explanations, and the flip *rate* is the flattest of any rank relative to
> random.
>
> Testing H7 properly needs a conditional statistic Phase 2 does not collect: how many
> reactivatable allies were face-up in the King's lane at the moment it was flipped, against
> the number available. **Add that probe in Phase 4** — it is cheap, and it is the only way to
> separate "flipped late" from "flipped when the lane was ready".

### H8 — First-player advantage is small and possibly negative
The first player gets one fewer action on turn one. Combined with H1 (nothing is decided
early), the tempo edge may not compensate. Prediction: near-even, and the split-deck variant
is where we can measure it cleanly.

> **First data point (F1.4): small, and positive rather than negative.** Under random play
> P0 scores 0.5139 ± 0.0022 in the split variant — a real edge, ~12σ from even, but a
> genuinely small one. "Small" holds; "possibly negative" does not, at least here.
>
> This is weak evidence for the hypothesis as stated, because H8's reasoning is about
> *tempo*, and a random agent cannot use tempo. What F1.4 really shows is that moving first
> is worth more than the missing opening action costs even when neither side is trying.
> Whether skilled play widens or narrows that gap is the actual question, and it is open.
> Re-measure in Phase 2 against the Elo ladder and again in Phase 3.

> **Phase 2 (F2.8): "small" holds and the edge disappears entirely.** In greedy self-play at
> 4,000 games per variant, P0 scores 0.4946 ± 0.0217 (base), 0.4918 ± 0.0217 (split), 0.5095
> ± 0.0216 (mirrored) — all three cover 0.5, two below it. Every other rung agrees at its own
> (wider) precision. F1.4's +1.4 points was a property of random play, not of the game.
>
> H8 is now the best-supported hypothesis in this file, and the one whose *reasoning* is still
> unverified: it predicted the wash but attributed it to tempo, and greedy has no more grasp
> of tempo than random does. Phase 3 re-runs it against an agent that does.

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
| base | bottom | 200000 | 0.5154 ± 0.0022 | 0.4% | 0.0% | 45 | +0.054 | 20 |
| split | bottom | 200000 | 0.5139 ± 0.0022 | 0.5% | 0.0% | 45 | +0.001 | 19 |
| mirrored | bottom | 200000 | 0.5148 ± 0.0022 | 0.5% | 0.0% | 45 | −0.001 | 19 |
| base | discard | 200000 | 0.5232 ± 0.0022 | 0.4% | 0.0% | 43 | +0.523 | 19 |
| split | discard | 200000 | 0.5147 ± 0.0022 | 0.5% | 0.0% | 44 | +0.000 | 19 |
| mirrored | discard | 200000 | 0.5166 ± 0.0022 | 0.5% | 0.0% | 44 | +0.000 | 19 |

"P0 draw edge" is the mean number of extra cards P0 drew relative to P1, per game.

**F1.1 — Games are short and remarkably uniform in length.** Mean 45 plies, median 45,
p10 39, p90 52, across every configuration. That is ~22 turns apiece, of which the first
~12 are the draw phase. The distribution is tight enough that game length is essentially
not a strategic variable under random play.
→ **Re-measured after the §4 mandatory-action ruling: mean 40.1, median 40, p10 35, p90 45.**
The shape is the same and the conclusion is unchanged; the centre moved because a third of
random play's actions used to go unspent. See F2.4b.

**F1.2 — The stalemate rule never fires under random play.** Zero games out of 1.2M hit
the quiet-ply cutoff, in any configuration. **Every** draw was a mutual lane win. This does
*not* vindicate the threshold: `game_rules.md` §7 is explicit that the reachable stall is
*strategic* — "neither player wants to attack first because attacking exposes the
attacker" — and random agents attack constantly, so they cannot produce it. The rule is
untested until Phase 2 puts a non-suicidal agent on both sides. **Re-measure then.**
→ Re-measured in **F2.4**, which found it firing for greedy alone; then **F2.4b**, where the
mandatory-action ruling removed the stall entirely. F1.2's zero is now the *expected* result
for every agent, not an artefact of random play.

**F1.3 — The mutual lane win is not astronomically rare.** `game_rules.md` §7 calls it
"astronomically rare"; it is 0.4–0.5% of games under random play — roughly 1 in 220, and
the *only* source of draws here. `duel52 demo --seed 47` replays one: P0's last card in a
lane attacks P1's last card, an 8, and retaliate kills the attacker as the attack kills the
8. Random play throws away material freely, so this rate should fall sharply with agent
strength; the point is that the case is reachable and the terminal check has to be total.

**Still 0.4% after the §4 mandatory-action ruling** (20,000 games, seed 1: 88 mutual lane
wins, 0 stalemates, 0 ply-cap draws), and it is now the *only* way any Duel 52 game draws,
at every level of play — see F2.4b. The replay seed changed from 86 to 47 with the ruling:
removing `Pass` from the legal-action list re-shuffles every random game, so any finding
that names a seed had to be re-derived rather than re-read.

**F1.4 — First-player advantage is real but small: P0 scores ≈ 0.514–0.515.** About a
+1.4 to +1.5 percentage-point edge, ~13σ from even at this sample size, and statistically
indistinguishable across the three variants under the house rule. So the opening turn's
missing action does *not* compensate for moving first. See H8 below — this is a first data
point, not a verdict: random play is a poor proxy for whether tempo matters.

**F1.5 — The `two_power` house rule is vindicated on its own terms, and only in the base
game.** `game_rules.md` §10a adopted bottoming over discarding on the argument that the RAW
discard shrinks a *shared* pile and thereby "turns a filtering effect into a lever on the
draw count". Measured directly rather than inferred:

| | base (shared pile) | split / mirrored (own pile) |
|---|---:|---:|
| `bottom` (house) | +0.054 | +0.001 / −0.001 |
| `discard` (RAW) | **+0.523** | +0.000 / +0.000 |

The RAW discard hands the first player half an extra card per game in the base variant, ten
times the structural residual, and it moves P0's score from 0.5154 to 0.5232 (+0.0078,
≈5σ). The mechanism is exactly as §10a described: P0 draws first from the shared pile, so
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

### Baseline statistics — Phase 1: closed by Phase 2

- ~~Re-measure the stalemate rate against a non-suicidal agent~~ → **F2.4**.
- ~~Measure hand size at pile-empty against agent strength~~ → **F2.5**. Random-play
  baseline for reference: median 6 cards combined across both hands at the unlock.

---

### F2 — The Phase 2 agent ladder

**Provenance.** Engine 0.1.0. Split-deck variant, `two_power = bottom`, stalemate 20 plies.
Five agents, 400 colour-paired games per pairing over seeds 1–200 — each deal played twice
with the seats swapped and both agents on the same random stream in both halves, so deal
luck cancels within the pair. 4,000 games, 26 minutes on 8 cores. Reproduce with:

```
duel52 ladder --games 400 --seed 1 --markdown
```

The frozen roster is `AgentSpec::LADDER`. Changing a budget there invalidates every number
below, so later phases add a rung rather than editing one.

Every *score* here is exactly reproducible — results do not depend on the thread count, which
`phase2_the_ladder_is_thread_count_independent` pins. Every *throughput* figure is not: they
were taken on a shared 8-core M-series Mac under other load, so read them as ratios between
agents rather than as absolute rates.

**F2.1 — The ladder is cleanly ordered and fully transitive.** No cycles: every agent beats
everything below it in the table and loses to everything above.

⚠️ **This table predates the §4 mandatory-action ruling** (2026-09-03) and was fitted on a
game where players could pass. **F3.7 carries the first post-ruling ladder.** The two do not
compare — F3.7 also adds a trained agent and swaps `pimc:32x1` for `pimc:8x1`, and Elo is
roster-relative — so `ismcts:800` reading +1186 here and +981 there is two different fits of
two different games, not a rung that got worse. The *ordering* below survives except for PIMC,
which has dropped to last; see F2.3.

| agent | Elo | ± | vs. random | vs. greedy | vs. flatmc | vs. pimc |
|---|---:|---:|---:|---:|---:|---:|
| ismcts:800 | +1186 | 16 | 1.000 | 0.944 | 0.796 | 0.912 |
| flatmc:600 | +952 | 12 | 1.000 | 0.859 | — | 0.689 |
| pimc:32x1 | +784 | 12 | 1.000 | 0.600 | 0.311 | — |
| greedy | +682 | 13 | 0.965 | — | 0.141 | 0.400 |
| random | 0 | — | — | 0.035 | 0.000 | 0.000 |

Ratings are a batch Bradley–Terry fit anchored at random, not the incremental Elo update, so
the table does not depend on the order games were played in.

> **Read the top four's rating *against random* with suspicion, and their gaps *from each
> other* with confidence.** Three rungs went 400–0 against random, and an undefeated record
> has no upper bound on the likelihood — those ratings are pinned almost entirely by the
> transitive chain through greedy, not by the pairing they look like they came from. The
> head-to-head columns are the primary evidence; the Elo column is a summary of them.

**F2.2 — PIMC is the weakest search, and the way it fails is the textbook one.** This is the
result the phase was really for. `DESIGN.md` §6 called PIMC "a useful control rather than a
target" on the grounds that it suffers **strategy fusion** — it solves each sampled world as
though the hidden cards were about to be revealed, so it will pick a line that wins against
world A one way and world B another, even though at the real decision node it must commit to
one. Measured, at matched compute (`pimc:32x1` and `flatmc:600` both run ~2–6 games/sec
against the same opponents):

- `flatmc:600` beats `pimc:32x1` **0.689 ± 0.045**, a 168-Elo gap.
- `ismcts:800` beats `pimc:32x1` **0.912**, versus **0.796** against flat MC.

So PIMC is beaten by an agent with *no evaluation function at all* and no tree — flat Monte
Carlo is nothing but random playouts averaged per action. Whatever PIMC's alpha–beta and
hand-written evaluation buy inside each world, they do not survive averaging across worlds.

**F2.3 — For PIMC, depth buys strength and determinizations do not.** Against a fixed greedy
opponent, 200 colour-paired games each on seeds 1–100
(`duel52 match --a pimc:WxD --b greedy --games 200 --seed 1`):

⚠️ **The `pimc:8x1` row is stale and now measurably wrong.** It predates the §4
mandatory-action ruling. Re-run post-ruling over 400 games at seed 1, `pimc:8x1` vs `greedy`
scores **0.5050 ± 0.0488** — dead even — against the 0.638 ± 0.066 below. The intervals do not
overlap. Removing the pass cost PIMC its edge over `greedy` outright, which is why F3.7's
ladder puts PIMC last. **The conclusions drawn from the table survive and one is stronger:**
`flatmc:600` now beats `pimc:8x1` **0.835 ± 0.036** post-ruling (400 games, seed 1) against
the 0.689 recorded here. The depth-vs-width conclusion has *not* been re-measured.

| PIMC budget | score vs. greedy (95% CI) | throughput | cost vs. 8×1 |
|---|---:|---:|---:|
| pimc:8x1 | 0.638 ± 0.066 | 22.4 g/s | 1× |
| pimc:32x1 | 0.533 ± 0.069 | 5.4 g/s | 4× |
| pimc:64x1 | 0.580 ± 0.068 | 2.6 g/s | 9× |
| pimc:16x2 | **0.863 ± 0.047** | 1.1 g/s | 20× |

**Eight-fold more sampled worlds at depth 1 buys nothing measurable.** The three depth-1 rows
scatter around 0.58 with overlapping intervals and no trend — 8 worlds is, if anything, the
best of them. Adding a single ply of search instead takes the score from 0.58 to 0.86.

This is exactly the signature strategy fusion predicts, and it is worth stating plainly
because it is the kind of thing that is easy to mistake for noise: **PIMC's error is bias,
not variance.** More samples of independently-solved perfect-information worlds cannot fix an
estimator that is averaging the wrong quantity. Deeper search makes each world's answer
better and so moves the biased average; more worlds only estimates the same biased average
more precisely. A Phase 3 agent that models beliefs is therefore buying something real, not
just a bigger sample.

Practical note for Phase 3: PIMC at depth `d` costs `b^(d+1)`, and `b` here is not bounded —
F1.7's 20-card lane is hundreds of attack actions. `pimc:16x2` is unrunnable against random
without the node budget the agent now carries. The branching factor is a live constraint on
this game, not a detail.

**Self-play instrumentation.** Everything from F2.4 down is measured from each rung playing
**itself**, 300 colour-paired games on seeds 1–150, split-deck. A mixed pairing measures how
an agent copes with a weaker opponent, which is a different and less interesting question
than how it plays when the opposition is competent. Reproduce with
`duel52 probe --games 300 --seed 1`.

⚠️ **This table predates the §4 mandatory-action ruling** (2026-09-03) and its `passes/game`
column describes an action that was never in the game. **F2.4b carries the re-run**, same
command, same seed. The findings drawn from it below mostly survive; where they do not, the
entry says so.

| agent | first-player score | draws | stalemate | mean plies | max lane | hand@unlock | flip rate | lane conc. | passes/game |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.520 ± 0.080 | 0.7% | 0.0% | 44.5 | **17** | 2.91 | 0.66 | 0.778 | 3.95 |
| greedy | 0.475 ± 0.079 | 1.7% | **0.7%** | 46.1 | **8** | 0.51 | 0.34 | 0.756 | 2.89 |
| pimc:32x1 | 0.480 ± 0.079 | 1.3% | 0.0% | 40.5 | 9 | 0.57 | 0.55 | 0.749 | 0.37 |
| flatmc:600 | 0.495 ± 0.080 | 0.3% | 0.0% | 45.9 | 11 | 2.51 | 0.64 | 0.777 | 1.62 |
| ismcts:800 | 0.523 ± 0.079 | 1.3% | 0.0% | 46.0 | 12 | 2.19 | 0.68 | 0.783 | 0.78 |

Two columns need a caveat before anything is read off them. **greedy and pimc share the same
hand-written evaluation** (`agents/eval.rs`), and that evaluation prices a card in hand at 0.9
against roughly 2.6 for the same card face-down on the table. So their near-empty hands and
their low flip rates are properties of *my weights*, not discoveries about Duel 52. The
evaluation-free rungs — flat MC and ISMCTS, which use nothing but random playouts — are the
ones whose behaviour is evidence, and random is the null.

**F2.4 — The stalemate rule fires, and only for the agent that has something to lose.**
`FINDINGS.md` F1.2 left the quiet-ply cutoff untested after 1.2M random games: "the reachable
stall is *strategic* … and random agents attack constantly." Now measured. Greedy self-play
reaches the cutoff in **0.7% of games** (2 in 300); random, flat MC, PIMC and ISMCTS reach it
in none.

That split is the interesting part, and it is exactly §7's mechanism: "neither player wants
to attack first because attacking exposes the attacker". Greedy is the only rung that
*evaluates material*, so it is the only one that can decline a trade — and it declines often
enough to pass 2.9 times a game and to stall outright in one game in 150. The playout-based
rungs have no notion of material at all, so they never develop the reluctance the rule guards
against.

At 4,000 greedy self-play games per variant the rate is pinned, and it varies with the
variant in a way that makes sense:

| variant | stalemate rate | all draws | mean plies |
|---|---:|---:|---:|
| base | 0.7% | 1.5% | 45.9 |
| split | 1.0% | 2.1% | 46.4 |
| mirrored | **1.7%** | 2.5% | 48.6 |

Mirrored removal is the most stall-prone, and that is the variant designed to make the two
decks rank-identical (§9b). Symmetric material produces symmetric standoffs: neither side
gets the asymmetry that would make attacking first worthwhile. Worth remembering, because
§9b is also "the cleanest target for equilibrium analysis" — the variant chosen for its
symmetry is the one where the draw rule does the most work.

**The 20-ply default is therefore validated as necessary but is still not calibrated.** It
fires, so it is not dead code; but the agent that produces the stall is the one whose
behaviour is most obviously an artifact of its weights. Re-measure in Phase 3, where a
trained value net will have a real opinion about whether a trade is good.

⚠️ **Superseded by F2.4b.** Everything above is measured on a game in which passing was
legal, which it never was. The stall was not greedy's weights; it was a phantom action.

**F2.4b — F2.4 is superseded: with actions mandatory, the stall is not reachable at all.**
The owner's ruling of 2026-09-03 (`game_rules.md` §4) is that **there is no pass** — three
actions means three, and the only short turn is the first player's opening one. F2.4's
finding was that greedy is "the only rung that can decline a trade … it declines often
enough to pass 2.9 times a game and to stall outright in one game in 150." Passing was the
whole mechanism. Re-measured at the same 4,000 greedy self-play games per variant:

| variant | stalemate rate (F2.4) | stalemate rate (now) | all draws | mean plies (F2.4 → now) |
|---|---:|---:|---:|---|
| base | 0.7% | **0.0%** | 1.5% | 45.9 → 42.9 |
| split | 1.0% | **0.0%** | 1.4% | 46.4 → 43.1 |
| mirrored | **1.7%** | **0.0%** | 1.6% | 48.6 → 43.5 |

Config: `two_power=bottom stalemate=20plies`, seed 1, 4,000 games per variant. Reproduce
with `duel52 probe --agents greedy,random --games 4000 --seed 1 --variant <v> --markdown`.

**The reason is structural, not statistical.** A player who prefers not to attack must still
spend the action, and the three non-attacking actions are each finite: a hand drains, a card
flips exactly once and nothing ever turns it face-down again, and a card joins one pair and
cannot leave it (§5). Refusal is a resource, and it runs out. Every game that used to end in
mutual refusal now ends in a fight. F2.4's variant ordering — mirrored most stall-prone,
because symmetric material produces symmetric standoffs — is gone with it; the three
variants are now indistinguishable on this axis.

The residual 1.4–1.6% draws are **not** stalemates. They are the mutual-lane-win case of §7:
the last card in a lane attacks an 8, retaliate kills the attacker as its damage kills the 8,
both sides of the lane empty, and both players win it. §7 called that "astronomically rare"
as a *game-ending* event; at 1.5% it is rare but perfectly ordinary, and it is now the only
way a Duel 52 game draws.

**What this costs.** The 20-ply quiet-ply rule is no longer validated as necessary by F2.4 —
nothing reaches it. It stays as a backstop for the position where *neither* player has a
legal action (everything frozen or out of range), which the engine must still be able to
end. Random-vs-random over the full F1 sweep — 200,000 games in each of six configurations,
1.2M total — gives **0 stalemates and 0 ply-cap draws everywhere**, with every draw a mutual
lane win at 0.4–0.5%. Reproduce with `duel52 stats --all --games 200000 --seed 1`.

**The whole F2 probe table, re-run** (`duel52 probe --games 300 --seed 1 --markdown`, the
same command that produced the original). The `passes/game` column is gone — the metric it
measured no longer exists — and `stuck/game` counts the replacement: turns that ended with
actions unspent because nothing was legal.

| agent | first-player score | draws | stalemate | mean plies | max lane | hand@unlock | flip rate | lane conc. | stuck/game |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.525 ± 0.080 | 1.0% | **0.0%** | 39.8 | 16 | 2.38 | 0.68 | 0.777 | 0.26 |
| greedy | 0.490 ± 0.079 | 2.0% | **0.0%** | 42.8 | 8 | 0.52 | 0.39 | 0.763 | 0.26 |
| pimc:32x1 | 0.513 ± 0.080 | 0.7% | **0.0%** | 40.4 | 10 | 0.57 | 0.56 | 0.749 | 0.27 |
| flatmc:600 | 0.492 ± 0.080 | 0.3% | **0.0%** | 44.6 | 12 | 2.28 | 0.64 | 0.771 | 0.21 |
| ismcts:800 | 0.518 ± 0.080 | 1.0% | **0.0%** | 45.8 | 12 | 1.94 | 0.68 | 0.786 | 0.22 |

**The last column is the one to look at.** `passes/game` ranged from 0.37 (pimc) to 3.95
(random) — an order of magnitude, and F2.4 read that spread as *behaviour*: the agents that
priced material declined actions, the ones that did not spent them. `stuck/game` is flat at
0.21–0.27 across all five rungs, including the two that used to sit at either extreme. It is
no longer a property of the agent, because it is no longer a decision. Running out of legal
actions happens to everyone at the same rate, which is what a *position* property looks like.

That flatness is what settled the follow-up question. The first fix kept `Pass` in the action
space as a forced single-option node; the owner's call was that **a non-choice does not belong
in the action space at all**, and the column is the evidence — an "action" every agent takes
at the same rate regardless of how it plays is not an action. The engine now ends such a turn
itself (`apply.rs`'s `skip_turns_with_nothing_to_do`), `Action` has no variant for it, and the
policy head lost its `PASS` logit: **1325 → 1324**, every logit now something a player
chooses. `stuck/game` survives as instrumentation, counted from outside by watching the ply
advance while the acting player still had an allowance.

**Re-measured after that removal, same command and seed**, with the stalemate column still
0.0% everywhere and every difference inside one confidence interval:

| agent | first-player score, with `Pass` | without | stuck/game |
|---|---:|---:|---:|
| random | 0.525 ± 0.080 | 0.527 ± 0.079 | 0.26 → 0.26 |
| greedy | 0.490 ± 0.079 | 0.490 ± 0.079 | 0.26 → 0.23 |
| pimc:32x1 | 0.513 ± 0.080 | 0.513 ± 0.080 | 0.27 → 0.27 |
| flatmc:600 | 0.492 ± 0.080 | 0.445 ± 0.079 | 0.21 → 0.21 |
| ismcts:800 | 0.518 ± 0.080 | 0.467 ± 0.080 | 0.22 → 0.20 |

The two rungs that moved most are ±0.05 on a statistic whose CI is ±0.08 and whose true
value is 0.5 by the symmetry of self-play, so this is noise — but noise with a specific
cause worth recording, because it is the one way a "pure removal" can fail to be pure.
**Every agent's `choose` was being called at the forced node**, and `greedy`, `pimc`,
`flatmc` and `ismcts` all determinize there, which consumes randomness. Deleting those nodes
shifts each agent's RNG stream, so identical seeds now walk different games. The aggregates
that average over thousands of events per run (flip rate, lane concentration, hand@unlock)
are unmoved to three decimals; the per-game outcome statistics resample. Over 20,000 random
games the decision count fell by ~10,000 — 0.5 per game, exactly the forced-pass nodes that
stopped existing.

`stuck/game` is now inferred rather than counted, since there is no action to intercept: the
probe watches the ply advance while the acting player still had an allowance, and attributes
any turn nobody was offered to the player who was skipped. Verified against an independent
recount of the same games —
`phase2_the_stuck_turn_count_matches_an_independent_recount` replays each game, tallies the
action-costing decisions in each ply, and calls a ply stuck when it holds fewer than its
allowance. The terminal ply is excluded from both: a turn cut short by the game ending was
not short of options.

Two other columns moved, and both matter later. **`hand@unlock` fell for every rung that
carries a hand** — random 2.91 → 2.38, flatmc 2.51 → 2.28, ismcts 2.19 → 1.94 — because a
turn with nothing better to do is now a turn that plays a card. Hoarding is materially harder
than it was when F2.5 tested it, so **H2 will need re-testing rather than re-reading**. And
**mean plies fell for the two rungs that passed most** (random 44.5 → 39.8, greedy 46.1 →
42.8) while the rungs that already spent their actions barely moved (pimc 40.5 → 40.4,
ismcts 46.0 → 45.8) — a clean confirmation that the shortening is the forfeited actions
coming back, not a change in how the game is played.

**Two side effects worth having on the record**, both from the same sweep:

- **Games got shorter: mean 45.0 → 40.1 plies** (38.5 under `two_power=discard`). Nothing
  about the game changed except that turns are now spent in full. Random play had been
  passing 3.95 times a game (F2's probe table), forfeiting roughly a third of its actions,
  and the ply count was measuring that as much as the game. F1.1's "tightly clustered around
  45" holds in shape — p10 35, median 40, p90 45 — at a lower centre.
- **The encoding bound did not move.** Max cards on one side of one lane is **19** across all
  1.2M games, against 17–20 before and an `encoding_slots` of 21. Forced actions mean more
  cards on the table per ply, so this was the number most likely to break `encoding_slots`;
  it did not. F3.1's two slots of headroom survive intact, and they are still only two.

**F2.5 — H2 is not supported: hand size at pile-empty does not predict winning.** This is the
hypothesis the project cared most about — "arguably the most valuable single output" — so it
gets the sharpest test available. Comparing hand size *across* agents confounds it with
strength and with my evaluation weights. Comparing **winners against losers inside one
agent's own self-play** holds both fixed:

| agent | hand@unlock, games it won | games it lost | difference (95% CI) |
|---|---:|---:|---:|
| random | 2.85 | 2.97 | −0.12 ± 0.21 |
| greedy | 0.53 | 0.49 | +0.04 ± 0.08 |
| pimc:32x1 | 0.60 | 0.54 | +0.06 ± 0.09 |
| flatmc:600 | 2.61 | 2.42 | +0.19 ± 0.19 |
| ismcts:800 | 2.27 | 2.11 | +0.16 ± 0.19 |

Every interval covers zero. The two evaluation-free rungs lean positive — the direction H2
predicts — but neither reaches significance, and the effect if real is under a fifth of a
card. **Stated honestly: this test can resolve a difference of about ±0.2 cards and finds
nothing above that.** H2 is not refuted; it is unsupported, and the ceiling on the effect at
this level of play is now low enough to be worth knowing.

Greedy at 4,000 games per variant tightens its own row by a factor of three and still finds
nothing: **+0.03 ± 0.03** (base), **+0.02 ± 0.03** (split), **−0.01 ± 0.03** (mirrored). That
is a genuinely tight null — but on an agent whose hand is nearly empty at the unlock either
way, so it bounds the effect only over the narrow range greedy actually explores.

There is a structural reason to keep the hypothesis alive anyway. Every rung plays **18.0
cards per game**, out of the 18 it receives — hands empty because there is nothing else to
spend actions on, not because anyone decided to spend them. Hoarding is a choice none of
these agents is capable of making deliberately, since none of them can represent "this card
is worth more unplayed". A trained value net can. Phase 3 should re-run this exact test.

**The §4 mandatory-action ruling raised the price of the hypothesis rather than settling it**
(F2.4b). "Hands empty because there is nothing else to spend actions on" is now the *rule*,
not an observation about weak agents: a turn with no flip, attack or pair available must
spend itself on a play. Measured, hand@unlock fell across the board — random 2.91 → 2.38,
flatmc 2.51 → 2.28, ismcts 2.19 → 1.94. Hoarding is still legal and still a choice, but it
is a choice that has to be *paid for* in other actions, which is a sharper and more
interesting version of H2 than the one F2.5 tested. The Phase 3 re-run is now the only test
of it that will mean anything; every number in this section was measured on the other game.

**F2.6 — H3 is not supported: strong play does not concentrate on two lanes.** Share of a
player's card-plays landing in their busiest two of three lanes: random 0.778, greedy 0.756,
pimc 0.749, flat MC 0.777, ISMCTS 0.783. The measure has no meaningful absolute scale —
picking the busiest lanes after the fact inflates it even for a uniform player, which is
exactly what random's 0.778 is showing — so read it only against that baseline. **No rung
exceeds random.** The two that use the evaluation concentrate slightly *less*.

Attack concentration tells the same story (random 0.864, ISMCTS 0.852). If the commitment H3
describes exists, no Phase 2 agent has found it, and the natural reading is that it is a
*late* decision — §7 makes lane wins an endgame event, so there is no reason to commit during
the ~25 plies of draw phase, and by the time commitment pays the game is nearly over.

**F2.7 — F1.7 reverses: the encoding bound belongs to competent play, not to the game tree.**
See also *Things we got wrong*.

F1.7 recorded 20 cards on one side of one lane under random play and concluded that
`DESIGN.md` §3's proposed 8-slot bound "is not defensible". Measured against agents:

| self-play | games | max cards on one side of one lane |
|---|---:|---:|
| random | 300 | 17 |
| greedy | 300 | 8 |
| pimc:32x1 | 300 | 9 |
| flatmc:600 | 300 | 11 |
| ismcts:800 | 300 | **12** |
| greedy | 4,000 | 9 |

§3's guess was right about *human-like* play and wrong only about random play — which is the
opposite of what F1.7 concluded from random games alone. Random agents spread cards evenly
because they have no reason not to; agents that kill things keep lanes short.

**For Phase 3 this is a real saving**: the observation tensor is `lanes × sides × slots ×
features`, so a bound of 12 rather than 21 cuts it by 43%.

Two reasons not to take 12 as the answer. First, **a maximum is not a statistic — it grows
with the sample.** Greedy peaks at 8 over 300 games and 9 over 4,000; ISMCTS's 12 comes from
300 games and would very likely be higher over 4,000. Second, a stronger agent could stack a
lane deliberately in a way none of these can. The defensible move is to **encode a bound of
16 with a hard assertion** — comfortably above every observed value, comfortably below the
theoretical 21 — and re-measure the *distribution*, not the maximum, against the trained
agent before tightening it. **Do not use 8**: greedy has already reached it.

**F2.8 — First-player advantage vanishes once both sides can play.** F1.4 measured P0 at
0.5139 ± 0.0022 under random play. In self-play at every Phase 2 rung it is indistinguishable
from even:

| agent | first-player score (95% CI) |
|---|---:|
| random | 0.520 ± 0.080 |
| greedy | 0.475 ± 0.079 |
| pimc:32x1 | 0.480 ± 0.079 |
| flatmc:600 | 0.495 ± 0.080 |
| ismcts:800 | 0.523 ± 0.079 |

Every interval covers 0.5, and the point estimates scatter on both sides of it. These
intervals are deliberately conservative: colour-paired deals mean both games of a pair deal
P0 the identical cards, so the effective sample is 150 deals rather than 300 games and the
interval is computed on that.

Greedy is cheap enough to settle it properly — 4,000 games per variant, seeds 1–2000
(`duel52 match --a greedy --b greedy --variant V --games 4000 --seed 1`):

| variant | first-player score (95% CI) | random-play value (F1.4) |
|---|---:|---:|
| base | 0.4946 ± 0.0217 | 0.5154 |
| split | 0.4918 ± 0.0217 | 0.5139 |
| mirrored | 0.5095 ± 0.0216 | 0.5148 |

**All three cover 0.5, and two sit below it.** So the +1.4-point edge F1.4 found is a
property of random play, not of the game: it survives neither the arrival of an opponent that
defends nor the change of variant.

That is a partial vindication of **H8** and a correction to F1.4's reading of it. H8 argued
from *tempo* — "the first player gets one fewer action on turn one … the tempo edge may not
compensate" — and F1.4 rightly noted that a random agent cannot use tempo, so its +1.4 points
said nothing about the hypothesis. With an agent that can, the edge is gone. The honest
summary is that P0's missing opening action and P0's head start cancel to within ±0.02 at
this level of play, in every variant.

Caveat, and it matters: greedy is 500 Elo below ISMCTS. First-player advantage is exactly the
kind of quantity that can reappear at higher skill, because using tempo is a skill. Phase 3
should re-run this against the trained agent before anyone calls it settled.

**F2.9 — Flip discipline: ISMCTS learns *which* cards to turn face-up, from random rollouts
alone.** The single most informative Phase 2 result, and the first one that is about Duel 52
rather than about search.

ISMCTS spends about as many flip actions as a random player — overall flip rate 0.68 against
random's 0.66 — but it distributes them completely differently. 300 self-play games each,
seeds 1–150 (`duel52 probe --agents random,ismcts:800 --games 300 --markdown`):

| rank | power | flipped (ISMCTS) | flipped (random) | mean flip ply (ISMCTS) | (random) |
|---|---|---:|---:|---:|---:|
| 5 | Flip | **0.86** | 0.71 | 20.3 | 21.8 |
| 8 | Retaliate *(constant)* | 0.79 | 0.62 | **18.2** | 20.2 |
| 10 | Twinstrike *(constant)* | 0.79 | 0.65 | **18.6** | 22.3 |
| 7 | Heal All | 0.77 | 0.66 | 23.0 | 20.2 |
| J | Taunt *(constant)* | 0.77 | 0.66 | **18.6** | 21.5 |
| K | Empower | 0.71 | 0.69 | 22.5 | 21.3 |
| Q | Move | 0.69 | 0.63 | 21.1 | 20.8 |
| 9 | Nimble *(constant)* | 0.64 | 0.66 | 22.4 | 21.5 |
| 2 | View | 0.63 | 0.66 | 23.4 | 21.1 |
| A | Action | 0.62 | 0.65 | 22.7 | 21.6 |
| 6 | Freeze | 0.62 | 0.68 | 22.4 | 21.9 |
| 4 | Foresight | 0.58 | 0.67 | 24.2 | 19.3 |
| 3 | Trap | **0.39** | 0.67 | **26.6** | 19.2 |

Random's column is flat by construction — 0.62 to 0.71, a spread of 0.09, and flip plies all
within ±1.5 of 21. ISMCTS's spread is **0.47, five times wider**, and the ordering is not
arbitrary. Three things fall out of it:

**It discovers that the 3 must stay face-down.** The 3's Trap is the one power in the game
that works *only while the card is face-down* (§6: "if killed while FACE-DOWN, returns
face-up at full HP"). Flipping a 3 voluntarily throws its power away. ISMCTS flips 3s at
**0.39 against random's 0.67**, and when it does, it does so six plies later than random.
Nothing told it this. It has no evaluation function, no card table, and no notion of what a 3
does — only legal moves and the win/loss at the end of uniform-random playouts. The rule is
visible in the outcomes, and the search found it.

**Constant powers get flipped first; one-shots get held.** The three ranks flipped earliest
are 8, 10 and J at plies 18.2–18.6 — three of the four *constant* powers (§6), which pay
continuously for every turn they are face-up. Every one-shot sits at ply 20.3 or later. That
is the correct shape: a constant power is worth flipping the moment it is safe to, while a
one-shot is a stored option whose value is in choosing the moment.

**The exception proves it.** The fourth constant power, the **9**, breaks the pattern — 0.64
flipped, ply 22.4, indistinguishable from random. It is the one constant power that is purely
*conditional*: Nimble does nothing unless someone tries to freeze you or you are hitting a
Jack. So it behaves like a one-shot, and the search treats it like one. A "flip your constant
powers early" heuristic would have got the 9 wrong; the search does not, because it is not
using a heuristic.

**What this does and does not license.** It is evidence about *flip timing*, which is what
`PLAN.md` Phase 2 predicted this phase would expose. It is **not** a card-value table —
nothing here says the Jack is worth more than the King, only that a Jack gets turned up
sooner. Phase 4's learned values are still the thing that answers that. Two caveats on the
numbers themselves: mean flip ply is conditioned on the card being flipped at all, and a 3
that springs its Trap returns face-up without a `Flip` action, so it is counted in neither
column — which affects random and ISMCTS identically and so leaves the comparison sound.

### F3 — The Phase 3 encoding path

Step 1 of Phase 3 builds the encoders, the network and the inference path — no training. The
one result worth recording is about the encoding bound, and it corrects a **methodological**
mistake in F2.7 rather than its number.

**F3.1 — Lane sprawl is set by the *weakest* agent in a pairing, so a self-play maximum does
not bound the ladder.**

F2.7 measured the widest lane side across **self-play** and concluded that 16 sat comfortably
above every competent agent (ISMCTS peaked at 12 over 300 games). It does. But the Phase 3
deliverable is an Elo ladder, and a ladder is not self-play: it keeps `random` as its
permanent anchor rung, forever.

| pairing | agent | games | seed | max cards on one side of one lane |
|---|---|---:|---:|---:|
| self-play | `netpolicy` (random init, 5.1M) | 100 | 1 | **10** |
| vs `random` | `netpolicy` (random init, 5.1M, ×4 inits) | 200 each | 1 | 13, 15, 15, 15 |
| vs `random` | `netpolicy` (random init, 64-wide, ×8 inits) | 200 each | 1 | 14–16 |
| vs `random` | `netpolicy` (`checkpoints/init.d52nn`, seed 0) | 100 | 1 | **17 — asserts** |
| self-play (F2.7) | `random` | 300 | — | 17 |
| self-play (F2.7) | `ismcts:800` | 300 | — | 12 |

Config: `variant=split two_power=bottom`, `encoding_slots=21` except the asserting row, which
is the default 16. Reproduce the failure exactly with:

```bash
.venv/bin/python -m duel52.nn init --out checkpoints/init.d52nn
./target/release/duel52 match --a netpolicy:checkpoints/init.d52nn --b random \
    --games 100 --seed 1        # panics: lane 2 side P0 holds 17 cards
```

The mechanism is not subtle once stated: **`random` never kills anything.** It plays cards
into lanes at the same rate as any agent and removes them at a far lower one, so a lane fills
up. Two competent agents clear each other's boards and lanes stay short; one competent agent
against a passive one produces the longest lanes of all, because one side is building and
neither is demolishing. F2.7 could not see this because every row in its table was a
*self-play* row.

So the encoding bound is really two questions, and F2.7 answered only the first:

1. **What does self-play need?** 10 here, 12 for ISMCTS in F2.7. 16 is comfortable, and the
   43% tensor saving F2.7 identified is real.
2. **What does the evaluation ladder need?** Higher, and not bounded by any measurement of
   how the *net* plays — it is bounded by how `random` plays, which F1.7 put at 17–20 and
   which no amount of training will improve.

**Recommendation, not yet applied.** `encoding_slots` should be **21** — the theoretical
maximum, and the value `max_slots_per_side` already uses, so the encoder provably cannot
assert. The cost is `obs_dim` 3300 → 4290 (+30%) and `action_dim` 1324 → 2194. That is a real
cost and it is smaller than the alternative, which is a training or evaluation run that dies
partway through a ladder. `PHASE3_STEP1.md` §1.1 fixed the default at 16 and said not to
relitigate it, so **the default is still 16** and this is flagged rather than changed; the
call belongs to the owner. Both `duel52 --encoding-slots N` and `python -m duel52.nn init
--encoding-slots N` exist, and the two must match — `encoding_slots` is what fixes `obs_dim`,
so a mismatch is a refused checkpoint rather than a silent misread.

Caveat on the agent used: a randomly-initialised policy played by argmax is not a trained
agent and is not competent. It is included here because it is what step 3's *early*
generations will look like, and because the mechanism above does not depend on the net being
bad — it depends on `random` being passive. F2.7's instruction stands: re-measure the
**distribution** against the trained agent, both in self-play and against the anchor rung.

`encode::observed_max_slots()` is a process-wide high-water mark for exactly this, and
`duel52 nn-dump` reports it.

**F3.2 — Rust and PyTorch agree on the forward pass to well inside the thresholds.** Over 256
sampled decision nodes from 20 seeded random games, with a 64-wide 3-block network:
`max |Δlogit| < 1e-3`, `max |Δvalue| < 1e-4`, masked-argmax agreement on every row, and
masked-softmax total-variation distance `< 1e-4`. Not a strategic finding — a claim that the
two implementations of `DESIGN.md` §5 are the same function, which is what makes "train in
Python, evaluate in Rust" safe. `py/tests/test_parity.py`.

**F3.3 — The observation is 4.8% dense and a position offers ~21 actions, and both numbers
paid for themselves immediately.** Over 954 decision nodes sampled from 20 seeded
random-vs-random games at `encoding_slots = 21`:

| quantity | mean | min | max |
|---|---:|---:|---:|
| non-zero observation features (of 4290) | 205 | 77 | 294 |
| legal actions (of 2195 encoded) | 20.7 | 1 | 78 |

Config: `variant=split two_power=bottom encoding_slots=21`, seed 1, from
`duel52 nn-dump --games 20 --max-rows 1024`. Self-play at 64 simulations gives the same
density to within a feature (208 per sample over 10,612 decisions).

Both are structural rather than incidental — the encoder is one-hot over rank, slot and
phase, and Duel 52 gives a player three actions from a small hand — so both were turned into
inference speed rather than merely noted:

- **The input layer walks the non-zeros.** `W_in` is kept transposed in `MlpEvaluator`, so
  each non-zero input adds one contiguous `width`-long row. Accumulation order per output is
  unchanged, so it is **bit-identical**, not approximately equal: the 954-row `nn-dump` above
  is byte-for-byte the same file before and after the change. It cut a 512-wide forward pass
  from 3.9 ms to 2.5 ms.
- **Search computes only the logits it will look at.** `MlpEvaluator::eval_masked_with`
  evaluates the policy head for the ~21 legal actions instead of all 2195, which removes
  almost the whole head from the search hot path.

Also worth recording because it sets the storage design: 4.8% density means a generation of
self-play stored as dense f32 observations is gigabytes and stored sparsely is hundreds of
megabytes, which is why `replay_shard` hands the trainer CSR triples rather than a matrix.

**F3.4 — Net-guided search throughput, and what a local training session actually buys.**
Self-play, 8 cores (M-series, 16 GB), `encoding_slots = 21`, a 128-wide 3-block network
(949,396 parameters), `variant=split`:

| simulations per decision | games/sec (8 threads) | games in 15 min |
|---:|---:|---:|
| 16 | 30.8 | 27,700 |
| 32 | 16.6 | 14,900 |
| 64 | 8.3 | 7,500 |
| 128 | 4.8 | 4,300 |

`ismcts:64`, the rollout agent at the same budget, runs at 10.7 games/sec — so a network
evaluation at this size costs about what a random playout costs, and the two agents are
comparable per simulation rather than the network being an order of magnitude dearer. Cost
is very close to **linear in simulations**, which means sims and games trade one for one and
the choice is a modelling decision rather than a budget one.

A self-play game at 64 simulations records **166 decisions** — far more than the ~65 "plies"
a match reports, because a ply is a turn's worth of the three actions plus the free
sub-decisions `DESIGN.md` §4 splits out, and every one of those is a policy target.

Two things this makes concrete for `PLAN.md` step 3:

1. **A 2–3 hour session is ~15 generations of 3000 games**, so ~45,000 self-play games and
   ~7.5M decisions. That is an assessment run — enough to see whether the loop learns, not
   enough to expect it to pass `ismcts:800`.
2. **A randomly-initialised network draws 69% of its self-play games** (64 games, seed 1,
   sims 64: P0 11%, P1 20%, draw 69%, all stalemates) and *loses to `random`*. Both are
   expected from an untrained prior — the policy is arbitrary, so it passes constantly and
   the stalemate rule at `game_rules.md` §7 fires — but they have a consequence for the
   loop: **the gate cannot be AlphaGo Zero's 0.55.** With most of the score mass exactly at
   0.5, a 0.55 bar rejects nearly every candidate and the run's teacher never advances while
   every other number looks healthy. The gate ships at 0.5, which still rejects a candidate
   that is measurably worse. Re-measure the draw rate once the run is real: if it stays near
   69% for several generations, the loop is stuck rather than slow.

**Both numbered claims above were then measured properly and one of them was wrong — see
F3.5.** They are left standing because the mistake is instructive: a 64-game sample from a
single random initialisation is not a measurement of the game.

**F3.5 — The first generation already beats the hand-written heuristic, and the fitting is
4% of the loop.** Generation 1 of `configs/train-fast.toml`, from a PyTorch-default random
initialisation, 3000 self-play games at 64 simulations, 1200 optimisation steps of batch 512
on MPS:

| | |
|---|---|
| self-play outcome mix | P0 45% · P1 46% · **draw 9%** |
| decisions recorded | 439,746 (147 per game) |
| policy loss | 7.696 → 3.116 (uniform over 2195 actions is `ln 2195` = 7.694) |
| value loss | 0.939 → 0.400 |
| gate: candidate vs the random-init incumbent | **0.858 ± 0.033** (W145 L2 D53) |
| vs `random`, 120 games | **0.929 ± 0.035** |
| vs `greedy`, 120 games | **0.558 ± 0.052** |

Config: `variant=split two_power=bottom encoding_slots=21`, run seed 1,000,000; gate and
benchmark matches at seed 1. **One generation is not a trained agent and this is not a
strength claim** — it is the claim that the loop is *connected*, which is what a first pass
is for. A policy loss starting at exactly `ln(action_dim)` and falling is the tightest
available evidence that each observation is paired with its own policy target rather than
with someone else's; a scatter bug or an off-by-one in the replay would sit at 7.69 forever.

**Where the 8m01s of one generation goes:** 6m57s self-play, 5s replay and encode, 20s
training, ~40s of gate and benchmark matches. Gradients are 4% of the wall clock, so a
larger budget goes into `selfplay.games` and `selfplay.sims` long before `net.width`.

Two corrections to what the small sample above suggested:

- **The draw rate at initialisation is a property of the initialisation, not of the game.**
  64 games from one random-init checkpoint drew 69%, all stalemates. The shipped
  configuration's generation 1, a different random init over 3000 games, drew **9%**. An
  untrained policy is arbitrary, so whether it passes constantly is arbitrary too.
- **The gate still ships at 0.5, but not for the reason F3.4 gave.** Generation 1 scored
  0.858 and would have cleared any bar. The real concern is the *late* generations, where
  gains shrink and the stalemate draw (`game_rules.md` §7) compresses the score onto 0.5 — a
  fixed margin then rejects small-but-real improvements. 0.5 still rejects a candidate that
  is measurably worse, which is the failure that compounds. Raise it once gate scores are
  routinely well above 0.5.

**F3.6 — The engine's stalemate draw is a stable equilibrium, and the first training run
walked straight into it.** Three generations of `configs/train-fast.toml` as originally
shipped (`stalemate_value = 0.5`, single-mirror gate at 0.5):

| gen | self-play draws | value targets (buffer) | policy loss | value loss | mirror gate | vs `random` | vs `greedy` |
|---:|---:|---|---:|---:|---|---:|---:|
| 1 | 9% | 46 / 10 / 44 | 7.696 → 3.116 | 0.939 → 0.400 | 0.858 (W145 L2 D53) | **0.929** | **0.558** |
| 2 | 55% | 29 / 44 / 27 | 2.955 → 2.826 | 0.547 → 0.269 | 0.502 (W1 L0 D199) | 0.600 | 0.496 |
| 3 | 88% | 13 / 75 / 12 | 2.748 → 2.659 | 0.317 → 0.163 | 0.500 (W0 L0 D200) | 0.525 | 0.500 |

Config: `variant=split two_power=bottom encoding_slots=21`, run seed 1,000,000, 3000
self-play games per generation at 64 simulations. Every generation was **promoted**.

**The mechanism.** `game_rules.md` §7 records that the published game defines no draw and
that the stalemate rule is **[ENGINE]** — added because the reachable stall never ends on its
own. Scoring it at half a point makes "neither player attacks" a stable equilibrium *of the
modified game*: a certain 0.5 beats a risky fight, for both players, at every decision. Duel
52 makes the trade especially attractive, because attacking costs material to retaliate
(§5) and lane wins are endgame-only (§2), so there is nothing to lose by waiting. The learner
found it in two generations.

It then compounded three ways:

1. **Stalled games are long**, so they are over-represented in the corpus relative to their
   share of games. Generation 2 drew 55% of its *games* and contributed ~70% of its
   *samples* — 147 decisions per game became 172.
2. **A value head trained on 75% zeros learns to predict zero**, which is why the value loss
   looks best exactly when the run is worst. `0.317 → 0.163` at generation 3 is the loss of
   a head that has learned the game is a draw.
3. **The gate could not see any of it.** Two stalling agents draw against each other, so the
   mirror match scored 0.500, 0.502, 0.500 — indistinguishable from a dead-even fight, and
   above the 0.5 threshold every time. Three promotions on no evidence at all, while the
   only honest measurement in the readout (`vs random`) fell by 0.40.

**Three fixes, and each one addresses a different link.**

- **`config.stalemate_value`** (default 0.5, training `0.0`) — what a stalemate is worth *to
  a learner*, to **both** players. At 0.0, refusing to play is no better than losing, so a
  player who is behind always prefers a gamble. Read by exactly two places, the terminal
  backup in `net_mcts` and the value targets in `selfplay`. **`Outcome::value_for` is
  untouched**, so `match`, `ladder`, the Elo fit and every number in F1 and F2 still mean
  what they meant. A *mutual lane win* keeps its half point — that one is a rule (§7), not
  an artefact. `rule_7_a_stalemate_is_not_worth_half_a_point_to_a_learner`.
- **The gate reads decisive games.** `W / (W + L)`, draws discarded, threshold back to 0.55.
  Generation 3's match then reports "0 decisive of 200", which is *no evidence* rather than
  a tie — a different thing, and the gate now treats it as one.
- **A reference panel with a veto.** The candidate plays `random` and `greedy` — opponents
  with no incentive to stall — *before* the promotion decision, and is refused if it falls
  more than 0.05 below the best score any promoted checkpoint has managed. Measured against
  a **high-water mark**, not the incumbent, so a slow give-back cannot ratchet the baseline
  down one tolerance at a time. This is the check that catches generation 2 at the moment it
  happens: mirror 0.502, `random` 0.929 → 0.600.

Shard format version 2 exists because of this: version 1 recorded one byte for "draw" and
could not distinguish an engine stalemate from a mutual lane win, so its corpus cannot be
re-valued and is refused rather than read.

**The generalisable lesson, and it is not about Duel 52.** Every *engine-defined* terminal
condition is a potential equilibrium, and the ones added for the trainer's convenience are
the most dangerous, because nothing about the real game constrains what they are worth. The
draw here was added so games would terminate; scoring it like a real draw quietly changed
which game was being solved. Anything else marked **[ENGINE]** in `game_rules.md` deserves
the same question: *what does an agent get for exploiting this, and did we mean to offer it?*

**Postscript, 2026-09-03 — the three fixes treated the symptom.** A second run under
`stalemate_value = 0.0` still collapsed: three consecutive refusals, self-play draws 16–18%,
`vs random` down from 0.963 to 0.673. Pricing the draw at zero removes the *incentive* to
stall but not the *ability*, and an agent that cannot find a plan still fills its turns with
nothing. The actual defect was upstream of the value: `Pass` was a legal action, and it was
not a rule. §4 says three actions; the engine let a player decline them. F2.4b has the
measurement after that was fixed. The lesson survives intact and gains a sharper edge — the
[ENGINE] audit should have asked not only *what is this worth?* but *is this in the rules at
all?* The pass had no ruling behind it, in any section, and nobody had looked.

### F3.7 — The first trained agent, and why the run stopped

`configs/train-fast.toml`, seed `1000000`, run dir `runs/third`. 57,000 self-play games, 19
generations on top of a random init, 1.94 h on an 8-core M-series Mac. Thirteen candidates
passed the gate; **generation 16 was the last of them** and is published as
`models/duel52-split-gen016.d52nn` (SHA-256 `03de8583…`, 949,267 parameters). `models/README.md`
carries the full provenance.

The first ladder fitted since the §4 mandatory-action ruling — 200 games per pairing, seeds
from 1, `split`:

| agent | Elo | ± | vs. anchor |
|---|---:|---:|---:|
| **netmcts:gen016@64** | **+1476** | 42 | 1.000 |
| ismcts:800 | +981 | 19 | 0.996 |
| flatmc:600 | +835 | 17 | 0.992 |
| greedy | +581 | 17 | 0.966 |
| pimc:8x1 | +547 | 17 | 0.959 |
| random | +0 | 0 | 0.500 |

**+495 Elo clear of the previous best, on one twelfth its simulation budget.** Head to head,
200 games each at seed 1: `netmcts@64` beats `ismcts:800` **0.9300 ± 0.0354** and `greedy`
0.9675 ± 0.0241; `netpolicy` — the policy head's argmax, no search at all — beats `greedy`
**0.9400 ± 0.0329** and `random` 1.0000. The fit and the head-to-head agree: a +495 gap
predicts 0.945 and the match measured 0.930.

⚠️ **This table does not compare to F2.1**, which was fitted before the §4 ruling and used
`pimc:32x1`. `ismcts:800` reading +1186 there and +981 here is two different games.

**Why the run stopped, and it is not what the log looks like.** Generations 17–19 all failed
the gate (0.503, 0.543, 0.528), which trips `max_consecutive_refusals = 3`. But at
`gate.games = 200` the decisive-score standard error is 0.035, and the threshold is 0.55:

| a generation whose true strength is… | passes the gate |
|---|---:|
| 0.50 — no improvement | 7.9% |
| 0.54 — real improvement | **38.9%** |
| 0.56 | 61.2% |
| 0.58 | 80.4% |

P(three consecutive refusals while genuinely improving at 0.54) = **22.9%**. The three refused
generations average **0.525 ± 0.040**, an interval covering both "dead flat" and "real +2.5%
per generation". **The run was ended by a measurement with no power to make the call.** Policy
loss was still falling (2.02 → 1.94) and never plateaued.

Reproduce: `duel52 ladder --games 200 --markdown --variant split --encoding-slots 21 --agents
random,greedy,flatmc:600,pimc:8x1,ismcts:800,netmcts:models/duel52-split-gen016.d52nn@64`.

### F3.8 — Search pays to 4096 sims, flat then halving, and there is no fusion signature

gen016 played against itself at different budgets, `--seed 1`, `--encoding-slots 21`:

| step (each 4× compute) | games | score | Elo | 95% CI |
|---|---:|---:|---:|---|
| `netpolicy` → `@64` | 200 | 0.8100 ± 0.054 | +252 | [+196, +321] |
| `@64` → `@256` | 200 | 0.6925 ± 0.064 | **+141** | [+91, +197] |
| `@256` → `@1024` | 200 | 0.7100 ± 0.088 | **+156** | [+87, +239] |
| `@1024` → `@4096` | 300 | 0.5983 ± 0.055 | **+69** | [+30, +110] |

**Every step excludes even, so search is still paying at 4096 — but the last step's return is
about half the two before it.** The first evidence of a knee, and it should be read carefully:
the point estimate halves, but [+87, +239] and [+30, +110] overlap, so the decline itself is
*suggestive rather than established*. What is established is that 4× more search at 4096 is
still worth a real +69 Elo.

⚠️ Measured at 60 games first, which gave 0.5667 ± 0.1254 — an interval covering even, and
useless. 300 games is the right size for this question; the power calculation says ~210. Two
experiments on the same day were under-powered the same way (the other is F3.9's first pass at
200 games). **Size the run before starting it, not after reading it.**

Two consequences.

**For training.** The policy target *is* the visit distribution, so a run at
`selfplay.sims = 64` teaches the network to imitate a search **~370 Elo** weaker than the same
weights produce at 4096. The teacher was capped, and `train-fast.toml`'s reasoning — "the
policy target only needs to be better than the current policy, not good" — is right from a
random init and stops being right once the policy is decent. The knee also says where the
useful ceiling is: the 64 → 256 step is the cheapest large gain and the one the next run
should buy.

**For the method, and this is the more important half.** F2.3 ran this exact test on PIMC:
8× more sampled worlds bought **nothing measurable**, the signature of strategy fusion. The
same test on `netmcts` buys +141, +156 and +69 Elo across three successive 4× steps — all
three excluding even. Fusion saturates under more sampling; this decays but does not saturate.
Per-simulation determinization (`net_mcts.rs`, `state.determinize` *inside* the simulation
loop) is genuinely avoiding the trap that killed PIMC. **The failure mode that would justify
abandoning AZ-over-ISMCTS is measurably not occurring**, though the 1024 → 4096 step is the
first place to look for it if the knee deepens. See `PLAN.md` Phase 4's tripwire.

**For playing it.** `@4096` is the strongest setting measured and is worth a real +69 Elo over
`@1024`, at well under a second a move — which is why the README offers it as the default
human opponent.

**Capacity is also misallocated.** Of gen016's 949,267 parameters, the input projection is
549,248 (57.9%) and the policy head 283,026 (29.8%) — both pinned by `obs_dim = 4290` and
`action_dim = 2194` — leaving **99,840, or 10.5%, in the residual trunk that does the actual
reasoning.** `blocks` 3 → 10 costs +25% parameters and takes the trunk to 28.2%; `width`
128 → 256 costs +120% for 18.9%. Depth is much the better buy here, and the config comment
recommending width first is wrong.

### F3.9 — H2 at last: the hoard is real, the causation is not established

`duel52 probe --agents netmcts:models/duel52-split-gen016.d52nn@64,greedy --games 1000
--seed 1 --encoding-slots 21`. Self-play, no exploration noise. `greedy` is the control: it is
the Phase 2 rung that prices material and still cannot hoard deliberately.

| agent | hand@unlock | won − lost | P0 score |
|---|---:|---:|---:|
| netmcts:gen016@64 | 6.64 | **+1.25 ± 0.17** | 0.5380 ± 0.0435 |
| greedy | 0.52 | −0.04 ± 0.07 | 0.5245 ± 0.0435 |

**The trained agent reaches the endgame holding 6.6 cards where `greedy` holds 0.5, and within
its own games the side holding more is the side that wins — by 1.25 cards, at ~14σ.** The
control shows −0.04 ± 0.07, reproducing F2.5's null exactly.

**F2.5's null was never a null about the game. It was a null about the agents.** No Phase 2
rung could hoard on purpose, so the within-agent test had nothing to detect. This is the
strongest support H2 has ever had, and it replicates across every pairing measured on the same
day: +1.51 ± 0.73 vs `ismcts:800`, +1.53 ± 0.40 vs `netpolicy`, +1.40 ± 0.42 and +1.09 ± 0.42
in self-play at mixed budgets.

⚠️ **Supported, not confirmed.** The measurement is correlational and the causal arrow is
ambiguous: holding cards may win games, or a winning position may simply be one that never
forces you to commit. F2.5 carried the same confound and never had to face it because its
effect was zero. `PLAN.md` Phase 4 names the two interventions that would close it.

**Also settled here: the P0 drift is not an alarm.** Training-time self-play P0 rose
monotonically 51.4% → 56.7% across the 19 generations, which looked like a policy pair sliding
off equilibrium. It does not survive clean play — 0.5380 ± 0.0435 covers even — and `greedy`
shows the same tilt at 0.5245 under identical conditions, so nothing about it is
net-specific. The training-time figure is measured *with* Dirichlet noise and temperature
sampling; the most likely explanation is that exploration damages the more delicate side.
H8 survives contact with a strong agent.

### F3.10 — The net learned flip timing keyed to power *type*

Same probe, 200 games. Mean ply at which each rank is turned face-up:

| agent | 8 | J | A | 5 | 10 | K | 6 | 3 | Q | spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| netmcts:gen016@64 | **12.9** | **15.7** | 18.1 | 20.7 | 25.3 | 27.1 | 30.3 | **32.0** | **33.8** | **21** |
| ismcts:800 | 18.9 | 20.6 | 21.2 | 20.2 | 18.9 | 22.0 | 23.9 | 26.6 | 20.4 | 5 |

`ismcts` flips everything around ply 20 regardless of rank. The net spans 21 plies, and the
ordering tracks what *kind* of power the card has:

- **Constant powers first** — the 8 (Retaliate) at 12.9 and the Jack (Taunt) at 15.7. They do
  nothing face-down, so every turn unflipped is wasted.
- **One-shot powers last** — the Queen (Move) at 33.8 and the 6 (Freeze) at 30.3. Flipping
  spends them, so hold until the board makes them worth spending.
- **The 3 latest of all, and least flipped.** Fraction of each rank flipped once played:

| agent | A | 3 | 8 | J |
|---|---:|---:|---:|---:|
| netmcts@64 | 0.93 | **0.63** | 0.98 | 0.97 |
| netmcts@256 | 0.93 | **0.59** | 0.97 | 0.98 |
| ismcts:800 | 0.68 | 0.49 | 0.78 | 0.79 |

The 3's Trap fires *if killed while face-down* (`game_rules.md` §5). It is the one card whose
power is strictly better unflipped, it is the least-flipped rank in the net's repertoire by a
wide margin against ~0.95 for everything else, and the net holds it ~6 plies longer than
`ismcts` does. **This is the clearest evidence the agent has learned structure rather than
tactics** — the ordering is not something a flat search discovers, and it is legible enough to
hand to a human player as advice.

Note `ismcts` flips less *overall* (0.49–0.79 against the net's ~0.95); the finding is in the
per-rank spread, not the level.

### Learned card values — Phase 3/4
From the value net and from ablation. F2.9 gives flip *priority* per rank, which is a
different quantity and should not be quoted as a value table. F3.10 now gives a *timing*
curve from a strong agent, which is closer but still not a value table.

### Policy characterization — Phase 4
Opening frequencies, flip timing, lane commitment, hand-hoarding curve. Three probes Phase 2
found it needed and did not have, all cheap to add:

- **Lane share restricted to post-unlock plies** (H3). A whole-game share cannot see a
  commitment that only pays in the endgame — see F2.6. **Now the priority**: H3 is the last
  Phase 2 null still resting on agents that could not deliberately do the thing, and the
  trained net's lane concentration is 0.904–0.910 against `ismcts:800`'s 0.774. That gap is
  large enough to deserve a real test rather than an eyeball.
- **King readiness at flip** (H7): reactivatable allies face-up in the King's lane when it is
  flipped, against the number available. Separates "flipped late" from "flipped when ready".
- ~~**F2.5's within-agent hand-size test, re-run against the trained net** (H2).~~ **Done —
  F3.9.** The effect is +1.25 ± 0.17 cards with `greedy` as a control at −0.04 ± 0.07. What
  remains is causal, not correlational: see `PLAN.md` Phase 4 for the two interventions.

---

## Things we got wrong

Log falsified hypotheses here rather than quietly deleting them. Knowing which intuitions
the game defeats is itself a finding, and it is the part a human player would most want to
read.

### F1.7 was backwards about the encoding bound (corrected by F2.7)

F1.7 measured 20 cards on one side of one lane under random play and concluded that
`DESIGN.md` §3's proposed 8-slot bound "is not defensible" — that §3 had described human play
and mistaken it for the game. The correction is that §3 was describing the right thing and
F1.7 was measuring the wrong one. Competent self-play tops out at **8–12** cards per side; it
is *random* play that sprawls to 17–20, because a random agent never kills anything and has
no reason to prefer one lane over another.

The methodological lesson generalises past this number: **random play is not a conservative
upper bound on the shape of real play, it is a different distribution.** F1.7 assumed the
first. Any Phase 3 sizing decision taken from Phase 1 statistics should be re-derived from
F2.7's table instead.

### F2.7 measured the wrong pairing (corrected by F3.1)

The number was right and the experiment was not general enough. F2.7's table is entirely
**self-play** rows, and it drew a conclusion about the encoding bound — a bound that has to
hold everywhere the encoder runs, which includes the Elo ladder, which includes `random`.
Sprawl turns out to be set by the *weakest* agent in a pairing, because `random` never kills
anything, so a competent agent against `random` produces longer lanes than either produces
against itself. 16 survives self-play and does not survive the ladder.

The lesson is a sibling of F1.7's rather than a repeat: F1.7 measured the wrong *agent*, F2.7
measured the wrong *pairing*. Both times the failure was assuming one distribution stood in
for another. A bound that must hold under all conditions has to be measured under the
condition that stresses it, not the condition that is most interesting.

### "More determinizations" is not a knob (F2.3)

The implicit assumption in building PIMC with a `worlds` parameter was that it was the
strength dial, with depth as a cost problem. It is the other way round: eight-fold more
worlds is flat, one more ply of search is worth +0.28 score against the same opponent.
Sampling more of a biased estimator estimates the bias more precisely.

### The greedy agent was quietly cheating, and nothing about search caused it

Not a hypothesis, but the most useful mistake of the phase. Greedy does no search and reads
only ranks it is entitled to know, so it looked exempt from `DESIGN.md` §6's determinization
discipline. It was not: one-ply lookahead *applies* a candidate action to the real state, and
applying reveals things — flipping your own base card turns it face-up (§3 says you did not
know it), and killing a face-down card sends its rank to the public discard (§5). Greedy was
choosing whether to flip after seeing what it would flip.

It was caught by `phase2_no_agent_reads_hidden_information` on the first run, not by review.
The property that makes that test exact is worth remembering: a determinized world is in the
same information set as the real one, so **any honest agent must return the same action from
either**. Every future agent gets that test for free.
