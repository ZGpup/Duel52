# Duel 52 — Findings

What we actually learn about the game. **This file is the point of the project.**

**Status: Phase 2 measured.** Five agents on a frozen Elo ladder, plus instrumented
self-play. Six of the eight hypotheses now have data:

| | verdict | where |
|---|---|---|
| H1 — the draw phase is positional | untested directly, but implied by H3's null | F2.6 |
| H2 — hand size at pile-empty is the resource | **unsupported**, effect bounded under ±0.2 cards | F2.5 |
| H3 — optimal play concentrates on two lanes | **unsupported**, no rung beats the random baseline | F2.6 |
| H4 — information is worth less than tempo | **split** — the 4 is weak, but the framing is wrong | F2.9 |
| H5 — the Jack is the strongest card | flip-timing half **confirmed**, value half open | F2.9 |
| H6 — the 7 scales with board commitment | timing half **supported** | F2.9 |
| H7 — the King is a combo enabler | consistent, too weak to call | F2.9 |
| H8 — first-player advantage is small | **confirmed**, and it is zero, not small | F2.8 |

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

### Learned card values — Phase 3/4
From the value net and from ablation. F2.9 gives flip *priority* per rank, which is a
different quantity and should not be quoted as a value table.

### Policy characterization — Phase 4
Opening frequencies, flip timing, lane commitment, hand-hoarding curve. Three probes Phase 2
found it needed and did not have, all cheap to add:

- **Lane share restricted to post-unlock plies** (H3). A whole-game share cannot see a
  commitment that only pays in the endgame — see F2.6.
- **King readiness at flip** (H7): reactivatable allies face-up in the King's lane when it is
  flipped, against the number available. Separates "flipped late" from "flipped when ready".
- **F2.5's within-agent hand-size test, re-run against the trained net** (H2). The Phase 2
  agents cannot hoard deliberately, so the null is not yet a fair test of the hypothesis.

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
