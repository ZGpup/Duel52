# Duel 52 — Findings

What we actually learn about the game. **This file is the point of the project.**

Newest first. Phase 3 produced the first strong agent and everything above the Phase 2
section was measured on it; Phase 1 and Phase 2 are kept below as baselines and controls,
not as results in their own right.

**Status: Phase 3 trained and measured.** `models/duel52-split-gen016.d52nn` is the first
trained agent and the first strong Duel 52 player that exists — **+495 Elo clear** of the
five hand-written rungs (F3.7). Six of the eight hypotheses have data, and the one the
project cared most about has moved.

| | verdict | where |
|---|---|---|
| H1 — the draw phase is positional | **supported** — the strong agent spends the phase accumulating, not scoring | §Strong play, F3.9 |
| H2 — hand size at pile-empty is the resource | **supported** — +1.25 ± 0.17 cards, ~14σ, with a control. Causation open | **F3.9** |
| H3 — optimal play concentrates on two lanes | **reopened** — Phase 2's null was agent-limited; the trained net concentrates far above the random baseline | §Strong play, F2.6 |
| H4 — information is worth less than tempo | **supported**, and the framing sharpened — reveal is cheap, per-rank timing is the real quantity | F3.10, F2.9 |
| H5 — the Jack is the strongest card | flip-timing half **confirmed**, value half still open | F3.10, F2.9 |
| H6 — the 7 scales with board commitment | timing half **supported** | F2.9 |
| H7 — the King is a combo enabler | consistent, too weak to call | F2.9 |
| H8 — first-player advantage is small | **confirmed**, and it is zero — survives contact with a strong agent | F3.9, F2.8 |

⚠️ **Read every Phase 2 null with this in mind.** H2 was recorded as "unsupported, effect
bounded under ±0.2 cards" for the whole of Phase 2. It was wrong, and not because the
measurement was bad — F3.9 reproduces F2.5's null *exactly* on `greedy`, using the same test.
**No Phase 2 agent could hoard deliberately, so the within-agent test had nothing to detect.**
A null from an agent that cannot do the thing is not a null about the game. H3 is the other
hypothesis in that position, and the trained agent's numbers say it is about to move the same
way.

## Recording standard

Every finding gets: the **claim**, the **config** (variant, rules version), the **agent** that
produced it, the **seed range**, the **sample size**, and a **confidence interval**. A number
without reproducible provenance is not a finding — it is a vibe.

Two standing traps, both of which have caught this file already:

- **Size the experiment before running it, not after reading it.** F3.8 and F3.9 were each
  first measured at a sample size whose interval covered even.
- **A statistic with no absolute scale must be read against `random` in the same table.**
  Lane and attack concentration are both of this kind.

---

## What strong play looks like

A portrait of `models/duel52-split-gen016.d52nn` — **one agent, from one two-hour run**, and
every claim below inherits that. It is nonetheless the only strong Duel 52 player that exists,
+495 Elo clear of anything hand-written (F3.7), and it plays a game that no agent before it
played. The evidence is F3.7–F3.10 plus the run below; what follows is what those numbers appear
to *mean*, which is a weaker kind of claim and is flagged where it gets weakest.

**Provenance for every table in this section.** One post-§4-ruling self-play probe, all five
agents in the same table so the scale-free statistics are comparable, 400 games each, seeds
from 1:

```bash
duel52 probe --games 400 --seed 1 --encoding-slots 21 --markdown --agents \
  random,greedy,ismcts:800,netmcts:models/duel52-split-gen016.d52nn@64,netmcts:…@256
```

| agent | hand@unlock | won − lost | flip rate | lane conc | attack conc | mean plies | stuck/game | max lane |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 2.36 | −0.09 ± 0.24 | 0.678 | 0.777 | 0.869 | 39.8 | 0.25 | 17 |
| greedy | 0.52 | −0.02 ± 0.10 | 0.385 | 0.762 | 0.871 | 42.8 | 0.23 | 8 |
| ismcts:800 | 1.96 | **+0.37 ± 0.22** | 0.690 | 0.774 | 0.843 | 45.6 | 0.21 | 13 |
| **netmcts@64** | **6.60** | **+1.14 ± 0.27** | **0.875** | **0.904** | **0.914** | 46.8 | 0.57 | 8 |
| **netmcts@256** | **6.55** | **+1.00 ± 0.26** | **0.869** | **0.906** | **0.914** | 47.3 | 0.54 | 7 |

**Read the two net rows against each other first.** Quadrupling the search budget moves every
behavioural statistic by less than its noise — hand@unlock 6.60 → 6.55, lane concentration
0.904 → 0.906, flip rate 0.875 → 0.869. **The strategy is in the policy, not in the tree.**
Search buys +141 Elo across that step (F3.8) by executing the same plan more accurately, not by
adopting a different one. Everything below is therefore a property of the trained weights and
should survive a change of search budget.

### The clock: ply 25 partitions the game, and the partition is exact

Under the house rule for the 2, the split-deck pile empties on **ply 25 in every single game,
with no variance at all** (F1.6). §7 makes lane wins impossible before that. So Duel 52 is not
one game but two, joined at a seam whose position is known to both players from the deal:

- **Plies 1–25 — the draw phase.** Nothing can be won. Nothing can be *lost*.
- **Plies 26–~47 — the endgame.** No new resources; the net's self-play runs 46.9 plies, so
  this is about 22 plies, roughly 11 turns each.

Every player begins with the same budget: 26 cards, of which 3 go to base, 5 to the opening
hand, 5 are removed unseen, and 13 form the personal pile (§9a). **18 cards pass through a
player's hand over a whole game**, and the seam falls after the last of them is drawn.

### 1. The trained agent does not spend its hand, and this is the largest behavioural gap in the project

At the seam, out of the 18 cards that pass through a hand:

| agent | hand@unlock | share of its 18 |
|---|---:|---:|
| **netmcts:gen016** | **6.6** | **37%** |
| random | 2.36 | 13% |
| ismcts:800 | 1.96 | 11% |
| greedy | 0.52 | 3% |

Every Phase 2 agent plays **18.0 of its 18 cards** over a game (F2.5); the net plays 15.5–17.6.
It is the only agent measured in this project that finishes a game with cards unplayed.

**The reason this matters is a piece of §7 arithmetic that the rules document states
defensively and that the agent appears to have found the offensive reading of.** §7 requires
three things for a lane win, and the third is *the opponent's hand is empty*. So consider two
players at the seam, one holding cards and one not:

- The player whose hand is **empty** can be scored against: their opponent's third condition is
  already satisfied, permanently.
- The player still **holding** cards cannot be scored against at all: the condition fails on
  their side until the hand runs out.

That is not a defensive resource. It is a **unilateral scoring window**, and it opens the moment
one player's hand empties before the other's. `game_rules.md` §7's own summary — "a held card
buys time; it does not buy safety" — is true about the *absolute* count and misses what the
*differential* does. Time is not what the last card buys. Exclusivity is.

F3.9 measures the payoff inside the net's own games and it is the largest effect in this file:
the side holding more at the seam is the side that wins, by **+1.25 ± 0.17 cards (~14σ)** over
1000 games, with `greedy` reproducing F2.5's null at −0.04 ± 0.07 as a control.

**And the same-run probe adds something F3.9's two-agent design could not show: the effect is
graded with strength.** The within-agent hand-size gap across the whole ladder, one table, one
seed range:

| agent | won − lost, cards at the seam |
|---|---:|
| random | −0.09 ± 0.24 |
| greedy | −0.02 ± 0.10 |
| ismcts:800 | **+0.37 ± 0.22** |
| netmcts@256 | **+1.00 ± 0.26** |
| netmcts@64 | **+1.14 ± 0.27** |

This is a dose–response curve, and it is much better evidence for H2 than the null-versus-effect
framing F3.9 used. Two agents show nothing, and then the effect appears and grows monotonically
with playing strength. **Note in particular that `ismcts:800` now excludes zero** — pre-ruling,
F2.5 measured it at +0.16 ± 0.19 and could not separate it from noise. The strongest hand-written
rung was always slightly on the right side of this; it took 400 post-ruling games to see it.

A confound that a graded result does *not* remove: strength and hand size are correlated across
agents by construction, since stronger agents both hoard more and win more. The dose–response
is within-agent at each row, which is the right design, but the *ordering* of the rows is not an
experiment. See F3.9's caveat.

**The limit is real too, and the rules name it.** A held card is a *wasting* asset: §4 makes
three actions mandatory, so the window lasts only as long as you have flips, attacks or pairs to
spend actions on instead. That is consistent with the net's other statistics — it flips far more
than anything else (below), which is precisely the supply of non-card actions that lets it keep
the hand closed.

**H1 and H2 are the same finding seen from two sides.** H1 says the draw phase is positional;
H2 says hand size at the seam is the resource. An agent that believed the draw phase decided
anything would spend cards during it. This one does not, and it wins.

### 2. Reveal is cheap; concealment is worth almost nothing

Face-down cards are blank 2-HP bodies (§5) — concealment buys exactly one thing, that the
opponent does not know the rank, and costs the whole of the card's power. The trained agent's
verdict on that trade is close to unanimous:

| agent | overall flip rate | per-rank range, 3 and 2 excluded | the 3 | the 2 |
|---|---:|---:|---:|---:|
| **netmcts:gen016** | **0.87** | **0.80–0.97** | **0.59–0.64** | 0.71–0.74 |
| ismcts:800 | 0.69 | 0.49–0.86 | 0.49 | 0.62 |
| random | 0.68 | 0.64–0.71 | 0.67 | 0.66 |
| greedy | 0.39 | 0.13–0.93 | 0.15 | 0.13 |

The net flips 0.80–0.97 of every rank it plays with exactly two exceptions, and both are
informative. **The 3** is the one card whose power *requires* darkness — Trap fires only if it
is killed face-down — and it is the lowest at 0.59–0.64. **The 2** is next-lowest at 0.71–0.74;
its View power is a one-shot that reads hidden information, so holding it is holding an option.
Everything else the net turns face-up almost on sight.

`greedy`'s row is in the table as a warning rather than a data point: its 0.13–0.93 spread is
the hand-written evaluation's opinion (it flips 6s at 0.93 and 7s at 0.82 and almost nothing
else), which is exactly why this file does not treat greedy's preferences as evidence about
Duel 52.

So the honest statement of H4 is not "information is worth less than tempo" but something more
specific and more useful: **there is no general information/tempo trade in Duel 52.** Hiding a
card is simply bad, with one rank-specific exception, and the exception exists because a power
is conditioned on hiddenness rather than because hiddenness is valuable. A human player looking
for edges in concealment is looking in a place the strongest available agent says is empty.

### 3. Commitment: H3 is standing exactly where H2 was

| | lane concentration | attack concentration |
|---|---:|---:|
| **netmcts:gen016** | **0.904–0.906** | **0.914** |
| random *(the only meaningful baseline)* | 0.777 | 0.869 |
| ismcts:800 | 0.774 | 0.843 |
| greedy | 0.762 | 0.871 |

F2.6 recorded H3 as unsupported because **no Phase 2 rung exceeded random** — and this run
reproduces that exactly, post-ruling and in the same table: `random` 0.777, `ismcts` 0.774,
`greedy` 0.762, all three inside a whisker of each other. **The trained agent is 0.13 above the
baseline, and it is the only agent in the project's history to exceed it at all.** Attack
concentration moves the same way, 0.914 against 0.869.

This is H2's situation repeated one hypothesis later: a null measured on a population incapable
of the behaviour. H3 should be read as **reopened, not refuted**, and it is now the last Phase 2
null still resting on that population.

⚠️ **One reason to hold back from calling it.** Lane share over a whole game is the wrong
statistic. H3's own second clause predicts commitment is a *late* decision, and §7 gives no
reason to commit before the seam at all — so a whole-game share dilutes whatever happens in the
endgame across 25 plies where nothing is at stake. **The measurement that would settle H3 is
lane share restricted to post-unlock plies.** Nothing collects it yet; it is the cheapest
outstanding item in `PLAN.md` Phase 5.

The pre/post-ruling worry that an earlier draft of this section carried is now discharged: the
baseline above is from the same run as the net's numbers, and it lands within 0.001 of F2.6's
pre-ruling 0.778. The §4 ruling did not move this statistic.

### 4. The powers sort by *when they start paying*, and that is not the constant/one-shot line

The full thirteen-rank curve, mean ply of the flip, same run:

| agent | 8 | J | A | 4 | 2 | 7 | 5 | 9 | 10 | K | 6 | 3 | Q | spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| netmcts@64 | **12.2** | **14.9** | 18.4 | 19.0 | 19.1 | 20.6 | 20.9 | 21.1 | 24.3 | 27.2 | 30.5 | 32.5 | **34.2** | **22.3** |
| netmcts@256 | **12.0** | **14.2** | 17.8 | 20.1 | 20.7 | 20.8 | 21.5 | 21.5 | 23.7 | 27.2 | 29.3 | 34.5 | **33.6** | **22.5** |
| ismcts:800 | 18.0 | 19.8 | 22.0 | 23.8 | 24.4 | 22.6 | 20.1 | 23.0 | 17.6 | 23.3 | 23.2 | 27.2 | 20.8 | 9.6 |
| random | 18.8 | 20.0 | 19.2 | 18.3 | 18.9 | 19.4 | 19.4 | 20.1 | 18.4 | 19.5 | 18.3 | 19.1 | 19.7 | 1.8 |

Random is flat by construction (1.8 plies). ISMCTS spans 9.6. **The net spans 22.3 and the order
is stable between its two search budgets** — the two net rows agree on every rank to within
about two plies, which is what makes the ordering worth interpreting at all.

The ordering does *not* split on the constant/one-shot line that F2.9 proposed. It splits on
**whether the power pays without you spending anything to make it pay**:

| class | trigger | net's timing | ranks |
|---|---|---|---|
| **Passive constant** | pays every turn, no action, no condition | **flip on sight** | 8 Retaliate (12.2), J Taunt (14.9) |
| **Conditional constant** | constant, but needs a board state to matter | mid | 9 Nimble (21.1), 10 Twinstrike (24.3) |
| **Cheap one-shot** | fires on flip, low value | early — spend it before it is wasted | A (18.4), 4 (19.0), 2 (19.1) |
| **Expensive one-shot** | fires on flip, high value | **hold** | K Empower (27.2), 6 Freeze (30.5), Q Move (34.2) |
| **Hidden-trigger** | pays *because* it is face-down | never, if avoidable | 3 Trap (32.5, flipped 0.59–0.64) |

Three things fall out of that table which no previous finding could see:

**The refinement is about conditionality, not duration.** F2.9 read the split as constant
powers early, one-shots late, with the 9 as an anomaly. The net's curve says the 9 was never an
anomaly: **10 Twinstrike sits beside it** at 24.3, and both are constant powers that need
something to be true — Twinstrike needs two targets, Nimble needs an opponent trying to freeze
you or a Jack to hit. Only the two *unconditionally* passive powers get flipped immediately. The
rule is "flip it when it starts paying", and for the 8 and the Jack that is instantly.

**The two strong agents disagree about the 10, and the disagreement is legible.** ISMCTS flips it
*earliest of all thirteen* (17.6); the net flips it ninth (24.3). Twinstrike's value depends on
having two enemy targets in a lane, which is a board condition a random rollout is poor at
evaluating and a value head is not. This is the one place the two methods contradict each other,
and it is a reason to trust the net's curve over ISMCTS's where they differ.

**Among one-shots, the hold time looks like a value ranking.** The net spends A, 4 and 2 at
18–19 and holds K, 6 and Q to 27–34. Those are the powers whose worth depends on arranging the
board first — Empower, Freeze, Move — against three whose worth is small and fixed. H4 already
established the 4 is weak; this puts it in a group. **It is still not a card-value table** — hold
time is not value, and H5 stays open — but it is the closest thing this project has produced, and
it is the first ordering of the one-shots by anything other than a hand-written weight.

**Credit for the discovery is not the net's.** ISMCTS found the shape first (F2.9) from uniform
random rollouts, with no evaluation function and no card knowledge. What the trained agent adds
is **resolution and conviction** — a 22-ply spread against 9.6, per-rank flip rates of 0.80–0.97
against a mild 0.49–0.86 preference, and a curve that holds steady when the search budget
quadruples. Two independent methods agreeing on most of the ordering is much better evidence
about *Duel 52* than either alone: this is the only conclusion in this section that does not rest
on a single training run.

### 5. One anomaly, unexplained

The net is left with nothing to do **more than twice as often** as any other agent: 0.54–0.57
stuck turns per game against 0.21–0.25 for random, greedy and ISMCTS alike. `probe` counts a
stuck turn when a turn ended with action allowance unspent, which requires the legal-action list
to have gone empty mid-turn — and since §4 made actions mandatory, `legal_actions()` is empty
only when there is genuinely nothing available.

This is the opposite of what the hoarding story predicts. An agent holding 6.6 cards should
*always* have a card to play, and playing one is an action. So either the net reaches
hand-empty-and-board-locked states more often than agents that empty their hands 25 plies
earlier — which is strange — or something about how it spends its last cards leaves it in
positions with no legal action.

**It is flagged rather than explained.** It may be nothing: 0.57 stuck turns in a 47-ply game is
rare. But it is the only statistic in the table where the strongest agent looks worse than
`random`, and that is exactly the shape a residual bug takes. Worth an hour with
`duel52 play --reveal` before the next training run.

### What this says about the game

1. **It is an endgame game with a long, deterministic prologue.** Twenty-five plies in which
   nothing can be decided, and the strongest agent uses them to accumulate rather than to fight.
2. **The scarce resource is not material — it is the right to refuse.** The win condition is
   gated on your opponent's hand being empty, so the last player still holding cards owns a
   window in which only they can score. That is the sharpest thing the agent has taught us and
   it is legible enough to give a human player tonight.
3. **The hidden-information layer is thinner than it looks.** A game where cards are played
   face-down reads as a game about concealment. The strongest agent flips 0.80–0.97 of every
   rank but two. The hiding is a *cost* the game imposes, not a resource it offers — except for
   the 3, whose power is conditioned on it, and mildly for the 2.
4. **The card powers are not thirteen special cases, and the axis is conditionality.** Flip a
   power the moment it starts paying. For the 8 and the Jack that is immediately; for Twinstrike
   and Nimble it is when the board supplies the condition; for Empower, Freeze and Move it is
   when you have arranged what they will act on; for Trap it is never. That is a rule a human
   can hold in their head, and it is derived rather than designed.
5. **Concentration is real, and it was invisible to every method that preceded this one.**
   0.90 against a 0.777 baseline that four other agents sit on top of.

### What this cannot tell us

- **It is one agent from one 2-hour laptop run**, stopped by a promotion gate with no power to
  make the call (F3.7). Nothing here is an equilibrium; it is what one training run found.
- **F3.9 is correlational.** The hand-size effect could run either way, and `PLAN.md` Phase 5
  names the interventions that would separate them. The §7 argument in §1 above makes the causal
  direction *plausible* — it is not a substitute for the experiment.
- **There is still no card-value table.** Flip *priority* and flip *timing* are not value, and
  this file has said so since F2.9. H5 — is the Jack the strongest card — remains open, and the
  timing data does not touch it.
- **The lane-commitment picture is a whole-game statistic** answering a question about the
  endgame, and needs the post-unlock probe before it is worth calling.
- **The dose–response across agents is not an experiment.** Strength and hoarding are correlated
  by construction. Each row is a valid within-agent test; the ordering of the rows is
  observational.
- **Everything here describes `split` only.** The observation layout is per-variant, so gen016
  cannot even be loaded against `base` or `mirrored` (`models/README.md`). Whether the seam, the
  hoard and the concentration survive a shared draw pile is untested, and F1.5 is a warning that
  pile *sharing* changes the game in ways that are easy to miss.
- **The owner still beats it.** That is the most informative unrecorded signal in the project:
  if a human wins the same way repeatedly, that is a systematic blind spot and it is diagnosable.
  Nothing captures human games yet.

---

## Phase 3 — the trained agent

### F3.10 — The net learned flip timing keyed to power *type*

`duel52 probe --games 400 --seed 1 --encoding-slots 21`, all thirteen ranks, mean ply of the
flip. Full table and the classification it implies are in **§Strong play 4**; this is the
finding, that section is the reading.

| agent | 8 | J | A | 4 | 2 | 7 | 5 | 9 | 10 | K | 6 | 3 | Q | spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| netmcts@64 | **12.2** | **14.9** | 18.4 | 19.0 | 19.1 | 20.6 | 20.9 | 21.1 | 24.3 | 27.2 | 30.5 | 32.5 | **34.2** | **22.3** |
| netmcts@256 | **12.0** | **14.2** | 17.8 | 20.1 | 20.7 | 20.8 | 21.5 | 21.5 | 23.7 | 27.2 | 29.3 | 34.5 | **33.6** | **22.5** |
| ismcts:800 | 18.0 | 19.8 | 22.0 | 23.8 | 24.4 | 22.6 | 20.1 | 23.0 | 17.6 | 23.3 | 23.2 | 27.2 | 20.8 | 9.6 |
| random | 18.8 | 20.0 | 19.2 | 18.3 | 18.9 | 19.4 | 19.4 | 20.1 | 18.4 | 19.5 | 18.3 | 19.1 | 19.7 | 1.8 |

Random is flat by construction. **The net spans 22.3 plies against ISMCTS's 9.6, and its
ordering is stable across a 4× change in search budget** — every rank agrees to within about two
plies between the two net rows, which is what licenses reading the order at all.

The split is on **conditionality, not duration**: the only two powers flipped immediately are the
8 (Retaliate) and the Jack (Taunt), the two that pay every turn with no action and no board
condition. The other two constant powers — 9 Nimble and 10 Twinstrike — need something to be
true, and the net puts both mid-table at 21.1 and 24.3.

**The 3 is the one card the net keeps face-down.** Fraction of each rank flipped once played:

| agent | overall | the 3 | the 2 | everything else |
|---|---:|---:|---:|---:|
| netmcts@64 | 0.875 | **0.64** | 0.74 | 0.85–0.97 |
| netmcts@256 | 0.869 | **0.59** | 0.71 | 0.80–0.97 |
| ismcts:800 | 0.690 | 0.49 | 0.62 | 0.60–0.86 |
| random | 0.678 | 0.67 | 0.66 | 0.64–0.71 |

Trap fires *if killed while face-down* (§5), so it is the one power strictly better unflipped.
The 2's View is next-lowest and is a one-shot that reads hidden information, so holding it holds
an option.

⚠️ **Correction to the original wording of this finding, 2026-09-04.** It claimed the ordering
"is not something a flat search discovers". **That is false and F2.9 is the counter-example** —
ISMCTS found the same structure from uniform random rollouts with no card knowledge at all: the
3 at 0.39 against random's 0.67, and constant powers flipped earliest. The net did not discover
this. What it adds is **resolution**: a 22.3-ply spread against 9.6, per-rank rates of 0.80–0.97
against a 0.49–0.86 preference, and a curve that holds when the search budget quadruples. A weak
search finds the *shape*; a trained policy finds the *conviction*.

**And one place they disagree, which is worth more than the agreements.** ISMCTS flips the 10
*earliest of all thirteen* (17.6); the net flips it ninth (24.3). Twinstrike needs two enemy
targets in a lane to be worth anything — a board condition random rollouts evaluate badly and a
value head does not. F2.9 treated the 9 as the lone anomaly in a constant-powers-first rule; the
net's curve says the 9 and the 10 are a *class* — conditional constants — and the rule was never
about duration. §Strong play 4 has the full classification.

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

**Extended 2026-09-04: the effect is graded with strength, and ISMCTS is no longer null.** The
same test over the whole ladder in one 400-game run (§Strong play 1) gives random −0.09 ± 0.24,
greedy −0.02 ± 0.10, **`ismcts:800` +0.37 ± 0.22 — excluding zero**, netmcts@256 +1.00 ± 0.26,
netmcts@64 +1.14 ± 0.27. F2.5 measured ISMCTS at +0.16 ± 0.19 pre-ruling and could not separate
it from noise. A monotone dose–response across five agents is better evidence than the
null-versus-effect contrast this finding originally rested on — though the *ordering* of the
rows is observational, since strength and hoarding are correlated by construction.

⚠️ **Supported, not confirmed.** The measurement is correlational and the causal arrow is
ambiguous: holding cards may win games, or a winning position may simply be one that never
forces you to commit. F2.5 carried the same confound and never had to face it because its
effect was zero. `PLAN.md` Phase 5 names the two interventions that would close it.

**Also settled here: the P0 drift is not an alarm.** Training-time self-play P0 rose
monotonically 51.4% → 56.7% across the 19 generations, which looked like a policy pair sliding
off equilibrium. It does not survive clean play — 0.5380 ± 0.0435 covers even — and `greedy`
shows the same tilt at 0.5245 under identical conditions, so nothing about it is net-specific.
The training-time figure is measured *with* Dirichlet noise and temperature sampling; the most
likely explanation is that exploration damages the more delicate side. H8 survives contact
with a strong agent.

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

Three consequences.

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
first place to look for it if the knee deepens. See `PLAN.md` §4.8, the tripwire.

**For playing it.** `@4096` is the strongest setting measured and is worth a real +69 Elo over
`@1024`, at well under a second a move — which is why the README offers it as the default
human opponent.

**Capacity is also misallocated.** Of gen016's 949,267 parameters, the input projection is
549,248 (57.9%) and the policy head 283,026 (29.8%) — both pinned by `obs_dim = 4290` and
`action_dim = 2194` — leaving **99,840, or 10.5%, in the residual trunk that does the actual
reasoning.** `blocks` 3 → 10 costs +25% parameters and takes the trunk to 28.2%; `width`
128 → 256 costs +120% for 18.9%. Depth is much the better buy here, and the config comment
recommending width first is wrong.

### F3.7 — The first trained agent, and why the run stopped

`configs/train-fast.toml`, seed `1000000`, run dir `runs/third`. 57,000 self-play games, 19
generations on top of a random init, 1.94 h on an 8-core M-series Mac. Thirteen candidates
passed the gate; **generation 16 was the last of them** and is published as
`models/duel52-split-gen016.d52nn` (SHA-256 `03de8583…`, 949,267 parameters).
`models/README.md` carries the full provenance.

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

---

## Phase 3 — building the loop

Infrastructure findings, newest first. Nothing here is about Duel 52; F3.6 is about training
in general and is the one worth reading twice.

### F3.11 — The trunk costs self-play throughput roughly linearly, and the GPU is irrelevant at this model size

Two measurements taken to size the Phase 4 run. Config `variant=split two_power=bottom
encoding_slots=21 stalemate_value=0.0`, seed 1, 8-core M-series Mac, 16 GB.

**Self-play throughput against trunk size.** 128 games per row at 64 simulations, 8 threads,
one random-init checkpoint per shape (`python -m duel52.nn init --seed 7`). Decisions/sec
rather than games/sec, because game length varies by 125–137 decisions across random inits and
the per-decision cost is the quantity of interest:

| trunk | params | games/sec | decisions/game | decisions/sec | throughput |
|---|---:|---:|---:|---:|---:|
| 128 × 3 — gen016's shape | 949,267 | 7.9 | 136.5 | 1078 | 1.00× |
| 128 × 6 | 1,049,107 | 5.6 | 128.7 | 721 | 0.67× |
| 128 × 10 | 1,182,227 | 3.5 | 134.6 | 471 | 0.44× |
| 256 × 3 | 2,092,691 | 2.6 | 124.7 | 324 | 0.30× |

**This is the throughput half of F3.8's capacity argument, and it changes the conclusion's
size without changing its direction.** F3.8 counted parameters and found `blocks` 3 → 10 costs
+25% of them against `width` 128 → 256's +120%, so depth is the better buy. It is — but
parameters are not what inference costs, because the input projection is walked sparsely
(F3.3) and only the legal logits are evaluated. What inference costs is the *dense* trunk, and
3 → 10 blocks costs **2.3× the wall clock of every self-play game** while 128 → 256 costs
3.3×. Both ratios track the FLOP count almost exactly (2.6× and 3.2× predicted), which implies
the forward pass is **~80% of per-simulation cost** — worth recording on its own, because the
note filed when the sparsity work landed said the network was no longer the dominant cost.
It is, again, now that the cheap wins are taken.

Consequence for Phase 4: depth trades against games at about 1.5× per 3 blocks, so the trunk
should be picked from the compute budget rather than from the parameter table. `PLAN.md` §4.3
carries the sizing formula this feeds.

**The gradient step does not need a GPU.** One forward+backward+AdamW step at the real shapes
(`obs_dim` 4290, `action_dim` 2194, batch 512, 205 non-zeros and 21 target actions per row,
100 timed steps after 5 warm-up):

| trunk | 8 CPU cores | MPS | speedup |
|---|---:|---:|---:|
| 128 × 3 | 8.3 ms | 5.9 ms | 1.4× |
| 128 × 10 | 13.8 ms | 9.1 ms | 1.5× |

A laptop GPU beats eight laptop CPU cores by 40% on a 1.2M-parameter MLP, which is what a
model far too small to fill a GPU looks like. Combined with F3.5's breakdown — gradients are
4% of a generation's wall clock and self-play is 87% — **a GPU changes ~1.5% of this loop.**
The rented-hardware decision is therefore a core-count decision; see `PLAN.md` §4.1. A GPU
becomes relevant only behind a batched `Evaluator` doing *self-play* inference, which is
filed as Phase 6 work and is not worth building until the trunk is large enough that a
forward pass dominates a determinization.

### F3.6 — The engine's stalemate draw is a stable equilibrium, and the first training run walked straight into it

Three generations of `configs/train-fast.toml` as originally shipped (`stalemate_value = 0.5`,
single-mirror gate at 0.5):

| gen | self-play draws | value targets (buffer) | policy loss | value loss | mirror gate | vs `random` | vs `greedy` |
|---:|---:|---|---:|---:|---|---:|---:|
| 1 | 9% | 46 / 10 / 44 | 7.696 → 3.116 | 0.939 → 0.400 | 0.858 (W145 L2 D53) | **0.929** | **0.558** |
| 2 | 55% | 29 / 44 / 27 | 2.955 → 2.826 | 0.547 → 0.269 | 0.502 (W1 L0 D199) | 0.600 | 0.496 |
| 3 | 88% | 13 / 75 / 12 | 2.748 → 2.659 | 0.317 → 0.163 | 0.500 (W0 L0 D200) | 0.525 | 0.500 |

Config: `variant=split two_power=bottom encoding_slots=21`, run seed 1,000,000, 3000 self-play
games per generation at 64 simulations. Every generation was **promoted**.

**The mechanism.** `game_rules.md` §7 records that the published game defines no draw and that
the stalemate rule is **[ENGINE]** — added because the reachable stall never ends on its own.
Scoring it at half a point makes "neither player attacks" a stable equilibrium *of the modified
game*: a certain 0.5 beats a risky fight, for both players, at every decision. Duel 52 makes
the trade especially attractive, because attacking costs material to retaliate (§5) and lane
wins are endgame-only (§2), so there is nothing to lose by waiting. The learner found it in two
generations.

It then compounded three ways:

1. **Stalled games are long**, so they are over-represented in the corpus relative to their
   share of games. Generation 2 drew 55% of its *games* and contributed ~70% of its *samples* —
   147 decisions per game became 172.
2. **A value head trained on 75% zeros learns to predict zero**, which is why the value loss
   looks best exactly when the run is worst. `0.317 → 0.163` at generation 3 is the loss of a
   head that has learned the game is a draw.
3. **The gate could not see any of it.** Two stalling agents draw against each other, so the
   mirror match scored 0.500, 0.502, 0.500 — indistinguishable from a dead-even fight, and
   above the 0.5 threshold every time. Three promotions on no evidence at all, while the only
   honest measurement in the readout (`vs random`) fell by 0.40.

**Three fixes, and each one addresses a different link.**

- **`config.stalemate_value`** (default 0.5, training `0.0`) — what a stalemate is worth *to a
  learner*, to **both** players. At 0.0, refusing to play is no better than losing, so a player
  who is behind always prefers a gamble. Read by exactly two places, the terminal backup in
  `net_mcts` and the value targets in `selfplay`. **`Outcome::value_for` is untouched**, so
  `match`, `ladder`, the Elo fit and every number in F1 and F2 still mean what they meant. A
  *mutual lane win* keeps its half point — that one is a rule (§7), not an artefact.
  `rule_7_a_stalemate_is_not_worth_half_a_point_to_a_learner`.
- **The gate reads decisive games.** `W / (W + L)`, draws discarded, threshold back to 0.55.
  Generation 3's match then reports "0 decisive of 200", which is *no evidence* rather than a
  tie — a different thing, and the gate now treats it as one.
- **A reference panel with a veto.** The candidate plays `random` and `greedy` — opponents with
  no incentive to stall — *before* the promotion decision, and is refused if it falls more than
  0.05 below the best score any promoted checkpoint has managed. Measured against a **high-water
  mark**, not the incumbent, so a slow give-back cannot ratchet the baseline down one tolerance
  at a time. This is the check that catches generation 2 at the moment it happens: mirror 0.502,
  `random` 0.929 → 0.600.

Shard format version 2 exists because of this: version 1 recorded one byte for "draw" and could
not distinguish an engine stalemate from a mutual lane win, so its corpus cannot be re-valued
and is refused rather than read.

**The generalisable lesson, and it is not about Duel 52.** Every *engine-defined* terminal
condition is a potential equilibrium, and the ones added for the trainer's convenience are the
most dangerous, because nothing about the real game constrains what they are worth. The draw
here was added so games would terminate; scoring it like a real draw quietly changed which game
was being solved. Anything else marked **[ENGINE]** in `game_rules.md` deserves the same
question: *what does an agent get for exploiting this, and did we mean to offer it?*

**Postscript, 2026-09-03 — the three fixes treated the symptom.** A second run under
`stalemate_value = 0.0` still collapsed: three consecutive refusals, self-play draws 16–18%,
`vs random` down from 0.963 to 0.673. Pricing the draw at zero removes the *incentive* to stall
but not the *ability*, and an agent that cannot find a plan still fills its turns with nothing.
The actual defect was upstream of the value: `Pass` was a legal action, and it was not a rule.
§4 says three actions; the engine let a player decline them. The lesson survives intact and
gains a sharper edge — the [ENGINE] audit should have asked not only *what is this worth?* but
*is this in the rules at all?* The pass had no ruling behind it, in any section, and nobody had
looked.

### F3.5 — The first generation already beats the hand-written heuristic, and the fitting is 4% of the loop

Generation 1 of `configs/train-fast.toml`, from a PyTorch-default random initialisation, 3000
self-play games at 64 simulations, 1200 optimisation steps of batch 512 on MPS:

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
benchmark matches at seed 1. **One generation is not a trained agent and this is not a strength
claim** — it is the claim that the loop is *connected*, which is what a first pass is for. A
policy loss starting at exactly `ln(action_dim)` and falling is the tightest available evidence
that each observation is paired with its own policy target rather than with someone else's; a
scatter bug or an off-by-one in the replay would sit at 7.69 forever.

**Where the 8m01s of one generation goes:** 6m57s self-play, 5s replay and encode, 20s
training, ~40s of gate and benchmark matches. Gradients are 4% of the wall clock, so a larger
budget goes into `selfplay.games` and `selfplay.sims` long before `net.width`.

**The draw rate at initialisation is a property of the initialisation, not of the game.** 64
games from one random-init checkpoint drew 69%, all stalemates; the shipped configuration's
generation 1, a different random init over 3000 games, drew **9%**. An untrained policy is
arbitrary, so whether it passes constantly is arbitrary too. A 64-game sample from a single
random initialisation is not a measurement of the game.

### F3.4 — Net-guided search throughput, and what a local training session buys

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
comparable per simulation rather than the network being an order of magnitude dearer. Cost is
very close to **linear in simulations**, which means sims and games trade one for one and the
choice is a modelling decision rather than a budget one.

A self-play game at 64 simulations records **166 decisions** — far more than the ~65 "plies" a
match reports, because a ply is a turn's worth of the three actions plus the free sub-decisions
`DESIGN.md` §4 splits out, and every one of those is a policy target.

### F3.3 — The observation is 4.8% dense and a position offers ~21 actions

Over 954 decision nodes sampled from 20 seeded random-vs-random games at `encoding_slots = 21`:

| quantity | mean | min | max |
|---|---:|---:|---:|
| non-zero observation features (of 4290) | 205 | 77 | 294 |
| legal actions (of 2195 encoded) | 20.7 | 1 | 78 |

Config: `variant=split two_power=bottom encoding_slots=21`, seed 1, from `duel52 nn-dump
--games 20 --max-rows 1024`. Self-play at 64 simulations gives the same density to within a
feature (208 per sample over 10,612 decisions).

Both are structural rather than incidental — the encoder is one-hot over rank, slot and phase,
and Duel 52 gives a player three actions from a small hand — so both were turned into inference
speed rather than merely noted:

- **The input layer walks the non-zeros.** `W_in` is kept transposed in `MlpEvaluator`, so each
  non-zero input adds one contiguous `width`-long row. Accumulation order per output is
  unchanged, so it is **bit-identical**, not approximately equal: the 954-row `nn-dump` above is
  byte-for-byte the same file before and after the change. It cut a 512-wide forward pass from
  3.9 ms to 2.5 ms.
- **Search computes only the logits it will look at.** `MlpEvaluator::eval_masked_with`
  evaluates the policy head for the ~21 legal actions instead of all 2195, which removes almost
  the whole head from the search hot path.

Also worth recording because it sets the storage design: 4.8% density means a generation of
self-play stored as dense f32 observations is gigabytes and stored sparsely is hundreds of
megabytes, which is why `replay_shard` hands the trainer CSR triples rather than a matrix.

The ~21 legal actions is also the number that makes F3.8 legible: at 64 simulations the root
gets about three visits per action, which is why raising `sims` moves the teacher so much.

### F3.2 — Rust and PyTorch agree on the forward pass

Over 256 sampled decision nodes from 20 seeded random games, with a 64-wide 3-block network:
`max |Δlogit| < 1e-3`, `max |Δvalue| < 1e-4`, masked-argmax agreement on every row, and
masked-softmax total-variation distance `< 1e-4`. Not a strategic finding — a claim that the two
implementations of `DESIGN.md` §5 are the same function, which is what makes "train in Python,
evaluate in Rust" safe. `py/tests/test_parity.py`.

### F3.1 — Lane sprawl is set by the *weakest* agent in a pairing

F2.7 measured the widest lane side across **self-play** and concluded that 16 slots sat
comfortably above every competent agent (ISMCTS peaked at 12 over 300 games). It does. But the
Phase 3 deliverable is an Elo ladder, and a ladder keeps `random` as its permanent anchor rung,
forever.

| pairing | agent | games | max cards on one side of one lane |
|---|---|---:|---:|
| self-play | `netpolicy` (random init) | 100 | **10** |
| vs `random` | `netpolicy` (random init, ×4 inits) | 200 each | 13, 15, 15, 15 |
| vs `random` | `netpolicy` (`init.d52nn`, seed 0) | 100 | **17 — asserts at 16** |
| self-play (F2.7) | `random` | 300 | 17 |
| self-play (F2.7) | `ismcts:800` | 300 | 12 |

**The mechanism is not subtle once stated: `random` never kills anything.** It plays cards into
lanes at the same rate as any agent and removes them at a far lower one, so a lane fills up. Two
competent agents clear each other's boards and lanes stay short; one competent agent against a
passive one produces the longest lanes of all, because one side is building and neither is
demolishing. F2.7 could not see this because every row in its table was a *self-play* row.

`encoding_slots` is therefore **21** — the theoretical maximum, so the encoder provably cannot
assert. The cost is `obs_dim` 3300 → 4290 and `action_dim` 1324 → 2194, and F3.8 shows that cost
again: those two dimensions are what pin 88% of the network's parameters into fixed projections.
`encode::observed_max_slots()` is a process-wide high-water mark for exactly this.

---

## Hypotheses

Derived from reading the rules, not from data. Each carries its prediction and where it now
stands. Phase 2's verdicts are kept only where Phase 3 has not superseded them.

### H1 — The draw phase is entirely positional
Lane wins require an empty pile *and* an empty opponent hand (§7), so nothing can be decided
during the draw phase. Prediction: strong agents treat the first ~25 plies as setup, and killing
enemy cards early is worth far less than intuition suggests.

> **Supported.** Phase 2 could only infer this from H3's null. The trained agent gives it
> directly: it spends the draw phase accumulating rather than trading, arriving at the unlock
> with 6.6 cards against every hand-written rung's under-3, and the accumulation is what
> correlates with winning (F3.9). See §Strong play.

### H2 — Hand size at pile-empty is a primary resource
Every card in hand is a turn the opponent cannot close a lane. Prediction: strong agents
**hoard cards** approaching the pile-empty transition, and hand size at that moment correlates
strongly with winning.

**Counter-consideration:** cards in hand do nothing on the board, and a card played early has
more turns to generate value. There is a crossover point. Finding it is arguably the most
valuable single output of this project.

> **Supported (F3.9): +1.25 ± 0.17 cards, ~14σ, with `greedy` as a control at −0.04 ± 0.07.**
> Phase 2 recorded this as unsupported with the effect bounded under ±0.2 cards. That verdict
> was about the agents, not the game: every Phase 2 rung plays **18.0 of its 18 cards**, and
> none of them can represent "this card is worth more unplayed".
>
> The effect is also **graded with strength** — random −0.09, greedy −0.02, `ismcts:800` +0.37 ±
> 0.22, netmcts +1.00 to +1.14 — which is a dose–response rather than a single contrast, and
> which moves ISMCTS off the null it sat on pre-ruling. §Strong play 1 has the table, and §7's
> win condition supplies a mechanism: the player still holding cards at the seam owns a window
> in which only they can win a lane. **Causation is still open** — see F3.9's caveat and
> `PLAN.md` Phase 5. The mechanism makes the direction plausible; it is not the experiment.

### H3 — Optimal play concentrates on two lanes
You need two lanes, not three. Prediction: strong play identifies a lane to concede and commits,
but does so *later* than human intuition — because conceding early lets the opponent redeploy
via the Queen.

> **Reopened.** Phase 2 found no concentration at all (F2.6): every rung sat at 0.749–0.783
> against a random baseline of 0.778, and the two using the hand-written evaluation concentrated
> *less* than random. A post-ruling run reproduces that in one table — random 0.777, ISMCTS
> 0.774, greedy 0.762 — and puts **the trained agent at 0.904–0.906**, the only agent in the
> project's history to exceed the baseline (§Strong play 3).
>
> This is H2's situation exactly, one hypothesis later, and H3 is now the **last** Phase 2 null
> still resting on agents that could not perform the behaviour. What stops it being called is
> the statistic, not the result: H3's own second clause predicts commitment is a late decision,
> and a whole-game share dilutes it across 25 plies where §7 makes nothing winnable. **Lane
> share restricted to post-unlock plies** is the measurement, and it is the cheapest outstanding
> item in `PLAN.md` Phase 5.

### H4 — Information is worth less than tempo
Playing face-down hides information but costs a second action to flip. Prediction: the 4
(Foresight) is among the weakest cards, and holding cards face-down for concealment is overrated
relative to flipping to get powers online.

> **Supported, with the framing corrected.** The trained agent's flip rate is **0.87** against
> ISMCTS's 0.69 and random's 0.68, and it flips 0.80–0.97 of every rank but two (F3.10) —
> concealment is worth very little, which is the hypothesis's second clause and Phase 2 could
> not confirm it. The 4 is confirmed weak from both directions: ISMCTS flips it at 0.58,
> second-lowest of any rank (F2.9), and the net spends it early (ply 19.0) alongside the other
> two cheap one-shots rather than holding it as an option.
>
> But the sharper statement cuts across the hypothesis: **there is no global tempo/information
> trade in this game.** The two cards the net keeps face-down are the 3, whose power *requires*
> it, and the 2, whose power is an option worth holding. Neither is information being valuable —
> one is a hidden-only trigger and one is a stored one-shot. Flip timing is a property of how the
> rank's power is *triggered*; see §Strong play 4.

### H5 — The Jack is the strongest card
3 HP plus taunt means the opponent must spend three attack actions before touching anything
else. Prediction: Jack tops the learned rank values; the 9's real value is mostly its
Jack-counter clause.

> **Flip-timing half confirmed; value half still open.** A face-down card is a blank 2-HP card
> (§5), so **both** halves of the Jack arrive only on the flip — which makes H5 a claim about
> *flipped* Jacks and entangles it with flip timing. Both strong agents turn Jacks up as early
> as they can: ISMCTS at ply 19.8, third-earliest of thirteen behind the 10 and the 8; the net at
> **14.9, behind only the 8** (F3.10). Neither holds them back for a threat.
>
> That is consistent with the Jack being strong, since all of its value is locked behind the
> flip. **It is still not evidence that the Jack is the strongest card.** Flip priority and card
> value are different quantities and nothing yet measures the second.

### H6 — The 7 scales with board commitment
Heal-all across every lane is a blowout when you have many damaged cards and nearly dead
otherwise. Prediction: the 7's value has the highest variance of any card, and strong agents
time it rather than flipping on sight.

> **Timing half supported (F2.9).** The 7 is the clearest "flipped often, but late" card:
> ISMCTS flips it at 0.77, fifth-highest of thirteen, at ply 23.0 — nearly five plies later than
> the constant powers it flips at a similar rate. That is exactly "time it rather than flipping
> on sight". The variance half needs per-card values and has to wait for Phase 5's ablations.

### H7 — The King is a combo enabler, not a body
King + Ace is a free action; King + Queen is a second move; King + 5 is a mass flip. Prediction:
King value depends almost entirely on lane composition, and strong agents arrange the lane
*before* flipping the King.

> **Consistent, too weak to call.** ISMCTS flips Kings at 0.71 against random's 0.69 —
> essentially no preference — at ply 22.5 (F2.9); the net flips at ply 27.1, in the late third
> (F3.10). "Arrange the lane first" predicts a late flip and that is what shows up, but so would
> half a dozen other explanations.
>
> Testing H7 needs a conditional statistic nothing collects: how many reactivatable allies were
> face-up in the King's lane at the moment it was flipped, against the number available. **Add
> that probe in Phase 5** — it is the only way to separate "flipped late" from "flipped when the
> lane was ready".

### H8 — First-player advantage is small and possibly negative
The first player gets one fewer action on turn one. Combined with H1, the tempo edge may not
compensate. Prediction: near-even, and the split-deck variant is where we can measure it cleanly.

> **Confirmed, and it is zero rather than merely small.** F1.4's +1.4 points under random play
> was a property of random play. Greedy self-play over 4,000 games per variant covers 0.5 in all
> three (F2.8), and the trained agent does too — 0.5380 ± 0.0435, against `greedy`'s 0.5245
> under identical conditions (F3.9).
>
> The one live scare was training-time self-play drifting 51.4% → 56.7% over 19 generations.
> That does not survive clean play and is not net-specific; F3.9 has the resolution. H8 is the
> best-supported hypothesis in this file and the only one that has now been checked against an
> agent that can actually use tempo.

---

## Phase 2 — the hand-written ladder, kept as a control

Five agents — random, greedy, flat Monte Carlo, PIMC, SO-ISMCTS — on a frozen Elo ladder.
400 colour-paired games per pairing, seeds 1–200, `split`/`bottom`, engine 0.1.0. Reproduce
with `duel52 ladder --games 400 --seed 1 --markdown`.

**Why this section is short now.** Phase 2 was mostly a phase about *search*, not about Duel 52.
Its two headline strategic results were nulls (H2, H3) and both turned out to be nulls about the
agents rather than about the game — no rung could hoard or concede deliberately, so the tests had
nothing to detect. Two of the five rungs are steered by hand-written weights this file does not
trust. What survives below is the numbers that still do work as **controls and baselines** for
Phase 3. The full Phase 2 text, with all the narrative, is in git history:
`git show 8f4d2f8:FINDINGS.md`.

⚠️ **Every number in this section predates the §4 mandatory-action ruling.** They were fitted on
a game where players could pass, which is not a rule and never was. Where a Phase 3 measurement
exists, prefer it.

**F2.1 — the frozen ladder, cleanly ordered and fully transitive.** No cycles.

| agent | Elo | ± | vs. random | vs. greedy | vs. flatmc | vs. pimc |
|---|---:|---:|---:|---:|---:|---:|
| ismcts:800 | +1186 | 16 | 1.000 | 0.944 | 0.796 | 0.912 |
| flatmc:600 | +952 | 12 | 1.000 | 0.859 | — | 0.689 |
| pimc:32x1 | +784 | 12 | 1.000 | 0.600 | 0.311 | — |
| greedy | +682 | 13 | 0.965 | — | 0.141 | 0.400 |
| random | 0 | — | — | 0.035 | 0.000 | 0.000 |

Ratings are a batch Bradley–Terry fit anchored at random, not the incremental Elo update, so the
table does not depend on the order games were played in. **Superseded by F3.7**, which is fitted
post-ruling and includes the net; the two do not compare.

**F2.3 — for PIMC, depth buys strength and determinizations do not.** Against a fixed greedy
opponent: `pimc:8x1` 0.638 ± 0.066, `pimc:32x1` 0.533 ± 0.069, `pimc:64x1` 0.580 ± 0.068 —
eight-fold more sampled worlds buys nothing measurable — while `pimc:16x2`, one extra ply, scores
**0.863 ± 0.047**. The signature of strategy fusion: sampling more of a biased estimator
estimates the bias more precisely.

→ **This is now the control for F3.8.** The same test on `netmcts` buys +141, +156 and +69 Elo
across three successive 4× steps, so per-simulation determinization is avoiding the trap that
killed PIMC.

⚠️ The 0.638 figure is one of the numbers the §4 ruling moved: re-measured post-ruling,
`pimc:8x1` vs `greedy` is **0.505 ± 0.049** over 400 games, and the intervals do not overlap.
PIMC has fallen to the bottom of the ladder, level with `greedy`. Meanwhile flat MC beats
`pimc:8x1` **0.835 ± 0.036** post-ruling against F2's 0.689 — the one Phase 2 result that came
back *stronger*.

**F2.5 — the within-agent hand-size null.** Comparing winners against losers inside one agent's
own self-play, which holds strength and evaluation weights fixed:

| agent | hand@unlock, won | lost | difference (95% CI) |
|---|---:|---:|---:|
| random | 2.85 | 2.97 | −0.12 ± 0.21 |
| greedy | 0.53 | 0.49 | +0.04 ± 0.08 |
| pimc:32x1 | 0.60 | 0.54 | +0.06 ± 0.09 |
| flatmc:600 | 2.61 | 2.42 | +0.19 ± 0.19 |
| ismcts:800 | 2.27 | 2.11 | +0.16 ± 0.19 |

→ **This is now the control for F3.9**, and it is a good one: F3.9 reproduces `greedy`'s null
exactly (−0.04 ± 0.07) while the trained agent shows +1.25 ± 0.17. Every rung above plays 18.0
of its 18 cards.

**F2.6 — the lane-concentration baseline.** Share of a player's plays landing in their busiest
two of three lanes: random 0.778, greedy 0.756, pimc 0.749, flat MC 0.777, ISMCTS 0.783. Attacks
the same: random 0.864, ISMCTS 0.852. The measure has no absolute scale, so **random's 0.778 is
the number that matters** — nothing else in Phase 2 exceeds it.

→ **This is now the baseline for §Strong play**, where the trained agent does exceed it.

**F2.8 — first-player advantage vanishes once both sides can play.** Greedy self-play, 4,000
games per variant, seeds 1–2000: 0.4946 ± 0.0217 (base), 0.4918 ± 0.0217 (split), 0.5095 ± 0.0216
(mirrored). All three cover 0.5. F1.4's +1.4 points was a property of random play.

→ Confirmed against a strong agent in **F3.9**.

**F2.9 — flip discipline: ISMCTS learns *which* cards to turn face-up, from random rollouts
alone.** The best Phase 2 result and the direct antecedent of F3.10 — which it beat to the
finding. 300 self-play games each, seeds 1–150.

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

Random's column is flat by construction — a 0.09 spread, flip plies within ±1.5 of 21. ISMCTS's
spread is 0.47, five times wider. Three things fall out:

- **It discovers that the 3 must stay face-down**, at 0.39 against random's 0.67, six plies
  later. Nothing told it this: no evaluation function, no card table, only legal moves and the
  win/loss at the end of uniform-random playouts.
- **Constant powers get flipped first; one-shots get held.** The three earliest are 8, 10 and J
  at plies 18.2–18.6 — three of the four constant powers. Every one-shot sits at 20.3 or later.
- **The exception proves it.** The fourth constant power, the **9**, breaks the pattern (0.64,
  ply 22.4, indistinguishable from random) because Nimble is purely *conditional* — it does
  nothing unless someone tries to freeze you or you are hitting a Jack. So it behaves like a
  one-shot and the search treats it like one. A "flip your constant powers early" heuristic gets
  the 9 wrong; the search does not, because it is not using a heuristic.

Caveats: mean flip ply is conditioned on the card being flipped at all, and a 3 that springs its
Trap returns face-up without a `Flip` action, so it is counted in neither column — which affects
both agents identically.

**Compressed to one-liners**, because the codebase cites them by number. Full text in
`git show 8f4d2f8:FINDINGS.md`.

- **F2.2 — PIMC is the weakest search, and it fails in the textbook way.** Flat Monte Carlo —
  no evaluation function, no tree — scores 0.689 against a compute-matched PIMC, and `ismcts:800`
  beats `pimc:32x1` 0.912 against 0.796 for flat MC. Whatever PIMC's alpha–beta and hand-written
  evaluation buy inside each world does not survive averaging across worlds. F2.3 has the
  budget-scaling measurement that names the mechanism.
- **F2.4 — the stalemate rule fires, and only for the agent that has something to protect.**
  Greedy — the only rung that prices material, so the only one that could decline a trade —
  stalled in 0.7–1.7% of its self-play games, most often in `mirrored` where symmetric decks
  make neither side want to attack first. Every other rung: zero.
- **F2.4b — F2.4 is superseded: with actions mandatory, the stall is not reachable.** Chasing
  F2.4 turned up the actual defect — the engine had been letting players **pass**, which §4 does
  not permit and never did. With the three actions mandatory, greedy's stalemate rate is **0 in
  4,000 games per variant**, and the only draw left in Duel 52 is the mutual lane win. Every
  other number in this Phase 2 section predates that ruling.
- **F2.7 — F1.7 reverses: the encoding bound belongs to competent play, not to the game tree.**
  Competent self-play tops out at 8–12 cards on one side of a lane (ISMCTS peaked at 12 over 300
  games); it is *random* play that sprawls to 17–20. This set the default `encoding_slots = 16`,
  which F3.1 then showed does not survive a ladder that keeps `random` as its anchor. See *Things
  we got wrong* for both halves.

---

## Phase 1 — the random-play baseline

1.2M games, uniform random on both sides, 200,000 per configuration, seeds 1–200000.
`duel52 stats --all --games 200000 --seed 1 --markdown`.

| variant | 2's power | P0 score (95% CI) | draw | stalemate | mean plies | P0 draw edge | max lane |
|---|---|---:|---:|---:|---:|---:|---:|
| base | bottom | 0.5154 ± 0.0022 | 0.4% | 0.0% | 45 | +0.054 | 20 |
| split | bottom | 0.5139 ± 0.0022 | 0.5% | 0.0% | 45 | +0.001 | 19 |
| mirrored | bottom | 0.5148 ± 0.0022 | 0.5% | 0.0% | 45 | −0.001 | 19 |
| base | discard | 0.5232 ± 0.0022 | 0.4% | 0.0% | 43 | +0.523 | 19 |
| split | discard | 0.5147 ± 0.0022 | 0.5% | 0.0% | 44 | +0.000 | 19 |
| mirrored | discard | 0.5166 ± 0.0022 | 0.5% | 0.0% | 44 | +0.000 | 19 |

**F1 characterises the game tree, not strategy.** Nothing here speaks to how the game should be
played, and two of its conclusions were later reversed by measuring a competent agent instead
(see *Things we got wrong*). Four numbers still do work:

- **F1.6 — the unlock is deterministic.** Under the house rule for the 2, the split-deck pile
  empties on **ply 25 in every single game**, no variance. §10a predicted it. This is what makes
  "hand size at the unlock" a clean measurement rather than a noisy one, and it is why the
  variant was chosen. Under `discard` it varies (mean 24.1).
- **F1.3 — the mutual lane win is not astronomically rare.** §7 calls it that; it is 0.4–0.5% of
  games, ~1 in 220, and since the §4 ruling it is the *only* way any Duel 52 game draws at any
  level of play. `duel52 demo --seed 47` replays one.
- **F1.5 — the `two_power` house rule fixes a real artifact, and it is an artifact of pile
  *sharing*.** RAW discard hands the first player half an extra card per game in the base
  variant (+0.523 draw edge, P0 0.5154 → 0.5232, ≈5σ) because P0 draws first from a shared pile,
  so every card the 2 destroys flips the pile's parity. In the split variants you discard into
  your own pile and the effect is exactly **zero**.
- **F1.4 — P0 scores 0.514 under random play**, which F2.8 and F3.9 then showed is a property of
  random play rather than of the game. Kept only as the thing those two findings correct.

The rest are kept as one-liners because the codebase cites them by number; the full text is in
git history (`git show 8f4d2f8:FINDINGS.md`).

- **F1.1 — games are short and uniform.** Mean 45 plies, median 45, p10 39, p90 52, in every
  configuration. Re-measured after the §4 ruling: mean 40.1, median 40, p10 35, p90 45 — same
  shape, and the centre moved because a third of random play's actions used to go unspent.
- **F1.2 — the stalemate rule never fires under random play.** Zero games in 1.2M; every draw
  was a mutual lane win. This never vindicated the threshold: §7's stall is *strategic* and
  random agents attack constantly. F2.4 found the one agent that could produce it, and the §4
  ruling then made it unreachable for everyone (F2.4b). F1.2's zero is now the expected result
  at every level of play rather than an artefact of random play.
- **F1.7 — `DESIGN.md` §3's 8-slot encoding bound is too small.** Observed maximum 20 cards on
  one side of one lane under random play. The conclusion drawn from it was backwards — see
  *Things we got wrong*, and F2.7/F3.1 for where the bound actually landed.
- **F1.8 — throughput ~16,800 random games/sec/core**, single-threaded, release build.
  `DESIGN.md` §8 targets ≥10k.

---

## Things we got wrong

Log falsified hypotheses here rather than quietly deleting them. Knowing which intuitions the
game defeats is itself a finding, and it is the part a human player would most want to read.

### The two strategic nulls of Phase 2 were nulls about the agents

The most expensive mistake in the file, and it was made twice. F2.5 tested H2 — does hand size at
the unlock predict winning — inside each agent's own self-play, which is the right design, and
found nothing at every rung. F2.6 tested H3 the same way and found nothing. Both were recorded as
evidence about **Duel 52**.

Neither was. Every Phase 2 agent plays 18.0 of its 18 cards, because there is nothing else to
spend three mandatory actions a turn on; none of them can represent "this card is worth more
unplayed". F3.9 ran F2.5's exact test on an agent that can, and the effect is +1.25 ± 0.17 cards
against greedy's −0.04 ± 0.07 — with the *same* test reproducing the *same* null on the *same*
control. The measurement was never wrong. The population was.

**The general form: a null result from an agent that cannot perform the behaviour is not evidence
about the game.** It is evidence about the agent. Before recording any null here, the question to
ask is whether the agent could have produced the positive result had it wanted to. H3 is still
sitting in this position and has not been re-run.

### F3.10 claimed a discovery that F2.9 had already made

F3.10 originally said the net's flip-timing ordering "is not something a flat search discovers".
F2.9 is a flat search discovering it — a phase earlier, from uniform random rollouts, with no
evaluation function and no card knowledge at all. The claim was written without re-reading the
finding it was implicitly contradicting. The finding survives with a smaller claim — magnitude and conviction, not
discovery — and the lesson is that "no previous method could do this" is a claim about the
previous methods, and needs checking against them rather than against memory.

### F1.7 was backwards about the encoding bound (corrected by F2.7)

F1.7 measured 20 cards on one side of one lane under random play and concluded `DESIGN.md` §3's
proposed 8-slot bound "is not defensible" — that §3 had described human play and mistaken it for
the game. The correction is that §3 was describing the right thing and F1.7 was measuring the
wrong one. Competent self-play tops out at **8–12** cards per side; it is *random* play that
sprawls to 17–20, because a random agent never kills anything.

**Random play is not a conservative upper bound on the shape of real play, it is a different
distribution.** F1.7 assumed the first.

### F2.7 measured the wrong pairing (corrected by F3.1)

The number was right and the experiment was not general enough. F2.7's table is entirely
**self-play** rows, and it drew a conclusion about a bound that has to hold everywhere the
encoder runs — which includes the ladder, which includes `random`. Sprawl is set by the *weakest*
agent in a pairing. 16 survives self-play and does not survive the ladder.

A sibling of F1.7's rather than a repeat: F1.7 measured the wrong *agent*, F2.7 the wrong
*pairing*. Both times the failure was assuming one distribution stood in for another. **A bound
that must hold under all conditions has to be measured under the condition that stresses it, not
the condition that is most interesting.**

### "More determinizations" is not a knob (F2.3)

The implicit assumption in building PIMC with a `worlds` parameter was that it was the strength
dial, with depth as a cost problem. It is the other way round: eight-fold more worlds is flat,
one more ply of search is worth +0.28 score against the same opponent. Sampling more of a biased
estimator estimates the bias more precisely.

F3.8 is the payoff: the same test run on `netmcts` is the evidence that the Phase 3 method is
*not* in this trap, and it is only interpretable because F2.3 established what the trap looks
like.

### The pass was never a rule, and nobody had looked

Two training runs collapsed into mutual passivity before anyone checked whether `Pass` was in
`game_rules.md`. It was not — §4 says three actions, mandatory, and the engine had been letting
players decline them since Phase 1. F3.6's three fixes all treated the symptom.

The [ENGINE] audit that F3.6 recommends should have asked not only *what is this worth to an
agent?* but *is this in the rules at all?*

### The greedy agent was quietly cheating, and nothing about search caused it

Greedy does no search and reads only ranks it is entitled to know, so it looked exempt from
`DESIGN.md` §6's determinization discipline. It was not: one-ply lookahead *applies* a candidate
action to the real state, and applying reveals things — flipping your own base card turns it
face-up (§3 says you did not know it), and killing a face-down card sends its rank to the public
discard (§5). Greedy was choosing whether to flip after seeing what it would flip.

It was caught by `phase2_no_agent_reads_hidden_information` on the first run, not by review. The
property that makes that test exact: a determinized world is in the same information set as the
real one, so **any honest agent must return the same action from either**. Every future agent
gets that test for free.
