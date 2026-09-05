# Duel 52 Findings

What we have actually learned about the game. **This file is the point of the project**; the
engine and the agents are the instrument that produced it.

Everything here is measured on the three trained agents. The five hand-written agents that
came before them (random, greedy, flat Monte Carlo, PIMC, information set MCTS) were the Elo
benchmark for two phases and are no longer competitive enough to say anything about the game;
what they measured is in `archive/FINDINGS.md`. `random` still appears in tables below, for one
reason only: several of these statistics have no absolute scale, and a number like "lane
concentration 0.907" means nothing without knowing that uniform play scores 0.777.

## How to read this file

Every finding carries the config, the agent, the seed range, the sample size, and a confidence
interval. A number without reproducible provenance is not a finding.

**Units.** Every game length here is counted in **turns**: one player's turn of three actions.
The code and config keys call this a "ply" for historical reasons (`GameState::ply`,
`max_plies`, `stalemate_quiet_plies`), and those names are frozen because they are written
verbatim into every game record. When a table says "mean plies 47.6", read "47.6 turns".

**Numbering is inherited.** Findings keep the identifiers they were published under, because
code comments and checkpoint notes cite them. The gaps (F1.x, F2.x, most of F3.x) are the
Phase 1 and Phase 2 results and the build history of the training loop, all of which are in
`archive/FINDINGS.md`.

**Two standing traps, both of which have caught this file before.**

- **Size the experiment before running it, not after reading it.**
- **A null from an agent that cannot do the thing is not a null about the game.** This has
  happened twice, and both times the "finding" was about the agents.

---

## The scale

Elo here is an internal coordinate. Every agent on it was written for this project, so a
rating is a distance between two of our own agents and never an absolute.

**The floor is `gen016`, the first trained agent**, and the ladder was re-anchored on it on
2026-09-05. Round robin, `split`, `encoding_slots = 21`, 400 games per pairing, seeds from 1,
1,200 games in 686 s on 8 cores:

| agent | Elo | +/- | expected vs. anchor |
|---|---:|---:|---:|
| `netmcts:gen031@256` | **+157** | 13 | 0.711 |
| `netmcts:gen022@256` | +91 | 13 | 0.628 |
| `netmcts:gen016@256` | 0 | 0 | 0.500 |

The three pairings the fit is made of, at equal simulations:

| | score | W-L-D |
|---|---:|---|
| gen022 vs gen016 | 0.636 | 253-144-3 |
| gen031 vs gen016 | 0.704 | 281-118-1 |
| gen031 vs gen022 | 0.603 | 238-156-6 |

```bash
./target/release/duel52 ladder --games 400 --seed 1 --variant split \
  --encoding-slots 21 --stalemate-value 0.0 \
  --anchor netmcts:models/duel52-split-gen016.d52nn@256 \
  --agents netmcts:models/duel52-split-gen016.d52nn@256,\
netmcts:models/duel52-split-gen022.d52nn@256,netmcts:models/duel52-split-gen031.d52nn@256
```

### Why the floor moved (F4.2, extended)

The old ladder was anchored at `random` and carried the five hand-written rungs. It has
saturated completely. `gen031` beats `ismcts:800`, the strongest of them, **200 games to 0**,
and `gen022` already beat `greedy` and `pimc:8x1` at essentially 1.000. A rung that loses every
game carries no information about the winner, and the last full fit showed it: the top agent
came out at +1788 with a standard error of 58, against 13 to 15 for every hand-written rung.
A rating driven by a handful of losses is an extrapolation, and it was literal. Against
`ismcts:800` over 400 games, `gen022` lost **three**, and one game either way in that region
moves the implied rating by more than 100 Elo. The rung went from 28 games of signal to 3 in a
single training run, and then to 0.

Two consequences worth keeping:

- **Elo is not comparable across two fits.** Bradley-Terry pins the anchor at 0 and fits the
  rest to the whole graph, so pulling the top agent away stretches every rung beneath it. Every
  hand-written rung moved up 35 to 70 points between two successive fits without a line of code
  changing. Compare gaps over a common rung, never row to row. This trap has caught this
  project twice.
- **The fit smooths the pairings, and the direct matches disagree slightly.** Measured
  head-to-head with the sides swapped, `gen031` scores 0.7475 +/- 0.0426 on `gen016` and
  0.6162 +/- 0.0475 on `gen022`, against the ladder's 0.704 and 0.603. Same games, opposite
  seat assignment, so the two differ by which agent draws which RNG stream. The intervals
  overlap; the ladder's numbers are the ones to quote because they are one internally
  consistent run.

---

## What strong play looks like

Provenance for every table in this section: one self-play probe, 400 games per agent, seeds
from 1, `split`, `two_power = bottom`, `encoding_slots = 21`, 2026-09-05.

```bash
./target/release/duel52 probe --games 400 --seed 1 --markdown --variant split \
  --encoding-slots 21 --stalemate-value 0.0 --agents \
  netmcts:models/duel52-split-gen031.d52nn@256,netmcts:models/duel52-split-gen016.d52nn@256,random
```

| agent | hand@unlock | won - lost | flip rate | lane conc | attack conc | mean turns | stuck/game | max lane |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **gen031@256** | **6.94** | **+1.13 ± 0.22** | **0.943** | **0.907** | 0.912 | 47.6 | 0.51 | 6 |
| gen016@256 | 6.55 | +1.00 ± 0.26 | 0.869 | 0.906 | 0.914 | 47.3 | 0.54 | 7 |
| random *(scale only)* | 2.36 | -0.09 ± 0.24 | 0.678 | 0.777 | 0.869 | 39.8 | 0.25 | 17 |

### The game is two games joined at a seam both players can see coming

A lane cannot be won until **all** of three things are true: the opponent's side of the lane is
empty, every draw pile is empty, and the opponent's hand is empty. The first two of those make
the entire draw phase unwinnable. Nothing can be decided, and nothing can be lost.

**The seam is at a fixed turn and both players can compute it from the deal.** Each player
draws exactly one card per turn while their pile is non-empty, and the only power that touches
the pile (the 2's View) draws one and bottoms one for no net change. Thirteen pile cards is
therefore thirteen turns of drawing, whatever anyone does. Games run about 47 turns, so roughly
the first half is positioning and the second half is the entire game.

Every player begins with 26 cards: 3 to base, 5 to the opening hand, 5 removed unseen, and 13
to the personal pile. **18 cards pass through a hand over a whole game**, and the last of them
is drawn at the seam.

### 1. Strong play does not spend its hand

At the seam, out of those 18 cards:

| agent | cards in hand | share of its 18 |
|---|---:|---:|
| **gen031** | **6.94** | **39%** |
| gen016 | 6.55 | 36% |
| random | 2.36 | 13% |

The trained agents are the only players measured in this project that finish a game with cards
unplayed. Every hand-written agent plays 18.0 of its 18.

**Inside their own games, the side holding more at the seam is the side that wins.** gen031
shows a +1.13 ± 0.22 card gap between the games it won and the games it lost; gen016 shows
+1.00 ± 0.26; `random` shows -0.09 ± 0.24. The head-to-head ladder reproduces the same gap in
all six agent-sides, ranging +0.81 to +1.06, every one of them clearing its own interval.

The mechanism is in the win condition. The third requirement for a lane win is *the opponent's
hand is empty*. So at the seam:

- The player whose hand is **empty** can be scored against. Their opponent's third condition is
  satisfied, permanently.
- The player still **holding cards** cannot be scored against at all, until the hand runs out.

That is not a defensive resource. It is a one-sided scoring window, and it opens the moment one
player's hand empties before the other's. The limit is that a held card is a wasting asset:
actions are mandatory, so the window lasts only as long as there are flips, attacks and pairs
to spend actions on instead. That is consistent with the other statistic in the table, since
the trained agents flip far more than anything else.

**This is correlational.** Holding cards may win games, or a winning position may simply be one
that never forces you to commit cards. Two experiments in `PLAN.md` separate them, and until
one lands this stays supported rather than confirmed.

### 2. Strong play flips almost everything

A face-down card is a blank 2 HP body whatever its rank. Concealment buys exactly one thing,
that the opponent does not know the rank, and it costs the whole of the card's power.

The verdict is close to unanimous, and it has hardened with each agent. Fraction of each rank
played from hand that was then turned face up:

| agent | A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | J | Q | K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **gen031** | 0.99 | 0.89 | **0.69** | 0.91 | 0.97 | 0.94 | 0.98 | 0.99 | 0.99 | 0.98 | 1.00 | 0.94 | 0.95 |
| gen016 | 0.94 | 0.71 | **0.59** | 0.80 | 0.92 | 0.90 | 0.91 | 0.97 | 0.90 | 0.94 | 0.97 | 0.87 | 0.86 |
| random | 0.68 | 0.66 | 0.67 | 0.65 | 0.71 | 0.67 | 0.64 | 0.71 | 0.69 | 0.68 | 0.68 | 0.67 | 0.69 |

gen031 flips **twelve of thirteen ranks at 0.89 or above**, and the exception is the one card
whose power requires darkness: the 3's Trap fires only if it is killed face-down. The stronger
agent flips more than the weaker one on every single rank.

So the useful statement is not "information is worth less than tempo". It is that **there is no
general information/tempo trade in Duel 52.** Hiding a card is simply bad, with exactly one
rank-specific exception, and that exception exists because a power is conditioned on hiddenness
rather than because hiddenness is worth anything. A player looking for edges in concealment is
looking in a place the strongest available agent says is empty.

### 3. Strong play concentrates on two lanes

| | lane concentration | attack concentration |
|---|---:|---:|
| gen031 | 0.907 | 0.912 |
| gen016 | 0.906 | 0.914 |
| random *(the scale)* | 0.777 | 0.869 |

You need two lanes, not three, and the trained agents put 0.90 of their plays into their
busiest two. No hand-written agent ever exceeded the random baseline at all; the two that used
a hand-written evaluation concentrated *less* than random.

**The statistic is the wrong one and the result should not be called yet.** Lane share over a
whole game averages across the draw phase, where nothing can be won and committing costs
nothing. The measurement that would settle this is lane share restricted to post-seam turns,
and nothing collects it. It is the cheapest open item in `PLAN.md`.

Note also that the two trained agents are indistinguishable here, 0.907 against 0.906, while
they differ by 157 Elo. Whatever separates them, it is not lane commitment.

### 4. Constant powers get flipped first, one-shots last

Mean turn on which each rank is turned face up, ranks sorted by gen031's timing:

| rank | power | type | gen031 | gen016 |
|---|---|---|---:|---:|
| 8 | Retaliate | constant | **11.8** | 12.0 |
| J | Taunt | constant | **12.2** | 14.2 |
| A | Action | one-shot | 14.0 | 17.8 |
| 9 | Nimble | constant | **15.8** | 21.5 |
| 10 | Twinstrike | constant | **19.1** | 23.7 |
| 7 | Heal All | one-shot | 19.4 | 20.8 |
| 2 | View | one-shot | 22.1 | 20.7 |
| 4 | Foresight | one-shot | 24.4 | 20.1 |
| K | Empower | one-shot | 24.4 | 27.2 |
| 5 | Flip | one-shot | 25.8 | 21.5 |
| 6 | Freeze | one-shot | 33.4 | 29.3 |
| Q | Move | one-shot | 35.3 | 33.6 |
| 3 | Trap | condition | **36.0** | 34.5 |

`random` spans 1.8 turns and is flat by construction. gen016 spans 22.5 turns and gen031 spans
**24.2**, so the effect sharpens with strength.

**All four constant powers are in gen031's earliest five, and the only one-shot among them is
the Ace, which buys an action.** Everything else the agent holds. The 3 is last and least
flipped.

This is cleaner than the same curve on gen016, where the 9 and the 10 sat ninth and tenth,
mixed in among one-shots. The stronger agent has sorted the ranks by a rule that can be stated
in one line: **a constant power is wasted every turn it spends face-down, so flip it as soon as
it is played; a one-shot is spent when you flip it, so hold it until it pays.** The 3 inverts
the rule because its power only works face-down.

**This is flip timing, not card value.** Nothing here says the Jack is the strongest card or
the 4 is the weakest. A value table is a separate measurement and it does not exist yet.

### 5. First-player advantage is zero

Self-play, first player's score:

| agent | P0 score (95% CI) |
|---|---|
| gen031 | 0.5363 ± 0.0689 |
| gen016 | 0.5062 ± 0.0692 |
| random | 0.5075 ± 0.0689 |

The first player takes one fewer action on turn one, which could have gone either way. Both
trained agents cover even, and so does uniform play. At 400 games the interval is +/- 0.069,
which is wide; a 1,000-game run on gen016 previously gave 0.5380 ± 0.0435, also covering even.

This is the best-supported result in the file and the only balance question close to answered.
It is answered for `split` only.

### 6. One anomaly, unexplained

The trained agents are left with nothing to do about twice as often as uniform play: 0.51 and
0.54 stuck turns per game against 0.25. A stuck turn is one that ended with action allowance
unspent, which since actions became mandatory means the legal action list genuinely went empty.

This is the opposite of what the hoarding result predicts. An agent holding seven cards should
always have a card to play, and playing one is an action.

It is flagged rather than explained. It is the only statistic in the table where the strongest
agent looks worse than `random`, and that is the shape a residual bug takes.

---

## Hypotheses

Written from reading the rules, before any data.

| | verdict |
|---|---|
| H1: the draw phase is entirely positional | **Supported.** Strong play accumulates through it rather than trading |
| H2: hand size at pile-empty is the primary resource | **Supported, not confirmed.** +1.13 ± 0.22 within-agent, mechanism in the win condition, causation open |
| H3: optimal play concentrates on two lanes | **Reopened.** 0.907 against a 0.777 baseline, but measured on the wrong window |
| H4: information is worth less than tempo | **Supported, framing corrected.** There is no general trade; hiding is just bad |
| H5: the Jack is the strongest card | **Open.** Flipped second-earliest of thirteen, which is not the same claim |
| H6: the 7 scales with board commitment | **Open.** Flipped mid-table; the variance half needs card values |
| H7: the King is a combo enabler | **Open.** Flipped late, which is consistent with "arrange the lane first" and with much else |
| H8: first-player advantage is small | **Confirmed, and it is zero.** On `split` |

Three of the eight are waiting on the same missing measurement, a per-rank value table, and a
fourth is waiting on a windowed version of a statistic that already exists. Neither needs a
stronger agent.

**H5, H6 and H7 all illustrate the same trap.** Flip priority is not card value. A face-down
card is a blank 2 HP body, so all of a Jack's value arrives on the flip, which is a good reason
to flip it early and no reason at all to think it is the best card. This file has been careful
about that distinction since the first flip-timing curve and should stay careful.

---

## What the training runs established

These are findings about the instrument rather than the game, kept because they decide what to
do next.

### F4.5: lane relabelling is worth as much as uncapping the teacher

Duel 52 is invariant under all six permutations of its three lanes. No rule names a lane,
orders them, or tells one from another. So every training sample has five more exactly correct
views, free.

gen022 had not learned this. Over 128 positions that are exact relabellings of one another, its
opening prior split .320 / .277 / .403 where it must be .333 each, and it preferred lane 3 in
24 of 24 seeds. Training gen031 with a random relabelling per sample per draw fixed it:

| 128 pairs | gen016 | gen022 | **gen031** | equivariant |
|---|---:|---:|---:|---:|
| opening prior on lane 1 / 2 / 3 | .317 / .328 / .354 | .320 / .277 / **.403** | **.328 / .331 / .341** | .333 each |
| value-head spread (median / max) | 0.068 / 0.240 | 0.068 / 0.178 | **0.034 / 0.088** | 0 |
| policy TV between lane pairs (median / max) | 0.133 / 0.290 | 0.152 / 0.362 | **0.039 / 0.103** | 0 |
| top second action agrees across all three lanes | 24/128 | 82/128 | **114/128** | 128/128 |

It is not a flatter policy, which is the obvious confound: gen031 is the *sharper* net, median
top prior 0.384 against gen022's 0.380. And the cost of the defect, scored on the same shard
with and without a relabelling, falls from 0.195 nats to 0.016.

The Elo gain was +82 at equal simulations, the same as the previous run bought by quadrupling
teacher search. **It does not price the lane bias**, because the run also spent three more
hours of self-play on a checkpoint that had not plateaued, and nothing here divides the two.

Two operational notes for the next run. **Watch the policy TV row at generation 1, not the
argmax agreement row**: agreement read 86/128 after one generation against gen022's 82/128,
inside noise, while TV had already moved 0.152 to 0.113. And **114/128 is not 128/128**;
nothing in the architecture enforces the symmetry, so a data augmentation cannot close the
last of it.

### F4.4: better weights did not buy more search

Search still pays, and it pays a better net the same amount. Elo gained per 4x step in
simulations, each net played against itself, 300 games a step:

| step | gen022 | gen016 |
|---|---:|---:|
| `netpolicy` to `@64` | +222 | +252 |
| `@64` to `@256` | +144 | +141 |
| `@256` to `@1024` | +139 | +156 |
| `@1024` to `@4096` | +104 | +69 |
| **end to end** | **+609** | **+618** |

Every step lands inside the other net's interval. The two nets convert the same range of
search into the same total, so a better policy and a policy a tree can do more with are
separable quantities, and the runs so far have moved only the first.

Search is still paying at 4096, +104 with an interval of [+64, +146], so the knee is not
established. Nothing here looks like search fusion, which is the failure this method would be
vulnerable to: PIMC bought nothing measurable from 8x more sampled worlds, and this buys +104
from 4x.

The practical consequence: **teacher search stays at 256.** Used as a teacher rather than at
play time, 64 to 256 returned +81, a bit over half the play-time figure, and 4x the teacher
costs 4x the self-play second. Games are what the value head needs, and the value head is the
weaker half of every checkpoint so far.

### F4.1: a deeper teacher was worth +81 Elo

The policy target *is* the search's visit distribution, so generating self-play at 64
simulations teaches the network to imitate a search hundreds of Elo weaker than the same
weights already produce. Raising it to 256 produced gen022, +81 Elo on gen016 at equal
simulations over 400 games.

This is the finding that made the run cadence: change one thing, measure it against the
checkpoint it came from at equal simulations, over enough games to have an interval.

### F4.3: the lane bias, as originally measured

Superseded by F4.5 and kept because it is what motivated the fix, and because
`py/tests/test_encoding.py` pins gen022's numbers as a regression test. The full entry is in
`archive/FINDINGS.md`.

---

## What none of this can tell us

- **These are three agents from one lineage**, each warm started from the last, all on a
  `128 x 3` trunk, all trained on `split`. They share whatever blind spot the first one had.
- **Nothing here is an equilibrium.** No exploitability measurement exists, so "optimal play"
  is not a phrase this file has earned. That is `PLAN.md` item 6.
- **The hand-size result is correlational.** The mechanism in the win condition makes the
  causal direction plausible and is not a substitute for the experiment.
- **There is still no card value table.** Flip priority and flip timing are not value, and
  three of the eight hypotheses are waiting on one.
- **The lane-commitment picture answers an endgame question with a whole-game statistic.**
- **Everything describes `split` only.** The observation layout is per variant, so these
  checkpoints cannot even be loaded against `base` or `mirrored`. Whether the seam, the hoard
  and the concentration survive a shared draw pile is untested.
- **The owner still beats it.** The record against gen016 is 0-5 and no series has been played
  against gen022 or gen031. That is the most informative unrecorded signal in the project, and
  `PLAN.md` puts recording it first.

---

## Things we got wrong

Kept so nobody re-derives the superseded version.

**Two strategic nulls were nulls about the agents, not the game.** Hand size at the seam was
recorded as "unsupported, effect bounded under ±0.2 cards" for a whole phase. It was wrong, and
not because the measurement was bad: the same test reproduces the same null on `greedy` today.
**No hand-written agent could hoard deliberately**, so the within-agent test had nothing to
detect. Lane concentration is the same story one hypothesis later. The general rule is in "How
to read this file" and it is the most expensive lesson in the project.

**The pass was never a rule, and nobody had looked.** The engine offered a fifth main-phase
action from the first commit. It came from no ruling and nothing in the published rules, which
say "take three actions" and stop. It survived the ruleset being written, the encoder, the
whole hand-written ladder and two training runs. Removing it deleted the strategic stalemate
outright and shifted the policy head from 1325 to 1324 logits, which invalidated every
checkpoint written before that date.

**The greedy agent was quietly cheating, and search had nothing to do with it.** Applying a
candidate action to the real state to score it reveals ranks, because flipping your own base
card or killing a face-down card into the public discard are both observable. The guard that
caught it is that an honest agent must return the same action from a sampled world as from the
real one.

**Elo was compared across two fits, twice.** See "The scale".

**A layout hash was trusted without being checked.** Trajectory shards store indices into
`legal_actions()`, so an encoder change silently repoints every one of them. The header had
carried the layout hashes from the beginning and nothing read them back until it mattered.
