# Duel 52 Findings

What we have actually learned about the game. **This file is the point of the project**; the
engine and the agents are the instrument that produced it.

Everything here is measured on the trained agents. The five hand-written agents that
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
2026-09-05. Refitted 2026-09-09 with `lane-gen032` added — six pairings, `split`,
`encoding_slots = 21`, 400 games per pairing, seeds from 1:

| agent | Elo | +/- | expected vs. anchor |
|---|---:|---:|---:|
| `netmcts:lane-gen032@256` | **+360** | 13 | 0.888 |
| `netmcts:gen031@256` | +161 | 11 | 0.717 |
| `netmcts:gen022@256` | +96 | 11 | 0.634 |
| `netmcts:gen016@256` | 0 | 0 | 0.500 |

⚠️ **gen031 and gen022 moved by +4 and +5 from the three-agent fit** (+157, +91). Nothing about
those agents changed: a Bradley–Terry fit is over the whole table at once, so adding a fourth
agent re-conditions every rating. Quote a rating with the fit it came from, and never mix rows
across two fits.

**The last step, two ways.** The fit puts `lane-gen032 − gen031` at +199; measured directly at
400 games it is **+167** (0.7238 ± 0.0436, W288 L109 D3). +199 lies inside the head-to-head
interval, so they agree — but they are different estimators. The direct number is the better
statement about *those two agents*; the fit is the better statement about the scale.

```bash
./target/release/duel52 ladder --games 400 --seed 1 --variant split \
  --encoding-slots 21 --stalemate-value 0.0 \
  --anchor netmcts:models/duel52-split-gen016.d52nn@256 \
  --agents netmcts:models/duel52-split-gen016.d52nn@256,\
netmcts:models/duel52-split-gen022.d52nn@256,\
netmcts:models/duel52-split-gen031.d52nn@256,\
netmcts:models/duel52-split-lane-gen032.d52nn@256
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

### F4.8: the lane network passes gen031 by +167 Elo, in one seven-hour sitting

`runs/seventh`, `configs/train-7h.toml`, split, `encoding_slots = 21`, `run.seed = 6000000`
so the games span seeds **6,000,000–6,080,000** (4,000 a generation), warm-started from
`runs/sixth/checkpoints/best.d52nn`. **20 generations, 11 promoted, 80,000 self-play games,
6.96 hours** on the 8-core M2 laptop, stopped by the clock rather than by the gate.

**The result, 400 games at equal simulations:**

| | score | | Elo |
|---|---:|---|---:|
| vs `gen031@256` **before** the run | 0.3175 ± 0.064 | (200 games) | −133 |
| vs `gen031@256` **after** | **0.7238 ± 0.044** | W288 L109 D3 | **+167** |
| vs `runs/sixth` best, after | 0.8700 ± 0.033 | W347 L51 D2 | +330 |

**A +300 Elo swing in one sitting**, against +189 for the entire preceding flat lineage
(gen016 → gen022 → gen031, three runs). The lane-equivariant architecture was never the
problem; `runs/sixth` was simply a from-scratch run that had not been given enough games.

⚠️ **The two rows are not perfectly consistent, and the direct one is the one to quote.**
+330 over `runs/sixth` and `runs/sixth` at −133 would predict +197 against gen031, not +167.
Elo is not transitive across a game that is not, and each row carries ±35–40 Elo of its own —
they are compatible, but a chained estimate is not a measurement.

**What the shape of the run says.** The panel row against `gen031@64`, by generation:

```
0.52 0.65 0.58 0.71 0.76 0.78 0.74 0.77 0.70 0.77 │ 0.83 0.81 0.84 0.80 0.83 0.84 0.77 0.85 0.83 0.81
                                       lr drop ──┘
```

It plateaued at ~0.77 from generation 5 and broke out to ~0.83 exactly when `lr_schedule`
dropped the rate to 5e-4 at generation 10 — the clearest evidence in this project so far that
the schedule earns its keep, and a vindication of keying tiers to the generations a run will
*finish* (`PLAN.md` §4.2 change 5). Held-out value MSE sat at 0.54–0.58 throughout while
policy loss fell 2.098 → 1.818: it learned steadily without overfitting. Gate scores drifting
to ~0.50 over the last five generations is convergence at this data scale, not a fault — three
of the last five were refused and the incumbent held.

⚠️ **`reference_tolerance = 0.10` saved this run, and 0.05 would have damaged it.** At
generation 9 a candidate that won its 294-game gate **0.605** posted a 0.700 panel row against
a 0.780 high-water mark. At `train-3h-new`'s inherited 0.05 the veto floor was 0.730 and that
candidate would have been refused outright. `configs/train-12h.toml` predicted this from
simulation; it happened for real, once, in twenty generations.

**Sizing.** 4,000 games a generation rather than `train-3h-new`'s 1,400, because batched
self-play (F4.7) made the old ratio wrong: at 1,400 the gate would have been the majority of
the clock. At 4,000 self-play is ~55% and the run got 80,000 games where `runs/sixth` got
18,200.

### F4.7: self-play is 94% neural network, and 3.26x of it was free

Measured 2026-09-09 on the 8-core M2 laptop (Mac14,7, 4 performance + 4 efficiency cores),
`runs/sixth/checkpoints/best.d52nn` (lane 128×3), split, `encoding_slots = 21`, 512 games from
seed 1 at `--sims 256 --cap-sims 32 --full-search-fraction 0.25`.

**Where the time goes.** A `sample` profile of self-play, discounting the main thread parked
in `pthread_join`, puts **89.3% of worker CPU in `LaneBody::trunk`** and 94.2% in the forward
pass as a whole. Determinization, legal-action enumeration, encoding and the game logic
together are under 5%. Any statement of the form "self-play is slow because of X" where X is
not the network is wrong on this architecture.

**Why one position at a time is slow.** The trunk runs at **1.95 GMAC/s, about 14% of the
chip's four-wide f32 throughput**. Not a coding defect: a dot product is a reduction, each
step needing the previous accumulator, and the standard escape — several accumulators — is
the reassociation `nn/mlp.rs`'s determinism contract forbids. Its whole value is 1.7x
(2.22 → 3.87 GMAC/s), which is not worth the contract. Note that this is *not* a bandwidth
wall, which was the first hypothesis and was wrong: a 64 KB L1-resident weight set runs at
2.22 GMAC/s and a 576 KB one at 1.95, near enough the same.

**Batching across games.** With activations laid out `[feature][batch]` the batch index goes
in the inner loop, which vectorises without touching the summation order, so each row still
sums ascending `j` from the same bias and is **bit-identical**. Trunk alone: 1.95 GMAC/s at
one position, 3.80 at 16, 7.56 at 32, **11.93 at 64** (~85% of the ceiling). End to end, 512
games on 8 threads:

| `--eval-batch` | wall clock | speed-up |
|---:|---:|---:|
| 1 | 225.5 s | 1.00x |
| 32 | 88.6 s | 2.54x |
| 64 | 69.3 s | **3.26x** |

Every one of those shards is byte-identical to the `--eval-batch 1` shard, and to the shard
the pre-change binary wrote. That is the finding: it is a speed result with **no accompanying
strength claim to verify**, because the games are the same games.

⚠️ **Two ways to measure this wrong, both of which I did first.** The batch is clamped to
`games / threads`, so a 100-game benchmark on 8 threads caps it at 12 and reports 1.13x for a
change worth 3.26x. And a batched kernel that accumulates into a slice of its output buffer
runs at the *unbatched* speed (1.9 GMAC/s against 11.9), because the compiler cannot rule out
aliasing with the weights; the accumulators have to be a fixed-size stack array.

**The gate and panel were done next**, through the same machinery — `probe::MatchGame` is the
state machine `selfplay::GameRunner` is, and `Agent::begin_decision` lets a `Box<dyn Agent>`
opt in while every other agent decides inline. A match caps the gain in a way self-play does
not: its two agents hold two different checkpoints, so each round's suspended games are grouped
by the network they wait on, and a gate reaches about half self-play's batch.

⚠️ **Measured properly on an idle machine, batching the gate is a *regression*.** 300 games,
two lane nets at 256 simulations, 8 threads: **373.5 s unbatched against 401.2 s at
`--eval-batch 64`, 7% slower**, for bit-identical scores (0.5333, W158 L138 D4). The
mechanism is the two-checkpoint split — a gate reaches only half self-play's batch, and at
~18 rows per evaluator the trunk saving no longer covers interleaving ~37 live search trees
per worker and staging their observations into contiguous rows. Self-play at the same setting
is 3.26x faster, so the knob is worth having; it is simply not worth applying everywhere.

`TrainingLoop.selfplay` therefore passes `--eval-batch` to self-play alone. `runs/seventh` ran
before this was known and paid the 7% on every gate and panel — it still finished 20
generations, so the cost was real but small against self-play's gain.

The lesson generalises past this knob: **an optimisation that is a large win on one workload
can be a small loss on a neighbouring one, and "same machinery, same direction" is not
evidence.** I asserted a ~2x gate speed-up in a commit message on exactly that reasoning
before measuring it.

**Two bugs, both found by measuring rather than by reasoning, and both worth remembering.**

*`min(games, batch)` is worse than not batching.* A worker with 37 games told to keep 32 in
flight advances all 32 in lockstep — one simulation each per round, so they finish together —
then drains the last 5 at a batch of 5 for a full game's length. On a 300-game gate that made
`--eval-batch 32` **slower than `--eval-batch 1`**, at 295% CPU against 712%. `nn::batch_slots`
spreads a worker's games over the fewest waves instead: 37 games at a batch of 32 runs 19 in
flight. A smaller batch that stays whole beats a big one that collapses.

*A batched match must absorb its games in game order.* Games finish out of order once several
are in flight, and `AgentBehaviour::absorb` **pushes** each game's lane and attack concentration
into a `Vec<f64>` whose mean `probe` reports — so absorbing them as they land shifts that mean's
last bits and makes the probe tables irreproducible. The gate itself reads only integers and
would never have shown it. `rule_2_the_ladder_is_eval_batch_independent` now compares the
per-game f64s, and fails if the sort is removed — verified by removing it.

**And a measurement trap that cost the first three benchmarks.** The batch is clamped by the
games a worker owns, so a 100-game benchmark over 8 threads caps it at 12 and reports 1.13x for
a change worth 3.26x. Size the benchmark before believing it.

**The negative result alongside it.** Restricting self-play to the 4 performance cores is
*slower*, not faster, despite `selfplay.rs` sharding statically with no work stealing: an
efficiency core is 3.79x slower than a performance core (88.3 s against 23.3 s for the same 10
games) but still contributes, and macOS migrates the stragglers onto performance cores as the
fast shards drain. 100 games, median of three: 8 threads 52.6 s, 4 threads 70.0 s, 6 threads
~59 s equivalent. Oversubscribing to 12/16/32 threads to recover the remaining tail does not
pay either. `threads = 0` is right and needs no change.

**And the GPU, assessed and declined.** The 10-core GPU is already used — `train.device =
"auto"` resolves to MPS — for the gradient step, which is 2–4% of the loop; MPS is 1.82x
faster there (13.27 → 7.28 ms/step at batch 512), worth about four seconds of a 21-minute
generation. It cannot take the inference, which is the 94%, because at batch 1 **MPS is
slower than the CPU** (617 µs against 322 µs per evaluation) and only overtakes above batch
~128. Batched, MPS reaches ~303k evals/sec against a batched CPU's ~120–150k — so the GPU is
worth perhaps 1.7x *on top of* batching, for a rewrite of the engine's inference path in a
framework the engine deliberately has no dependency on. Batching was the whole prize; the GPU
is not worth it on this machine.

### F4.6: building the lane symmetry into the architecture closes it completely, and playout cap randomisation pays for the cost

`runs/sixth`, `configs/train-3h-new.toml`, split, `encoding_slots = 21`, `run.seed = 5000000`
so the games span seeds **5,001,400–5,019,600** (1,400 a generation), 13 generations over
**4.69 hours** on the 8-core laptop, from a random init. Two changes together, for a reason
given below.

**The symmetry result is unambiguous.** F4.5 got a long way with data augmentation and said
plainly that it could not finish the job: "nothing in the architecture enforces the symmetry,
so a data augmentation cannot close the last of it." The lane-equivariant network
(`PLAN.md` §4.2b) shares one set of weights across the three lanes and lets them interact only
through their mean, so no parameter is indexed by a lane at all:

| 128 pairs | gen022 | gen031 (augmented) | **runs/sixth gen013** | equivariant |
|---|---:|---:|---:|---:|
| opening prior on lane 1 / 2 / 3 | .320 / .277 / .403 | .328 / .331 / .341 | **.333 / .333 / .333** | .333 each |
| value-head spread (median / max) | 0.068 / 0.178 | 0.034 / 0.088 | **0.000 / 0.000** | 0 |
| policy TV between lane pairs (median / max) | 0.152 / 0.362 | 0.039 / 0.103 | **0.000 / 0.000** | 0 |
| top second action agrees across all three lanes | 82/128 | 114/128 | **128/128** | 128/128 |

⚠️ **These zeros are not a training result and must not be read as one.** They hold on a
*random init*, before any gradient step, and they held unchanged after 13 generations. The
architecture cannot represent a lane preference, so this row stops being a measurement and
becomes an assertion the build checks
(`engine/tests/encoding.rs::phase4_the_lane_network_is_exactly_equivariant`). The corollary is
that F4.5's `lane_augment` is now an exact no-op: on an equivariant network a relabelled sample
gives the identical loss *and* the identical gradient, so F4.5's +82 Elo does not carry over
and `train-3h-new.toml` sets `lane_augment = false`.

**Strength, honestly.** `runs/sixth` generation 13 against gen031, both at 256 simulations, 300
games, seed 1: **0.3233 ± 0.0527 (W96 L202 D2), which is −128 Elo.** From scratch in 4.69 hours
against an agent that is the product of three chained runs. That is a good result for the time
spent and it is **not** a stronger agent; nothing was shipped to `models/`.

⚠️ The run's own `gen031` reference column reached **0.515**, and that number is a trap. The
panel scores gen031 at **@64** against the candidate's @256 — a deliberate 4:1 search handicap,
because a from-scratch net would otherwise read 0.00 for most of the run. F3.8 and F4.4 price
64 → 256 at ~+141 Elo, which is very close to the 128 measured here. Read the slope of that
column, never its level.

**Playout cap randomisation** (`PLAN.md` §4.2c) is what made the architecture affordable. A
value target is the game's outcome and costs nothing extra however little search produced the
position; a policy target is the visit distribution and is worthless if the visits are few. So
25% of decisions got 256 simulations and a policy target and 75% got 32 and a value target
only. Measured on 40 games at 256 sims:

| network | games/sec | vs gen031 |
|---|---:|---:|
| flat `128x3` (gen031), no capping | 2.0 | 1.00x |
| lane `128x3` + capping | 2.0 | 1.00x |
| lane `128x4` + capping | 1.7 | 0.85x |
| lane `128x6` + capping | 1.2 | 0.60x |
| lane `128x6`, no capping | 0.5 | 0.25x |

The equivariant trunk runs once per lane and costs ~4x the flat one uncapped; capping hands
almost exactly that back. **This is why the two changes share a run** and why the usual
one-change-per-run rule was broken: separately, the first is a run with fewer generations and
the second is a run whose only change is a speed-up.

The fraction was chosen by measurement, not taste. At a fixed self-play budget the games
capping buys back partly replace the policy targets it costs:

| `full_search_fraction` | games/sec | games | policy targets | outcomes |
|---|---:|---:|---:|---:|
| 0.25 | 2.0 | 1400 | ~21,000 | **1400** |
| 0.40 | 1.4 | 980 | ~24,000 | 980 |

Nearly the same policy targets, 43% more outcomes — and outcomes are the scarce half.

**Learning speed.** Beat `greedy` (0.98) at **generation 3**. `runs/third`, the only other
from-scratch run, needed seven. 13 of 13 candidates promoted, zero refusals.

**⚠️ The learning-rate schedule was the binding constraint for three generations, and this is
the same mistake `PLAN.md` §4.2 change 5 already records.** The tiers were keyed to the 7–8
generations three hours buys; the run was extended to 4.69 hours, so a sixteenth rate from
generation 6 was decaying on a clock that no longer existed:

| generations | lr | Δ on the gen031 column |
|---|---:|---:|
| 6–8 (throttled) | 1.25e-4 | +0.055 over 3 |
| 9–11 (re-keyed) | 5.0e-4 | +0.180 over 3 |

Training loss fell monotonically throughout, so the flat stretch was the rate and not the data.
Moving one boundary tripled the slope. **`configs/train-big.toml`'s schedule is keyed to 50
generations it may well not reach**, and this is the cheapest mistake on the list to avoid.

**⚠️ The held-out set is broken for a from-scratch run, and this is the most important
operational finding here.** Held-out value MSE rose monotonically 0.723 → **1.002** while
training value loss fell 0.780 → 0.469 and external strength climbed the whole way. At 1.002 a
head that always predicted zero would score the same, so the number is noise. The cause is
structural: `holdout_samples` is carved from **generation 1**, which a from-scratch run plays
with a random-init network, and the agent leaves that distribution within a few generations.
`runs/fourth` and `runs/fifth` never saw it because they were warm-started and their generation
1 was already strong play.

So this run had **no trustworthy internal diagnostic** and was steered entirely on the gate and
the reference panel. `train-big` is also from scratch and will hit the same wall. The holdout
must be carved from a mid-run generation instead — `PLAN.md` §4.3 now says so.

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
- **The human check has started but is not a measurement yet.** The agent lost 0-5 to the owner
  at gen016, unrecorded. Five recorded games since, all played without hints: 2-0 to the agent
  against gen022 at 4096 simulations, and 2-1 to the owner against gen031 across 128 and 8192.
  So the agent can now take games off the one person who has played it, which it could not
  before. Five games across three budgets and unpaired seeds is a pilot, and the value of the
  corpus is the diagnosis of the losses rather than the score. `PLAN.md` item 1.

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
