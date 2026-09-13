# Duel 52 Plan

## What this project is for

Two questions, and the engine and the bot exist only to answer them.

1. **What does optimal play in Duel 52 look like?** Nobody has published an answer. There is
   no prior engine, bot, or strategy analysis for this game.
2. **Is the game balanced?** Is the first player favoured, are the thirteen card powers worth
   comparable amounts, and does the win condition reward a degenerate strategy?

The insight is the deliverable. The bot is the instrument, and a slightly weaker agent whose
play can be explained is worth more here than a stronger one that cannot.

That framing decides priority throughout this file. Training compute is worth spending only
when agent strength is the thing blocking a question, and right now it mostly is not.

## Status

| Phase | State |
|---|---|
| 0. Specification | Done |
| 1. Engine and rules validation | Done |
| 2. Hand written baselines | Done, and now retired as a benchmark |
| 3. Neural self play loop | Done |
| 4. Scale up | Done. Four laptop runs, the fourth a from scratch architecture change; then the rented run — 24 h on 32 cores, `128 × 6` from noise, **+190 Elo** and the current default |
| 5. Extract the insight | Started early, partly banked, and now the critical path |
| 6. Verification | Not started |
| 7. R-NaD | Built beside AlphaZero and tested on the laptop; the three-hour GPU run is next. Item 8 |

## What is done

### The engine

A rules exact Rust engine for the split deck variant and two others, with 330 tests named
after the rule sections they check, PyO3 bindings, and a text CLI that names the rule behind
every prompt. Everything is seeded and deterministic: same seed and config gives the same
game, so any non reproducible result is a bug.

`game_rules.md` is the spec. It is not a copy of the published rules; it is an engine ready
version where every claim is tagged as published, resolved by a player, or inferred. Every
rules question raised for this project has been answered and ported into it, which is why
`OPEN_QUESTIONS.md` no longer exists as a live document.

One ruling reshaped the game more than any other. **Actions are mandatory. There is no pass**,
and there never was one in the published rules. The engine had offered one from the first
commit. Removing it deleted the strategic stalemate outright: greedy self play went from
0.7 to 1.7 percent stalemates to zero in 4,000 games per variant.

### The hand written ladder, and why it is retired

Five agents, each reasoning from its own information set through determinization: random,
greedy, flat Monte Carlo, PIMC, and information set MCTS. They were built as the Elo
benchmark and served as one for two phases.

**They are now saturated and no longer measure anything.** The current agent beats the
strongest of them, `ismcts:800`, by 200 games to 0. A rung that loses every game carries no
information about the winner. The ladder is kept as history and as a sanity check that a
broken checkpoint is broken, not as a scale.

The scale is now anchored on the first trained agent instead. See `FINDINGS.md`.

### The training loop and four agents

An AlphaZero style loop over information set MCTS. There is exactly one encoder and it is in
Rust; the network is defined and trained in PyTorch, evaluated in Rust, and a test asserts the
two forward passes compute the same function. Self play writes trajectory shards, the trainer
replays and fits them, and a gate promotes a candidate only when it beats the incumbent.

Four agents have come out of it, all on the same laptop. The first three are one lineage, each
warm started from the one before and each changing one thing. The fourth is a **second lineage
from a random init**, because it changed the architecture and the two share no tensor names:

| Agent | Trunk | The one change | Result |
|---|---|---|---|
| `gen016` | flat | The loop itself, from a random init | The first strong Duel 52 player that exists |
| `gen022` | flat | Teacher search raised from 64 to 256 simulations | +81 Elo on gen016 |
| `gen031` | flat | Every training sample relabelled by a random lane permutation | +82 Elo on gen022 |
| `lane-gen032` | lane `128 x 3` | The lane symmetry built into the architecture, and 5x the games per hour from batched self play | **+167 Elo on gen031** |
| `32c-24h-best` | lane `128 x 6` | Off the laptop: 24 hours on 32 rented cores, twice the trunk depth, from a random init | **+190 Elo on lane-gen032** |

`32c-24h-best` is the current default. The measurements are in `FINDINGS.md`; the provenance of
each checkpoint is in `models/README.md`.

**What five runs have established about the method:** the loop works, and the binding
constraint is games per hour rather than ideas. The first three gained +81, +82 and stopped on
their own clocks. The fourth gained +167 in one sitting — not because the idea was better, but
because batching self play's forward passes across concurrent games made 80,000 games in a
night possible where the laptop had managed 18,200. The fifth gained +190 by taking that same
observation off the laptop entirely: four times the cores, and a trunk twice as deep to spend
them on. Nothing yet says the method is near its ceiling; what it says is that the hardware has
been the ceiling every time it was moved.

⚠️ The fifth run moved **two** things at once, so its +190 is a joint effect and not evidence
that depth was the right place to spend the cores. `FINDINGS.md` F4.9.

## What is next

Ordered by what each answers, not by what is easiest. The first four need no rented hardware
and none of them is blocked on a stronger agent.

### 1. Play and record a human series against `lane-gen032`

**Status: under way. Seven games recorded, and the agent now wins more than it loses.**

The corpus in `games/` as of 2026-09-09, all seven played without hints:

| opponent | budget | owner's record |
|---|---|---|
| `gen022` | @4096 | 0 wins, 2 losses |
| `gen031` | @128 | 1 win |
| `gen031` | @8192 | 1 win, 1 loss |
| `lane-gen032` | @8192 | **0 wins, 2 losses** |

**This is the criterion the project set for itself, and it has been met.** The agent has taken
five of seven recorded games off the owner, who beat `gen016` five out of five in the
unrecorded series. It is not yet a measurement: seven games is not a series, the seeds are not
paired, and three different budgets are mixed together. What it settles is that the agent is no
longer obviously below the one human who has played it, which is the thing the earlier drafts
of this file were waiting to find out.

The two `lane-gen032` games are the first against the second lineage, on seeds 18 and 19, and
the agent won both. That is consistent with its +167 Elo over `gen031` but it is two games, and
two games is an anecdote — the value in them is the *diagnosis* below, not the scoreline.

What is still owed is the *diagnosis*, which was always the point of recording rather than
the scoreline.

`duel52 play --record` writes a game as (config, seed, chosen indices), a few hundred bytes
that replay it exactly including hidden information, and `duel52 replay` walks it back showing
what the net thought at each decision.

⚠️ **Play the six series games without `--hint`.** The flag puts the agent's top few moves on
screen *before* you choose, which is the right tool for learning the game and the wrong one
for measuring it: the whole value of this series is that the human's move was made without
seeing the net's. A hinted game is flagged in the record and `replay` prints a warning over
the table, so a contaminated game cannot quietly join the six — but the flag is on the same
command line, so it is worth knowing before rather than after. Use it on throwaway seeds.

**Why this is first.** Every strength number in this project is scored against agents this
project wrote, on a scale this project anchored, so **the human series is the only external
measurement that exists.** A +157 Elo rating and a losing record against one person would not
have been in contradiction, and the only way to find out which described the agent was to play.

It also gates everything below it. `FINDINGS.md` reports what strong play looks like based on
what the trained agents do. If a human beats them consistently and in the same way, those
findings describe a flawed agent rather than the game, and the flaw is diagnosable from the
recordings in a way that no amount of self play can reproduce.

**What is left to do here**, in order:

- **Finish the series properly.** Six games on fixed seeds, paired on the deal so each seed is
  played from both sides, at one budget rather than three. The five games recorded so far are
  a pilot, not the measurement.
- **Diagnose the owner's two wins**, which is why the games were recorded at all. For each,
  find the nodes where the value head was confident in the side that went on to lose, and sort
  them into three buckets: moves more search fixes, moves the value head scores wrongly at any
  budget, and moves that look fine at every budget and are still wrong. The third bucket is the
  valuable one, because an error invisible from inside the system is exactly what self play
  cannot label, and a systematic error a strong search makes is a place the game rewards
  something the search cannot see. That is a finding about Duel 52, not only about the agent.
- **Turn the corpus into a fixed evaluation set.** Score every future checkpoint on the same
  positions and ask whether the value head still thinks it is winning at the node where it
  actually lost. Seconds to run, and it never goes stale.

⚠️ **Two housekeeping problems with the corpus as it stands**, both cheap and both worth
fixing before it grows. The records name their opponent as `runs/fifth/checkpoints/best.d52nn`,
and `runs/` is not tracked, so a fresh clone cannot re-score these games even though it ships
the byte-identical `models/duel52-split-gen031.d52nn`. And `games/owner-vs-gen006.jsonl` now
holds a game against `gen031`, so the filenames no longer say what is in them.

### 2. Turn the hand size result from a correlation into a cause

**Status: measured, correlational, and the largest effect in the project.**

The trained agents hold cards back through the entire draw phase and arrive at the endgame
with six to seven cards where every hand written agent arrives with under two. Within their
own games, the side holding more at the seam is the side that wins.

**Why this matters for balance, not just for strategy.** A lane cannot be won until the draw
pile and the opponent's hand are both empty. So the player who still holds cards cannot be
scored against at all, while the player who has emptied their hand can be. That is not a
defensive resource, it is a one sided scoring window, and if it is real then the published win
condition creates a strong incentive to stall that the rules text does not signal anywhere.
**A game whose optimal line is "do not spend your resources" is a balance finding**, and it is
the single most actionable thing this project could tell a player or a designer.

The causal arrow is genuinely ambiguous today. Holding cards may win games, or a winning
position may simply be one that never forces you to commit cards. Two experiments separate
them, both cheap and both needing only `testkit` and an existing checkpoint:

- **Intervention.** Build positions identical except for hand size at unlock and evaluate both
  with a fixed strong agent. If hand size causes the win rate, the constructed advantage
  survives.
- **Forced commitment.** Constrain the agent to play a card on turns where it would have
  hoarded, and measure what the constraint costs. That prices the resource instead of
  correlating with it.

Until one of these lands, this stays "supported, not confirmed".

### 3. Lane commitment measured after the seam, not across the whole game

**Status: the cheapest open item in the project.**

The trained agents concentrate their play on two lanes far more than any baseline. The
statistic collected is lane share over a whole game, which is the wrong window: nothing can be
won before the seam, so a whole game share dilutes the endgame across twenty five turns in
which commitment costs nothing and decides nothing.

**Why it matters.** Whether strong play commits to two lanes, and when it decides to, is one
of the few questions here with a direct answer a human can use at the table. It is also a
balance question in disguise: three lanes with a two lane win condition means one lane is
meant to be surrendered, and if optimal play surrenders it immediately then the third lane is
doing less work than the design implies.

The work is to collect lane share restricted to post unlock turns in `probe`, and re run.

### 4. A card value table

**Status: built 2026-09-10 — see §4a for the instrument and the first table.** The measurement
now exists and its control passes; what is still outstanding is a value head good enough for
the numbers to be about Duel 52 rather than about `lane-gen032`, which is what the long runs
are for. Re-run `duel52 card-value` when they land.

What exists is a flip *timing* curve: the order in which the agent turns each rank face up,
which spans twenty two turns from the 8 to the Queen and is stable across search budgets. That
ordering is about when a power starts paying, not what it is worth. Timing is not value, and
this project has been careful never to claim otherwise.

**Why this is the balance question.** "Are the thirteen powers worth comparable amounts" cannot
be answered from the rules text or from play experience, and it is exactly what a trained value
head can be asked. Duel 52 gives every rank a distinct power on a standard 52 card deck, so if
two or three ranks are worth multiples of the others, that is a designed imbalance nobody has
quantified.

The route is the value head plus hand built `testkit` positions: hold a position fixed, vary
the rank of one card, and read the change in the value head's score. That gives a value per
rank in units of win probability, with the flip timing curve as an independent ordering to
check it against. It needs a trustworthy value head, which is the one place a stronger agent
would genuinely help, and the value head is the weaker half of every checkpoint so far.

### 4a. The card value table — **built 2026-09-10**

`engine/src/cardvalue.rs` and `duel52 card-value`. Hold a position fixed, vary the rank of one
card **in the observer's hand**, read the value head's delta, report in win-probability points.
Paired — every rank on the identical position — so the position's own difficulty cancels and
the ± is the error on the comparison between cards.

Measured on `lane-gen032`, 400 positions, canonical rules: the Ace leads at **+3.50 ± 0.14**
and the 4 trails at **−3.83 ± 0.13**, a spread of **7.33 points**. Full table in
`MODULAR_RULES.md` §10.

**In hand, not on the board, and the difference decides the answer.** The first implementation
put the card face-up in a lane. Its top four came out 8, J, 10, 9 — exactly the four constant
powers — because a constant power is fully live face-up, a one-shot has already fired and is
spent, and the 3's Trap only works face-down. It ranked power *kind*. A card in hand has its
whole future ahead of it whatever power it carries, which is the only way to measure thirteen
cards on the same footing. The board number is still reported beside it, and the **gap** between
them says where a card's value lives — in the flip, or in the body.

Two reasons to believe it, and three not to over-read it:

- **The null control is clean.** Each rank is also substituted into the *opponent's* hand,
  which the observer cannot see — thirteen bit-identical tensors, so the value head must return
  one number. Spread `0.00000`, checked in tests against a hash of the observation rather than
  a network. `card-value` prints it first and says not to read the table without it.
- **The fix is visible in the output.** `in hand` interleaves the power kinds; `on board` still
  sorts by them. The artifact is isolated in the column that is labelled as carrying it.
- ⚠️ **It measures gen032's value head, not the game.** The prerequisite this section names.
- ⚠️ **The sample skews early.** Asking "what if I held an `R`" needs a copy of `R` to be
  somewhere unseen, so only 8.3% of `split` positions admit all thirteen substitutions, and
  those are the ones where least has been revealed.
- ⚠️ **`mirrored` cannot be measured this way at all** — 0.06% of positions survive, because
  §9b publishes the removed multiset. The tool detects the thin sample and refuses to print.

### 4b. The comparison document — **built 2026-09-10**

`duel52 analyze` and `python -m duel52.analysis`, writing `analysis/<variant>.md` and `.html`.
The same measurements for every agent, side by side, with almost no prose: first-player
advantage, when cards are played and flipped, how long they stay face-down, hand size at the
unlock and what a bigger hand is worth, win rate per rank held at the deal and at the unlock,
pairs, how cards die, and what becomes of a face-down card. Adding a model is one more
`--agents` entry; adding a measurement is one function.

**The design decision is that the engine does not compute statistics.** It plays an agent
against itself and writes down what happened — one row per player-game, one row per card that
entered play, plus a `meta.json` naming the agent, the seed range and the `rules_hash`.
Everything in the document is a fold over those two files. That is what makes a question
nobody anticipated cost a Python function and a re-render rather than another run of the
games, which at a thousand simulations is hours per model.

Four things it does that a first version would not have:

- **Intervals are clustered on the deal.** Both games of a colour-paired deal hold the same
  cards, and forty card rows come out of one shuffle. An unclustered interval is too narrow —
  about √2 for the paired games alone, more for anything counted per card.
- **The per-rank win rate is the *exclusive* one:** you held the card, the opponent did not.
  Pooling in the games where both held it adds symmetric win/loss pairs that carry no
  information about the card and drag every rank toward 0.500.
- **The opening hand is taken at the start of each player's own first turn.** `GameState::new`
  performs P0's opening draw (§2), so "the hand at setup" is six cards for P0 and five for P1.
  Recording that would have put a first-player edge into all thirteen rows.
- **Two card-value tables, side by side, in the same unit.** §4a's counterfactual, which needs
  a value head, and a logistic fit of the result on how many more of each rank you were dealt
  than your opponent, which needs nothing but the games. **The dealt hand is randomly
  assigned**, so that fit is a randomised comparison rather than a correlation — the closest
  thing in this project to an experiment, and it works for `random` too.

Open: the per-rank win rates are the sample-size driver, at ~0.46 exclusive observations per
game (±0.010 at 5,000 games, ±0.005 at 20,000), while everything per-card is tight by 2,000.
Adding games is moving `--seed` past the last chunk; the reader merges them.

### 5. First player advantage and the variant comparison

**Status: partly measured on one variant.**

First player advantage on the split deck variant covers even in 1,000 self play games, and a
strong agent does not move it away from even. That is a real balance result and it is the one
question already close to answered.

What is missing is the other two variants. Answering "is the split deck fairer than the rules as
written" costs a training run per variant rather than an evaluation. That is the honest price
and it is why this sits below the free work.

⚠️ **Corrected 2026-09-09.** This section used to say "the observation layout is per variant, so
a checkpoint cannot be loaded against a variant it was not trained on". That is false, and it
described a guard rail that does not exist. All three variants produce **identical** layout
hashes — `obs_dim 4290`, `action_dim 2194`, `obs b1355a841a1fdc4a`, `action 5169f9461d627b39` —
because the encoder is rank-agnostic and nothing in the layout depends on deck composition.
A split-trained checkpoint plays `--variant base` at full speed with no warning:

```
$ ./target/release/duel52 match --a netmcts:models/duel52-split-lane-gen032.d52nn@16 \
    --b random --games 4 --seed 1 --encoding-slots 21 --variant base
  config: variant=base two_power=bottom stalemate=20plies
  score for ...gen032...: 1.0000 +/- 0.0000 — W4 L0 D0
```

The conclusion above survives — a *fair* comparison still needs a run per variant, because an
agent trained on one variant is not the meta of another. The mechanism does not. What actually
stops the two from being confused is `rules_hash` (`MODULAR_RULES.md` §6), added on the
`ruleset-configs` branch; before that there was nothing.

Worth doing anyway, because the split deck variant is a house rule that the regular player
community adopted, and whether it actually improves the game is a question the community
cannot answer and this project can.

### 6. How far from optimal is any of this

**Status: not started, and it is the word "optimal" in question 1 that depends on it.**

Everything measured so far is relative. `gen031` is better than `gen022` is better than
`gen016`, and all three beat everything hand written. None of that says whether the play is
near optimal or merely the best of a small family that all share a blind spot.

Two instruments, in order of cost:

- **Local best response** against the full game, as an exploitability proxy. This is the
  honest version of "how strong is it really" and it is the detector for the tripwire below.
- **Duel52-mini**, a scaled down variant small enough for exact CFR, to validate the whole
  loop against a known equilibrium before any claim that the full game's policy is near one.

Until one of these exists, `FINDINGS.md` should keep saying "what the strongest available
agent does" and not "what optimal play is", which is the discipline it currently holds.

### 6b. Stage 1: the lane equivariant network and playout cap randomisation

**Status: DONE 2026-09-06. `runs/sixth`, 13 generations over 4.69 hours from scratch.
FINDINGS.md F4.6.** Both changes work. The lane symmetry is closed completely — policy TV
0.000 and agreement 128 of 128, against gen031's augmented 0.039 and 114 of 128 — and it holds
on a random init, so it is a property of the architecture rather than a training result.
Capping paid for the trunk exactly as intended: self play held at gen031's throughput despite a
forward pass roughly three times as expensive. 13 of 13 candidates promoted, no refusals, and
the equal simulation result was 0.3233 ± 0.0527 against gen031, or −128 Elo, from scratch.

Nothing was shipped to `models/`; the lineage is still gen016 → gen022 → gen031. **The answer
this run existed to give is yes**, and item 7 below is re-scoped around it.

**§4.2b, lane equivariance.** F4.3 measured the flat network not knowing its three lanes are
interchangeable, and F4.5 attacked that with six fold data augmentation: policy TV between
lane pairs fell 0.152 to 0.039 and bought +82 Elo, but argmax agreement stopped at 114 of 128,
because augmentation can only *ask* a network to be symmetric. `arch = "lane"` makes it
symmetric. The trunk runs once per lane with one shared set of weights and the lanes exchange
information only through their mean, so relabelling them permutes the policy exactly and
leaves the value alone. Measured on a trained checkpoint: opening prior `.333 / .333 / .333`,
policy TV `0.000`, agreement `128/128`. There is no parameter that could encode a lane
preference, so this is not a smaller defect but no defect.

Two consequences. It forces a from scratch run — the two architectures share no tensor name,
so `--init-from` refuses across them. And it makes `lane_augment` pointless: on an
equivariant net a relabelled sample gives the identical loss *and* the identical gradient, so
F4.5's result does not carry over.

**§4.2c, playout cap randomisation** (Wu 2019). A value target is the game's outcome and costs
nothing extra however little search produced the position; a policy target is the visit
distribution and is worthless if the visits are few. So a quarter of decisions get the full 256
simulations and a policy target and three quarters get 32 and a value target only, which makes
search 4.0 times cheaper as measured.

**The two are in one run on purpose, which breaks the one change per run rule.** They are not
independent: the equivariant trunk costs about 2.9 times the flat network's forward pass and
capping pays almost exactly that back. Separately, the first is a run with fewer generations
and the second is a run whose only change is a speed up. Attribution survives because each has
its own specific readout — `duel52.lanes` for the first, the policy target count for the
second.

**§4.2d, batched evaluation.** Done 2026-09-09, and the only change in this project so far
that is provably free. Profiling self-play found the lane trunk at **89% of worker CPU** —
determinization, legality and encoding together are under 5% — running at about **14% of the
chip's arithmetic width**. The cause is structural rather than sloppy: a dot product is a
reduction, every step needing the previous step's accumulator, so one position cannot fill
four-wide f32 units however the loop is written, and the escape that would work — several
accumulators — is precisely the reassociation `nn/mlp.rs`'s determinism contract forbids
(measured at 1.7x for the whole cost of breaking the contract).

The batch index has no such dependency. Evaluating `B` positions with the activations laid
out `[feature][batch]` puts the batch in the inner loop, which vectorises **without touching
the summation order**, so each row still sums ascending `j` from the same bias and comes out
bit-for-bit identical. Measured on the trunk alone: 1.95 GMAC/s at one position, 11.93 at 64.

The batch is taken **across games, never inside a search**. A worker keeps `G` games in
flight, advances each to the point it needs the network, and evaluates the round in one call;
no game's search is altered, so every game is still reproducible from its own seed. This is
the design `engine/src/nn/mod.rs`'s `Evaluator` header specified before there was a consumer
for it. The alternative — leaf parallelism with virtual loss — was rejected on measurement
grounds, not taste: it changes the visit distribution by an amount that depends on how peaked
the prior is, so it distorts the *differences* between agents rather than offsetting them,
which would invalidate F4.1's +81 and F4.5's +82 and unfreeze the gen031 reference row. It
also cannot fill a batch at `cap_sims = 32`.

Cost, on the 8-core laptop at 512 games, lane 128x3, 256 sims with capping:

| `--eval-batch` | wall clock | speed-up |
|---:|---:|---:|
| 1 | 225.5 s | 1.00x |
| 32 | 88.6 s | 2.54x |
| 64 | 69.3 s | **3.26x** |

⚠️ The batch is clamped to `selfplay.games / run.threads`, so a small generation on many
cores silently gets a smaller one — `train check` prints the effective number. It costs about
400 KB of live search tree per game in flight.

**The gate can use it and does not**, which is the measured answer rather than the assumed one:
batching a match is a 7% regression (`FINDINGS.md` F4.8), because its two checkpoints halve the
batch. The machinery went in anyway and `run_match` grew an `eval_batch` argument, `probe::MatchGame` is the same state machine `GameRunner` is, and
`Agent::begin_decision` is how a `Box<dyn Agent>` opts into being suspendable — every agent
but `netmcts` returns `None` and decides inline exactly as before.

One thing is genuinely different in a match, and it caps the gain: **the two agents hold two
different checkpoints**, so a round's suspended games are grouped by the network each is
waiting on and evaluated separately. The in-flight games split roughly evenly between the two,
so a gate reaches about half the batch self-play does. A panel row against `random` or
`greedy` does not pay that, because the non-network agent never suspends.

⚠️ **And one trap, which cost a benchmark before it was found.** The batch a worker can use is
capped by the games it owns, and the naive `min(games, batch)` is wrong in a way that is worse
than not batching: a worker with 37 games told to keep 32 in flight advances all 32 in lockstep
so they finish together, then drains the last 5 at a batch of 5 for a full game's length. On a
300-game gate over 8 threads that tail made `--eval-batch 32` **slower than `--eval-batch 1`**,
at 295% CPU against 712%. `nn::batch_slots` spreads a worker's games evenly over the fewest
waves instead — 37 games at a batch of 32 runs 19 in flight, not 32. A smaller batch that
stays whole beats a big one that collapses.

**The trap this uncovered, which item 7 must not walk into.** Depth on the equivariant net
costs far more than the parameter count suggests, because in the search path the input layer
is sparse and the policy head is masked, so the trunk is the whole cost. `lane 128 x 6` runs at
0.25 times gen031's throughput and `lane 128 x 3` at 1.00 with capping on. `blocks = 3` is what
fits in three hours on eight cores; `blocks = 6` belongs on the rented box. CLAUDE.md has the
measured table.

### 7. The from scratch run on rented cores

**Status: DONE, 24 hours on 32 rented cores. It produced the current default, by the largest
margin the project has measured.** `configs/train-24h-32c.toml` is the config that ran;
`train-24h-64c.toml` and `train-24h-128c.toml` are the same run sized for bigger boxes and
`train-big`/`train-12h` are the earlier drafts it superseded.

| | |
|---|---|
| Config | `configs/train-24h-32c.toml`, `run.hours = 24.0`, `run.threads = 32`, seed `30000000` |
| Network | lane-equivariant `128 × 6`, 604,005 parameters — **twice the depth** of every checkpoint before it |
| Started from | a random init; `--init-from` refuses a trunk of a different shape, so `blocks = 6` *meant* from scratch |
| Shipped | `models/duel52-32c-24h-best.d52nn` and `…-gen039.d52nn` (different files — `best` predates the last generation) |
| Result | **0.7488 ± 0.0424 against `lane-gen032`** at equal 256 simulations, 400 games, W299 L100 D1 — **+190 Elo** |

**Item 7's own question was whether rented cores produce something the laptop could not, and
the answer is yes.** For context the whole first lineage was +189 across three runs, and the
lane architecture's own step was +167; this is +190 in one sitting, from noise.

⚠️ **Read it for what it is: the run changed two things at once.** Four times the cores *and*
twice the trunk depth. The +190 does not decompose into a depth number and a compute number,
and separating them costs a second 24-hour run at `128 × 3` that has not been done. The config
itself says the fork was close — a `128 × 3` warm start from gen032 would have had 55% more
games and a +167 head start — so "depth was the right call" is **not** what this measures.

⚠️ **The run directory never came back from the box.** Generations played, generations
promoted, positions seen and the true wall clock are in `runs/eighth/log.jsonl` on a machine
that no longer exists, so `models/README.md` records the config's *plan* where it cannot record
the run's record. `log.jsonl` is 20 KB. Copy it back next time — it is the whole provenance,
and it is also the only place the value-curve question below could have been answered.

**What is still open from this item.** The value head is the half that has plateaued in every
run and the half playout cap randomisation is aimed at; 6b's value curve was the thing to look
at hardest when this run finished, and without the log there is nothing to look at. Item 4's
card-value table needs a value head worth trusting, so that question is now answered by
re-running a generation locally or not at all.

What follows is the reasoning as it stood before the run, kept because the sizing rules in it
still apply to the 64- and 128-core configs.

**What it now is.** A lane-equivariant `128 x N` trunk trained from a random init with playout
cap randomisation, against a gen031 progress column. Not the flat deeper trunk this item
described before: 6b showed the equivariant network learning far faster from scratch — beating
`greedy` at generation 3 where `runs/third` needed seven — and reaching within 128 Elo of
gen031 in 4.7 laptop hours. The instrument change is now the point of the run rather than a
side effect of it.

**The four things 6b fixed in the config, all of them mistakes this run would otherwise have
made at 24 hour scale** (FINDINGS.md F4.6):

1. `arch = "lane"` and `full_search_fraction = 0.25`, the two validated changes.
2. **`blocks` is a Stage 1 measurement, not a constant.** The throughput table this file used
   to quote is for the flat network. The equivariant trunk runs once per lane, and depth is
   the whole cost, because the input layer is sparse and the policy head is masked. On the
   laptop `lane 128 x 6` ran at 0.25 times gen031's uncapped throughput and 0.60 with capping.
   Stage 1 must re-measure `lane 128 x {3,4,6}` **with capping on** and set `blocks` from that.
3. **The holdout is carved from generation 8, not generation 1.** A from scratch run plays
   generation 1 with a random init, and 6b's held out value MSE rose 0.723 to 1.002 while the
   training loss fell and external strength climbed the whole way. At 1.002 the number is what
   predicting zero scores. 6b had no trustworthy internal diagnostic at all; this run must.
4. **`lr_schedule` is keyed to the generations Stage 1 says the run will finish**, not to the
   `generations` backstop. 6b lost three generations to a schedule keyed to a length the run
   outgrew, and moving one boundary tripled the slope. This is the third time the project has
   made this mistake and the cheapest one on the list to avoid.

**What success looks like, and what does not count.** The result is an **equal simulation match
against gen031**, run at the end. The reference column in the log scores gen031 at `@64`
against the candidate's `@256`, a deliberate 4:1 handicap so that a from scratch net's column
is not pinned at zero — in 6b that column read 0.515 while the honest number was
0.3233 ± 0.0527, which is −128 Elo. Read the column's slope during the run and never its
level, and do not ship on it.

**Why it is still last rather than first.** It is the only item on this list that costs money,
and it answers none of the four questions above. It makes the instrument better, and the
instrument is not what is blocking the insight. The one exception is item 4: the value table
needs a value head worth trusting, and the value head is the half that has plateaued in every
run — which is also the half playout cap randomisation is aimed at, so 6b's value curve is
the thing to look at hardest when this run finishes.

Rent cores, not a GPU. The gradient step is 2 to 4 percent of the loop and only 1.4 times
faster on a GPU than on eight CPU cores, so `run.threads` matters far more than
`train.device`. `runs/sixth` spent thirty seconds of a twenty one minute generation on it.

Everything else is Rust self play and the gate, and **the split between those two is a ratio
this run sets rather than a constant of the loop.** An earlier draft of this paragraph said 87
percent self play. That was correctly measured on Phase 3's shape — 3,000 self play games
against a 200 game gate — and stopped describing anything once generations shrank to 1,200 and
the gate grew to 300: the measured share runs from 91 percent in `runs/third` down to **53
percent in `runs/sixth`**. The mechanism is that a gate game is uncapped and net against net
while self play is capped to a mean of 88 simulations, so **a gate game costs about 3.3 times a
self play game**. CLAUDE.md's Architecture section carries the per run table.

The consequence for this item is a sizing rule rather than a fact: pick `selfplay.games` so the
gate is a tax and not a partner. Below roughly 6,000 games against a 600 game gate the run
spends more than half its wall clock evaluating itself, which is compute that buys precision on
a number the gate only needs to get roughly right.

### 8. R-NaD, a second learner beside AlphaZero

**Status: Stages 1 and 2 built 2026-09-13; Stage 3's smoke run passed on the laptop, and the
three-hour run on a GPU is next.** Scope: implement R-NaD (Perolat et al., *Science* 2022) as a
second learner beside the AlphaZero loop, test it on the laptop, and run it once for three hours
on a GPU. The AlphaZero loop keeps working as it does today: no existing run, config, shard or
checkpoint changes.

**What gets built.** A search-free policy learner. The actor samples moves from the network's
policy. The learner fits the policy with the NeuRD loss and the value against V-trace targets,
on rewards transformed by `η·log(π/π_reg)` against a regularisation policy that is replaced on a
fixed schedule. It holds four networks — online, an EMA target, and the last two regularisation
policies — and its output checkpoint is the target net. Play uses the target net's policy with
probabilities below a threshold zeroed and the rest rounded to a grid.

**Reference.** OpenSpiel's `open_spiel/python/algorithms/rnad/rnad.py`, removed from master on
2025-05-28 (`e165bbdd`). Pin the last version, **`d1dcdf5d`**. Starting values:

| key | reference default | here |
|---|---|---|
| `eta_reward_transform` | 0.2 | kept [ASSUMED] |
| `entropy_schedule_size` / `_repeats` | 20,000 / 1 learner steps | sized by `bench` for ~10 updates in the run's clock |
| `target_network_avg` | 0.001 | **raised with the schedule**, to about `10 / entropy_schedule_size` — see Stage 3 |
| `learning_rate` | 5e-5 | **5e-4** [ASSUMED] — see Stage 3 |
| Adam `b1` / `b2` | 0.0 / 0.999 | kept |
| `nerd.beta` / `nerd.clip`, `c_vtrace` | 2.0 / 10,000, 1.0 | kept |
| `finetune.policy_threshold` / `_discretization` | 0.03 / 32 | kept, as `netsample`'s defaults |
| `batch_size` / `trajectory_max` | 256 / 10 | whole games instead |

**Layout.** The two learners share the engine, rulesets, encoder, the `.d52nn` format with both
forward passes, and every agent and measuring command. R-NaD's own pieces are new files:

| piece | where |
|---|---|
| batched games | `GameBatch` in `bindings/src/lib.rs`: the engine's rules and encoder, many games at once |
| actor and learner | `py/duel52/rnad/`, beside `py/duel52/train/`, in one process on one device |
| configs, runs | `configs/rnad-*.toml`, `runs/rnad-*` |

There is no R-NaD shard: games are played and learned from in the same process, so `.d52sp`
and `duel52 selfplay` are untouched. Code both learners need — config key checking,
`resolve_device`, the network classes and checkpoint I/O — is imported from the existing
modules rather than duplicated.

**Tests that keep AlphaZero unchanged**, each landing before the code it protects:

1. `rnad_leaves_the_alphazero_shard_byte_identical`: a golden hash of a small fixed-seed
   `duel52 selfplay` shard, recorded before the first R-NaD commit.
2. `obs b1355a841a1fdc4a` at 21 slots is unchanged, every checkpoint in `models/` loads, and one
   rewritten by this build is byte-identical. New header keys are written only when they differ
   from the default.
3. Every existing `configs/train-*.toml` passes `train check`, and `duel52.train` imports nothing
   from `duel52.rnad`.
4. `train-fast` and `rnad-fast` both run at toy size in `py/tests`.

**Stage 1 — engine plumbing.**

- **`netsample:<path>`**, an agent: one forward pass, masked softmax, then the post-processing
  above (threshold 0.03, grid 32); `netsample:<path>@raw` skips it. It draws one uniform per
  decision from its own seeded stream, so `phase2_no_agent_reads_hidden_information` stays
  exact. Add it to `TEST_ROSTER` by hand.
- **`GameBatch`**, a binding holding `N` games. `observe(obs, mask, ids, players)` writes the
  observation and legal mask of every game awaiting a decision straight into the caller's
  numpy arrays, across threads with the GIL released. `apply(ids, actions)` takes encoded action
  indices, which the engine decodes and checks for legality, then plays any forced decisions
  itself, as `.d52sp` skips them. `returns()` gives each player's result on `-1..1` from
  `learning_value`. The config is resolved from the same fields as the CLI flags.
- **Optional checkpoint header keys**, absent meaning today's behaviour: `learner = rnad`, and
  `value_head = linear`, since the reference value head is linear and every forward pass here
  ends in `tanh`. `netmcts` clamps a linear value.

Exit: `netsample` plays a match on `32c-24h-best`; tests 1 and 2 pass; a game driven through
`GameBatch` reaches the same outcome as the same game driven through `Game`; and the actor's
decisions per second is measured on the laptop.

**Stage 2 — the learner.** Port `v_trace`, `get_loss_nerd` and the entropy schedule to PyTorch in
`py/duel52/rnad/`.

- `test_rnad_vtrace_matches_reference` and `test_rnad_nerd_matches_reference`: parity to 1e-6
  against fixtures generated once from `d1dcdf5d` and committed as `.npz`, so the suite does not
  depend on JAX. Include a trajectory with forced steps.
- `test_rnad_kuhn_poker_converges`: exploitability, computed exactly, from 0.458 at a random
  init to **0.007** after 3,000 steps; the test's bar is 0.02, because sampled games keep it
  oscillating around 0.01. Kuhn is a test fixture for the learner, not a second rules engine.

Exit: both green before any Duel 52 run.

**Stage 3 — the first three-hour run.** `configs/rnad-fast.toml` for a smoke run on the laptop
and `configs/rnad-3h.toml` for the real one on a GPU: lane `128 × 3`, from scratch, `split`,
`encoding_slots = 21`, `device = "auto"`.

```bash
.venv/bin/python -m duel52.rnad check --config configs/rnad-3h.toml
.venv/bin/python -m duel52.rnad run   --config configs/rnad-3h.toml --run-dir runs/rnad-3h
```

- **One process, strictly on-policy.** Each learner step plays a fresh batch of games in a
  `GameBatch` with the online net on the device, sampling on the device, then takes one learner
  step on exactly those games — the reference's own shape, so there is no policy lag to correct.
- **No promotion gate.** Every so many steps the target net is written as a checkpoint and plays
  `random` and `greedy` through `duel52 match --a netsample:<checkpoint>`, and the scores are
  logged.
- **Entropy schedule** sized in learner steps from the measured step rate, so that three hours
  covers several regularisation updates [ASSUMED].
- **Resume state**: all four networks, the optimiser, the step counter and the RNG state, in the
  run directory.
- ⚠️ **Games replay exactly from their seeds, but a GPU run does not reproduce bit for bit**, since
  GPU kernels are not deterministic the way the Rust forward pass is.

Exit: `rnad-fast` completes on the laptop and resumes after a stop; `rnad-3h` completes on a GPU
and writes a target-net checkpoint that `netsample` plays.

**Measured on the 8-core laptop, 2026-09-13** (MPS, lane `128 × 3`, from scratch):

- **The smoke run passes.** 40 steps, two regularisation updates, an evaluation match through
  `netsample`, Ctrl-C saves and `--resume` continues. On the CPU a resumed run is bit-identical to
  an uninterrupted one.
- **Speed.** A 256-game step is 3.2 s — 1.1 s acting, 2.0 s learning — so ~81 games per second,
  and it fits in 16 GB. The engine is not the constraint: `GameBatch` encodes and applies ~359,000
  decisions per second on eight threads.
- ⚠️ **The target average has to move with the schedule.** Each regularisation update copies the
  target net, which moves `target_network_avg` of the way to the online net per step. The
  reference pairs 0.001 with 20,000-step iterations. Shortened to a laptop-sized schedule without
  raising it, the regularisation policy stayed pinned to the random init and the policy did not
  learn at all. 200 steps of 128 games, scored at step 200:

  | `learning_rate` | `target_network_avg` × size | vs `random` | vs `greedy` |
  |---|---:|---:|---:|
  | 5e-5 | 0.05 | 0.435 | 0.050 |
  | 5e-4 | 0.05 | 0.525 | 0.055 |
  | 5e-4 | 5 | 0.935 | 0.295 |
  | 1e-3 | 5 | **0.990** | **0.350** |

  `check` now warns below 5, and `bench` prints both numbers for a machine's step rate.
- **The value head's targets leave ±1** — to ±1.6 in those runs — so `value_head = linear` stays.

## Held in reserve

Nothing. R-NaD, which waited here, is item 8.

## Constraints that shape every decision here

- The owner has limited time and is delegating implementation. Prefer making a defensible
  call, documenting it as `[ASSUMED]`, and flagging it, over blocking.
- Everything runs locally on an M series Mac and scales to rented hardware through config
  alone.
- **Elo here is an internal coordinate.** The scale is anchored on `gen016` and every agent on
  it was written for this project. A number from it is a distance between two of our own
  agents, never an absolute, and the human series is the only external check.
- Insight beats strength.
