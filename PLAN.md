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
| 4. Scale up | Four laptop runs done, the fourth a from scratch architecture change. The rented run is not started |
| 5. Extract the insight | Started early, partly banked, and now the critical path |
| 6. Verification | Not started |
| 7. R-NaD | Held in reserve, on a tripwire |

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

### The training loop and three agents

An AlphaZero style loop over information set MCTS. There is exactly one encoder and it is in
Rust; the network is defined and trained in PyTorch, evaluated in Rust, and a test asserts the
two forward passes compute the same function. Self play writes trajectory shards, the trainer
replays and fits them, and a gate promotes a candidate only when it beats the incumbent.

Three agents have come out of it, each warm started from the one before, each on the same
laptop, and each changing one thing:

| Agent | The one change | Result |
|---|---|---|
| `gen016` | The loop itself, from a random init | The first strong Duel 52 player that exists |
| `gen022` | Teacher search raised from 64 to 256 simulations | +81 Elo on gen016 |
| `gen031` | Every training sample relabelled by a random lane permutation | +82 Elo on gen022 |

`gen031` is the current default. The measurements are in `FINDINGS.md`; the provenance of each
checkpoint is in `models/README.md`.

**What three runs have established about the method:** the loop works, the gains are steady
and roughly equal per run, and neither run so far was stopped by running out of ideas. The
first stopped on a promotion gate with no statistical power, the second and third on their own
clocks. Nothing yet says the method is near its ceiling.

## What is next

Ordered by what each answers, not by what is easiest. The first four need no rented hardware
and none of them is blocked on a stronger agent.

### 1. Play and record a human series against `gen031`

**Status: under way. Five games recorded, and the agent has started winning them.**

The corpus in `games/` as of 2026-09-06, all five played without hints:

| opponent | budget | owner's record |
|---|---|---|
| `gen022` | @4096 | 0 wins, 2 losses |
| `gen031` | @128 | 1 win |
| `gen031` | @8192 | 1 win, 1 loss |

**This is the criterion the project set for itself, and it has been met.** The agent has taken
three of five recorded games off the owner, who beat `gen016` five out of five in the
unrecorded series. It is not yet a measurement: five games is not a series, the seeds are not
paired, and two different budgets are mixed together. What it settles is that the agent is no
longer obviously below the one human who has played it, which is the thing the earlier drafts
of this file were waiting to find out.

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

**Status: nothing exists. This is the main balance deliverable and it is missing.**

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

### 5. First player advantage and the variant comparison

**Status: partly measured on one variant.**

First player advantage on the split deck variant covers even in 1,000 self play games, and a
strong agent does not move it away from even. That is a real balance result and it is the one
question already close to answered.

What is missing is the other two variants. **The observation layout is per variant, so a
checkpoint cannot be loaded against a variant it was not trained on**, and answering "is the
split deck fairer than the rules as written" therefore costs a training run per variant rather
than an evaluation. That is the honest price and it is why this sits below the free work.

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

**Status: configured and re-scoped after 6b. Not run, and still deliberately last.**
`configs/train-big.toml` is a 24 hour from scratch run, and 6b changed what it is.

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

## Held in reserve

**R-NaD**, swapping the learner while leaving the engine and encoders alone, for an
approximate Nash policy rather than a merely strong one.

AlphaZero over determinized search has a real ceiling in an imperfect information game: no
equilibrium guarantee, and no way to learn to conceal or signal deliberately. Switch only on
one of two signals, and neither is present today:

1. Search keeps scaling but the network stops absorbing it, across a run whose gate has the
   power to tell. That is a representation limit, not a compute one.
2. Self play looks healthy while the score against a fixed external opponent flattens or
   falls. That is the exploitability signature and no amount of compute fixes it.

Signal 2 needs a detector that can still detect. Every hand written yardstick is now saturated,
so the reference opponent has to be a frozen *trained* net, promoted whenever the incumbent
reference stops losing.

A third condition is more likely than either: the run succeeds and the owner still wins. That
is not a reason to change learner. It is a reason to look hard at what the human does that the
agent does not, which is item 1.

## Constraints that shape every decision here

- The owner has limited time and is delegating implementation. Prefer making a defensible
  call, documenting it as `[ASSUMED]`, and flagging it, over blocking.
- Everything runs locally on an M series Mac and scales to rented hardware through config
  alone.
- **Elo here is an internal coordinate.** The scale is anchored on `gen016` and every agent on
  it was written for this project. A number from it is a distance between two of our own
  agents, never an absolute, and the human series is the only external check.
- Insight beats strength.
