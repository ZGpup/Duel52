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
| 4. Scale up | Three laptop runs done. The rented run is not started |
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

**Status: the tooling is done, the games are not played.**

`duel52 play --record` writes a game as (config, seed, chosen indices), a few hundred bytes
that replay it exactly including hidden information, and `duel52 replay` walks it back showing
what the net thought at each decision.

**Why this is first.** The owner beat `gen016` five games out of five. No series has been
played against `gen022` or `gen031`. Every strength number in this project is scored against
agents this project wrote, on a scale this project anchored, so **the human series is the only
external measurement that exists.** A +157 Elo rating and an 0 for 5 record against one person
are not in contradiction, and until the series is replayed we do not know which of them
describes the agent.

This also gates everything below it. `FINDINGS.md` reports what strong play looks like based
on what the trained agents do. If a human beats them consistently and in the same way, those
findings describe a flawed agent rather than the game, and the flaw is diagnosable from the
recordings in a way that no amount of self play can reproduce.

Six games on fixed seeds, paired on the deal, recorded. Then sort the losing nodes into three
buckets: moves more search fixes, moves the value head scores wrongly at any budget, and moves
that look fine at every budget and are still wrong. The third bucket is the valuable one,
because an error invisible from inside the system is exactly what self play cannot label, and
a systematic error a strong search makes is a place the game rewards something the search
cannot see. That is a finding about Duel 52, not only about the agent.

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

### 7. The from scratch run on rented cores

**Status: configured, not run, and deliberately last.**

Every agent so far is a `128 x 3` trunk, because a warm start cannot change the shape of the
network it inherits. `configs/train-big.toml` is a 24 hour from scratch run at a deeper trunk.

**Why it is last rather than first.** It is the only item on this list that costs money, and
it answers none of the four questions above. It makes the instrument better, and the
instrument is not currently what is blocking the insight. The one exception is item 4: the
value table needs a value head worth trusting, and the value head is the half that has
plateaued in every run.

Rent cores, not a GPU. 87 percent of the loop is Rust self play on CPU cores and 4 percent is
gradient work, and the gradient step is only 1.4 times faster on a GPU than on eight CPU
cores. `run.threads` matters more than `train.device`.

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
