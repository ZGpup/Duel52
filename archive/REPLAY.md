# Reading a replay

`duel52 replay` walks a game you have already played and prints, at every decision **you**
made, what the trained net thought of the position and what it would have done instead. It
is the tool `PLAN.md` §4.0 exists for: a self-play table tells you an agent is strong, and a
replay tells you *where a human and the agent disagree*, which is the only place a strategy
insight can come from.

Everything below is the output of `engine/src/bin/duel52.rs` (`cmd_replay`,
`print_replay_table`, `print_replay_summary`) and `engine/src/display.rs` (`render`,
`card_token`, `card_status`, `describe_move`). `duel52` with no arguments prints the flag
list; this file explains what the numbers mean.

---

## 0. Three counters, three names

A replay prints two different numbers that both used to be called "ply", which is why they
now are not. Get these straight before reading anything else:

| Term | Is | Where you see it | Game 2 of the corpus |
| --- | --- | --- | --- |
| **node** | One decision offered to one player. | The replay table's `node` column; `--node N`; the `nodes` column of the index. | 172 |
| **turn** | One player's turn: **3 actions**, 2 on the opening turn, 4 after an Ace. | `turn N` in a board's status line; `demo`'s action log. | 51 |
| **round** | Both players' turns. | Nothing counts it. Derive it as `turn / 2`. | ~25 |

A turn is *several* nodes, because sub-decisions — `2ND`, `NEXT`, `MOVE`, `BACK`, `PEEK` —
are separate zero-cost nodes (`DESIGN.md` §4). That is why the `node` column climbs about
3.4 per turn.

**"Ply" still exists, and it means a turn.** It is the standard sense — one player's
half-move — and it is what `game_rules.md` §7 defines when it counts "individual player turns
(plies)". It survives in three places where renaming it would be a change rather than a
rename: the `GameState::ply` field, the config keys `stalemate_quiet_plies` and `max_plies`
(written verbatim into every record, so old games would stop replaying), and the aggregate
statistics in `stats`, `probe`, `ladder` and `FINDINGS.md`, which are all denominated in it.
So when `FINDINGS.md` says "mean plies 45.2", read "45.2 turns". No user-facing line of the
replay tool or the board says "ply" any more.

## 1. What a record is

A game record is `(config, seed, chosen indices)` and nothing else. The engine is
deterministic, so those three things replay the game *exactly* — including the hidden
information, including the 10 cards removed unseen at setup. A 158-node game is under a
kilobyte.

```bash
./target/release/duel52 play --encoding-slots 21 --seed 123 \
    --record games/owner-vs-gen031.jsonl \
    --opponent netmcts:models/duel52-split-gen031.d52nn@4096
```

Two properties of the format are deliberate and worth knowing:

- **Only finished games are written.** A record that cannot be checked against an outcome is
  not evidence. Quitting mid-game loses it; the seed is printed either way, which is enough
  to deal the identical position again.
- **`walk` verifies rather than decodes.** Replaying re-runs the rules and re-derives the
  legal-action list at every node; a stored index that no longer names the same action, or a
  game that no longer reaches its recorded outcome, is refused. That is what stops a rules
  change from silently turning the corpus into games nobody played.

## 2. The index

`--record` without `--game` lists what is in the file. This is the first thing you want from
a file you have not opened in a month.

```
games/owner-vs-gen006.jsonl — 1 game(s)

game          seed    seat   nodes   result  opponent
   1           123      P0     158     loss  netmcts:runs/fourth/checkpoints/gen006.d52nn@4096
```

`result` is always **from your seat**. `nodes` is the number of decision nodes, which is not
the number of turns — see §0.

## 3. The walk

```bash
./target/release/duel52 replay --record games/owner-vs-gen006.jsonl --game 1
```

```
  seed 123 · you were P0 · opponent netmcts:…/gen006.d52nn@4096 · 158 nodes · P1 wins
  scoring with runs/fourth/checkpoints/gen006.d52nn · searching 4096 simulations per decision

node      actor   value  played                              prior  second opinion
   6     P0 you   -0.00  PLAY  7 face-down into lane 1       0.076  search: PLAY  8 face-down into lane 1 (90% of visits, v +0.09)
   7     P0 you   -0.27  PLAY  8 face-down into lane 2       0.053  search: PLAY  8 face-down into lane 3 (82% of visits, v -0.06)
   8     P0 you   -0.35  FLIP  lane 2 #2 (8 ²♥) -> reveals…  0.190  search agrees (58% of visits, v -0.18)
```

The checkpoint and the search budget **default to the agent that actually played the game**,
so a bare `replay --game 1` says what your opponent thought at the time. Overriding
`--checkpoint` scores the same game with a different net, which is how an old game becomes a
fixed evaluation set for a new one.

### `node`

An index into the decision nodes of the game, 1-based — §0's first row. **Not a turn, and
not an action.** This column climbs about 3.4 per turn and skips numbers, for three separate
reasons:

1. **A turn is three actions** (two on the first player's opening turn), and each is a node.
2. **Sub-decisions are separate zero-cost nodes** (`DESIGN.md` §4) — `2ND`, `NEXT`, `MOVE`,
   `BACK`, `PEEK`. A 10's twinstrike is one action but two nodes; a 5 that flips a King that
   re-empowers a lane is one action and a run of `NEXT`s. They cost no action, so a turn can
   be three actions and five nodes.
3. **An Ace grants a fourth action** when flipped, so even the action count per turn varies.

Here is one turn of game 2, with the bot's turn either side of it, at `--all-nodes`:

```
 133  P0 you  FLIP lane 1 #3 (10) → 10   ┐ your turn: 3 actions, 4 nodes —
 134  P0 you  ATK  your #3 [10] → opp #1 │ the twinstrike's second target
 135  P0 you  2ND  second target: #3     │ is a free sub-decision
 136  P0 you  ATK  your #1 [9]  → opp #3 ┘
 137  P1      ATK  …                     ┐
 138  P1      FLIP lane 1 #3 (2) → 2     │ its turn: 3 actions, 3 nodes —
 139  P1      ATK  …                     ┘ your gap from 136 to 140
 140  P0 you  ATK  …
```

The **gaps** are the opponent's nodes, which are not scored unless you pass `--all-nodes`.
That is a cost decision, not a display one: only the rows being analysed pay for a forward
pass, so a 172-node game at `--sims 4096` costs your half of it.

The footer prints both counters — `172 decision node(s) over 51 turn(s)` — so you never have
to hold the conversion in your head.

### `actor`

`P0 you` / `P1`. The seat that owns the decision. `you` is whichever seat the record says you
held.

### `value`

The value head's reading of the position, **from the perspective of the player to move**, on
`-1.0 .. +1.0` — the tanh output, not a probability. `+1` is a certain win for the actor,
`-1` a certain loss, `0` even.

The perspective flip is the single easiest thing to misread here. On your rows the number is
yours. On a `--all-nodes` opponent row it is *theirs*, so to plot one curve for the whole
game you have to negate the opponent's rows:

```bash
duel52 replay --record <file> --game 1 --sims 0 --all-nodes \
  | grep -E "^ *[0-9]+ +P[01]" \
  | awk '{if($2=="P0"){v=$4+0}else{v=-($3+0)}; print $1, v}'
```

### `prior`

The played action's share of the policy prior — a softmax over the **legal actions at that
node only**, in the order the record's indices refer to (`masked_softmax`). Read it against
uniform, not against 1.0: a node with 40 legal actions has a uniform prior of 0.025, so 0.078
there is three times the base rate and not a surprise at all. A prior in the low thousandths
in a narrow position is a real one.

### `second opinion`

Three different things can appear here, and which one you get depends on `--sims`:

| Output | Means |
|---|---|
| `search agrees (58% of visits, v -0.18)` | A search ran and its most-visited action was the one you played. |
| `search: FLIP lane 1 #3 … (73% of visits, v -0.59)` | A search ran and preferred something else. |
| `policy: PLAY J face-down into lane 1 (0.378)` | `--sims 0`: no search, this is just the policy argmax and its prior. |
| *(blank)* | No search, and the policy's argmax *was* your move. |

**The search has the last word when it ran**, so a policy disagreement it overturned never
appears — that is not the interesting fact about the node.

- **`% of visits`** is the winning action's share of the search tree's visits. It is a spread,
  not a probability: on a wide branching factor a confident search still reports 30%. Compare
  it against the number of legal actions, the same way you read `prior`.
- **`v`** is the search's root value, converted from its native win-probability
  (`0..1`) onto the value head's `-1..+1` scale so the two columns compare directly. A large
  gap between `value` and `v` is the diagnostic: the raw net misjudges the position and
  search fixes it.

⚠️ **The search in a replay is a fresh one, not a reproduction of the search that played the
game.** An agent's RNG stream depends on how many times it has been called, and the replay
rebuilds it from scratch. That is the right thing for analysing *your* decisions, which is
what the tool is for. It also means the bot's own recorded move and the replay's "second
opinion" on that move can differ without anything being wrong.

## 4. The footer

```
158 decision node(s) over 45 turn(s) — a turn is 3 actions (2 on the opening turn,
4 after an Ace), and sub-decisions are extra nodes that cost no action.

P1 wins. Value head confident (|v| > 0.6) in the side that lost: 0 of 78 scored node(s).
```

The first line is §0's conversion for this particular game, so the `node` column and the
boards' `turn N` never have to be reconciled by hand.

The second lists the nodes where the value head was **confident and wrong** — `|v| > 0.6`,
better than 4:1, and
backing the side that went on to lose. A value head that is merely uncertain is behaving
correctly; one that confidently backs a loser is the failure everything past the search
horizon inherits. `0 of 78` is a clean bill of health for the evaluation, and it means any
mistakes in the game were the human's, not the scorer's.

When the list is non-empty, sort it into three buckets — this is `PLAN.md` §4.0a, and it is
the whole point of the command:

1. **A higher `--sims` fixes it.** Re-run with a bigger budget; if the search changes its
   mind, it was a horizon problem.
2. **It plays the same at any budget but scores the position wrongly.** A value-function
   problem — training data, not compute.
3. **It looks fine at every budget and is still wrong.** The only bucket no amount of compute
   reaches, and the reason a human series is worth more than another self-play table.

## 5. The board (`--node N`)

`--node N` also prints the full position as it stood at that decision, drawn **from the acting
player's point of view** — what they could actually see when they chose. `--reveal` draws
from ground truth instead, which is how you check what a face-down card really was after the
fact.

```
 Deck: you 0 · opponent 1
 ═════════════════════════════════
 P1   hand 7   discard A A 7 8 9 10 10

    lane 1  │  lane 2  │  lane 3
    (? ²♥)  │  (? ²♥)  │  (? ²♥)      ← opponent's base cards
  ──────────┼──────────┼──────────
    [5 ¹♥]  │          │  [8 ²♥]      ← opponent's played cards
    [4 ²♥]  │          │
  ══════════╪══════════╪══════════    ← the front line
    [2 ²♥]  │  (3 ²♥)  │              ← your played cards
    [7 ¹♥]  │  [8 ²♥]  │
  ──────────┼──────────┼──────────
    (? ²♥)  │  (? ²♥)  │  (? ²♥)      ← your base cards

 P0   hand A 4 4 5 10 J   discard 2 3 5 6 6 7 10 J
 you know: P0's pile bottom-up: J
 ═════════════════════════════════
 turn 24 · base locked · quiet 0/20
```

**Orientation.** The observer is always at the bottom. Lanes are columns; each side's base
card sits at the far end of its column, on its own row. The double rule is the front line —
the only place cards can reach each other. An **empty base cell means that base card is
dead**, which matters, because a lane is won when the opponent's *entire* side of it is
empty, base card included.

**Card tokens** are `[rank HP♥]` face-up and `(rank HP♥)` face-down; `?` is a rank the
observer does not know. HP is a superscript. Every face-down card is a blank 2 HP card
whatever its rank; face-up it is 2 HP, or 3 for a Jack.

Two symbol columns follow a token, per `card_status`:

| Glyph | Means |
|---|---|
| `a`, `b`, … | Which declared pair the card belongs to, lettered within its side of its lane. |
| `*` | Frozen: cannot attack and cannot be flipped, by anyone (§8). |
| `·` | Out of attacks this turn. Only drawn on your own face-up cards, on your own turn. |
| `+` | More than one attack left this turn. |

**`you know:`** is private knowledge that is not on the board — the run of cards you put on
the bottom of a draw pile with a 2, and anything a 4's Foresight showed you. It appears only
when there is something to say.

**The status line.** `turn N` is §0's second row — one player's turn, `state.ply` — not the
table's node index. `quiet n/20` is §7's stalemate counter: turns with no damage and no
kill, reset by either. `base locked` / `base UNLOCKED` is the shape of the whole game: until
every draw pile is empty, nothing can be won. Once it unlocks, the line gains
`lanes won: you N · opp M`, **relative to the observer** — so the same position reads with the
numbers swapped depending on whose turn it was. Remember what `lanes_won_by` actually
computes (`apply.rs:831`): you win a lane when the opponent's side of it is empty *and* the
opponent's hand is empty *and* the piles are empty. Sweeping a lane while the opponent still
holds cards scores zero.

## 6. The verbs

| Verb | Action |
|---|---|
| `PLAY` | Put a card from hand, face-down, into a lane. |
| `FLIP` | Turn one of your face-down cards face-up, firing its power. |
| `ATK` | Attack: `lane L: your #a [card] -> opp #b [card]`. |
| `PAIR` | Declare two same-rank cards on your side of a lane as a pair (§5). |
| `2ND` | A 10's twinstrike: the second of its two targets. |
| `NEXT` | Adaptive resolution order (§8): which pending power resolves next. Often forced, and then it shows a prior of 1.000. |
| `MOVE` | A Queen's Move: pull an allied card from another lane into hers. |
| `PEEK` | A 4's Foresight: look privately at one face-down card, either side's. |
| `BACK` | A 2's View: put a card from hand on the bottom of your pile (house rule) or discard it, per `two_power`. |

Numbering (`#1`, `#2`, …) is `display.rs`'s `column_slots` order, the same order the board
draws a column in and the same order the CLI's menus use — the one place in the codebase
where lanes and cards are numbered from 1.

## 7. Recipes

```bash
# The index of a file.
duel52 replay --record games/g.jsonl

# The default walk: the opponent's own net, at the budget it actually played.
duel52 replay --record games/g.jsonl --game 1

# Fast pass, no search — policy and value only. Seconds instead of minutes.
duel52 replay --record games/g.jsonl --game 1 --sims 0

# Score an old game with a NEW net: the fixed evaluation set of PLAN.md §4.0a.
duel52 replay --record games/g.jsonl --game 1 \
    --checkpoint runs/fifth/checkpoints/best.d52nn --sims 4096

# Bucket (i): does a deeper search change its mind about the nodes it flagged?
duel52 replay --record games/g.jsonl --game 1 --sims 16384

# Both sides scored, for a value curve over the whole game.
duel52 replay --record games/g.jsonl --game 1 --sims 0 --all-nodes

# The board at one decision, as the player saw it — then as it really was.
duel52 replay --record games/g.jsonl --game 1 --node 82
duel52 replay --record games/g.jsonl --game 1 --node 82 --reveal
```

`--ply` and `--all-plies` are still accepted as undocumented aliases for `--node` and
`--all-nodes`, so a command line written before the rename keeps working.

⚠️ You never pass `--encoding-slots` to `replay`. The record carries the whole config,
`encoding_slots` included, so a game played at 21 replays at 21 on its own. What you do have
to get right is a `--checkpoint` you supply by hand: it must have been built at the same
`encoding_slots`, because that is what fixes `obs_dim`. A mismatch fails loudly at load,
which is the guard working.
