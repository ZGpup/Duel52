# Duel 52

An engine and self-play agent for [Duel 52](https://www.juddmadden.com/duel52/index.html),
the two-player combat card game by Judd Madden and Nina Riddell that uses a standard 52
card deck.

It exists to answer two questions. **What does optimal play actually look like**, and **is the game balanced**? As far as I can tell there is no existing
engine, bot, or strategy analysis for this game, so there is nothing to read and the only way
to find out is to build a player strong enough to ask.

## Try it

A Rust toolchain is all you need to play. The engine has zero dependencies, so the build
resolves nothing and takes about ten seconds. The trained agent ships with the repo,
[models/duel52-32c-24h-best.d52nn](models/duel52-32c-24h-best.d52nn), 2.3 MB, an ordinary git
blob with no LFS to install.

```bash
# No Rust yet? This is the whole install. On Windows, run the rustup-init.exe from
# https://rustup.rs instead. Then restart the shell, or `source "$HOME/.cargo/env"`,
# so that `cargo` is on PATH.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

git clone https://github.com/ZGpup/Duel52.git && cd Duel52
cargo build --release

# Play the trained agent. `--encoding-slots 21` is not optional: it is what fixes the
# size of the observation, and the checkpoint refuses to load against any other value.
./target/release/duel52 play --encoding-slots 21 \
    --opponent netmcts:models/duel52-32c-24h-best.d52nn@8192
```

Every prompt names the rule it is applying, so if you think the engine is wrong you can point
at the line. `duel52 powers` prints the card-power reference, and `duel52 demo --seed 47`
watches a game play out action by action.

Answer a prompt with the arrow keys or by typing its number, whichever suits the move: up and
down walk the lines you could actually pick, right or Enter takes the one you are on, and left
goes back a question. Every question opens with the first line you could pick already marked,
so Enter alone takes it. The line you are on carries a `*` and lights up red on the board
above — the `*` because the red is gone on a terminal without colour, and the arrow keys are
not.

Add `--hint` and the agent you are playing will show you the three moves it would consider
before each of your decisions, best first — the share of its search each one got, and what it
thinks your chances are after it.

```bash
./target/release/duel52 play --encoding-slots 21 --hint \
    --opponent netmcts:models/duel52-split-lane-gen032.d52nn@4096
```

```text
   PLAY   #1
   FLIP   #2
   ATTACK #3
   PAIR    —

 netmcts:models/duel52-split-lane-gen032.d52nn@4096 · it puts you at 52% from here
  #  the net would consider                          sims   after
  1  FLIP  lane 2 #1 (7 ²♥) -> reveals 7              41%     61%
  2  ATK   lane 1: your #2 (8 ²♥) -> opp #1           22%     55%
  3  PLAY  9 face-down into lane 3                    12%     52%
```

`--hint 5` lists five instead of three, and `--hint-agent <agent>` asks somebody other than
your opponent — a bigger budget than you are playing against, say, or anybody at all in a
hotseat game. 

## The game, briefly

Three lanes, three actions per turn, and every card has a power tied to its rank. Cards are
played face down and flipped to activate. You win a lane by clearing it once neither player
can play more cards, and you win the game by taking two lanes.

Three properties make it interesting, and all three shape everything below:

1. **Ten cards are removed unseen at setup**, so uncertainty about hidden cards never fully
   resolves, even at the end of the game.
2. **Lane wins require an empty draw pile and an empty opposing hand**, so the draw
   phase is somewhat positional. Nothing is decided until the deck runs dry.
3. **The abilities are flexible rules.** The abilities of each of the cards could be changed without changing the core of the game, so there is room to adjust the balance if the agents show that the published powers are not balanced. 

## Status

There is a trained agent in the repo and you can play it.

The AlphaZero style training loop runs end to end. There is exactly one encoder and it lives
in Rust; a network is defined and trained in PyTorch, evaluated in Rust.

Six runs have gone through it. The first five were on the same laptop: three of them are one
lineage, each warm started from the one before and each changing exactly one thing, and the
fourth started a **second lineage from scratch**, because it changed the network architecture
and the two share no tensor names — there was nothing to warm start from. The sixth is the
first that a laptop did not run: 24 hours on 32 rented cores, from scratch again, and it
produced the agent that is now the default.

| Agent | Trunk | The one change |
|---|---|---|
| [gen016](models/duel52-split-gen016.d52nn) | flat | The loop itself, from a random init. 57,000 self-play games in 1.94 hours |
| [gen022](models/duel52-split-gen022.d52nn) | flat | Teacher search raised from 64 simulations to 256 |
| [gen031](models/duel52-split-gen031.d52nn) | flat | Every training sample relabelled by a random permutation of the three lanes |
| — | lane | A new root. The lane symmetry built into the *architecture* rather than asked for by augmentation: the trunk runs once per lane with shared weights, so a lane preference is unrepresentable rather than merely small |
| [lane-gen032](models/duel52-split-lane-gen032.d52nn) | lane `128 x 3` | 80,000 self-play games in 6.96 hours, which is ~5x what the laptop could produce before self-play's network evaluations were batched across games |
| — | lane `128 x 6` | A third root. Twice the trunk depth, which `--init-from` refuses to warm start into because the shapes disagree — so from a random init again |
| **[32c-24h-best](models/duel52-32c-24h-best.d52nn)** — the default | lane `128 x 6` | Off the laptop: 24 hours on 32 rented cores, 24,000 self-play games a generation to the laptop's 4,000 |

The jump in the middle is not a better idea than the ones before it; it is the same loop given
five times the data per hour. Batching the forward pass across concurrent games made self-play
3.26x faster with **bit-identical** results — the batch is taken across games and never inside
a search, so no game's search is altered and a shard is byte-for-byte what an unbatched run
would have written. The detail is in [FINDINGS.md](FINDINGS.md) F4.7 and F4.8.

The jump at the end is compute, and it is the point of having rented it: **+190 Elo over
`lane-gen032`**, from noise, in one sitting. Depth is what the extra cores were spent on —
in the search path the input layer walks only the observation's ~205 non-zeros and the policy
head is masked to the legal logits, so the trunk is the whole cost, and a trunk's cost is
linear in its blocks but quadratic in its width. ⚠️ That run changed **two** things at once,
four times the cores and twice the depth, so the +190 does not split into a depth number and a
compute number. Separating them costs another 24 hours and has not been done.

Rated against each other at equal simulations, 400 games per pairing, with the first trained
agent pinned at zero:

| agent | Elo | +/- | expected vs. gen016 |
|---|---:|---:|---:|
| `netmcts:lane-gen032@256` | **+360** | 13 | 0.888 |
| `netmcts:gen031@256` | +161 | 11 | 0.717 |
| `netmcts:gen022@256` | +96 | 11 | 0.634 |
| `netmcts:gen016@256` | 0 | 0 | 0.500 |

The fit puts the last step at +199. Measured head-to-head instead — `lane-gen032` against
`gen031` directly, 400 games — it is **+167** (0.7238 ± 0.0436, W288 L109 D3). The two are
different estimators of the same quantity and they agree: +199 sits inside the head-to-head
interval. Elo is not transitive across a game that is not, so where they differ, the direct
measurement is the one to trust about *those two agents* and the fit is the one to trust about
the scale as a whole.

⚠️ **`32c-24h-best` is not in that fit.** It is measured only head-to-head so far, against the
agent it had to beat:

| pairing, 400 games at 256 simulations each | score | record | Elo |
|---|---:|---|---:|
| `32c-24h-best` vs `lane-gen032` | **0.7488 ± 0.0424** | W299 L100 D1 | **+190** |

Do not add +190 to `lane-gen032`'s +360 and read off +550. A Bradley–Terry fit is over the
whole table at once, so putting this agent on the scale re-conditions every rating on it — that
is what moved `gen022` and `gen031` by +5 and +4 when `lane-gen032` was added. The refit is a
`ladder` run that has not been done.

**gen016 is the floor of the elo system** Five hand-written agents
(random, greedy, flat Monte Carlo, PIMC, information set MCTS) were the benchmark for two
phases. Already at gen031 the strongest of them lost 200 games to 0, and a rung that loses
every game measures nothing about the winner.

What the agents have taught us about the game is in [FINDINGS.md](FINDINGS.md). That file is
the point of the project.

## Recording a game and replaying it

A self-play table tells you an agent is strong. A replay tells you where a human and the agent
disagree, which is the only place a strategy insight can come from.

```bash
# Play, and append the finished game to a file.
./target/release/duel52 play --encoding-slots 21 --seed 123 \
    --record games/mine.jsonl \
    --opponent netmcts:models/duel52-32c-24h-best.d52nn@8192

./target/release/duel52 replay --record games/mine.jsonl            # what is in the file
./target/release/duel52 replay --record games/mine.jsonl --game 1   # walk it
./target/release/duel52 replay --record games/mine.jsonl --game 1 --node 34   # and the board
```

A record is `(config, seed, chosen indices)`. The engine is deterministic, so
those three things replay the game exactly. 

The walk prints one row per decision you made:

```
node      actor   value  played                              prior  second opinion
   6     P0 you   -0.00  PLAY  7 face-down into lane 1       0.076  search: PLAY  8 face-down into lane 1 (90% of visits, v +0.09)
   8     P0 you   -0.35  FLIP  lane 2 #2 (8 2H) -> reveals   0.190  search agrees (58% of visits, v -0.18)
```

- **`value`** is the net's score for the position from your seat, on -1 to +1.
- **`prior`** is the probability its policy head put on the move you actually chose.
- **`second opinion`** is what a full search would have played instead, with its share of
  visits.

The checkpoint and search budget default to the agent that actually played the game, so a bare
`replay --game 1` says what your opponent was thinking at the time. Passing `--checkpoint`
scores the same game with a different net, which is how an old game becomes a permanent
evaluation set for a new one.

**`node`, `turn` and `round` are three different counters** and the replay prints two of them.
A node is one decision offered to one player. A turn is one player's turn of three actions
(two on the opening turn, four after an Ace). A round is both players' turns and nothing counts
it. Sub-decisions are separate nodes that cost no action, so the node column climbs about 3.4
per turn.

## Where the project is going

[PLAN.md](PLAN.md) has the detail. In short, the next work is not a bigger training run:

1. **Play and record a human series against `32c-24h-best`.** The only external measurement
   there is.
2. **Build a card value table.** Whether the thirteen powers are worth comparable amounts is
   the balance question, and nothing answers it yet.
3. **First-player advantage across all three variants**, which costs a training run per
   variant because the observation layout is per-variant.
4. **Exploitability**, so that "optimal" is a word the project is allowed to use.

The long from-scratch run on rented cores was fifth on this list, last because it answered
none of the others. It has since been run — 24 hours on 32 cores — and it produced the current
default by +190 Elo. That does not promote it back up the list: a stronger instrument still
answers none of the four questions above, which is what the list is ordered by.

## Docs

| File | Contents |
| --- | --- |
| [game_rules.md](game_rules.md) | The spec. The disambiguated ruleset the engine implements. |
| [PLAN.md](PLAN.md) | What is done, and in detail what is next and why. |
| [FINDINGS.md](FINDINGS.md) | What the trained agents have shown about the game. |
| `analysis/<variant>.md` and `.html` | The comparison document: the same measurements for every agent, side by side. Built by `python -m duel52.analysis`, never edited by hand. The `.html` carries the figures, and its tables sort on any column — a third click puts the table back in the order it was written in. |
| [models/README.md](models/README.md) | The shipped checkpoints: how each was trained, and what it scores. |
| [CLAUDE.md](CLAUDE.md) | Commands, repo layout, architecture, and the facts that are easy to get wrong. |
| [archive/](archive/) | The superseded working documents, frozen for provenance. |

## Layout

```
engine/      the rules engine (zero dependencies) and the `duel52` CLI
  tests/     one named test per ruling, named for its rule section
bindings/    PyO3 wrapper, kept separate so the engine never depends on Python
py/duel52/   the Python package: training loop and analysis, never an encoder
configs/     variant configs (split is the default) and training configs
models/      trained checkpoints, tracked in git, with their provenance
games/       recorded human games, a few hundred bytes each
analysis/    the comparison document, plus the per-game and per-card corpora it is built from
archive/     superseded working documents
```

Training output, `runs/` and `checkpoints/`, is deliberately not tracked. A run is reproducible
from its config and seed, and the one checkpoint worth keeping is copied into `models/` by
hand.

## A note on rules

`game_rules.md` is not a copy of the official rules. It is an engine ready version, with every
claim tagged as either published, resolved by a player, or inferred and pending confirmation.
It also specifies the red and black split deck variant common among regular players, which is
the default configuration here because symmetric material makes results much cleaner to
measure.

## License

GPL-3.0
