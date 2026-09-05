# Duel 52

An engine and self-play agent for [Duel 52](https://www.juddmadden.com/duel52/index.html),
the two-player combat card game by Judd Madden and Nina Riddell that uses a standard 52
card deck.

It exists to answer two questions nobody has published an answer to. **What does optimal play
actually look like**, and **is the game balanced**? As far as I can tell there is no existing
engine, bot, or strategy analysis for this game, so there is nothing to read and the only way
to find out is to build a player strong enough to ask.

The insight is the deliverable. The bot is the instrument.

## Try it

A Rust toolchain is all you need to play. The engine has zero dependencies, so the build
resolves nothing and takes about ten seconds. The trained agent ships with the repo,
[models/duel52-split-gen031.d52nn](models/duel52-split-gen031.d52nn), 3.6 MB, an ordinary git
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
    --opponent netmcts:models/duel52-split-gen031.d52nn@4096
```

Every prompt names the rule it is applying, so if you think the engine is wrong you can point
at the line. `duel52 powers` prints the card-power reference, and `duel52 demo --seed 47`
watches a game play out action by action.

## The game, briefly

Three lanes, three actions per turn, and every card has a power tied to its rank. Cards are
played face down and flipped to activate. You win a lane by clearing it once neither player
can play more cards, and you win the game by taking two lanes.

Three properties make it interesting, and all three shape everything below:

1. **Ten cards are removed unseen at setup**, so uncertainty about hidden cards never fully
   resolves, even at the end of the game.
2. **Lane wins require an empty draw pile and an empty opposing hand**, so the entire draw
   phase is positional. Nothing is decided until the deck runs dry.
3. **Suits do not matter.** Rank is the whole of a card's identity.

## Status

There is a trained agent in the repo and you can play it. The engine plays the full game to
spec, with 332 Rust tests named after the rule sections they check, 94 Python tests, PyO3
bindings, and a text CLI.

The AlphaZero style training loop runs end to end. There is exactly one encoder and it lives
in Rust; a network is defined and trained in PyTorch, evaluated in Rust, and a test asserts
the two forward passes compute the same function.

Three runs have gone through it, each on the same laptop, each warm started from the one
before, and each changing exactly one thing:

| Agent | The one change |
|---|---|
| [gen016](models/duel52-split-gen016.d52nn) | The loop itself, from a random init. 57,000 self-play games in 1.94 hours |
| [gen022](models/duel52-split-gen022.d52nn) | Teacher search raised from 64 simulations to 256 |
| [gen031](models/duel52-split-gen031.d52nn) | Every training sample relabelled by a random permutation of the three lanes |

Rated against each other at equal simulations, 400 games per pairing, with the first trained
agent pinned at zero:

| agent | Elo | +/- | expected vs. gen016 |
|---|---:|---:|---:|
| `netmcts:gen031@256` | **+157** | 13 | 0.711 |
| `netmcts:gen022@256` | +91 | 13 | 0.628 |
| `netmcts:gen016@256` | 0 | 0 | 0.500 |

**gen016 is the floor because the hand-written ladder is saturated.** Five hand-written agents
(random, greedy, flat Monte Carlo, PIMC, information set MCTS) were the benchmark for two
phases. gen031 beats the strongest of them 200 games to 0, and a rung that loses every game
measures nothing about the winner.

**Every number above is scored against agents written for this project.** The one external
check that exists says something different: the project owner beat gen016 five games out of
five, and no series has been played against the two agents since. That is the measurement the
project turns on, which is why [PLAN.md](PLAN.md) puts it first.

What the agents have taught us about the game is in [FINDINGS.md](FINDINGS.md). That file is
the point of the project.

## Recording a game and replaying it

A self-play table tells you an agent is strong. A replay tells you where a human and the agent
disagree, which is the only place a strategy insight can come from.

```bash
# Play, and append the finished game to a file.
./target/release/duel52 play --encoding-slots 21 --seed 123 \
    --record games/mine.jsonl \
    --opponent netmcts:models/duel52-split-gen031.d52nn@4096

./target/release/duel52 replay --record games/mine.jsonl            # what is in the file
./target/release/duel52 replay --record games/mine.jsonl --game 1   # walk it
./target/release/duel52 replay --record games/mine.jsonl --game 1 --node 34   # and the board
```

A record is `(config, seed, chosen indices)` and nothing else. The engine is deterministic, so
those three things replay the game exactly, including the hidden information and the ten cards
removed unseen. A 158-node game is under a kilobyte, which is why the games are committed to
the repo. Only finished games are written: a half-played game cannot be checked against an
outcome, and that check is what makes an old record trustworthy.

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

The footer counts the nodes where the value head was **confident and wrong**: `|v| > 0.6`,
better than four to one, backing the side that went on to lose. A value head that is uncertain
is behaving correctly. One that confidently backs a loser is the failure everything past the
search horizon inherits.

**`node`, `turn` and `round` are three different counters** and the replay prints two of them.
A node is one decision offered to one player. A turn is one player's turn of three actions
(two on the opening turn, four after an Ace). A round is both players' turns and nothing counts
it. Sub-decisions are separate nodes that cost no action, so the node column climbs about 3.4
per turn.

## Where the project is going

[PLAN.md](PLAN.md) has the detail. In short, the next work is not a bigger training run:

1. **Play and record a human series against gen031.** The only external measurement there is.
2. **Turn the hand-size result from a correlation into a cause.** The agents hoard cards
   through the whole draw phase and the side holding more at the seam is the side that wins.
   If that is causal, the published win condition rewards stalling, which is a balance finding.
3. **Measure lane commitment after the seam** rather than across the whole game.
4. **Build a card value table.** Whether the thirteen powers are worth comparable amounts is
   the balance question, and nothing answers it yet.
5. **First-player advantage across all three variants**, which costs a training run per
   variant because the observation layout is per-variant.
6. **Exploitability**, so that "optimal" is a word the project is allowed to use.
7. **The long from-scratch run on rented cores**, last, because it answers none of the above.

## Docs

| File | Contents |
| --- | --- |
| [game_rules.md](game_rules.md) | The spec. The disambiguated ruleset the engine implements. |
| [PLAN.md](PLAN.md) | What is done, and in detail what is next and why. |
| [FINDINGS.md](FINDINGS.md) | What the trained agents have shown about the game. |
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

One ruling is worth knowing before you play: **actions are mandatory and there is no pass.**
The published rules say "take three actions" and stop, so a player who can act must act.

## License

GPL-3.0
