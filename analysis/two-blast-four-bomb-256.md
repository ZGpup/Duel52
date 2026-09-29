# Duel 52 — self-play analysis: `split`

Generated 2026-09-15 04:02 UTC · corpus schema 1 · figures in [`two-blast-four-bomb-256.html`](two-blast-four-bomb-256.html)

- **variant** — split
- **ruleset** — two-blast-four-bomb/e108e420b08c8675
- **trained on other rules** — ⚠️ lane-gen032@256 (trained on `unstamped`) — a control, not a same-rules result
- **the 2's power** — bottom
- **lanes** — 3, 2 to win
- **hand size** — 5
- **stalemate** — 20 quiet turns
- **deal seeds** — 1–1000
- **models** — lane-gen032@256, gen019@256
- **games per model** — 2,000, 2,000

Every agent plays **itself**. Intervals are 95% and clustered on the deal, since both games of a colour-paired deal hold the same cards. Turn numbers are the player's own turns, 1-based.

## Contents

- [Corpora](#corpora)
- [First vs second player](#first-vs-second-player)
- [Game shape](#game-shape)
- [Average card play turn](#average-card-play-turn)
- [Average card flip turn](#average-card-flip-turn)
- [Turns spent face-down](#turns-spent-face-down)
- [Hand size at the unlock](#hand-size-at-the-unlock)
- [Win rate with each card in the opening hand](#win-rate-with-each-card-in-the-opening-hand)
- [Win rate with each card in hand at the unlock](#win-rate-with-each-card-in-hand-at-the-unlock)
- [Pairs](#pairs)
- [How cards die: face-up or face-down](#how-cards-die-face-up-or-face-down)
- [What becomes of a face-down card](#what-becomes-of-a-face-down-card)
- [What a card is worth](#what-a-card-is-worth)

## Corpora

One agent per column, each playing **itself**. Every corpus in this document was played under the ruleset named in the header, and the reader refuses to merge two that were not. *Trained on* is the ruleset the checkpoint was stamped with; ⚠️ marks a control that played rules it was not trained on (`duel52 analyze --allow-cross-ruleset`). A deal is played twice with the seats swapped, so games = 2 × deals.

| model | agent | trained on | games | deals | chunks | card rows | games/sec | cpu time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| lane-gen032@256 | netmcts:models/duel52-split-lane-gen032.d52nn@256 | ⚠️ unstamped | 2,000 | 1,000 | 8 | 81,914 | 0.770 | 0.72 h |
| gen019@256 | netmcts:runs/mod-traps/checkpoints/gen019.d52nn@256 | two-blast-four-bomb/e108e420b08c8675 | 2,000 | 1,000 | 8 | 82,274 | 0.711 | 0.78 h |

## First vs second player

The score of whoever moved first, pooled over both halves of every colour-paired deal. 0.500 is no advantage. The interval is clustered on the deal, which is the unit that was randomised — both games of a pair hold the same cards.

| model | first-player score | P0 wins | P1 wins | draws | draw rate | separated from 0.500 |
|---|---:|---:|---:|---:|---:|---:|
| lane-gen032@256 | 0.5310 ± 0.0244 | 1,053 | 929 | 18 | 0.90% | yes |
| gen019@256 | 0.4985 ± 0.0238 | 988 | 994 | 18 | 0.90% | no |

*The figure is in [`two-blast-four-bomb-256.html`](two-blast-four-bomb-256.html#first-vs-second-player).*

## Game shape

Turns here are the game's, not one player's: a game of 42 turns is 21 each. The unlock is the turn the last draw pile emptied and base cards became attackable (`game_rules.md` §3) — until then a lane cannot be won, so it divides the game in two.

| model | mean turns | median | p10 – p90 | draw rate | stalemate / mutual / cap | reached unlock | mean unlock turn |
|---|---:|---:|---:|---:|---:|---:|---:|
| lane-gen032@256 | 46.6 | 47 | 42 – 51 | 0.90% | 0 / 18 / 0 | 100.0% | 13.0 |
| gen019@256 | 46.5 | 47 | 42 – 50 | 0.90% | 0 / 18 / 0 | 100.0% | 13.0 |

## Average card play turn

The owner's own turn on which a card is played from hand, face-down. Base cards are excluded — they were never played. A low number is a card that goes down early, which is not the same as a card that goes face-up early.

| model | mean play turn | cards |
|---|---:|---:|
| lane-gen032@256 | 10.680 ± 0.046 | 69,914 |
| gen019@256 | 10.604 ± 0.048 | 70,274 |

| rank | power | lane-gen032@256 | n | gen019@256 | n |
|---|---|---:|---:|---:|---:|
| A | Action | 7.55 ± 0.19 | 5,382 | 7.81 ± 0.19 | 5,398 |
| 2 | Blast | 13.14 ± 0.21 | 5,446 | 10.82 ± 0.21 | 5,582 |
| 3 | Trap | 12.87 ± 0.22 | 5,343 | 10.48 ± 0.23 | 5,403 |
| 4 | Bomb | 15.68 ± 0.25 | 5,045 | 13.32 ± 0.26 | 5,239 |
| 5 | Flip | 10.85 ± 0.23 | 5,187 | 13.37 ± 0.27 | 5,169 |
| 6 | Freeze | 12.94 ± 0.20 | 5,524 | 14.25 ± 0.21 | 5,495 |
| 7 | Heal All | 8.46 ± 0.18 | 5,456 | 8.41 ± 0.18 | 5,469 |
| 8 | Retaliate | 6.24 ± 0.18 | 5,595 | 6.32 ± 0.17 | 5,590 |
| 9 | Nimble | 8.03 ± 0.20 | 5,425 | 7.57 ± 0.19 | 5,446 |
| 10 | Twinstrike | 11.14 ± 0.20 | 5,410 | 11.81 ± 0.23 | 5,403 |
| J | Taunt | 6.05 ± 0.16 | 5,442 | 6.22 ± 0.16 | 5,432 |
| Q | Move | 16.19 ± 0.15 | 5,212 | 15.95 ± 0.17 | 5,220 |
| K | Empower | 10.39 ± 0.19 | 5,447 | 12.06 ± 0.20 | 5,428 |

*The figure is in [`two-blast-four-bomb-256.html`](two-blast-four-bomb-256.html#average-card-play-turn).*

## Average card flip turn

The turn a card goes face-up. The main table counts **only flips its owner chose** — a card turned up by a 5's cascade or by springing a 3's Trap went face-up without anyone deciding to, and averaging those in answers a different question. Base cards are tabled separately: they cannot be flipped before the unlock, so their timing is a fact about the unlock rather than about the card.

| model | mean flip turn (chosen) | flips | any cause |
|---|---:|---:|---:|
| lane-gen032@256 | 10.99 ± 0.05 | 62,387 | 11.02 ± 0.05 |
| gen019@256 | 11.17 ± 0.05 | 61,613 | 11.13 ± 0.05 |

| rank | power | lane-gen032@256 | n | gen019@256 | n |
|---|---|---:|---:|---:|---:|
| A | Action | 7.61 ± 0.19 | 5,308 | 7.90 ± 0.19 | 5,329 |
| 2 | Blast | 14.09 ± 0.26 | 4,129 | 13.19 ± 0.26 | 3,618 |
| 3 | Trap | 16.31 ± 0.25 | 3,096 | 13.67 ± 0.28 | 3,158 |
| 4 | Bomb | 16.35 ± 0.29 | 4,058 | 14.74 ± 0.29 | 3,882 |
| 5 | Flip | 11.29 ± 0.23 | 4,834 | 13.98 ± 0.27 | 4,690 |
| 6 | Freeze | 13.95 ± 0.22 | 4,717 | 15.68 ± 0.21 | 4,521 |
| 7 | Heal All | 8.58 ± 0.18 | 5,291 | 8.57 ± 0.18 | 5,311 |
| 8 | Retaliate | 6.37 ± 0.18 | 5,543 | 6.43 ± 0.17 | 5,542 |
| 9 | Nimble | 8.05 ± 0.21 | 5,109 | 7.70 ± 0.20 | 5,122 |
| 10 | Twinstrike | 11.37 ± 0.20 | 5,256 | 12.12 ± 0.22 | 5,162 |
| J | Taunt | 6.21 ± 0.16 | 5,307 | 6.35 ± 0.16 | 5,340 |
| Q | Move | 16.48 ± 0.15 | 4,996 | 16.16 ± 0.16 | 5,083 |
| K | Empower | 11.22 ± 0.21 | 4,743 | 12.80 ± 0.21 | 4,855 |

*Flips the owner chose, played cards only.*

| rank | power | lane-gen032@256 | gen019@256 |
|---|---|---:|---:|
| A | Action | 7.60 ± 0.19 | 7.88 ± 0.19 |
| 2 | Blast | 13.90 ± 0.24 | 12.60 ± 0.23 |
| 3 | Trap | 14.95 ± 0.21 | 12.78 ± 0.23 |
| 4 | Bomb | 16.12 ± 0.28 | 14.25 ± 0.28 |
| 5 | Flip | 11.26 ± 0.23 | 13.85 ± 0.27 |
| 6 | Freeze | 13.71 ± 0.21 | 15.27 ± 0.21 |
| 7 | Heal All | 8.50 ± 0.18 | 8.48 ± 0.18 |
| 8 | Retaliate | 6.37 ± 0.18 | 6.43 ± 0.17 |
| 9 | Nimble | 8.12 ± 0.21 | 7.78 ± 0.19 |
| 10 | Twinstrike | 11.35 ± 0.20 | 12.09 ± 0.22 |
| J | Taunt | 6.16 ± 0.16 | 6.32 ± 0.16 |
| Q | Move | 16.45 ± 0.15 | 16.14 ± 0.16 |
| K | Empower | 11.13 ± 0.20 | 12.70 ± 0.20 |

*Any cause — chosen, cascaded, or sprung.*

| rank | power | lane-gen032@256 | gen019@256 |
|---|---|---:|---:|
| A | Action | 15.94 ± 0.19 | 15.88 ± 0.18 |
| 2 | Blast | 16.23 ± 0.21 | 16.17 ± 0.22 |
| 3 | Trap | 16.11 ± 0.19 | 16.25 ± 0.18 |
| 4 | Bomb | 16.19 ± 0.21 | 16.10 ± 0.20 |
| 5 | Flip | 16.25 ± 0.20 | 16.16 ± 0.21 |
| 6 | Freeze | 16.27 ± 0.20 | 16.23 ± 0.21 |
| 7 | Heal All | 15.94 ± 0.20 | 15.84 ± 0.19 |
| 8 | Retaliate | 15.95 ± 0.21 | 15.83 ± 0.20 |
| 9 | Nimble | 16.05 ± 0.21 | 16.12 ± 0.21 |
| 10 | Twinstrike | 16.10 ± 0.22 | 16.16 ± 0.22 |
| J | Taunt | 15.92 ± 0.19 | 15.94 ± 0.18 |
| Q | Move | 16.40 ± 0.22 | 16.06 ± 0.19 |
| K | Empower | 16.11 ± 0.21 | 16.02 ± 0.20 |

*Base cards, any cause.*

*The figure is in [`two-blast-four-bomb-256.html`](two-blast-four-bomb-256.html#average-card-flip-turn).*

## Turns spent face-down

Measured in the owner's own turns, so a card flipped on the turn it was played is **0**. Three columns because one number would be a lie by omission: a rank that is flipped fast *and* killed fast has a short tenure for two different reasons.

* **among flipped** — cards that were eventually turned face-up. The decision.
* **never flipped** — the share that were not, whether killed hidden or still hidden at the end. This is the censoring, stated rather than dropped.
* **to exit** — every played card, counting a hidden death or the end of the game as the end of its tenure. How long a card actually spends hidden.

| model | mean turns face-down (among flipped) | never flipped | mean turns to exit (all cards) |
|---|---:|---:|---:|
| lane-gen032@256 | 0.58 ± 0.01 | 5.0% | 0.67 ± 0.02 |
| gen019@256 | 0.69 ± 0.02 | 5.4% | 0.79 ± 0.02 |

| rank | power | lane-gen032@256 | n | gen019@256 | n |
|---|---|---:|---:|---:|---:|
| A | Action | 0.08 ± 0.02 | 5,355 | 0.09 ± 0.02 | 5,373 |
| 2 | Blast | 1.11 ± 0.06 | 4,612 | 1.92 ± 0.10 | 4,525 |
| 3 | Trap | 2.46 ± 0.10 | 4,970 | 2.57 ± 0.11 | 5,147 |
| 4 | Bomb | 0.81 ± 0.06 | 4,364 | 1.04 ± 0.07 | 4,488 |
| 5 | Flip | 0.53 ± 0.03 | 4,927 | 0.66 ± 0.04 | 4,815 |
| 6 | Freeze | 0.81 ± 0.05 | 5,112 | 0.98 ± 0.06 | 4,988 |
| 7 | Heal All | 0.10 ± 0.01 | 5,423 | 0.11 ± 0.01 | 5,426 |
| 8 | Retaliate | 0.12 ± 0.02 | 5,569 | 0.11 ± 0.01 | 5,554 |
| 9 | Nimble | 0.32 ± 0.03 | 5,253 | 0.39 ± 0.03 | 5,279 |
| 10 | Twinstrike | 0.26 ± 0.04 | 5,311 | 0.34 ± 0.05 | 5,244 |
| J | Taunt | 0.11 ± 0.02 | 5,421 | 0.11 ± 0.02 | 5,409 |
| Q | Move | 0.28 ± 0.03 | 5,029 | 0.21 ± 0.02 | 5,115 |
| K | Empower | 0.82 ± 0.06 | 5,102 | 0.73 ± 0.05 | 5,121 |

*Turns face-down before being flipped, by rank.*

| rank | power | lane-gen032@256 | gen019@256 |
|---|---|---:|---:|
| A | Action | 0.005 ± 0.002 | 0.005 ± 0.002 |
| 2 | Blast | 0.153 ± 0.010 | 0.189 ± 0.011 |
| 3 | Trap | 0.070 ± 0.007 | 0.047 ± 0.006 |
| 4 | Bomb | 0.135 ± 0.010 | 0.143 ± 0.010 |
| 5 | Flip | 0.050 ± 0.006 | 0.068 ± 0.007 |
| 6 | Freeze | 0.075 ± 0.007 | 0.092 ± 0.008 |
| 7 | Heal All | 0.006 ± 0.002 | 0.008 ± 0.002 |
| 8 | Retaliate | 0.005 ± 0.002 | 0.006 ± 0.003 |
| 9 | Nimble | 0.032 ± 0.005 | 0.031 ± 0.005 |
| 10 | Twinstrike | 0.018 ± 0.004 | 0.029 ± 0.005 |
| J | Taunt | 0.004 ± 0.002 | 0.004 ± 0.002 |
| Q | Move | 0.035 ± 0.005 | 0.020 ± 0.004 |
| K | Empower | 0.063 ± 0.007 | 0.057 ± 0.007 |

*Share of played cards of each rank never turned face-up.*

| rank | power | lane-gen032@256 | gen019@256 |
|---|---|---:|---:|
| A | Action | 0.09 ± 0.02 | 0.10 ± 0.02 |
| 2 | Blast | 1.28 ± 0.06 | 2.14 ± 0.09 |
| 3 | Trap | 2.60 ± 0.10 | 2.74 ± 0.12 |
| 4 | Bomb | 0.93 ± 0.06 | 1.19 ± 0.07 |
| 5 | Flip | 0.58 ± 0.03 | 0.68 ± 0.04 |
| 6 | Freeze | 0.94 ± 0.05 | 1.08 ± 0.06 |
| 7 | Heal All | 0.10 ± 0.01 | 0.12 ± 0.01 |
| 8 | Retaliate | 0.13 ± 0.02 | 0.12 ± 0.02 |
| 9 | Nimble | 0.37 ± 0.03 | 0.44 ± 0.04 |
| 10 | Twinstrike | 0.30 ± 0.04 | 0.40 ± 0.05 |
| J | Taunt | 0.11 ± 0.02 | 0.12 ± 0.02 |
| Q | Move | 0.33 ± 0.03 | 0.23 ± 0.03 |
| K | Empower | 0.97 ± 0.07 | 0.84 ± 0.06 |

*Turns face-down counting hidden deaths and the game's end.*

*The figure is in [`two-blast-four-bomb-256.html`](two-blast-four-bomb-256.html#turns-spent-face-down).*

## Hand size at the unlock

`FINDINGS.md` H2: every card in hand after the piles empty is a turn the opponent cannot close a lane. The score column is the **larger-hand side's**, over the games where the two hands differed — one observation per game, not per player, so a game cannot vote twice. 0.500 would mean holding more cards is worth nothing.

| model | mean hand at unlock | median | score of the larger hand | games | tied |
|---|---:|---:|---:|---:|---:|
| lane-gen032@256 | 6.51 ± 0.04 | 7 | 0.7266 ± 0.0231 | 1,547 | 22.6% |
| gen019@256 | 6.41 ± 0.04 | 6 | 0.7315 ± 0.0226 | 1,581 | 20.9% |

| margin | lane-gen032@256 | gen019@256 |
|---|---:|---:|
| +1 | 0.6198 ± 0.0360 | 0.6562 ± 0.0357 |
| +2 | 0.7925 ± 0.0361 | 0.7431 ± 0.0394 |
| +3 | 0.8568 ± 0.0450 | 0.8299 ± 0.0477 |
| +4 or more | 0.8595 ± 0.0630 | 0.9216 ± 0.0448 |

*Score of the side holding this many more cards at the unlock.*

*The figure is in [`two-blast-four-bomb-256.html`](two-blast-four-bomb-256.html#hand-size-at-the-unlock).*

## Win rate with each card in the opening hand

The opening hand is the one held at the start of that player's **own** first turn — the deal plus the draw that opens a turn — so both players are measured on the same number of cards. (`GameState::new` performs P0's opening draw, so 'the hand at setup' would give P0 six cards and P1 five.)

**Read the exclusive table.** It is the score of the games where you held the card and your opponent did not. The inclusive one pools in the games where both held it, and those contribute a win and a loss in symmetric pairs — they pull every rank toward 0.500 without saying anything about the card.

| rank | power | lane-gen032@256 | n | gen019@256 | n |
|---|---|---:|---:|---:|---:|
| A | Action | 0.5669 ± 0.0347 | 1,002 | 0.5554 ± 0.0330 | 1,002 |
| 2 | Blast | 0.4668 ± 0.0348 | 980 | 0.5209 ± 0.0340 | 980 |
| 3 | Trap | 0.4891 ± 0.0348 | 966 | 0.4865 ± 0.0339 | 966 |
| 4 | Bomb | 0.4360 ± 0.0343 | 968 | 0.4427 ± 0.0336 | 968 |
| 5 | Flip | 0.4792 ± 0.0353 | 960 | 0.5000 ± 0.0348 | 960 |
| 6 | Freeze | 0.4252 ± 0.0353 | 922 | 0.4496 ± 0.0344 | 922 |
| 7 | Heal All | 0.5936 ± 0.0344 | 978 | 0.5261 ± 0.0339 | 978 |
| 8 | Retaliate | 0.5489 ± 0.0338 | 1,002 | 0.5220 ± 0.0336 | 1,002 |
| 9 | Nimble | 0.5513 ± 0.0350 | 964 | 0.5135 ± 0.0345 | 964 |
| 10 | Twinstrike | 0.5044 ± 0.0367 | 912 | 0.4912 ± 0.0354 | 912 |
| J | Taunt | 0.5379 ± 0.0353 | 962 | 0.5047 ± 0.0342 | 962 |
| Q | Move | 0.4813 ± 0.0343 | 1,016 | 0.5094 ± 0.0340 | 1,016 |
| K | Empower | 0.5005 ± 0.0349 | 1,002 | 0.4775 ± 0.0343 | 1,002 |

*Score when you hold this rank and the opponent does not.*

| rank | power | lane-gen032@256 | gen019@256 |
|---|---|---:|---:|
| A | Action | 0.5418 ± 0.0219 | 0.5346 ± 0.0208 |
| 2 | Blast | 0.4802 ± 0.0209 | 0.5125 ± 0.0203 |
| 3 | Trap | 0.4938 ± 0.0199 | 0.4923 ± 0.0194 |
| 4 | Bomb | 0.4640 ± 0.0195 | 0.4677 ± 0.0191 |
| 5 | Flip | 0.4881 ± 0.0202 | 0.5000 ± 0.0199 |
| 6 | Freeze | 0.4580 ± 0.0201 | 0.4717 ± 0.0195 |
| 7 | Heal All | 0.5523 ± 0.0196 | 0.5146 ± 0.0190 |
| 8 | Retaliate | 0.5293 ± 0.0204 | 0.5132 ± 0.0202 |
| 9 | Nimble | 0.5320 ± 0.0219 | 0.5084 ± 0.0215 |
| 10 | Twinstrike | 0.5024 ± 0.0197 | 0.4953 ± 0.0191 |
| J | Taunt | 0.5227 ± 0.0212 | 0.5028 ± 0.0204 |
| Q | Move | 0.4887 ± 0.0207 | 0.5056 ± 0.0205 |
| K | Empower | 0.5003 ± 0.0211 | 0.4864 ± 0.0208 |

*Inclusive: score whenever you hold at least one, whatever the opponent holds. Kept for contrast.*

*The figure is in [`two-blast-four-bomb-256.html`](two-blast-four-bomb-256.html#win-rate-with-each-card-in-the-opening-hand).*

## Win rate with each card in hand at the unlock

The same two estimators, on the hand held when the last pile emptied. Restricted to games that reached the unlock.

The second table is the one to trust. Holding a particular rank at the unlock is partly just holding *more cards*, and the section above shows that is worth something on its own; **adjusted** subtracts the mean score at the same hand size, leaving what is associated with the card rather than with the size of the hand it sits in. It is a difference from 0, not a win rate: `+0.02` is two points of win probability above an average hand of that size.

| rank | power | lane-gen032@256 | n | gen019@256 | n |
|---|---|---:|---:|---:|---:|
| A | Action | 0.5085 ± 0.0428 | 585 | 0.5383 ± 0.0417 | 627 |
| 2 | Blast | 0.5917 ± 0.0326 | 1,003 | 0.5732 ± 0.0317 | 1,025 |
| 3 | Trap | 0.5803 ± 0.0320 | 1,028 | 0.5694 ± 0.0327 | 929 |
| 4 | Bomb | 0.5553 ± 0.0367 | 814 | 0.5747 ± 0.0327 | 1,018 |
| 5 | Flip | 0.6566 ± 0.0302 | 1,057 | 0.6435 ± 0.0298 | 1,042 |
| 6 | Freeze | 0.6349 ± 0.0323 | 975 | 0.6106 ± 0.0348 | 836 |
| 7 | Heal All | 0.4493 ± 0.0374 | 789 | 0.4962 ± 0.0371 | 781 |
| 8 | Retaliate | 0.5570 ± 0.0497 | 412 | 0.5736 ± 0.0507 | 401 |
| 9 | Nimble | 0.5517 ± 0.0387 | 744 | 0.5855 ± 0.0389 | 678 |
| 10 | Twinstrike | 0.5082 ± 0.0319 | 1,039 | 0.5490 ± 0.0316 | 1,001 |
| J | Taunt | 0.5526 ± 0.0643 | 266 | 0.5719 ± 0.0634 | 278 |
| Q | Move | 0.5433 ± 0.0448 | 543 | 0.5422 ± 0.0436 | 569 |
| K | Empower | 0.5482 ± 0.0323 | 1,017 | 0.4857 ± 0.0319 | 1,051 |

*Score when you hold this rank at the unlock and the opponent does not.*

| rank | power | lane-gen032@256 | gen019@256 |
|---|---|---:|---:|
| A | Action | -0.0166 ± 0.0386 | +0.0278 ± 0.0377 |
| 2 | Blast | +0.0447 ± 0.0309 | +0.0102 ± 0.0302 |
| 3 | Trap | +0.0205 ± 0.0297 | +0.0049 ± 0.0308 |
| 4 | Bomb | +0.0233 ± 0.0347 | +0.0165 ± 0.0312 |
| 5 | Flip | +0.0723 ± 0.0294 | +0.0669 ± 0.0282 |
| 6 | Freeze | +0.0929 ± 0.0304 | +0.0705 ± 0.0331 |
| 7 | Heal All | -0.0427 ± 0.0356 | +0.0010 ± 0.0352 |
| 8 | Retaliate | -0.0005 ± 0.0465 | +0.0179 ± 0.0465 |
| 9 | Nimble | +0.0049 ± 0.0352 | +0.0344 ± 0.0360 |
| 10 | Twinstrike | -0.0163 ± 0.0294 | +0.0145 ± 0.0294 |
| J | Taunt | +0.0646 ± 0.0613 | +0.0779 ± 0.0569 |
| Q | Move | +0.0881 ± 0.0410 | +0.0783 ± 0.0405 |
| K | Empower | +0.0250 ± 0.0301 | -0.0029 ± 0.0302 |

*The same, minus the mean score at that hand size.*

*The figure is in [`two-blast-four-bomb-256.html`](two-blast-four-bomb-256.html#win-rate-with-each-card-in-hand-at-the-unlock).*

## Pairs

A pair is two face-up same-rank cards on one side of one lane, declared with an action (§5). Rates are **per player-game**, so 'pairs per game' is what one player declares in one game; the game as a whole sees twice that.

| model | pairs declared per player-game | per game (both sides) | player-games with a pair | cards that were ever paired |
|---|---:|---:|---:|---:|
| lane-gen032@256 | 0.058 ± 0.008 | 0.12 | 5.6% | 0.6% |
| gen019@256 | 0.059 ± 0.008 | 0.12 | 5.7% | 0.6% |

| rank | power | lane-gen032@256 | n | gen019@256 | n |
|---|---|---:|---:|---:|---:|
| A | Action | 0.002 ± 0.002 | 6,314 | 0.003 ± 0.002 | 6,330 |
| 2 | Blast | 0.002 ± 0.001 | 6,300 | 0.002 ± 0.002 | 6,436 |
| 3 | Trap | 0.006 ± 0.003 | 6,339 | 0.009 ± 0.003 | 6,399 |
| 4 | Bomb | 0.008 ± 0.003 | 5,941 | 0.003 ± 0.002 | 6,135 |
| 5 | Flip | 0.005 ± 0.003 | 6,149 | 0.002 ± 0.002 | 6,131 |
| 6 | Freeze | 0.002 ± 0.001 | 6,436 | 0.006 ± 0.003 | 6,407 |
| 7 | Heal All | 0.002 ± 0.002 | 6,360 | 0.003 ± 0.002 | 6,373 |
| 8 | Retaliate | 0.028 ± 0.007 | 6,483 | 0.022 ± 0.006 | 6,478 |
| 9 | Nimble | 0.001 ± 0.001 | 6,315 | 0.001 ± 0.001 | 6,336 |
| 10 | Twinstrike | 0.001 ± 0.001 | 6,324 | 0.004 ± 0.002 | 6,317 |
| J | Taunt | 0.001 ± 0.001 | 6,448 | 0.002 ± 0.001 | 6,438 |
| Q | Move | 0.013 ± 0.004 | 6,150 | 0.013 ± 0.004 | 6,158 |
| K | Empower | 0.004 ± 0.002 | 6,355 | 0.005 ± 0.003 | 6,336 |

*Share of the cards of each rank that entered play and were ever a member of a declared pair. The rate that is comparable across ranks.*

| rank | power | lane-gen032@256 | gen019@256 |
|---|---|---:|---:|
| A | Action | 0.002 ± 0.001 | 0.002 ± 0.001 |
| 2 | Blast | 0.001 ± 0.001 | 0.002 ± 0.001 |
| 3 | Trap | 0.005 ± 0.002 | 0.007 ± 0.003 |
| 4 | Bomb | 0.006 ± 0.002 | 0.002 ± 0.001 |
| 5 | Flip | 0.004 ± 0.002 | 0.002 ± 0.001 |
| 6 | Freeze | 0.002 ± 0.001 | 0.004 ± 0.002 |
| 7 | Heal All | 0.002 ± 0.001 | 0.003 ± 0.002 |
| 8 | Retaliate | 0.023 ± 0.005 | 0.018 ± 0.005 |
| 9 | Nimble | 0.001 ± 0.001 | 0.001 ± 0.001 |
| 10 | Twinstrike | 0.001 ± 0.001 | 0.003 ± 0.002 |
| J | Taunt | 0.001 ± 0.001 | 0.002 ± 0.001 |
| Q | Move | 0.010 ± 0.003 | 0.010 ± 0.003 |
| K | Empower | 0.003 ± 0.002 | 0.004 ± 0.002 |

*Pairs of each rank declared per player-game. Depends on how often the rank is drawn as well as on how pairable it is.*

*The figure is in [`two-blast-four-bomb-256.html`](two-blast-four-bomb-256.html#pairs).*

## How cards die: face-up or face-down

Every card that entered play and was killed, split by which side it was showing when it died. A face-down card is a blank 2-HP card whatever its rank (§5), so dying face-down means its power never did anything — the flip that would have paid for it never happened.

One group of ranks cannot die face-down at all: a face-down card with a death trigger springs face-up instead of dying (§6), so it is either killed face-up later or not killed at all. In these corpora that is **3**, which is why they read 1.000 below.

| model | deaths per game | share of cards that die | died face-up | died face-down |
|---|---:|---:|---:|---:|
| lane-gen032@256 | 33.24 | 81.2% | 0.9516 ± 0.0018 | 0.0484 |
| gen019@256 | 33.18 | 80.7% | 0.9423 ± 0.0020 | 0.0577 |

| rank | power | lane-gen032@256 | n | gen019@256 | n |
|---|---|---:|---:|---:|---:|
| A | Action | 0.977 ± 0.004 | 5,694 | 0.979 ± 0.004 | 5,706 |
| 2 | Blast | 0.862 ± 0.010 | 4,930 | 0.817 ± 0.012 | 5,157 |
| 3 | Trap | 1.000 ± 0.000 | 4,603 | 1.000 ± 0.000 | 4,776 |
| 4 | Bomb | 0.895 ± 0.010 | 4,138 | 0.869 ± 0.010 | 4,655 |
| 5 | Flip | 0.939 ± 0.007 | 5,137 | 0.921 ± 0.008 | 4,501 |
| 6 | Freeze | 0.928 ± 0.007 | 5,237 | 0.910 ± 0.008 | 4,900 |
| 7 | Heal All | 0.981 ± 0.004 | 5,511 | 0.975 ± 0.004 | 5,392 |
| 8 | Retaliate | 0.978 ± 0.005 | 4,585 | 0.974 ± 0.005 | 4,831 |
| 9 | Nimble | 0.962 ± 0.005 | 5,659 | 0.961 ± 0.005 | 5,726 |
| 10 | Twinstrike | 0.970 ± 0.005 | 5,701 | 0.959 ± 0.006 | 5,545 |
| J | Taunt | 0.978 ± 0.004 | 5,853 | 0.977 ± 0.004 | 5,803 |
| Q | Move | 0.957 ± 0.006 | 4,334 | 0.958 ± 0.006 | 4,478 |
| K | Empower | 0.928 ± 0.007 | 5,097 | 0.934 ± 0.007 | 4,899 |

*Of the cards of this rank that died, the share that were face-up at the time.*

| rank | power | lane-gen032@256 | gen019@256 |
|---|---|---:|---:|
| A | Action | 0.902 ± 0.008 | 0.901 ± 0.008 |
| 2 | Blast | 0.783 ± 0.011 | 0.801 ± 0.010 |
| 3 | Trap | 0.726 ± 0.011 | 0.746 ± 0.011 |
| 4 | Bomb | 0.697 ± 0.012 | 0.759 ± 0.011 |
| 5 | Flip | 0.835 ± 0.010 | 0.734 ± 0.011 |
| 6 | Freeze | 0.814 ± 0.010 | 0.765 ± 0.011 |
| 7 | Heal All | 0.867 ± 0.009 | 0.846 ± 0.010 |
| 8 | Retaliate | 0.707 ± 0.012 | 0.746 ± 0.011 |
| 9 | Nimble | 0.896 ± 0.008 | 0.904 ± 0.008 |
| 10 | Twinstrike | 0.901 ± 0.007 | 0.878 ± 0.008 |
| J | Taunt | 0.908 ± 0.008 | 0.901 ± 0.008 |
| Q | Move | 0.705 ± 0.011 | 0.727 ± 0.012 |
| K | Empower | 0.802 ± 0.011 | 0.773 ± 0.011 |

*Share of the cards of each rank that entered play and were killed at all.*

*The figure is in [`two-blast-four-bomb-256.html`](two-blast-four-bomb-256.html#how-cards-die-face-up-or-face-down).*

## What becomes of a face-down card

Every card is played face-down, so this is the whole population: of the cards you put on the board, how many ever come up, how many are killed before they do, and how many are still hidden when the game ends. The five outcomes partition the cards played from hand — base cards are excluded, since nobody chose to play them.

| model | flipped by choice | flipped by a cascade | sprang its trap | killed face-down | face-down at the end | ever face-up |
|---|---:|---:|---:|---:|---:|---:|
| lane-gen032@256 | 0.892 | 0.045 | 0.014 | 0.029 | 0.020 | 0.950 |
| gen019@256 | 0.877 | 0.055 | 0.014 | 0.036 | 0.018 | 0.946 |

| rank | power | lane-gen032@256 | n | gen019@256 | n |
|---|---|---:|---:|---:|---:|
| A | Action | 0.995 ± 0.002 | 5,382 | 0.995 ± 0.002 | 5,398 |
| 2 | Blast | 0.847 ± 0.010 | 5,446 | 0.811 ± 0.011 | 5,582 |
| 3 | Trap | 0.930 ± 0.007 | 5,343 | 0.953 ± 0.006 | 5,403 |
| 4 | Bomb | 0.865 ± 0.010 | 5,045 | 0.857 ± 0.010 | 5,239 |
| 5 | Flip | 0.950 ± 0.006 | 5,187 | 0.932 ± 0.007 | 5,169 |
| 6 | Freeze | 0.925 ± 0.007 | 5,524 | 0.908 ± 0.008 | 5,495 |
| 7 | Heal All | 0.994 ± 0.002 | 5,456 | 0.992 ± 0.002 | 5,469 |
| 8 | Retaliate | 0.995 ± 0.002 | 5,595 | 0.994 ± 0.003 | 5,590 |
| 9 | Nimble | 0.968 ± 0.005 | 5,425 | 0.969 ± 0.005 | 5,446 |
| 10 | Twinstrike | 0.982 ± 0.004 | 5,410 | 0.971 ± 0.005 | 5,403 |
| J | Taunt | 0.996 ± 0.002 | 5,442 | 0.996 ± 0.002 | 5,432 |
| Q | Move | 0.965 ± 0.005 | 5,212 | 0.980 ± 0.004 | 5,220 |
| K | Empower | 0.937 ± 0.007 | 5,447 | 0.943 ± 0.007 | 5,428 |

*Share of each rank played from hand that was ever face-up, by any cause.*

| rank | power | lane-gen032@256 | gen019@256 |
|---|---|---:|---:|
| A | Action | 0.004 ± 0.002 | 0.004 ± 0.002 |
| 2 | Blast | 0.107 ± 0.009 | 0.149 ± 0.010 |
| 3 | Trap | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 4 | Bomb | 0.067 ± 0.007 | 0.098 ± 0.009 |
| 5 | Flip | 0.042 ± 0.006 | 0.045 ± 0.006 |
| 6 | Freeze | 0.050 ± 0.006 | 0.058 ± 0.006 |
| 7 | Heal All | 0.003 ± 0.002 | 0.007 ± 0.002 |
| 8 | Retaliate | 0.004 ± 0.002 | 0.006 ± 0.002 |
| 9 | Nimble | 0.020 ± 0.004 | 0.021 ± 0.004 |
| 10 | Twinstrike | 0.013 ± 0.003 | 0.019 ± 0.004 |
| J | Taunt | 0.004 ± 0.002 | 0.004 ± 0.002 |
| Q | Move | 0.018 ± 0.004 | 0.011 ± 0.003 |
| K | Empower | 0.049 ± 0.006 | 0.041 ± 0.006 |

*Share killed while still face-down — the power never fired.*

*The 2 figures are in [`two-blast-four-bomb-256.html`](two-blast-four-bomb-256.html#what-becomes-of-a-face-down-card).*

## What a card is worth

Two answers to the same question, by different routes, and they are worth reading against each other.

**Counterfactual** holds a real position fixed and swaps the card in hand, asking the value head what changed. It is exact about the position and only as good as that head. It is the one measurement in this document that cannot come from played games.

**Corpus-derived** fits the result on how many more of each rank you held than your opponent, so each rank is measured with the rest of the hand held fixed. It needs no network, so it works for any agent — including `random`. Both tables are **relative to an average card**, which is the counterfactual's convention, so the two are on the same scale.

The *dealt* table is the one with an identification argument behind it: the opening hand is dealt at random, so how many of a rank you were dealt is randomly assigned and its coefficient is a causal effect rather than a correlation. The *unlock* table is what you were still holding, which you chose, and it is descriptive.

| rank | power | lane-gen032@256 | gen019@256 |
|---|---|---:|---:|
| A | Action | +3.52 ± 0.14 | +2.25 ± 0.13 |
| 2 | Blast | -3.06 ± 0.13 | -1.19 ± 0.11 |
| 3 | Trap | -1.64 ± 0.16 | -2.21 ± 0.14 |
| 4 | Bomb | -3.78 ± 0.13 | +0.24 ± 0.13 |
| 5 | Flip | +1.05 ± 0.15 | +1.17 ± 0.15 |
| 6 | Freeze | -1.04 ± 0.15 | -3.81 ± 0.12 |
| 7 | Heal All | +2.01 ± 0.15 | +0.73 ± 0.13 |
| 8 | Retaliate | +0.72 ± 0.15 | +2.18 ± 0.15 |
| 9 | Nimble | -0.02 ± 0.11 | +0.73 ± 0.15 |
| 10 | Twinstrike | -1.14 ± 0.12 | -2.38 ± 0.14 |
| J | Taunt | +2.51 ± 0.20 | +4.13 ± 0.18 |
| Q | Move | +2.40 ± 0.13 | -0.64 ± 0.13 |
| K | Empower | -1.53 ± 0.11 | -1.20 ± 0.12 |

*`duel52 card-value`: holding the position fixed, what is this card worth in hand rather than an average card, in win-probability points? Each column is that checkpoint's own value head.*

| rank | power | lane-gen032@256 | gen019@256 |
|---|---|---:|---:|
| A | Action | +5.09 ± 2.89 | +4.30 ± 2.66 |
| 2 | Blast | -4.30 ± 2.74 | +1.49 ± 2.58 |
| 3 | Trap | -2.34 ± 2.75 | -1.35 ± 2.60 |
| 4 | Bomb | -5.21 ± 2.67 | -3.71 ± 2.63 |
| 5 | Flip | -1.95 ± 2.80 | +0.23 ± 2.68 |
| 6 | Freeze | -6.58 ± 2.89 | -4.61 ± 2.70 |
| 7 | Heal All | +7.41 ± 2.92 | +1.19 ± 2.68 |
| 8 | Retaliate | +3.91 ± 2.65 | +2.50 ± 2.64 |
| 9 | Nimble | +4.76 ± 2.86 | +1.28 ± 2.72 |
| 10 | Twinstrike | -0.48 ± 2.84 | -0.93 ± 2.69 |
| J | Taunt | +3.56 ± 2.85 | +1.15 ± 2.74 |
| Q | Move | -2.45 ± 2.74 | +0.23 ± 2.71 |
| K | Empower | -1.41 ± 2.63 | -1.79 ± 2.63 |

***Dealt.** A logistic fit of the result on how many more of each rank you were dealt than your opponent, relative to an average card, in win-probability points. The opening hand is dealt at random, so this is a randomised comparison rather than a correlation — it is the closest thing here to an experiment.*

| rank | power | lane-gen032@256 | gen019@256 |
|---|---|---:|---:|
| A | Action | -0.46 ± 4.17 | +3.77 ± 3.78 |
| 2 | Blast | -2.66 ± 2.88 | -3.09 ± 2.92 |
| 3 | Trap | -2.83 ± 2.55 | -4.20 ± 2.95 |
| 4 | Bomb | -1.29 ± 2.57 | -2.51 ± 2.56 |
| 5 | Flip | +1.59 ± 3.04 | +0.03 ± 2.64 |
| 6 | Freeze | +5.53 ± 2.92 | +1.61 ± 2.81 |
| 7 | Heal All | -6.59 ± 3.60 | -1.70 ± 3.44 |
| 8 | Retaliate | -0.28 ± 4.94 | +1.66 ± 5.08 |
| 9 | Nimble | -1.73 ± 3.67 | +0.01 ± 3.82 |
| 10 | Twinstrike | -1.73 ± 2.69 | -0.86 ± 2.55 |
| J | Taunt | +5.22 ± 6.79 | +4.27 ± 6.42 |
| Q | Move | +4.31 ± 2.79 | +2.46 ± 2.75 |
| K | Empower | +0.93 ± 3.22 | -1.46 ± 2.86 |

***Held at the unlock.** The same fit on the hand held when the piles emptied. Descriptive rather than randomised: you chose what to still be holding, so a card that gets kept in positions that are already won looks good for that reason.*

*The 2 figures are in [`two-blast-four-bomb-256.html`](two-blast-four-bomb-256.html#what-a-card-is-worth).*
