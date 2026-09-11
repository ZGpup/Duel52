# Duel 52 — self-play analysis: `split`

Generated 2026-09-11 00:11 UTC · corpus schema 1 · figures in [`split-64.html`](split-64.html)

- **variant** — split
- **ruleset** — canonical-2026-09/8a9767ea88977b84
- **the 2's power** — bottom
- **lanes** — 3, 2 to win
- **hand size** — 5
- **stalemate** — 20 quiet turns
- **deal seeds** — 1–1000
- **models** — gen031@64, lane-gen032@64, 32c-24h-best@64
- **games per model** — 2,000, 2,000, 2,000

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

One agent per column, each playing **itself**. Every corpus in this document was played under the ruleset named in the header, and the reader refuses to merge two that were not. A deal is played twice with the seats swapped, so games = 2 × deals.

| model | agent | games | deals | chunks | card rows | games/sec | cpu time |
|---|---|---:|---:|---:|---:|---:|---:|
| gen031@64 | netmcts:models/duel52-split-gen031.d52nn@64 | 2,000 | 1,000 | 8 | 80,422 | 8.452 | 0.07 h |
| lane-gen032@64 | netmcts:models/duel52-split-lane-gen032.d52nn@64 | 2,000 | 1,000 | 8 | 81,611 | 2.965 | 0.19 h |
| 32c-24h-best@64 | netmcts:models/duel52-32c-24h-best.d52nn@64 | 2,000 | 1,000 | 8 | 80,932 | 1.614 | 0.34 h |

## First vs second player

The score of whoever moved first, pooled over both halves of every colour-paired deal. 0.500 is no advantage. The interval is clustered on the deal, which is the unit that was randomised — both games of a pair hold the same cards.

| model | first-player score | P0 wins | P1 wins | draws | draw rate | separated from 0.500 |
|---|---:|---:|---:|---:|---:|---:|
| gen031@64 | 0.5353 ± 0.0248 | 1,067 | 926 | 7 | 0.35% | yes |
| lane-gen032@64 | 0.5385 ± 0.0242 | 1,071 | 917 | 12 | 0.60% | yes |
| 32c-24h-best@64 | 0.4730 ± 0.0245 | 941 | 1,049 | 10 | 0.50% | yes |

*The figure is in [`split-64.html`](split-64.html#first-vs-second-player).*

## Game shape

Turns here are the game's, not one player's: a game of 42 turns is 21 each. The unlock is the turn the last draw pile emptied and base cards became attackable (`game_rules.md` §3) — until then a lane cannot be won, so it divides the game in two.

| model | mean turns | median | p10 – p90 | draw rate | stalemate / mutual / cap | reached unlock | mean unlock turn |
|---|---:|---:|---:|---:|---:|---:|---:|
| gen031@64 | 47.2 | 47 | 43 – 51 | 0.35% | 0 / 7 / 0 | 100.0% | 13.0 |
| lane-gen032@64 | 46.8 | 47 | 42 – 51 | 0.60% | 0 / 12 / 0 | 100.0% | 13.0 |
| 32c-24h-best@64 | 46.1 | 47 | 40 – 51 | 0.50% | 0 / 10 / 0 | 100.0% | 13.0 |

## Average card play turn

The owner's own turn on which a card is played from hand, face-down. Base cards are excluded — they were never played. A low number is a card that goes down early, which is not the same as a card that goes face-up early.

| model | mean play turn | cards |
|---|---:|---:|
| gen031@64 | 10.948 ± 0.035 | 68,422 |
| lane-gen032@64 | 10.775 ± 0.045 | 69,611 |
| 32c-24h-best@64 | 10.482 ± 0.052 | 68,932 |

| rank | power | gen031@64 | n | lane-gen032@64 | n | 32c-24h-best@64 | n |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Action | 6.23 ± 0.15 | 5,445 | 7.62 ± 0.18 | 5,375 | 6.53 ± 0.16 | 5,401 |
| 2 | View | 9.76 ± 0.17 | 5,637 | 13.12 ± 0.20 | 5,431 | 9.61 ± 0.22 | 5,393 |
| 3 | Trap | 16.81 ± 0.21 | 4,490 | 13.07 ± 0.23 | 5,347 | 10.82 ± 0.23 | 4,894 |
| 4 | Foresight | 12.01 ± 0.20 | 5,429 | 16.38 ± 0.23 | 4,978 | 12.24 ± 0.23 | 5,433 |
| 5 | Flip | 12.88 ± 0.19 | 5,305 | 11.15 ± 0.23 | 5,147 | 10.73 ± 0.27 | 5,212 |
| 6 | Freeze | 17.69 ± 0.12 | 5,258 | 12.81 ± 0.19 | 5,514 | 18.92 ± 0.17 | 5,046 |
| 7 | Heal All | 9.72 ± 0.17 | 5,448 | 8.43 ± 0.18 | 5,452 | 8.63 ± 0.18 | 5,432 |
| 8 | Retaliate | 5.40 ± 0.15 | 5,601 | 6.17 ± 0.17 | 5,598 | 5.91 ± 0.16 | 5,539 |
| 9 | Nimble | 7.50 ± 0.18 | 5,325 | 8.24 ± 0.20 | 5,408 | 7.25 ± 0.18 | 5,340 |
| 10 | Twinstrike | 9.55 ± 0.20 | 5,209 | 11.39 ± 0.20 | 5,383 | 12.63 ± 0.24 | 5,237 |
| J | Taunt | 5.64 ± 0.15 | 5,435 | 5.87 ± 0.15 | 5,443 | 5.95 ± 0.15 | 5,421 |
| Q | Move | 18.80 ± 0.09 | 4,746 | 16.37 ± 0.16 | 5,132 | 15.44 ± 0.17 | 5,316 |
| K | Empower | 12.82 ± 0.21 | 5,094 | 10.38 ± 0.19 | 5,403 | 12.49 ± 0.22 | 5,268 |

*The figure is in [`split-64.html`](split-64.html#average-card-play-turn).*

## Average card flip turn

The turn a card goes face-up. The main table counts **only flips its owner chose** — a card turned up by a 5's cascade or by springing a 3's Trap went face-up without anyone deciding to, and averaging those in answers a different question. Base cards are tabled separately: they cannot be flipped before the unlock, so their timing is a fact about the unlock rather than about the card.

| model | mean flip turn (chosen) | flips | any cause |
|---|---:|---:|---:|
| gen031@64 | 11.14 ± 0.04 | 64,792 | 11.20 ± 0.04 |
| lane-gen032@64 | 11.02 ± 0.05 | 62,412 | 11.07 ± 0.05 |
| 32c-24h-best@64 | 10.86 ± 0.05 | 59,849 | 10.72 ± 0.06 |

| rank | power | gen031@64 | n | lane-gen032@64 | n | 32c-24h-best@64 | n |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Action | 6.40 ± 0.16 | 5,396 | 7.69 ± 0.19 | 5,314 | 6.76 ± 0.16 | 5,222 |
| 2 | View | 10.54 ± 0.18 | 5,215 | 13.97 ± 0.24 | 4,250 | 10.01 ± 0.25 | 4,482 |
| 3 | Trap | 18.75 ± 0.20 | 3,241 | 16.11 ± 0.27 | 3,226 | 14.30 ± 0.30 | 2,546 |
| 4 | Foresight | 12.48 ± 0.21 | 5,062 | 16.95 ± 0.27 | 4,007 | 13.13 ± 0.26 | 4,232 |
| 5 | Flip | 13.20 ± 0.20 | 5,064 | 11.51 ± 0.24 | 4,802 | 10.98 ± 0.28 | 4,623 |
| 6 | Freeze | 18.33 ± 0.11 | 4,887 | 13.71 ± 0.22 | 4,726 | 19.74 ± 0.16 | 4,145 |
| 7 | Heal All | 9.88 ± 0.17 | 5,370 | 8.56 ± 0.18 | 5,348 | 8.85 ± 0.19 | 5,086 |
| 8 | Retaliate | 5.56 ± 0.15 | 5,580 | 6.32 ± 0.17 | 5,531 | 6.08 ± 0.16 | 5,382 |
| 9 | Nimble | 7.56 ± 0.18 | 5,240 | 8.18 ± 0.21 | 5,067 | 7.52 ± 0.19 | 4,857 |
| 10 | Twinstrike | 9.72 ± 0.20 | 5,119 | 11.56 ± 0.21 | 5,198 | 12.89 ± 0.24 | 4,879 |
| J | Taunt | 5.78 ± 0.16 | 5,405 | 5.98 ± 0.15 | 5,343 | 6.34 ± 0.15 | 4,963 |
| Q | Move | 19.12 ± 0.09 | 4,378 | 16.65 ± 0.16 | 4,872 | 16.01 ± 0.16 | 4,827 |
| K | Empower | 13.16 ± 0.21 | 4,835 | 11.08 ± 0.20 | 4,728 | 12.86 ± 0.24 | 4,605 |

*Flips the owner chose, played cards only.*

| rank | power | gen031@64 | lane-gen032@64 | 32c-24h-best@64 |
|---|---|---:|---:|---:|
| A | Action | 6.40 ± 0.16 | 7.70 ± 0.19 | 6.75 ± 0.16 |
| 2 | View | 10.56 ± 0.17 | 13.88 ± 0.23 | 9.59 ± 0.23 |
| 3 | Trap | 17.92 ± 0.19 | 14.94 ± 0.22 | 12.96 ± 0.23 |
| 4 | Foresight | 12.48 ± 0.21 | 16.78 ± 0.26 | 12.70 ± 0.25 |
| 5 | Flip | 13.19 ± 0.20 | 11.50 ± 0.24 | 10.88 ± 0.29 |
| 6 | Freeze | 18.30 ± 0.11 | 13.48 ± 0.21 | 19.55 ± 0.17 |
| 7 | Heal All | 9.88 ± 0.17 | 8.50 ± 0.18 | 8.77 ± 0.18 |
| 8 | Retaliate | 5.56 ± 0.15 | 6.33 ± 0.17 | 6.05 ± 0.16 |
| 9 | Nimble | 7.57 ± 0.18 | 8.26 ± 0.21 | 7.51 ± 0.18 |
| 10 | Twinstrike | 9.72 ± 0.20 | 11.55 ± 0.21 | 12.81 ± 0.24 |
| J | Taunt | 5.78 ± 0.16 | 5.96 ± 0.15 | 6.10 ± 0.15 |
| Q | Move | 19.11 ± 0.09 | 16.62 ± 0.16 | 15.97 ± 0.16 |
| K | Empower | 13.16 ± 0.21 | 10.98 ± 0.19 | 12.75 ± 0.23 |

*Any cause — chosen, cascaded, or sprung.*

| rank | power | gen031@64 | lane-gen032@64 | 32c-24h-best@64 |
|---|---|---:|---:|---:|
| A | Action | 15.32 ± 0.19 | 15.97 ± 0.20 | 15.38 ± 0.18 |
| 2 | View | 15.41 ± 0.20 | 16.20 ± 0.21 | 15.29 ± 0.17 |
| 3 | Trap | 15.46 ± 0.16 | 16.40 ± 0.21 | 15.28 ± 0.15 |
| 4 | Foresight | 15.43 ± 0.18 | 15.88 ± 0.20 | 15.35 ± 0.17 |
| 5 | Flip | 15.31 ± 0.18 | 16.18 ± 0.20 | 15.31 ± 0.16 |
| 6 | Freeze | 15.22 ± 0.19 | 16.08 ± 0.21 | 15.42 ± 0.17 |
| 7 | Heal All | 14.97 ± 0.16 | 15.61 ± 0.18 | 15.10 ± 0.15 |
| 8 | Retaliate | 15.29 ± 0.18 | 15.96 ± 0.20 | 15.30 ± 0.17 |
| 9 | Nimble | 15.45 ± 0.21 | 16.14 ± 0.22 | 15.54 ± 0.18 |
| 10 | Twinstrike | 15.17 ± 0.18 | 16.07 ± 0.23 | 15.38 ± 0.17 |
| J | Taunt | 15.03 ± 0.16 | 15.75 ± 0.19 | 15.18 ± 0.15 |
| Q | Move | 15.24 ± 0.17 | 15.93 ± 0.22 | 15.42 ± 0.16 |
| K | Empower | 15.49 ± 0.20 | 16.09 ± 0.21 | 15.13 ± 0.14 |

*Base cards, any cause.*

*The figure is in [`split-64.html`](split-64.html#average-card-flip-turn).*

## Turns spent face-down

Measured in the owner's own turns, so a card flipped on the turn it was played is **0**. Three columns because one number would be a lie by omission: a rank that is flipped fast *and* killed fast has a short tenure for two different reasons.

* **among flipped** — cards that were eventually turned face-up. The decision.
* **never flipped** — the share that were not, whether killed hidden or still hidden at the end. This is the censoring, stated rather than dropped.
* **to exit** — every played card, counting a hidden death or the end of the game as the end of its tenure. How long a card actually spends hidden.

| model | mean turns face-down (among flipped) | never flipped | mean turns to exit (all cards) |
|---|---:|---:|---:|
| gen031@64 | 0.44 ± 0.01 | 3.6% | 0.49 ± 0.01 |
| lane-gen032@64 | 0.55 ± 0.01 | 5.1% | 0.63 ± 0.02 |
| 32c-24h-best@64 | 0.52 ± 0.01 | 6.0% | 0.60 ± 0.01 |

| rank | power | gen031@64 | n | lane-gen032@64 | n | 32c-24h-best@64 | n |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Action | 0.20 ± 0.02 | 5,396 | 0.10 ± 0.02 | 5,350 | 0.23 ± 0.02 | 5,336 |
| 2 | View | 0.82 ± 0.05 | 5,309 | 1.09 ± 0.06 | 4,693 | 0.39 ± 0.03 | 5,031 |
| 3 | Trap | 1.52 ± 0.09 | 4,119 | 2.25 ± 0.09 | 4,989 | 2.31 ± 0.09 | 4,745 |
| 4 | Foresight | 0.59 ± 0.04 | 5,145 | 0.81 ± 0.05 | 4,274 | 0.78 ± 0.05 | 4,747 |
| 5 | Flip | 0.48 ± 0.03 | 5,074 | 0.52 ± 0.04 | 4,875 | 0.53 ± 0.03 | 4,767 |
| 6 | Freeze | 0.59 ± 0.03 | 4,915 | 0.75 ± 0.05 | 5,078 | 0.50 ± 0.04 | 4,279 |
| 7 | Heal All | 0.27 ± 0.01 | 5,374 | 0.11 ± 0.02 | 5,426 | 0.26 ± 0.02 | 5,261 |
| 8 | Retaliate | 0.17 ± 0.01 | 5,580 | 0.16 ± 0.02 | 5,552 | 0.16 ± 0.01 | 5,492 |
| 9 | Nimble | 0.15 ± 0.02 | 5,247 | 0.30 ± 0.03 | 5,203 | 0.43 ± 0.04 | 5,131 |
| 10 | Twinstrike | 0.22 ± 0.02 | 5,123 | 0.25 ± 0.04 | 5,253 | 0.31 ± 0.04 | 4,996 |
| J | Taunt | 0.15 ± 0.01 | 5,405 | 0.10 ± 0.02 | 5,418 | 0.17 ± 0.02 | 5,383 |
| Q | Move | 0.40 ± 0.03 | 4,383 | 0.28 ± 0.03 | 4,911 | 0.49 ± 0.05 | 4,883 |
| K | Empower | 0.44 ± 0.03 | 4,870 | 0.75 ± 0.05 | 5,042 | 0.45 ± 0.03 | 4,768 |

*Turns face-down before being flipped, by rank.*

| rank | power | gen031@64 | lane-gen032@64 | 32c-24h-best@64 |
|---|---|---:|---:|---:|
| A | Action | 0.009 ± 0.003 | 0.005 ± 0.002 | 0.012 ± 0.003 |
| 2 | View | 0.058 ± 0.007 | 0.136 ± 0.010 | 0.067 ± 0.007 |
| 3 | Trap | 0.083 ± 0.008 | 0.067 ± 0.007 | 0.030 ± 0.005 |
| 4 | Foresight | 0.052 ± 0.006 | 0.141 ± 0.010 | 0.126 ± 0.010 |
| 5 | Flip | 0.044 ± 0.006 | 0.053 ± 0.006 | 0.085 ± 0.008 |
| 6 | Freeze | 0.065 ± 0.007 | 0.079 ± 0.008 | 0.152 ± 0.011 |
| 7 | Heal All | 0.014 ± 0.003 | 0.005 ± 0.002 | 0.031 ± 0.005 |
| 8 | Retaliate | 0.004 ± 0.002 | 0.008 ± 0.003 | 0.008 ± 0.003 |
| 9 | Nimble | 0.015 ± 0.003 | 0.038 ± 0.005 | 0.039 ± 0.005 |
| 10 | Twinstrike | 0.017 ± 0.003 | 0.024 ± 0.004 | 0.046 ± 0.006 |
| J | Taunt | 0.006 ± 0.002 | 0.005 ± 0.002 | 0.007 ± 0.003 |
| Q | Move | 0.076 ± 0.008 | 0.043 ± 0.006 | 0.081 ± 0.008 |
| K | Empower | 0.044 ± 0.006 | 0.067 ± 0.007 | 0.095 ± 0.009 |

*Share of played cards of each rank never turned face-up.*

| rank | power | gen031@64 | lane-gen032@64 | 32c-24h-best@64 |
|---|---|---:|---:|---:|
| A | Action | 0.20 ± 0.02 | 0.10 ± 0.02 | 0.24 ± 0.02 |
| 2 | View | 0.95 ± 0.05 | 1.23 ± 0.06 | 0.46 ± 0.03 |
| 3 | Trap | 1.56 ± 0.09 | 2.36 ± 0.10 | 2.45 ± 0.10 |
| 4 | Foresight | 0.67 ± 0.05 | 0.89 ± 0.05 | 0.90 ± 0.05 |
| 5 | Flip | 0.51 ± 0.03 | 0.55 ± 0.04 | 0.58 ± 0.03 |
| 6 | Freeze | 0.64 ± 0.03 | 0.87 ± 0.05 | 0.63 ± 0.04 |
| 7 | Heal All | 0.27 ± 0.01 | 0.11 ± 0.02 | 0.29 ± 0.02 |
| 8 | Retaliate | 0.17 ± 0.01 | 0.16 ± 0.02 | 0.17 ± 0.02 |
| 9 | Nimble | 0.17 ± 0.02 | 0.35 ± 0.03 | 0.48 ± 0.04 |
| 10 | Twinstrike | 0.23 ± 0.02 | 0.29 ± 0.04 | 0.36 ± 0.04 |
| J | Taunt | 0.15 ± 0.01 | 0.12 ± 0.02 | 0.18 ± 0.03 |
| Q | Move | 0.43 ± 0.03 | 0.34 ± 0.03 | 0.65 ± 0.06 |
| K | Empower | 0.51 ± 0.04 | 0.87 ± 0.06 | 0.54 ± 0.03 |

*Turns face-down counting hidden deaths and the game's end.*

*The figure is in [`split-64.html`](split-64.html#turns-spent-face-down).*

## Hand size at the unlock

`FINDINGS.md` H2: every card in hand after the piles empty is a turn the opponent cannot close a lane. The score column is the **larger-hand side's**, over the games where the two hands differed — one observation per game, not per player, so a game cannot vote twice. 0.500 would mean holding more cards is worth nothing.

| model | mean hand at unlock | median | score of the larger hand | games | tied |
|---|---:|---:|---:|---:|---:|
| gen031@64 | 6.96 ± 0.03 | 7 | 0.8118 ± 0.0195 | 1,634 | 18.3% |
| lane-gen032@64 | 6.66 ± 0.04 | 7 | 0.7086 ± 0.0231 | 1,582 | 20.9% |
| 32c-24h-best@64 | 6.50 ± 0.04 | 7 | 0.7508 ± 0.0226 | 1,549 | 22.6% |

| margin | gen031@64 | lane-gen032@64 | 32c-24h-best@64 |
|---|---:|---:|---:|
| +1 | 0.7025 ± 0.0359 | 0.6049 ± 0.0355 | 0.6874 ± 0.0331 |
| +2 | 0.8074 ± 0.0371 | 0.7454 ± 0.0416 | 0.7457 ± 0.0412 |
| +3 | 0.9377 ± 0.0284 | 0.8366 ± 0.0453 | 0.8575 ± 0.0473 |
| +4 or more | 0.9677 ± 0.0233 | 0.9323 ± 0.0428 | 0.9760 ± 0.0268 |

*Score of the side holding this many more cards at the unlock.*

*The figure is in [`split-64.html`](split-64.html#hand-size-at-the-unlock).*

## Win rate with each card in the opening hand

The opening hand is the one held at the start of that player's **own** first turn — the deal plus the draw that opens a turn — so both players are measured on the same number of cards. (`GameState::new` performs P0's opening draw, so 'the hand at setup' would give P0 six cards and P1 five.)

**Read the exclusive table.** It is the score of the games where you held the card and your opponent did not. The inclusive one pools in the games where both held it, and those contribute a win and a loss in symmetric pairs — they pull every rank toward 0.500 without saying anything about the card.

| rank | power | gen031@64 | n | lane-gen032@64 | n | 32c-24h-best@64 | n |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Action | 0.5519 ± 0.0353 | 1,002 | 0.5519 ± 0.0339 | 1,002 | 0.5534 ± 0.0346 | 1,002 |
| 2 | View | 0.4622 ± 0.0347 | 980 | 0.4668 ± 0.0349 | 980 | 0.5051 ± 0.0357 | 980 |
| 3 | Trap | 0.4974 ± 0.0359 | 966 | 0.4907 ± 0.0347 | 966 | 0.5285 ± 0.0352 | 966 |
| 4 | Foresight | 0.4132 ± 0.0354 | 968 | 0.4205 ± 0.0340 | 968 | 0.4194 ± 0.0339 | 968 |
| 5 | Flip | 0.4724 ± 0.0357 | 960 | 0.4958 ± 0.0358 | 960 | 0.5240 ± 0.0353 | 960 |
| 6 | Freeze | 0.4685 ± 0.0370 | 922 | 0.4555 ± 0.0358 | 922 | 0.4528 ± 0.0354 | 922 |
| 7 | Heal All | 0.5578 ± 0.0358 | 978 | 0.5823 ± 0.0341 | 978 | 0.5741 ± 0.0348 | 978 |
| 8 | Retaliate | 0.6063 ± 0.0341 | 1,002 | 0.5464 ± 0.0344 | 1,002 | 0.5339 ± 0.0351 | 1,002 |
| 9 | Nimble | 0.5638 ± 0.0354 | 964 | 0.5322 ± 0.0346 | 964 | 0.5410 ± 0.0346 | 964 |
| 10 | Twinstrike | 0.4726 ± 0.0365 | 912 | 0.4627 ± 0.0363 | 912 | 0.4825 ± 0.0367 | 912 |
| J | Taunt | 0.4875 ± 0.0360 | 962 | 0.5457 ± 0.0351 | 962 | 0.5265 ± 0.0358 | 962 |
| Q | Move | 0.5118 ± 0.0355 | 1,016 | 0.5074 ± 0.0338 | 1,016 | 0.4508 ± 0.0347 | 1,016 |
| K | Empower | 0.4960 ± 0.0355 | 1,002 | 0.4915 ± 0.0339 | 1,002 | 0.4656 ± 0.0349 | 1,002 |

*Score when you hold this rank and the opponent does not.*

| rank | power | gen031@64 | lane-gen032@64 | 32c-24h-best@64 |
|---|---|---:|---:|---:|
| A | Action | 0.5325 ± 0.0222 | 0.5325 ± 0.0213 | 0.5334 ± 0.0218 |
| 2 | View | 0.4774 ± 0.0208 | 0.4802 ± 0.0209 | 0.5030 ± 0.0213 |
| 3 | Trap | 0.4985 ± 0.0205 | 0.4947 ± 0.0198 | 0.5163 ± 0.0202 |
| 4 | Foresight | 0.4512 ± 0.0202 | 0.4552 ± 0.0194 | 0.4547 ± 0.0194 |
| 5 | Flip | 0.4842 ± 0.0204 | 0.4976 ± 0.0205 | 0.5137 ± 0.0202 |
| 6 | Freeze | 0.4823 ± 0.0208 | 0.4750 ± 0.0202 | 0.4735 ± 0.0200 |
| 7 | Heal All | 0.5323 ± 0.0201 | 0.5460 ± 0.0194 | 0.5414 ± 0.0197 |
| 8 | Retaliate | 0.5638 ± 0.0209 | 0.5278 ± 0.0207 | 0.5204 ± 0.0211 |
| 9 | Nimble | 0.5397 ± 0.0222 | 0.5200 ± 0.0216 | 0.5255 ± 0.0216 |
| 10 | Twinstrike | 0.4853 ± 0.0196 | 0.4800 ± 0.0196 | 0.4906 ± 0.0197 |
| J | Taunt | 0.4925 ± 0.0215 | 0.5273 ± 0.0211 | 0.5158 ± 0.0214 |
| Q | Move | 0.5071 ± 0.0214 | 0.5045 ± 0.0204 | 0.4703 ± 0.0210 |
| K | Empower | 0.4976 ± 0.0214 | 0.4949 ± 0.0205 | 0.4792 ± 0.0211 |

*Inclusive: score whenever you hold at least one, whatever the opponent holds. Kept for contrast.*

*The figure is in [`split-64.html`](split-64.html#win-rate-with-each-card-in-the-opening-hand).*

## Win rate with each card in hand at the unlock

The same two estimators, on the hand held when the last pile emptied. Restricted to games that reached the unlock.

The second table is the one to trust. Holding a particular rank at the unlock is partly just holding *more cards*, and the section above shows that is worth something on its own; **adjusted** subtracts the mean score at the same hand size, leaving what is associated with the card rather than with the size of the hand it sits in. It is a difference from 0, not a win rate: `+0.02` is two points of win probability above an average hand of that size.

| rank | power | gen031@64 | n | lane-gen032@64 | n | 32c-24h-best@64 | n |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Action | 0.6494 ± 0.0649 | 231 | 0.5415 ± 0.0436 | 603 | 0.5892 ± 0.0578 | 314 |
| 2 | View | 0.5787 ± 0.0377 | 775 | 0.5682 ± 0.0333 | 968 | 0.6463 ± 0.0316 | 1,029 |
| 3 | Trap | 0.5448 ± 0.0424 | 625 | 0.5762 ± 0.0315 | 1,017 | 0.6369 ± 0.0312 | 1,001 |
| 4 | Foresight | 0.6727 ± 0.0296 | 1,132 | 0.5120 ± 0.0390 | 747 | 0.6181 ± 0.0313 | 1,071 |
| 5 | Flip | 0.6606 ± 0.0306 | 1,024 | 0.6526 ± 0.0305 | 1,029 | 0.6500 ± 0.0315 | 990 |
| 6 | Freeze | 0.5756 ± 0.0470 | 516 | 0.6314 ± 0.0320 | 963 | 0.4604 ± 0.0469 | 492 |
| 7 | Heal All | 0.4631 ± 0.0345 | 908 | 0.4684 ± 0.0378 | 775 | 0.4509 ± 0.0356 | 835 |
| 8 | Retaliate | 0.6615 ± 0.0770 | 161 | 0.5896 ± 0.0578 | 335 | 0.7214 ± 0.0560 | 280 |
| 9 | Nimble | 0.7511 ± 0.0335 | 701 | 0.5727 ± 0.0360 | 805 | 0.6030 ± 0.0400 | 636 |
| 10 | Twinstrike | 0.7870 ± 0.0259 | 1,115 | 0.5461 ± 0.0316 | 1,064 | 0.5155 ± 0.0333 | 998 |
| J | Taunt | 0.7633 ± 0.0721 | 169 | 0.6042 ± 0.0744 | 192 | 0.6919 ± 0.0697 | 198 |
| Q | Move | 0.5804 ± 0.0607 | 317 | 0.5911 ± 0.0454 | 505 | 0.5270 ± 0.0428 | 629 |
| K | Empower | 0.5786 ± 0.0341 | 910 | 0.5318 ± 0.0319 | 1,054 | 0.4790 ± 0.0329 | 998 |

*Score when you hold this rank at the unlock and the opponent does not.*

| rank | power | gen031@64 | lane-gen032@64 | 32c-24h-best@64 |
|---|---|---:|---:|---:|
| A | Action | -0.0132 ± 0.0539 | -0.0070 ± 0.0388 | +0.0132 ± 0.0509 |
| 2 | View | -0.0180 ± 0.0314 | +0.0235 ± 0.0313 | +0.0691 ± 0.0298 |
| 3 | Trap | -0.0109 ± 0.0352 | +0.0153 ± 0.0295 | +0.0388 ± 0.0297 |
| 4 | Foresight | +0.0365 ± 0.0256 | -0.0255 ± 0.0359 | +0.0512 ± 0.0299 |
| 5 | Flip | +0.0229 ± 0.0255 | +0.0671 ± 0.0295 | +0.0677 ± 0.0297 |
| 6 | Freeze | +0.0410 ± 0.0408 | +0.0831 ± 0.0300 | -0.0466 ± 0.0429 |
| 7 | Heal All | -0.0358 ± 0.0270 | -0.0261 ± 0.0349 | -0.0307 ± 0.0322 |
| 8 | Retaliate | +0.0234 ± 0.0596 | +0.0137 ± 0.0516 | +0.1022 ± 0.0513 |
| 9 | Nimble | +0.0741 ± 0.0294 | +0.0221 ± 0.0329 | +0.0420 ± 0.0375 |
| 10 | Twinstrike | +0.1174 ± 0.0243 | +0.0094 ± 0.0296 | -0.0064 ± 0.0303 |
| J | Taunt | +0.0980 ± 0.0594 | +0.1017 ± 0.0684 | +0.1211 ± 0.0632 |
| Q | Move | +0.1084 ± 0.0532 | +0.1315 ± 0.0417 | +0.0600 ± 0.0399 |
| K | Empower | +0.0244 ± 0.0293 | +0.0098 ± 0.0298 | -0.0245 ± 0.0303 |

*The same, minus the mean score at that hand size.*

*The figure is in [`split-64.html`](split-64.html#win-rate-with-each-card-in-hand-at-the-unlock).*

## Pairs

A pair is two face-up same-rank cards on one side of one lane, declared with an action (§5). Rates are **per player-game**, so 'pairs per game' is what one player declares in one game; the game as a whole sees twice that.

| model | pairs declared per player-game | per game (both sides) | player-games with a pair | cards that were ever paired |
|---|---:|---:|---:|---:|
| gen031@64 | 0.025 ± 0.005 | 0.05 | 2.4% | 0.2% |
| lane-gen032@64 | 0.051 ± 0.007 | 0.10 | 4.9% | 0.5% |
| 32c-24h-best@64 | 0.091 ± 0.010 | 0.18 | 8.7% | 0.9% |

| rank | power | gen031@64 | n | lane-gen032@64 | n | 32c-24h-best@64 | n |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Action | 0.001 ± 0.001 | 6,377 | 0.003 ± 0.002 | 6,307 | 0.003 ± 0.002 | 6,333 |
| 2 | View | 0.002 ± 0.001 | 6,491 | 0.002 ± 0.001 | 6,285 | 0.002 ± 0.001 | 6,247 |
| 3 | Trap | 0.005 ± 0.003 | 5,486 | 0.003 ± 0.002 | 6,343 | 0.007 ± 0.003 | 5,890 |
| 4 | Foresight | 0.001 ± 0.001 | 6,325 | 0.004 ± 0.002 | 5,874 | 0.003 ± 0.002 | 6,329 |
| 5 | Flip | 0.000 ± 0.001 | 6,267 | 0.002 ± 0.002 | 6,109 | 0.007 ± 0.003 | 6,174 |
| 6 | Freeze | 0.005 ± 0.003 | 6,170 | 0.001 ± 0.001 | 6,426 | 0.027 ± 0.006 | 5,958 |
| 7 | Heal All | 0.001 ± 0.001 | 6,352 | 0.003 ± 0.002 | 6,356 | 0.007 ± 0.003 | 6,336 |
| 8 | Retaliate | 0.005 ± 0.002 | 6,489 | 0.023 ± 0.006 | 6,486 | 0.027 ± 0.006 | 6,427 |
| 9 | Nimble | 0.001 ± 0.001 | 6,215 | 0.000 ± 0.001 | 6,298 | 0.002 ± 0.002 | 6,230 |
| 10 | Twinstrike | 0.000 ± 0.001 | 6,123 | 0.001 ± 0.001 | 6,297 | 0.003 ± 0.002 | 6,151 |
| J | Taunt | 0.002 ± 0.001 | 6,441 | 0.001 ± 0.001 | 6,449 | 0.002 ± 0.002 | 6,427 |
| Q | Move | 0.010 ± 0.004 | 5,684 | 0.017 ± 0.005 | 6,070 | 0.018 ± 0.005 | 6,254 |
| K | Empower | 0.000 ± 0.001 | 6,002 | 0.003 ± 0.002 | 6,311 | 0.009 ± 0.003 | 6,176 |

*Share of the cards of each rank that entered play and were ever a member of a declared pair. The rate that is comparable across ranks.*

| rank | power | gen031@64 | lane-gen032@64 | 32c-24h-best@64 |
|---|---|---:|---:|---:|
| A | Action | 0.001 ± 0.001 | 0.003 ± 0.002 | 0.002 ± 0.001 |
| 2 | View | 0.001 ± 0.001 | 0.001 ± 0.001 | 0.001 ± 0.001 |
| 3 | Trap | 0.004 ± 0.002 | 0.003 ± 0.002 | 0.005 ± 0.002 |
| 4 | Foresight | 0.001 ± 0.001 | 0.003 ± 0.002 | 0.003 ± 0.002 |
| 5 | Flip | 0.000 ± 0.000 | 0.002 ± 0.001 | 0.005 ± 0.002 |
| 6 | Freeze | 0.004 ± 0.002 | 0.001 ± 0.001 | 0.020 ± 0.004 |
| 7 | Heal All | 0.001 ± 0.001 | 0.003 ± 0.002 | 0.006 ± 0.003 |
| 8 | Retaliate | 0.004 ± 0.002 | 0.018 ± 0.004 | 0.022 ± 0.005 |
| 9 | Nimble | 0.001 ± 0.001 | 0.000 ± 0.000 | 0.002 ± 0.001 |
| 10 | Twinstrike | 0.000 ± 0.000 | 0.001 ± 0.001 | 0.002 ± 0.001 |
| J | Taunt | 0.001 ± 0.001 | 0.001 ± 0.001 | 0.002 ± 0.001 |
| Q | Move | 0.007 ± 0.003 | 0.013 ± 0.004 | 0.014 ± 0.004 |
| K | Empower | 0.000 ± 0.000 | 0.003 ± 0.002 | 0.007 ± 0.003 |

*Pairs of each rank declared per player-game. Depends on how often the rank is drawn as well as on how pairable it is.*

*The figure is in [`split-64.html`](split-64.html#pairs).*

## How cards die: face-up or face-down

Every card that entered play and was killed, split by which side it was showing when it died. A face-down card is a blank 2-HP card whatever its rank (§5), so dying face-down means its power never did anything — the flip that would have paid for it never happened.

One group of ranks cannot die face-down at all: a face-down card with a death trigger springs face-up instead of dying (§6), so it is either killed face-up later or not killed at all. In these corpora that is **3**, which is why they read 1.000 below.

| model | deaths per game | share of cards that die | died face-up | died face-down |
|---|---:|---:|---:|---:|
| gen031@64 | 33.99 | 84.5% | 0.9424 ± 0.0018 | 0.0576 |
| lane-gen032@64 | 33.00 | 80.9% | 0.9502 ± 0.0019 | 0.0498 |
| 32c-24h-best@64 | 32.04 | 79.2% | 0.9333 ± 0.0025 | 0.0667 |

| rank | power | gen031@64 | n | lane-gen032@64 | n | 32c-24h-best@64 | n |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Action | 0.955 ± 0.006 | 5,922 | 0.981 ± 0.004 | 5,638 | 0.966 ± 0.005 | 5,597 |
| 2 | View | 0.916 ± 0.008 | 5,680 | 0.878 ± 0.010 | 4,939 | 0.926 ± 0.008 | 5,084 |
| 3 | Trap | 1.000 ± 0.000 | 4,079 | 1.000 ± 0.000 | 4,659 | 1.000 ± 0.000 | 4,555 |
| 4 | Foresight | 0.923 ± 0.008 | 5,304 | 0.896 ± 0.010 | 4,023 | 0.880 ± 0.010 | 4,885 |
| 5 | Flip | 0.929 ± 0.007 | 5,140 | 0.933 ± 0.007 | 5,041 | 0.897 ± 0.009 | 4,660 |
| 6 | Freeze | 0.912 ± 0.009 | 4,774 | 0.922 ± 0.008 | 5,291 | 0.881 ± 0.011 | 3,847 |
| 7 | Heal All | 0.961 ± 0.005 | 5,616 | 0.983 ± 0.004 | 5,549 | 0.947 ± 0.007 | 5,169 |
| 8 | Retaliate | 0.961 ± 0.006 | 5,079 | 0.972 ± 0.005 | 4,401 | 0.964 ± 0.006 | 4,852 |
| 9 | Nimble | 0.954 ± 0.006 | 5,793 | 0.956 ± 0.006 | 5,635 | 0.943 ± 0.007 | 5,555 |
| 10 | Twinstrike | 0.949 ± 0.006 | 5,701 | 0.966 ± 0.005 | 5,605 | 0.937 ± 0.007 | 5,319 |
| J | Taunt | 0.957 ± 0.006 | 5,876 | 0.978 ± 0.004 | 5,844 | 0.969 ± 0.005 | 5,713 |
| Q | Move | 0.903 ± 0.010 | 4,029 | 0.949 ± 0.007 | 4,216 | 0.914 ± 0.009 | 4,273 |
| K | Empower | 0.926 ± 0.008 | 4,996 | 0.922 ± 0.008 | 5,154 | 0.881 ± 0.010 | 4,579 |

*Of the cards of this rank that died, the share that were face-up at the time.*

| rank | power | gen031@64 | lane-gen032@64 | 32c-24h-best@64 |
|---|---|---:|---:|---:|
| A | Action | 0.929 ± 0.007 | 0.894 ± 0.008 | 0.884 ± 0.008 |
| 2 | View | 0.875 ± 0.008 | 0.786 ± 0.010 | 0.814 ± 0.011 |
| 3 | Trap | 0.744 ± 0.011 | 0.735 ± 0.011 | 0.773 ± 0.011 |
| 4 | Foresight | 0.839 ± 0.008 | 0.685 ± 0.011 | 0.772 ± 0.010 |
| 5 | Flip | 0.820 ± 0.009 | 0.825 ± 0.010 | 0.755 ± 0.011 |
| 6 | Freeze | 0.774 ± 0.010 | 0.823 ± 0.009 | 0.646 ± 0.013 |
| 7 | Heal All | 0.884 ± 0.008 | 0.873 ± 0.009 | 0.816 ± 0.010 |
| 8 | Retaliate | 0.783 ± 0.011 | 0.679 ± 0.012 | 0.755 ± 0.011 |
| 9 | Nimble | 0.932 ± 0.007 | 0.895 ± 0.008 | 0.892 ± 0.008 |
| 10 | Twinstrike | 0.931 ± 0.006 | 0.890 ± 0.008 | 0.865 ± 0.009 |
| J | Taunt | 0.912 ± 0.008 | 0.906 ± 0.008 | 0.889 ± 0.008 |
| Q | Move | 0.709 ± 0.012 | 0.695 ± 0.013 | 0.683 ± 0.012 |
| K | Empower | 0.832 ± 0.010 | 0.817 ± 0.010 | 0.741 ± 0.012 |

*Share of the cards of each rank that entered play and were killed at all.*

*The figure is in [`split-64.html`](split-64.html#how-cards-die-face-up-or-face-down).*

## What becomes of a face-down card

Every card is played face-down, so this is the whole population: of the cards you put on the board, how many ever come up, how many are killed before they do, and how many are still hidden when the game ends. The five outcomes partition the cards played from hand — base cards are excluded, since nobody chose to play them.

| model | flipped by choice | flipped by a cascade | sprang its trap | killed face-down | face-down at the end | ever face-up |
|---|---:|---:|---:|---:|---:|---:|
| gen031@64 | 0.947 | 0.008 | 0.009 | 0.022 | 0.015 | 0.964 |
| lane-gen032@64 | 0.897 | 0.040 | 0.013 | 0.030 | 0.021 | 0.949 |
| 32c-24h-best@64 | 0.868 | 0.050 | 0.022 | 0.037 | 0.022 | 0.940 |

| rank | power | gen031@64 | n | lane-gen032@64 | n | 32c-24h-best@64 | n |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Action | 0.991 ± 0.003 | 5,445 | 0.995 ± 0.002 | 5,375 | 0.988 ± 0.003 | 5,401 |
| 2 | View | 0.942 ± 0.007 | 5,637 | 0.864 ± 0.010 | 5,431 | 0.933 ± 0.007 | 5,393 |
| 3 | Trap | 0.917 ± 0.008 | 4,490 | 0.933 ± 0.007 | 5,347 | 0.970 ± 0.005 | 4,894 |
| 4 | Foresight | 0.948 ± 0.006 | 5,429 | 0.859 ± 0.010 | 4,978 | 0.874 ± 0.010 | 5,433 |
| 5 | Flip | 0.956 ± 0.006 | 5,305 | 0.947 ± 0.006 | 5,147 | 0.915 ± 0.008 | 5,212 |
| 6 | Freeze | 0.935 ± 0.007 | 5,258 | 0.921 ± 0.008 | 5,514 | 0.848 ± 0.011 | 5,046 |
| 7 | Heal All | 0.986 ± 0.003 | 5,448 | 0.995 ± 0.002 | 5,452 | 0.969 ± 0.005 | 5,432 |
| 8 | Retaliate | 0.996 ± 0.002 | 5,601 | 0.992 ± 0.003 | 5,598 | 0.992 ± 0.003 | 5,539 |
| 9 | Nimble | 0.985 ± 0.003 | 5,325 | 0.962 ± 0.005 | 5,408 | 0.961 ± 0.005 | 5,340 |
| 10 | Twinstrike | 0.983 ± 0.003 | 5,209 | 0.976 ± 0.004 | 5,383 | 0.954 ± 0.006 | 5,237 |
| J | Taunt | 0.994 ± 0.002 | 5,435 | 0.995 ± 0.002 | 5,443 | 0.993 ± 0.003 | 5,421 |
| Q | Move | 0.924 ± 0.008 | 4,746 | 0.957 ± 0.006 | 5,132 | 0.919 ± 0.008 | 5,316 |
| K | Empower | 0.956 ± 0.006 | 5,094 | 0.933 ± 0.007 | 5,403 | 0.905 ± 0.009 | 5,268 |

*Share of each rank played from hand that was ever face-up, by any cause.*

| rank | power | gen031@64 | lane-gen032@64 | 32c-24h-best@64 |
|---|---|---:|---:|---:|
| A | Action | 0.009 ± 0.003 | 0.004 ± 0.002 | 0.012 ± 0.003 |
| 2 | View | 0.051 ± 0.006 | 0.091 ± 0.008 | 0.044 ± 0.006 |
| 3 | Trap | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 4 | Foresight | 0.037 ± 0.005 | 0.062 ± 0.007 | 0.084 ± 0.008 |
| 5 | Flip | 0.025 ± 0.004 | 0.045 ± 0.006 | 0.063 ± 0.007 |
| 6 | Freeze | 0.044 ± 0.006 | 0.055 ± 0.006 | 0.060 ± 0.007 |
| 7 | Heal All | 0.009 ± 0.002 | 0.003 ± 0.002 | 0.027 ± 0.005 |
| 8 | Retaliate | 0.004 ± 0.002 | 0.008 ± 0.003 | 0.008 ± 0.003 |
| 9 | Nimble | 0.011 ± 0.003 | 0.025 ± 0.004 | 0.035 ± 0.005 |
| 10 | Twinstrike | 0.014 ± 0.003 | 0.017 ± 0.004 | 0.032 ± 0.005 |
| J | Taunt | 0.006 ± 0.002 | 0.004 ± 0.002 | 0.006 ± 0.002 |
| Q | Move | 0.040 ± 0.006 | 0.021 ± 0.004 | 0.042 ± 0.006 |
| K | Empower | 0.032 ± 0.005 | 0.053 ± 0.006 | 0.074 ± 0.008 |

*Share killed while still face-down — the power never fired.*

*The 3 figures are in [`split-64.html`](split-64.html#what-becomes-of-a-face-down-card).*

## What a card is worth

Two answers to the same question, by different routes, and they are worth reading against each other.

**Counterfactual** holds a real position fixed and swaps the card in hand, asking the value head what changed. It is exact about the position and only as good as that head. It is the one measurement in this document that cannot come from played games.

**Corpus-derived** fits the result on how many more of each rank you held than your opponent, so each rank is measured with the rest of the hand held fixed. It needs no network, so it works for any agent — including `random`. Both tables are **relative to an average card**, which is the counterfactual's convention, so the two are on the same scale.

The *dealt* table is the one with an identification argument behind it: the opening hand is dealt at random, so how many of a rank you were dealt is randomly assigned and its coefficient is a causal effect rather than a correlation. The *unlock* table is what you were still holding, which you chose, and it is descriptive.

> The counterfactual table is not available for these corpora — no agent carries a checkpoint, the engine binary was not found, or `card-value` refused the ruleset (it needs a rank to be plausibly hidden, which `mirrored` almost never allows). It is the only measurement here that cannot come from played games.

| rank | power | gen031@64 | lane-gen032@64 | 32c-24h-best@64 |
|---|---|---:|---:|---:|
| A | Action | +4.18 ± 2.89 | +4.28 ± 2.77 | +4.72 ± 2.80 |
| 2 | View | -3.08 ± 2.72 | -3.63 ± 2.65 | -0.90 ± 2.78 |
| 3 | Trap | -1.79 ± 2.83 | -2.08 ± 2.62 | +0.69 ± 2.84 |
| 4 | Foresight | -7.01 ± 2.76 | -6.98 ± 2.68 | -6.02 ± 2.63 |
| 5 | Flip | -2.21 ± 2.78 | -0.21 ± 2.79 | +0.24 ± 2.75 |
| 6 | Freeze | -2.86 ± 3.00 | -3.75 ± 2.84 | -3.98 ± 2.83 |
| 7 | Heal All | +4.68 ± 2.82 | +5.83 ± 2.91 | +6.31 ± 2.82 |
| 8 | Retaliate | +8.62 ± 2.91 | +4.65 ± 2.72 | +3.29 ± 2.81 |
| 9 | Nimble | +4.42 ± 2.97 | +2.86 ± 2.89 | +3.90 ± 2.87 |
| 10 | Twinstrike | -2.79 ± 2.90 | -3.78 ± 2.81 | -2.69 ± 2.86 |
| J | Taunt | -2.54 ± 2.82 | +3.38 ± 2.81 | +2.18 ± 2.85 |
| Q | Move | +0.42 ± 2.93 | +0.33 ± 2.72 | -4.07 ± 2.87 |
| K | Empower | -0.05 ± 2.79 | -0.89 ± 2.63 | -3.66 ± 2.72 |

***Dealt.** A logistic fit of the result on how many more of each rank you were dealt than your opponent, relative to an average card, in win-probability points. The opening hand is dealt at random, so this is a randomised comparison rather than a correlation — it is the closest thing here to an experiment.*

| rank | power | gen031@64 | lane-gen032@64 | 32c-24h-best@64 |
|---|---|---:|---:|---:|
| A | Action | -0.05 ± 7.73 | -0.52 ± 4.13 | -0.09 ± 5.98 |
| 2 | View | -7.88 ± 4.33 | -3.30 ± 2.92 | +1.14 ± 3.33 |
| 3 | Trap | -9.15 ± 3.36 | -3.49 ± 2.53 | -1.99 ± 2.81 |
| 4 | Foresight | -2.53 ± 3.50 | -4.77 ± 2.61 | -1.97 ± 2.67 |
| 5 | Flip | -5.95 ± 3.70 | +1.67 ± 3.30 | +0.17 ± 3.13 |
| 6 | Freeze | -0.93 ± 2.85 | +4.36 ± 2.88 | -3.53 ± 2.62 |
| 7 | Heal All | -5.78 ± 3.93 | -5.51 ± 3.52 | -4.83 ± 3.49 |
| 8 | Retaliate | +0.67 ± 9.36 | +1.70 ± 5.54 | +5.32 ± 7.10 |
| 9 | Nimble | +5.19 ± 5.18 | -1.30 ± 3.58 | -0.13 ± 3.99 |
| 10 | Twinstrike | +11.86 ± 4.03 | -1.41 ± 2.73 | -3.68 ± 2.54 |
| J | Taunt | +12.06 ± 11.55 | +6.88 ± 7.86 | +9.99 ± 7.95 |
| Q | Move | +3.95 ± 3.48 | +6.13 ± 2.72 | +2.59 ± 2.83 |
| K | Empower | -1.45 ± 3.76 | -0.44 ± 3.00 | -3.00 ± 2.73 |

***Held at the unlock.** The same fit on the hand held when the piles emptied. Descriptive rather than randomised: you chose what to still be holding, so a card that gets kept in positions that are already won looks good for that reason.*

*The 2 figures are in [`split-64.html`](split-64.html#what-a-card-is-worth).*
