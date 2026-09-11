# Duel 52 — self-play analysis: `split`

Generated 2026-09-11 00:11 UTC · corpus schema 1 · figures in [`split.html`](split.html)

- **variant** — split
- **ruleset** — canonical-2026-09/8a9767ea88977b84
- **the 2's power** — bottom
- **lanes** — 3, 2 to win
- **hand size** — 5
- **stalemate** — 20 quiet turns
- **deal seeds** — 1–1000
- **models** — gen031@1000, lane-gen032@1000
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

One agent per column, each playing **itself**. Every corpus in this document was played under the ruleset named in the header, and the reader refuses to merge two that were not. A deal is played twice with the seats swapped, so games = 2 × deals.

| model | agent | games | deals | chunks | card rows | games/sec | cpu time |
|---|---|---:|---:|---:|---:|---:|---:|
| gen031@1000 | netmcts:models/duel52-split-gen031.d52nn@1000 | 2,000 | 1,000 | 8 | 81,457 | 0.395 | 1.41 h |
| lane-gen032@1000 | netmcts:models/duel52-split-lane-gen032.d52nn@1000 | 2,000 | 1,000 | 8 | 82,328 | 0.155 | 3.57 h |

## First vs second player

The score of whoever moved first, pooled over both halves of every colour-paired deal. 0.500 is no advantage. The interval is clustered on the deal, which is the unit that was randomised — both games of a pair hold the same cards.

| model | first-player score | P0 wins | P1 wins | draws | draw rate | separated from 0.500 |
|---|---:|---:|---:|---:|---:|---:|
| gen031@1000 | 0.5170 ± 0.0250 | 1,028 | 960 | 12 | 0.60% | no |
| lane-gen032@1000 | 0.4928 ± 0.0241 | 971 | 1,000 | 29 | 1.45% | no |

*The figure is in [`split.html`](split.html#first-vs-second-player).*

## Game shape

Turns here are the game's, not one player's: a game of 42 turns is 21 each. The unlock is the turn the last draw pile emptied and base cards became attackable (`game_rules.md` §3) — until then a lane cannot be won, so it divides the game in two.

| model | mean turns | median | p10 – p90 | draw rate | stalemate / mutual / cap | reached unlock | mean unlock turn |
|---|---:|---:|---:|---:|---:|---:|---:|
| gen031@1000 | 48.0 | 48 | 44 – 51 | 0.60% | 0 / 12 / 0 | 100.0% | 13.0 |
| lane-gen032@1000 | 47.2 | 48 | 43 – 51 | 1.45% | 0 / 29 / 0 | 100.0% | 13.0 |

## Average card play turn

The owner's own turn on which a card is played from hand, face-down. Base cards are excluded — they were never played. A low number is a card that goes down early, which is not the same as a card that goes face-up early.

| model | mean play turn | cards |
|---|---:|---:|
| gen031@1000 | 10.997 ± 0.036 | 69,457 |
| lane-gen032@1000 | 10.670 ± 0.042 | 70,328 |

| rank | power | gen031@1000 | n | lane-gen032@1000 | n |
|---|---|---:|---:|---:|---:|
| A | Action | 6.72 ± 0.16 | 5,448 | 7.42 ± 0.18 | 5,405 |
| 2 | View | 11.33 ± 0.20 | 5,612 | 12.60 ± 0.22 | 5,489 |
| 3 | Trap | 16.11 ± 0.21 | 4,773 | 13.02 ± 0.22 | 5,354 |
| 4 | Foresight | 11.95 ± 0.22 | 5,444 | 14.58 ± 0.27 | 5,148 |
| 5 | Flip | 12.84 ± 0.19 | 5,301 | 10.94 ± 0.22 | 5,261 |
| 6 | Freeze | 17.16 ± 0.15 | 5,360 | 13.71 ± 0.19 | 5,525 |
| 7 | Heal All | 9.44 ± 0.18 | 5,472 | 8.63 ± 0.18 | 5,479 |
| 8 | Retaliate | 5.61 ± 0.15 | 5,605 | 5.98 ± 0.16 | 5,604 |
| 9 | Nimble | 7.71 ± 0.18 | 5,343 | 7.71 ± 0.19 | 5,441 |
| 10 | Twinstrike | 9.15 ± 0.20 | 5,340 | 11.57 ± 0.21 | 5,404 |
| J | Taunt | 5.79 ± 0.15 | 5,446 | 5.87 ± 0.14 | 5,443 |
| Q | Move | 18.08 ± 0.12 | 5,039 | 16.05 ± 0.15 | 5,315 |
| K | Empower | 12.49 ± 0.20 | 5,274 | 11.08 ± 0.19 | 5,460 |

*The figure is in [`split.html`](split.html#average-card-play-turn).*

## Average card flip turn

The turn a card goes face-up. The main table counts **only flips its owner chose** — a card turned up by a 5's cascade or by springing a 3's Trap went face-up without anyone deciding to, and averaging those in answers a different question. Base cards are tabled separately: they cannot be flipped before the unlock, so their timing is a fact about the unlock rather than about the card.

| model | mean flip turn (chosen) | flips | any cause |
|---|---:|---:|---:|
| gen031@1000 | 11.23 ± 0.04 | 64,157 | 11.33 ± 0.04 |
| lane-gen032@1000 | 11.04 ± 0.05 | 61,753 | 11.03 ± 0.04 |

| rank | power | gen031@1000 | n | lane-gen032@1000 | n |
|---|---|---:|---:|---:|---:|
| A | Action | 6.85 ± 0.16 | 5,390 | 7.46 ± 0.18 | 5,272 |
| 2 | View | 12.37 ± 0.22 | 4,789 | 13.32 ± 0.28 | 4,260 |
| 3 | Trap | 19.79 ± 0.19 | 2,719 | 17.05 ± 0.27 | 2,836 |
| 4 | Foresight | 12.55 ± 0.24 | 4,769 | 15.72 ± 0.31 | 3,953 |
| 5 | Flip | 13.35 ± 0.20 | 4,970 | 11.45 ± 0.23 | 4,859 |
| 6 | Freeze | 17.59 ± 0.16 | 5,066 | 14.96 ± 0.20 | 4,594 |
| 7 | Heal All | 9.50 ± 0.18 | 5,365 | 8.71 ± 0.18 | 5,274 |
| 8 | Retaliate | 5.78 ± 0.16 | 5,566 | 6.12 ± 0.17 | 5,523 |
| 9 | Nimble | 7.82 ± 0.19 | 5,211 | 7.83 ± 0.20 | 5,000 |
| 10 | Twinstrike | 9.29 ± 0.20 | 5,232 | 11.68 ± 0.21 | 5,212 |
| J | Taunt | 5.91 ± 0.15 | 5,392 | 6.09 ± 0.15 | 5,101 |
| Q | Move | 18.27 ± 0.12 | 4,832 | 16.22 ± 0.15 | 5,152 |
| K | Empower | 12.90 ± 0.22 | 4,856 | 11.94 ± 0.21 | 4,717 |

*Flips the owner chose, played cards only.*

| rank | power | gen031@1000 | lane-gen032@1000 |
|---|---|---:|---:|
| A | Action | 6.86 ± 0.16 | 7.47 ± 0.18 |
| 2 | View | 12.34 ± 0.22 | 13.11 ± 0.25 |
| 3 | Trap | 17.70 ± 0.19 | 15.19 ± 0.21 |
| 4 | Foresight | 12.55 ± 0.24 | 15.11 ± 0.29 |
| 5 | Flip | 13.34 ± 0.20 | 11.39 ± 0.23 |
| 6 | Freeze | 17.57 ± 0.16 | 14.57 ± 0.19 |
| 7 | Heal All | 9.51 ± 0.18 | 8.67 ± 0.18 |
| 8 | Retaliate | 5.79 ± 0.16 | 6.13 ± 0.17 |
| 9 | Nimble | 7.85 ± 0.18 | 7.87 ± 0.20 |
| 10 | Twinstrike | 9.30 ± 0.20 | 11.68 ± 0.21 |
| J | Taunt | 5.91 ± 0.15 | 5.99 ± 0.15 |
| Q | Move | 18.26 ± 0.12 | 16.18 ± 0.15 |
| K | Empower | 12.91 ± 0.21 | 11.76 ± 0.20 |

*Any cause — chosen, cascaded, or sprung.*

| rank | power | gen031@1000 | lane-gen032@1000 |
|---|---|---:|---:|
| A | Action | 15.49 ± 0.19 | 16.24 ± 0.21 |
| 2 | View | 15.64 ± 0.23 | 16.43 ± 0.22 |
| 3 | Trap | 15.74 ± 0.17 | 16.49 ± 0.20 |
| 4 | Foresight | 15.76 ± 0.24 | 16.42 ± 0.24 |
| 5 | Flip | 15.65 ± 0.20 | 16.58 ± 0.22 |
| 6 | Freeze | 15.33 ± 0.19 | 16.43 ± 0.23 |
| 7 | Heal All | 15.03 ± 0.17 | 15.86 ± 0.18 |
| 8 | Retaliate | 15.43 ± 0.21 | 16.19 ± 0.21 |
| 9 | Nimble | 15.67 ± 0.21 | 16.46 ± 0.23 |
| 10 | Twinstrike | 15.68 ± 0.21 | 16.23 ± 0.21 |
| J | Taunt | 15.41 ± 0.19 | 16.01 ± 0.19 |
| Q | Move | 15.63 ± 0.22 | 16.27 ± 0.21 |
| K | Empower | 15.58 ± 0.20 | 16.32 ± 0.23 |

*Base cards, any cause.*

*The figure is in [`split.html`](split.html#average-card-flip-turn).*

## Turns spent face-down

Measured in the owner's own turns, so a card flipped on the turn it was played is **0**. Three columns because one number would be a lie by omission: a rank that is flipped fast *and* killed fast has a short tenure for two different reasons.

* **among flipped** — cards that were eventually turned face-up. The decision.
* **never flipped** — the share that were not, whether killed hidden or still hidden at the end. This is the censoring, stated rather than dropped.
* **to exit** — every played card, counting a hidden death or the end of the game as the end of its tenure. How long a card actually spends hidden.

| model | mean turns face-down (among flipped) | never flipped | mean turns to exit (all cards) |
|---|---:|---:|---:|
| gen031@1000 | 0.49 ± 0.01 | 3.8% | 0.55 ± 0.01 |
| lane-gen032@1000 | 0.58 ± 0.01 | 4.6% | 0.66 ± 0.02 |

| rank | power | gen031@1000 | n | lane-gen032@1000 | n |
|---|---|---:|---:|---:|---:|
| A | Action | 0.16 ± 0.02 | 5,417 | 0.06 ± 0.01 | 5,388 |
| 2 | View | 1.06 ± 0.06 | 5,068 | 0.96 ± 0.06 | 4,793 |
| 3 | Trap | 1.93 ± 0.09 | 4,452 | 2.48 ± 0.10 | 5,052 |
| 4 | Foresight | 0.76 ± 0.05 | 4,987 | 0.94 ± 0.06 | 4,492 |
| 5 | Flip | 0.63 ± 0.04 | 4,992 | 0.60 ± 0.04 | 4,980 |
| 6 | Freeze | 0.44 ± 0.03 | 5,124 | 0.84 ± 0.05 | 5,075 |
| 7 | Heal All | 0.15 ± 0.02 | 5,407 | 0.11 ± 0.01 | 5,436 |
| 8 | Retaliate | 0.19 ± 0.02 | 5,579 | 0.14 ± 0.02 | 5,572 |
| 9 | Nimble | 0.22 ± 0.02 | 5,253 | 0.37 ± 0.03 | 5,257 |
| 10 | Twinstrike | 0.19 ± 0.02 | 5,262 | 0.20 ± 0.02 | 5,297 |
| J | Taunt | 0.12 ± 0.01 | 5,416 | 0.11 ± 0.02 | 5,401 |
| Q | Move | 0.27 ± 0.02 | 4,852 | 0.18 ± 0.02 | 5,203 |
| K | Empower | 0.47 ± 0.04 | 4,978 | 0.80 ± 0.06 | 5,115 |

*Turns face-down before being flipped, by rank.*

| rank | power | gen031@1000 | lane-gen032@1000 |
|---|---|---:|---:|
| A | Action | 0.006 ± 0.002 | 0.003 ± 0.002 |
| 2 | View | 0.097 ± 0.008 | 0.127 ± 0.009 |
| 3 | Trap | 0.067 ± 0.007 | 0.056 ± 0.007 |
| 4 | Foresight | 0.084 ± 0.008 | 0.127 ± 0.009 |
| 5 | Flip | 0.058 ± 0.006 | 0.053 ± 0.006 |
| 6 | Freeze | 0.044 ± 0.006 | 0.081 ± 0.008 |
| 7 | Heal All | 0.012 ± 0.003 | 0.008 ± 0.002 |
| 8 | Retaliate | 0.005 ± 0.002 | 0.006 ± 0.002 |
| 9 | Nimble | 0.017 ± 0.004 | 0.034 ± 0.005 |
| 10 | Twinstrike | 0.015 ± 0.003 | 0.020 ± 0.004 |
| J | Taunt | 0.006 ± 0.002 | 0.008 ± 0.003 |
| Q | Move | 0.037 ± 0.005 | 0.021 ± 0.004 |
| K | Empower | 0.056 ± 0.006 | 0.063 ± 0.007 |

*Share of played cards of each rank never turned face-up.*

| rank | power | gen031@1000 | lane-gen032@1000 |
|---|---|---:|---:|
| A | Action | 0.16 ± 0.02 | 0.07 ± 0.01 |
| 2 | View | 1.24 ± 0.06 | 1.12 ± 0.06 |
| 3 | Trap | 1.99 ± 0.09 | 2.58 ± 0.10 |
| 4 | Foresight | 0.90 ± 0.06 | 1.05 ± 0.06 |
| 5 | Flip | 0.71 ± 0.04 | 0.65 ± 0.04 |
| 6 | Freeze | 0.50 ± 0.04 | 0.97 ± 0.05 |
| 7 | Heal All | 0.16 ± 0.02 | 0.12 ± 0.01 |
| 8 | Retaliate | 0.20 ± 0.02 | 0.15 ± 0.02 |
| 9 | Nimble | 0.24 ± 0.02 | 0.43 ± 0.03 |
| 10 | Twinstrike | 0.22 ± 0.02 | 0.23 ± 0.03 |
| J | Taunt | 0.12 ± 0.01 | 0.12 ± 0.02 |
| Q | Move | 0.29 ± 0.02 | 0.20 ± 0.02 |
| K | Empower | 0.57 ± 0.04 | 0.92 ± 0.06 |

*Turns face-down counting hidden deaths and the game's end.*

*The figure is in [`split.html`](split.html#turns-spent-face-down).*

## Hand size at the unlock

`FINDINGS.md` H2: every card in hand after the piles empty is a turn the opponent cannot close a lane. The score column is the **larger-hand side's**, over the games where the two hands differed — one observation per game, not per player, so a game cannot vote twice. 0.500 would mean holding more cards is worth nothing.

| model | mean hand at unlock | median | score of the larger hand | games | tied |
|---|---:|---:|---:|---:|---:|
| gen031@1000 | 6.81 ± 0.03 | 7 | 0.7717 ± 0.0213 | 1,610 | 19.5% |
| lane-gen032@1000 | 6.45 ± 0.04 | 6 | 0.6832 ± 0.0243 | 1,558 | 22.1% |

| margin | gen031@1000 | lane-gen032@1000 |
|---|---:|---:|
| +1 | 0.6766 ± 0.0345 | 0.5940 ± 0.0359 |
| +2 | 0.7985 ± 0.0347 | 0.7115 ± 0.0426 |
| +3 | 0.9021 ± 0.0374 | 0.8285 ± 0.0538 |
| +4 or more | 0.9500 ± 0.0365 | 0.8992 ± 0.0551 |

*Score of the side holding this many more cards at the unlock.*

*The figure is in [`split.html`](split.html#hand-size-at-the-unlock).*

## Win rate with each card in the opening hand

The opening hand is the one held at the start of that player's **own** first turn — the deal plus the draw that opens a turn — so both players are measured on the same number of cards. (`GameState::new` performs P0's opening draw, so 'the hand at setup' would give P0 six cards and P1 five.)

**Read the exclusive table.** It is the score of the games where you held the card and your opponent did not. The inclusive one pools in the games where both held it, and those contribute a win and a loss in symmetric pairs — they pull every rank toward 0.500 without saying anything about the card.

| rank | power | gen031@1000 | n | lane-gen032@1000 | n |
|---|---|---:|---:|---:|---:|
| A | Action | 0.5574 ± 0.0357 | 1,002 | 0.5215 ± 0.0342 | 1,002 |
| 2 | View | 0.4628 ± 0.0354 | 980 | 0.5041 ± 0.0348 | 980 |
| 3 | Trap | 0.4845 ± 0.0359 | 966 | 0.4798 ± 0.0336 | 966 |
| 4 | Foresight | 0.3771 ± 0.0346 | 968 | 0.4463 ± 0.0343 | 968 |
| 5 | Flip | 0.4615 ± 0.0361 | 960 | 0.5151 ± 0.0353 | 960 |
| 6 | Freeze | 0.4582 ± 0.0368 | 922 | 0.4648 ± 0.0340 | 922 |
| 7 | Heal All | 0.5746 ± 0.0353 | 978 | 0.5567 ± 0.0332 | 978 |
| 8 | Retaliate | 0.5883 ± 0.0354 | 1,002 | 0.5369 ± 0.0335 | 1,002 |
| 9 | Nimble | 0.5550 ± 0.0358 | 964 | 0.5228 ± 0.0348 | 964 |
| 10 | Twinstrike | 0.4874 ± 0.0371 | 912 | 0.4781 ± 0.0357 | 912 |
| J | Taunt | 0.4984 ± 0.0363 | 962 | 0.5223 ± 0.0350 | 962 |
| Q | Move | 0.5167 ± 0.0353 | 1,016 | 0.4887 ± 0.0336 | 1,016 |
| K | Empower | 0.4990 ± 0.0344 | 1,002 | 0.5065 ± 0.0337 | 1,002 |

*Score when you hold this rank and the opponent does not.*

| rank | power | gen031@1000 | lane-gen032@1000 |
|---|---|---:|---:|
| A | Action | 0.5359 ± 0.0224 | 0.5134 ± 0.0214 |
| 2 | View | 0.4777 ± 0.0212 | 0.5024 ± 0.0208 |
| 3 | Trap | 0.4911 ± 0.0205 | 0.4885 ± 0.0192 |
| 4 | Foresight | 0.4308 ± 0.0201 | 0.4698 ± 0.0194 |
| 5 | Flip | 0.4780 ± 0.0207 | 0.5086 ± 0.0202 |
| 6 | Freeze | 0.4766 ± 0.0207 | 0.4802 ± 0.0191 |
| 7 | Heal All | 0.5417 ± 0.0200 | 0.5317 ± 0.0187 |
| 8 | Retaliate | 0.5530 ± 0.0215 | 0.5222 ± 0.0201 |
| 9 | Nimble | 0.5342 ± 0.0224 | 0.5142 ± 0.0217 |
| 10 | Twinstrike | 0.4932 ± 0.0200 | 0.4882 ± 0.0192 |
| J | Taunt | 0.4991 ± 0.0217 | 0.5134 ± 0.0209 |
| Q | Move | 0.5101 ± 0.0213 | 0.4932 ± 0.0202 |
| K | Empower | 0.4994 ± 0.0208 | 0.5039 ± 0.0204 |

*Inclusive: score whenever you hold at least one, whatever the opponent holds. Kept for contrast.*

*The figure is in [`split.html`](split.html#win-rate-with-each-card-in-the-opening-hand).*

## Win rate with each card in hand at the unlock

The same two estimators, on the hand held when the last pile emptied. Restricted to games that reached the unlock.

The second table is the one to trust. Holding a particular rank at the unlock is partly just holding *more cards*, and the section above shows that is worth something on its own; **adjusted** subtracts the mean score at the same hand size, leaving what is associated with the card rather than with the size of the hand it sits in. It is a difference from 0, not a win rate: `+0.02` is two points of win probability above an average hand of that size.

| rank | power | gen031@1000 | n | lane-gen032@1000 | n |
|---|---|---:|---:|---:|---:|
| A | Action | 0.5175 ± 0.0554 | 343 | 0.5534 ± 0.0445 | 534 |
| 2 | View | 0.4878 ± 0.0339 | 943 | 0.5401 ± 0.0321 | 1,011 |
| 3 | Trap | 0.5240 ± 0.0398 | 729 | 0.5722 ± 0.0321 | 1,045 |
| 4 | Foresight | 0.6573 ± 0.0304 | 1,055 | 0.5137 ± 0.0351 | 910 |
| 5 | Flip | 0.6569 ± 0.0307 | 1,058 | 0.6220 ± 0.0314 | 1,004 |
| 6 | Freeze | 0.6281 ± 0.0389 | 679 | 0.5840 ± 0.0345 | 857 |
| 7 | Heal All | 0.4789 ± 0.0351 | 899 | 0.4901 ± 0.0361 | 810 |
| 8 | Retaliate | 0.5726 ± 0.0660 | 248 | 0.6507 ± 0.0572 | 302 |
| 9 | Nimble | 0.6773 ± 0.0365 | 719 | 0.5912 ± 0.0391 | 669 |
| 10 | Twinstrike | 0.7158 ± 0.0308 | 929 | 0.5743 ± 0.0314 | 1,043 |
| J | Taunt | 0.6622 ± 0.0730 | 188 | 0.6148 ± 0.0694 | 183 |
| Q | Move | 0.5050 ± 0.0539 | 403 | 0.5724 ± 0.0429 | 573 |
| K | Empower | 0.5187 ± 0.0339 | 965 | 0.4617 ± 0.0314 | 1,058 |

*Score when you hold this rank at the unlock and the opponent does not.*

| rank | power | gen031@1000 | lane-gen032@1000 |
|---|---|---:|---:|
| A | Action | -0.0230 ± 0.0478 | +0.0275 ± 0.0404 |
| 2 | View | -0.0604 ± 0.0305 | +0.0050 ± 0.0308 |
| 3 | Trap | -0.0224 ± 0.0353 | +0.0200 ± 0.0306 |
| 4 | Foresight | +0.0472 ± 0.0278 | -0.0162 ± 0.0334 |
| 5 | Flip | +0.0548 ± 0.0280 | +0.0467 ± 0.0304 |
| 6 | Freeze | +0.0774 ± 0.0341 | +0.0616 ± 0.0329 |
| 7 | Heal All | -0.0075 ± 0.0308 | -0.0151 ± 0.0336 |
| 8 | Retaliate | +0.0064 ± 0.0600 | +0.0660 ± 0.0520 |
| 9 | Nimble | +0.0621 ± 0.0334 | +0.0385 ± 0.0362 |
| 10 | Twinstrike | +0.0928 ± 0.0277 | +0.0566 ± 0.0290 |
| J | Taunt | +0.0821 ± 0.0629 | +0.1067 ± 0.0663 |
| Q | Move | +0.0660 ± 0.0468 | +0.1113 ± 0.0404 |
| K | Empower | -0.0007 ± 0.0306 | -0.0456 ± 0.0294 |

*The same, minus the mean score at that hand size.*

*The figure is in [`split.html`](split.html#win-rate-with-each-card-in-hand-at-the-unlock).*

## Pairs

A pair is two face-up same-rank cards on one side of one lane, declared with an action (§5). Rates are **per player-game**, so 'pairs per game' is what one player declares in one game; the game as a whole sees twice that.

| model | pairs declared per player-game | per game (both sides) | player-games with a pair | cards that were ever paired |
|---|---:|---:|---:|---:|
| gen031@1000 | 0.043 ± 0.007 | 0.09 | 4.2% | 0.4% |
| lane-gen032@1000 | 0.094 ± 0.010 | 0.19 | 9.1% | 0.9% |

| rank | power | gen031@1000 | n | lane-gen032@1000 | n |
|---|---|---:|---:|---:|---:|
| A | Action | 0.001 ± 0.001 | 6,380 | 0.004 ± 0.002 | 6,337 |
| 2 | View | 0.002 ± 0.001 | 6,466 | 0.006 ± 0.003 | 6,343 |
| 3 | Trap | 0.009 ± 0.003 | 5,769 | 0.008 ± 0.003 | 6,350 |
| 4 | Foresight | 0.002 ± 0.001 | 6,340 | 0.009 ± 0.003 | 6,044 |
| 5 | Flip | 0.001 ± 0.001 | 6,263 | 0.010 ± 0.004 | 6,223 |
| 6 | Freeze | 0.009 ± 0.003 | 6,272 | 0.004 ± 0.002 | 6,437 |
| 7 | Heal All | 0.004 ± 0.002 | 6,376 | 0.006 ± 0.003 | 6,383 |
| 8 | Retaliate | 0.011 ± 0.004 | 6,493 | 0.043 ± 0.008 | 6,492 |
| 9 | Nimble | 0.002 ± 0.001 | 6,233 | 0.001 ± 0.001 | 6,331 |
| 10 | Twinstrike | 0.002 ± 0.002 | 6,254 | 0.000 ± 0.001 | 6,318 |
| J | Taunt | 0.001 ± 0.001 | 6,452 | 0.001 ± 0.001 | 6,449 |
| Q | Move | 0.010 ± 0.004 | 5,977 | 0.013 ± 0.004 | 6,253 |
| K | Empower | 0.003 ± 0.002 | 6,182 | 0.013 ± 0.004 | 6,368 |

*Share of the cards of each rank that entered play and were ever a member of a declared pair. The rate that is comparable across ranks.*

| rank | power | gen031@1000 | lane-gen032@1000 |
|---|---|---:|---:|
| A | Action | 0.001 ± 0.001 | 0.003 ± 0.002 |
| 2 | View | 0.001 ± 0.001 | 0.005 ± 0.002 |
| 3 | Trap | 0.006 ± 0.002 | 0.006 ± 0.002 |
| 4 | Foresight | 0.001 ± 0.001 | 0.006 ± 0.002 |
| 5 | Flip | 0.001 ± 0.001 | 0.007 ± 0.003 |
| 6 | Freeze | 0.007 ± 0.003 | 0.003 ± 0.002 |
| 7 | Heal All | 0.004 ± 0.002 | 0.005 ± 0.002 |
| 8 | Retaliate | 0.009 ± 0.003 | 0.035 ± 0.006 |
| 9 | Nimble | 0.001 ± 0.001 | 0.001 ± 0.001 |
| 10 | Twinstrike | 0.002 ± 0.001 | 0.000 ± 0.000 |
| J | Taunt | 0.001 ± 0.001 | 0.001 ± 0.001 |
| Q | Move | 0.007 ± 0.003 | 0.010 ± 0.003 |
| K | Empower | 0.003 ± 0.002 | 0.010 ± 0.003 |

*Pairs of each rank declared per player-game. Depends on how often the rank is drawn as well as on how pairable it is.*

*The figure is in [`split.html`](split.html#pairs).*

## How cards die: face-up or face-down

Every card that entered play and was killed, split by which side it was showing when it died. A face-down card is a blank 2-HP card whatever its rank (§5), so dying face-down means its power never did anything — the flip that would have paid for it never happened.

One group of ranks cannot die face-down at all: a face-down card with a death trigger springs face-up instead of dying (§6), so it is either killed face-up later or not killed at all. In these corpora that is **3**, which is why they read 1.000 below.

| model | deaths per game | share of cards that die | died face-up | died face-down |
|---|---:|---:|---:|---:|
| gen031@1000 | 34.19 | 83.9% | 0.9409 ± 0.0019 | 0.0591 |
| lane-gen032@1000 | 33.37 | 81.1% | 0.9499 ± 0.0017 | 0.0501 |

| rank | power | gen031@1000 | n | lane-gen032@1000 | n |
|---|---|---:|---:|---:|---:|
| A | Action | 0.967 ± 0.005 | 5,831 | 0.982 ± 0.004 | 5,685 |
| 2 | View | 0.885 ± 0.009 | 5,534 | 0.882 ± 0.009 | 4,982 |
| 3 | Trap | 1.000 ± 0.000 | 4,333 | 1.000 ± 0.000 | 4,663 |
| 4 | Foresight | 0.889 ± 0.009 | 5,313 | 0.894 ± 0.009 | 4,349 |
| 5 | Flip | 0.915 ± 0.008 | 5,070 | 0.927 ± 0.008 | 5,051 |
| 6 | Freeze | 0.934 ± 0.007 | 5,010 | 0.914 ± 0.008 | 5,161 |
| 7 | Heal All | 0.966 ± 0.005 | 5,490 | 0.979 ± 0.004 | 5,434 |
| 8 | Retaliate | 0.965 ± 0.006 | 4,918 | 0.977 ± 0.005 | 4,612 |
| 9 | Nimble | 0.947 ± 0.007 | 5,727 | 0.955 ± 0.006 | 5,671 |
| 10 | Twinstrike | 0.954 ± 0.006 | 5,844 | 0.967 ± 0.005 | 5,716 |
| J | Taunt | 0.962 ± 0.005 | 5,912 | 0.975 ± 0.004 | 5,876 |
| Q | Move | 0.938 ± 0.008 | 4,389 | 0.963 ± 0.006 | 4,582 |
| K | Empower | 0.913 ± 0.008 | 5,005 | 0.920 ± 0.008 | 4,964 |

*Of the cards of this rank that died, the share that were face-up at the time.*

| rank | power | gen031@1000 | lane-gen032@1000 |
|---|---|---:|---:|
| A | Action | 0.914 ± 0.007 | 0.897 ± 0.008 |
| 2 | View | 0.856 ± 0.009 | 0.785 ± 0.011 |
| 3 | Trap | 0.751 ± 0.011 | 0.734 ± 0.011 |
| 4 | Foresight | 0.838 ± 0.009 | 0.720 ± 0.011 |
| 5 | Flip | 0.810 ± 0.009 | 0.812 ± 0.010 |
| 6 | Freeze | 0.799 ± 0.010 | 0.802 ± 0.010 |
| 7 | Heal All | 0.861 ± 0.009 | 0.851 ± 0.009 |
| 8 | Retaliate | 0.757 ± 0.011 | 0.710 ± 0.011 |
| 9 | Nimble | 0.919 ± 0.007 | 0.896 ± 0.008 |
| 10 | Twinstrike | 0.934 ± 0.006 | 0.905 ± 0.007 |
| J | Taunt | 0.916 ± 0.007 | 0.911 ± 0.008 |
| Q | Move | 0.734 ± 0.011 | 0.733 ± 0.012 |
| K | Empower | 0.810 ± 0.010 | 0.780 ± 0.011 |

*Share of the cards of each rank that entered play and were killed at all.*

*The figure is in [`split.html`](split.html#how-cards-die-face-up-or-face-down).*

## What becomes of a face-down card

Every card is played face-down, so this is the whole population: of the cards you put on the board, how many ever come up, how many are killed before they do, and how many are still hidden when the game ends. The five outcomes partition the cards played from hand — base cards are excluded, since nobody chose to play them.

| model | flipped by choice | flipped by a cascade | sprang its trap | killed face-down | face-down at the end | ever face-up |
|---|---:|---:|---:|---:|---:|---:|
| gen031@1000 | 0.924 | 0.018 | 0.020 | 0.025 | 0.013 | 0.962 |
| lane-gen032@1000 | 0.878 | 0.056 | 0.019 | 0.029 | 0.018 | 0.954 |

| rank | power | gen031@1000 | n | lane-gen032@1000 | n |
|---|---|---:|---:|---:|---:|
| A | Action | 0.994 ± 0.002 | 5,448 | 0.997 ± 0.002 | 5,405 |
| 2 | View | 0.903 ± 0.008 | 5,612 | 0.873 ± 0.009 | 5,489 |
| 3 | Trap | 0.933 ± 0.007 | 4,773 | 0.944 ± 0.007 | 5,354 |
| 4 | Foresight | 0.916 ± 0.008 | 5,444 | 0.873 ± 0.009 | 5,148 |
| 5 | Flip | 0.942 ± 0.006 | 5,301 | 0.947 ± 0.006 | 5,261 |
| 6 | Freeze | 0.956 ± 0.006 | 5,360 | 0.919 ± 0.008 | 5,525 |
| 7 | Heal All | 0.988 ± 0.003 | 5,472 | 0.992 ± 0.002 | 5,479 |
| 8 | Retaliate | 0.995 ± 0.002 | 5,605 | 0.994 ± 0.002 | 5,604 |
| 9 | Nimble | 0.983 ± 0.004 | 5,343 | 0.966 ± 0.005 | 5,441 |
| 10 | Twinstrike | 0.985 ± 0.003 | 5,340 | 0.980 ± 0.004 | 5,404 |
| J | Taunt | 0.994 ± 0.002 | 5,446 | 0.992 ± 0.003 | 5,443 |
| Q | Move | 0.963 ± 0.005 | 5,039 | 0.979 ± 0.004 | 5,315 |
| K | Empower | 0.944 ± 0.006 | 5,274 | 0.937 ± 0.007 | 5,460 |

*Share of each rank played from hand that was ever face-up, by any cause.*

| rank | power | gen031@1000 | lane-gen032@1000 |
|---|---|---:|---:|
| A | Action | 0.005 ± 0.002 | 0.003 ± 0.001 |
| 2 | View | 0.080 ± 0.007 | 0.084 ± 0.008 |
| 3 | Trap | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 4 | Foresight | 0.067 ± 0.007 | 0.064 ± 0.007 |
| 5 | Flip | 0.042 ± 0.005 | 0.046 ± 0.006 |
| 6 | Freeze | 0.027 ± 0.005 | 0.057 ± 0.006 |
| 7 | Heal All | 0.006 ± 0.002 | 0.005 ± 0.002 |
| 8 | Retaliate | 0.004 ± 0.002 | 0.005 ± 0.002 |
| 9 | Nimble | 0.015 ± 0.003 | 0.025 ± 0.004 |
| 10 | Twinstrike | 0.013 ± 0.003 | 0.015 ± 0.003 |
| J | Taunt | 0.006 ± 0.002 | 0.008 ± 0.003 |
| Q | Move | 0.017 ± 0.004 | 0.011 ± 0.003 |
| K | Empower | 0.045 ± 0.006 | 0.051 ± 0.006 |

*Share killed while still face-down — the power never fired.*

*The 2 figures are in [`split.html`](split.html#what-becomes-of-a-face-down-card).*

## What a card is worth

Two answers to the same question, by different routes, and they are worth reading against each other.

**Counterfactual** holds a real position fixed and swaps the card in hand, asking the value head what changed. It is exact about the position and only as good as that head. It is the one measurement in this document that cannot come from played games.

**Corpus-derived** fits the result on how many more of each rank you held than your opponent, so each rank is measured with the rest of the hand held fixed. It needs no network, so it works for any agent — including `random`. Both tables are **relative to an average card**, which is the counterfactual's convention, so the two are on the same scale.

The *dealt* table is the one with an identification argument behind it: the opening hand is dealt at random, so how many of a rank you were dealt is randomly assigned and its coefficient is a causal effect rather than a correlation. The *unlock* table is what you were still holding, which you chose, and it is descriptive.

| rank | power | gen031@1000 | lane-gen032@1000 |
|---|---|---:|---:|
| A | Action | +0.73 ± 0.11 | +3.50 ± 0.14 |
| 2 | View | -2.70 ± 0.07 | -2.99 ± 0.13 |
| 3 | Trap | +0.06 ± 0.11 | -1.62 ± 0.16 |
| 4 | Foresight | -2.01 ± 0.09 | -3.83 ± 0.13 |
| 5 | Flip | -0.69 ± 0.10 | +1.21 ± 0.17 |
| 6 | Freeze | -0.43 ± 0.08 | -0.97 ± 0.15 |
| 7 | Heal All | +0.92 ± 0.13 | +1.97 ± 0.15 |
| 8 | Retaliate | +2.82 ± 0.11 | +0.67 ± 0.15 |
| 9 | Nimble | +0.36 ± 0.09 | -0.12 ± 0.10 |
| 10 | Twinstrike | -0.72 ± 0.07 | -1.19 ± 0.12 |
| J | Taunt | +0.82 ± 0.10 | +2.46 ± 0.20 |
| Q | Move | +2.32 ± 0.15 | +2.24 ± 0.13 |
| K | Empower | -1.48 ± 0.09 | -1.31 ± 0.12 |

*`duel52 card-value`: holding the position fixed, what is this card worth in hand rather than an average card, in win-probability points? Each column is that checkpoint's own value head.*

| rank | power | gen031@1000 | lane-gen032@1000 |
|---|---|---:|---:|
| A | Action | +4.51 ± 2.96 | +2.30 ± 2.71 |
| 2 | View | -3.90 ± 2.75 | -0.59 ± 2.68 |
| 3 | Trap | -1.83 ± 2.85 | -2.21 ± 2.56 |
| 4 | Foresight | -8.97 ± 2.79 | -3.99 ± 2.62 |
| 5 | Flip | -3.54 ± 2.80 | +1.64 ± 2.74 |
| 6 | Freeze | -3.88 ± 2.90 | -3.55 ± 2.70 |
| 7 | Heal All | +6.69 ± 2.94 | +3.12 ± 2.73 |
| 8 | Retaliate | +7.56 ± 2.96 | +3.43 ± 2.64 |
| 9 | Nimble | +3.17 ± 2.97 | +0.82 ± 2.77 |
| 10 | Twinstrike | -1.29 ± 2.89 | -2.64 ± 2.76 |
| J | Taunt | -0.36 ± 2.93 | +1.54 ± 2.81 |
| Q | Move | +1.26 ± 2.89 | -0.74 ± 2.71 |
| K | Empower | +0.58 ± 2.70 | +0.87 ± 2.58 |

***Dealt.** A logistic fit of the result on how many more of each rank you were dealt than your opponent, relative to an average card, in win-probability points. The opening hand is dealt at random, so this is a randomised comparison rather than a correlation — it is the closest thing here to an experiment.*

| rank | power | gen031@1000 | lane-gen032@1000 |
|---|---|---:|---:|
| A | Action | -2.74 ± 5.37 | +2.53 ± 4.00 |
| 2 | View | -10.30 ± 3.51 | -4.14 ± 2.81 |
| 3 | Trap | -8.70 ± 2.90 | -3.54 ± 2.64 |
| 4 | Foresight | -1.02 ± 3.18 | -3.40 ± 2.61 |
| 5 | Flip | +1.45 ± 3.42 | +0.11 ± 3.18 |
| 6 | Freeze | +2.99 ± 2.61 | +1.02 ± 2.69 |
| 7 | Heal All | -0.99 ± 3.55 | -2.55 ± 3.23 |
| 8 | Retaliate | -1.60 ± 7.17 | +5.03 ± 5.35 |
| 9 | Nimble | +3.09 ± 4.44 | +0.22 ± 3.93 |
| 10 | Twinstrike | +8.29 ± 3.62 | +1.57 ± 2.46 |
| J | Taunt | +8.26 ± 9.35 | +5.23 ± 7.48 |
| Q | Move | +4.10 ± 3.04 | +3.48 ± 2.80 |
| K | Empower | -2.83 ± 3.32 | -5.56 ± 2.85 |

***Held at the unlock.** The same fit on the hand held when the piles emptied. Descriptive rather than randomised: you chose what to still be holding, so a card that gets kept in positions that are already won looks good for that reason.*

*The 2 figures are in [`split.html`](split.html#what-a-card-is-worth).*
