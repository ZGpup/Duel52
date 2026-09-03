# Duel 52 — Canonical Rules

**Status:** This document is the *specification the engine implements*. It is not a
transcript of the official rules page — it is a disambiguated, engine-ready version of
them, incorporating rulings from the project owner (a regular player) where the published
rules are silent or ambiguous.

**Precedence:** If the code and this document disagree, this document wins — unless the
point is listed in `OPEN_QUESTIONS.md`, in which case the code's current choice is a
placeholder pending resolution.

Source: <https://www.juddmadden.com/duel52/index.html> (Judd Madden & Nina Riddell, 2017).

Markers used below:
- **[RAW]** — stated on the official rules page.
- **[RULING]** — resolved by the project owner; treat as authoritative.
- **[ASSUMED]** — my inference, not yet confirmed. Cross-listed in `OPEN_QUESTIONS.md`.

---

## 1. Components and board

- A standard 52-card deck, jokers removed. **[RAW]**
- Three **lanes**. Each player has their own side of each lane. Cards are played to your
  own side of a lane and attack across into the opposing side of the *same* lane. Lanes
  are otherwise independent.
- No limit on cards per lane per side. **[RULING]** (Never a practical constraint; the
  engine caps slots at 8/side/lane purely as an encoding bound — see `DESIGN.md`.)
- **Suits are mechanically irrelevant.** Only rank matters. The one exception is the
  split-deck variant (§9), where *color* denotes deck ownership, and even there suit
  within a color never matters.

## 2. Setup

Base game (rules-as-written):

1. Shuffle the 52-card deck.
2. Deal one card **face-down** into each lane for each player — 3 base cards per player,
   6 total. These are **base cards**. **[RAW]**
3. Deal 5 cards to each player's hand. **[RAW]**
4. Remove 10 cards from the draw pile, face-down, **without revealing them**. They take no
   further part in the game. **[RAW]**
5. Remaining shared draw pile: 52 − 6 − 10 − 10 = **26 cards**.
6. The first player takes only **two actions** on their opening turn. Every turn
   thereafter is three actions. **[RAW]**

> **Modeling note.** Because those 10 cards are removed unseen, a player's belief over
> hidden cards *never fully resolves*, even at the end of the game. This is a defining
> feature of the game and must not be abstracted away.

## 3. Base cards

- Base cards are face-down and **hidden from both players, including their owner**.
  **[ASSUMED — high confidence]** The official text says base cards "cannot be looked at
  before being flipped," and the 4's Foresight power explicitly allows peeking at *your
  own* base cards, which would be pointless otherwise.
- While the draw pile is non-empty, base cards **cannot be attacked and cannot be
  flipped**. They are untouchable. **[RAW]**
- Once the draw pile is empty, base cards become **normal cards** in their lane: they can
  be attacked, flipped, targeted by a 5, healed by a 7, and moved by a Queen. They still
  cannot be looked at before being flipped. **[RAW]** + **[RULING]** (see §6 for 5/7/Q)

## 4. Turn structure

Each turn:

1. **Draw** one card from the draw pile, if it is non-empty. **[RAW]** (Draw occurs at the
   *start* of the turn — **[ASSUMED]**.)
2. Take **three actions** (two for the first player's first turn only).

Any combination of actions is allowed, including repeats. **[RAW]**

There is no hand-size limit. **[ASSUMED]**

### The four actions

| Action | Cost | Effect |
|---|---|---|
| **Play** | 1 | Put a card from hand **face-down** into one of your lanes. It is inactive: it can be attacked and killed, but cannot attack and has no power. |
| **Flip** | 1 | Turn one of your face-down cards face-up. Its power activates immediately (one-shot) or becomes live (constant). |
| **Attack** | 1 | One of your face-up cards deals 1 damage to an opposing card in its lane. |
| **Pair** | 1 | Declare a pair (see §5). |

- A card may be **played, flipped, and attack all in the same turn**, actions permitting.
  **[RULING]**
- **Each card may attack only once per turn.** **[RAW]** (Sole exception: a freshly
  flipped Ace — see §6.)
- You may look at your **own played face-down cards** at any time, for free. **[RAW]**
  (This does *not* extend to your own base cards — see §3.)

## 5. Combat, damage, and pairs

- Every card has **2 hit points**, except the Jack, which has **3**. **[RAW]**
- A normal attack deals **1 damage**. A damaged card is turned sideways but remains fully
  functional — powers, attacks, everything. **[RAW]**
- A card that has taken damage equal to its hit points is **killed** and goes to the
  discard pile. **[RAW]**
- Damage persists through flipping and through being moved by a Queen. **[ASSUMED]**
- You may attack face-down enemy cards (they are legal targets; you just don't know what
  they are). Base cards are exempt while the draw pile is non-empty (§3).

### Pairs

- A pair is **two face-up cards of matching rank that you control in the same lane**.
  Both members must be face-up. **[RULING]**
- Declaring a pair costs 1 action. **[RAW]**
- A pair attacks together as a **single action** dealing **2 damage to one target**. The
  damage **cannot be split**. **[RAW]**
- **Pairs must attack together** — a paired card cannot attack alone. **[RAW]**
- If a pair attacks an 8, **both members take 1 retaliate damage**. **[RULING]**
- A pair is **broken** if a Queen moves one member to another lane. **[RULING]** It is
  also broken if a member dies. **[ASSUMED]**

## 6. Card powers

Powers are inert while a card is face-down. **One-shot** powers fire when the card is
flipped face-up (and again if reactivated by a King). **Constant** powers are continuously
active while the card is face-up.

| Rank | Name | Type | Effect |
|---|---|---|---|
| **A** | Action | one-shot | Gain **1 action** this turn, usable however you like. On the turn it is flipped, the Ace itself **may attack twice** — an explicit exception to once-per-turn. **[RAW]** |
| **2** | View | one-shot | **Draw a card, then discard a card** from your hand. **[RAW]** |
| **3** | Trap | conditional | If killed **while face-down**, it **returns to play face-up** with full 2 HP instead of dying. **[RAW]** |
| **4** | Foresight | one-shot | **Look at any one face-down card on the board** — including base cards, yours or your opponent's. Private information. **[RAW]** |
| **5** | Flip | one-shot | **Flip all your face-down cards in its lane.** You choose the order in which their powers resolve, and they may attack if actions remain. **[RAW]** Includes your base card in that lane **once the draw pile is empty**. **[RULING]** |
| **6** | Freeze | one-shot | **All enemy cards in the lane are frozen for one turn**: they may not attack or flip themselves. **Cannot freeze a 9.** New cards may still be played into the lane. **[RAW]** |
| **7** | Heal All | one-shot | **Heal all your damaged cards 2 HP**, in all lanes, face-up and face-down. **[RAW]** Includes base cards **once the draw pile is empty**. **[RULING]** Healing is capped at the card's maximum HP. **[ASSUMED]** |
| **8** | Retaliate | constant | Any card that attacks this 8 **takes 1 damage** — except a 9. **[RAW]** A pair attacking an 8: both members take 1. **[RULING]** |
| **9** | Nimble | constant | Cannot be frozen by a 6. Takes no damage when attacking an 8. Blocks a 10's twinstrike from splitting (see 10). **Deals 2 damage to Jacks.** **[RAW]** |
| **10** | Twinstrike | constant | When attacking, deals **1 damage each to two cards** in the opposing lane. **If either intended target is a 9 or a Jack, only that card is damaged** — the 9 and the Jack block the split. **[RULING]** |
| **J** | Taunt | constant | **Must be killed before any other card in his lane can be attacked.** Has **3 HP**. **[RAW]** |
| **Q** | Move | one-shot | **Move one allied card from another lane into the Queen's lane**, face-down or face-up. The moved card's one-shot power does **not** reactivate; constant powers persist. **[RAW]** Can move a base card **once the draw pile is empty**. **[RULING]** Moving a paired card breaks the pair. **[RULING]** |
| **K** | Empower | one-shot | **All your face-up cards in this lane reactivate their powers.** Does **not** affect other Kings. Does **not** affect constant powers (8, 9, 10, J). **[RAW]** |

### King + Ace interaction **[RULING]**

A King reactivating an Ace **does grant another action** — **once**, at the moment the
King flips. It does not repeat. Because Kings cannot activate other Kings, **no infinite
loop is possible.**

Ranks a King can meaningfully reactivate: **A, 2, 4, 5, 6, 7, Q**.
Ranks a King cannot reactivate: **8, 9, 10, J** (constant), **K** (excluded by rule),
**3** (conditional, and only relevant face-down).

## 7. Winning

**[RULING] — this is the most important structural rule in the game.**

A lane is won when **all three** hold:

1. The opponent has **no cards remaining in that lane** (including their base card), and
2. The **draw pile is empty**, and
3. The opponent's **hand is empty**.

So long as the opponent holds any card in hand, they can defend the lane, and it cannot be
won. **Lane wins are therefore strictly an endgame event** — they cannot occur while cards
are still being drawn, since base cards are untouchable until the pile empties.

**Win two lanes to win the game.** **[RAW]**

### Consequences worth internalizing

- The entire draw phase (~26 turns of card flow in the base game) is **positioning**, not
  scoring. Nothing is decided until the pile runs dry.
- Once the pile empties, hands drain and the game converges to a finite, no-new-resources
  combat endgame.
- Hand size at the moment the pile empties is a **defensive resource** — every card in
  hand is a turn the opponent cannot close a lane.

### Draw / stalemate **[ASSUMED — engine requirement]**

The published rules define no draw. Once the pile and both hands are empty, it is possible
for neither player to reach two lanes. The engine declares a **draw (0.5/0.5)** after a
configurable number of consecutive turns (default 20) with no damage dealt and no kill.
This is an engine necessity for training, not a claim about the paper game.

## 8. Sequencing notes for implementers

- Flipping a card and resolving its power is a single action, but the power may open
  **sub-decisions that cost no action** (a 4's peek target, a 5's activation order, a 2's
  discard, a Queen's move source, a 10's second target, a King's reactivation order).
  These are modeled as separate decision nodes. See `DESIGN.md`.
- Freeze duration: an enemy card frozen by a 6 is unfrozen at the **end of the frozen
  player's next turn**. **[ASSUMED]**
- Retaliate (8) resolves *after* the attacker's damage is applied. **[ASSUMED]**
- A 9 attacking a Jack deals 2 damage in a single attack action.
- Jack taunt applies to *any* attack targeting the lane, including a 10's twinstrike (a 10
  facing a Jack deals 1 damage to the Jack only).

## 9. Variants — **the split-deck variant is this project's default**

The project owner and the game's Discord community commonly play a variant that differs
from the published rules. **We treat it as the primary configuration**, because it makes
the game symmetric and therefore far better behaved for equilibrium analysis.

### 9a. Split deck (red/black) **[RULING]**

The deck is split by color. Each player owns one color — 26 cards, two suits, ranks A–K
twice — and **draws only from their own deck**. Both players therefore have access to an
identical multiset of ranks.

Card counts **[ASSUMED — needs confirmation, see `OPEN_QUESTIONS.md`]**: per player,
26 − 3 base − 5 hand = 18, then remove 5 unseen → a **13-card personal draw pile**. This
preserves the base game's totals exactly (10 cards removed overall, 26 cards of draw
across both players), which is why I believe it is the intended split.

### 9b. Mirrored removal (deterministic variant) **[RULING]**

As 9a, but both players **remove the same set of ranks**, so the two decks are
rank-identical. This strips out the asymmetry in available material and makes the matchup
as close to a pure skill mirror as the game allows.

### Interaction with the "draw pile empty" trigger **[ASSUMED]**

Base cards unlock, and lane wins become possible, when **both** draw piles are empty. Since
both players draw one card per turn, the piles empty within one turn of each other, so the
choice barely matters — but it must be pinned down for the engine.

### Why this matters for the project

The base game's shared pile means the two players can face materially different card
distributions. The split-deck variant removes that source of variance:

- Symmetric material → **first-player advantage becomes cleanly measurable** rather than
  confounded by deal luck.
- Lower variance → **fewer self-play games needed** for the same signal.
- The mirrored variant (9b) is the most nearly deterministic form and is the best target
  for "what does optimal play look like" as a question about *skill* rather than luck.

All three configurations (base / 9a / 9b) will be supported by a single config flag so
results can be compared across them.
