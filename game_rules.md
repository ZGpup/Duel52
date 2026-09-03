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
- **[ENGINE]** — no "correct" answer exists in the paper game; the engine must pick
  something, and this is the pick. Config-driven where a threshold is involved.
- **[HOUSE]** — a deliberate *deviation* from the published rules, adopted because the
  published rule is considered broken. Distinct from **[RULING]**, which disambiguates rather
  than overrides. Every **[HOUSE]** rule is behind a config flag with the RAW behaviour
  available, so the deviation stays measurable rather than assumed.

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
- A base card that a Queen moves to another lane **stops being a base card** and is a normal
  card from then on. **[RULING]** It remains face-down, and its owner still may **not** look
  at it — the free look at your own face-down cards (§4) never applies to a card that
  entered play as a base card, moved or not. **[RULING]** Moving a base card is therefore
  *not* a back-door Foresight on your own base.
- A face-down base **3** post-unlock still springs its Trap when killed (§6). **[RULING]**

The engine tracks one global flag, `base_unlocked`. In the split-deck variant it is set when
**both** piles are empty (§9). Every "once the draw pile is empty" clause in this document —
base unlock, 5/7/Q reaching base cards, and lane-win condition 2 in §7 — reads off that one
flag. The lone exception is the 2's View power, which is tied to the pile **you personally
draw from** (§6) — so a 2 can go dead a turn before the global unlock. **[RULING]**

## 4. Turn structure

Each turn:

1. **Draw** one card from the draw pile, if it is non-empty. The draw happens at the
   **start** of the turn, including the first player's opening turn — that turn is a draw
   plus two actions, opening at **6 cards in hand**. Only the *action* count is reduced.
   **[RULING]**
2. Take **three actions** (two for the first player's first turn only).

Any combination of actions is allowed, including repeats. **[RAW]**

There is **no hand-size limit**. **[RULING]**

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
  flipped Ace — see §6. That exception lifts the per-card limit only; the Ace's second
  attack still costs its own action. **[RULING]**)
- You may look at your **own played face-down cards** at any time, for free. **[RAW]**
  (This does *not* extend to your own base cards — see §3.)

## 5. Combat, damage, and pairs

- Every card has **2 hit points**, except the Jack, which has **3**. **[RAW]**
- **A face-down card is a blank 2-HP card, whatever its rank.** **[RULING]** The Jack's
  third hit point arrives when it is flipped, exactly as its Taunt does — face-down, it
  dies to two hits like anything else. Hit points are not an exception to "powers are inert
  while face-down" (§6); they follow the same line.
  - This is load-bearing for hidden information, not a detail. Damage is **public** (see
    below), so if a face-down Jack really had 3 HP, watching a face-down card survive two
    hits would identify it as the Jack — for free, to both players. The blank-card rule
    removes that leak: attacking a face-down card can never tell you what it is.
  - **Damage persists through the flip** and the ceiling rises with it: a face-down Jack on
    1 damage becomes a face-up Jack on **2 of 3** hit points. Nothing ever turns a card
    face-down again (§7), so a card can never be retroactively killed by losing hit points
    it had already spent.
  - Everything that keys on a target *being a Jack* therefore reads the **live power**, not
    the bare rank: the Taunt, the 10's blocked split, and the 9's double damage all agree.
    A 9 attacking a face-down Jack deals its ordinary **1**.
- A normal attack deals **1 damage**. A damaged card is turned sideways but remains fully
  functional — powers, attacks, everything. **[RAW]**
- A card that has taken damage equal to its hit points is **killed** and goes to the
  discard pile. **[RAW]**
- Damage persists through flipping and through being moved by a Queen. **[RULING]**
- You may attack face-down enemy cards (they are legal targets; you just don't know what
  they are). Base cards are exempt while the draw pile is non-empty (§3).
- The attacker chooses the target among the legal ones. Where Jack taunt (§6) constrains the
  choice to Jacks and there is **more than one Jack in the lane, the attacker picks which
  Jack to hit** — there is no "oldest first" ordering. **[RULING]**

### Pairs

- A pair is **two face-up cards of matching rank that you control in the same lane**.
  Both members must be face-up. **[RULING]**
- Declaring a pair costs 1 action. **[RAW]**
- A pair attacks together as a **single action** dealing **2 damage to one target**. You
  **may not choose to split** the damage. **[RAW]** (A pair of 10s is forced to split it by
  twinstrike — see the modifier rules below. That is the power acting, not a choice.)
- **Pairs must attack together** — a paired card cannot attack alone. **[RAW]** A pair attack
  is one attack for **both** members' once-per-turn budget, so a pair may not attack if either
  member has already attacked this turn. **[RULING]** You therefore cannot squeeze extra
  damage out of a lane by attacking with two cards separately and *then* pairing them.
- If a pair attacks an 8, **both members take 1 retaliate damage** — except a pair of 9s,
  which is immune (see the modifier rules below). **[RULING]**
- A pair is **broken** if a Queen moves one member to another lane, and broken if a member
  **dies**. Those are the only two ways out: a pair **cannot be dissolved voluntarily**, and
  a card **cannot leave one pair to join another**. **[RULING]** Un-pairing therefore costs
  a Queen, which makes declaring a pair a real commitment rather than a free toggle.
- A card belongs to **at most one pair**, so pairs are a *matching*, not a graph. Two disjoint
  pairs of the same rank in one lane are legal, which can only arise in the base game — the
  split deck gives each player exactly two of each rank. Three same-rank cards do not form a
  bigger group; the third stays unpaired and attacks alone. **[RULING]**
- Rank-based attack modifiers **do** apply to a pair attack. **[RULING]**
  - A pair of **10s** twinstrikes: the pair's 2 damage is **split 1 + 1 across two targets**,
    not doubled. **Damage is never lost** — whenever the split cannot happen, because it is
    blocked (a 9, or a lone Jack, §6) or because the lane holds only one legal target, the
    full **2** lands on that single card. **[RULING]** A 10-pair therefore never loses raw
    damage — but the *forced* spread can still be worse than concentration: 1 + 1 across two
    fresh cards kills nothing, where a consolidated 2 kills one outright. Pairing 10s is a
    commitment to chip damage.
  - A pair of **9s** deals **4 damage to a Jack** (the 9's doubling applies to the pair's 2),
    which one-shots a 3-HP Jack. Against anything else it is the normal 2. **[RULING]**
  - A pair of **9s** attacking an **8** takes **no** retaliate damage — each member is
    individually immune, and pairing does not forfeit Nimble. **[RULING]** A 9-pair therefore
    kills an 8 outright for free.

### What is public

The engine treats the physical table as the definition of public information:

- **Damage is public**, including on face-down cards — a damaged card is turned sideways,
  which both players can see. So "face-down and damaged" is a visible state. **[RULING]**
  This is load-bearing for belief modeling: every chip on a face-down card leaks information,
  and what a player chooses to defend leaks more.
- **Hand sizes are public**; hand *contents* are private. **[ASSUMED]** — physically forced.
- **The discard pile is public and inspectable** by both players at any time. **[RULING]**
  Dead cards are therefore common knowledge, not a memory feat — belief tracking runs over
  genuinely hidden cards only, and the agent's edge over a human stays strategic rather than
  mnemonic.
- A **bottomed** card (from a 2, §10a) is known to the player who bottomed it and to nobody
  else. They know both its identity and its position — the bottom of that pile. **[RULING]**
- Which lanes hold how many cards, face-up ranks, and `is_base` status are public.
- Private: your hand, your own played face-down cards (which only *you* may look at), and
  anything a 4 revealed to you. Base cards are hidden from **everyone**, owner included.

## 6. Card powers

Powers are inert while a card is face-down. **One-shot** powers fire when the card is
flipped face-up (and again if reactivated by a King). **Constant** powers are continuously
active while the card is face-up.

| Rank | Name | Type | Effect |
|---|---|---|---|
| **A** | Action | one-shot | Gain **1 action** this turn, usable however you like. On the turn it is flipped, the Ace itself **may attack twice** — genuinely two attacks, at two targets or the same one twice. **[RAW]** + **[RULING]** Each attack costs its own action; the exception is to the once-per-card limit, not to the action cost. **[RULING]** |
| **2** | View | one-shot | **Draw a card, then put a card from your hand on the bottom of your draw pile** — a scry, not a discard. **[HOUSE]** (RAW: discard it. See §10.) You may bottom the card you just drew. Gated on the pile **you** draw from; if that pile is empty the power does **nothing at all** — no draw, and no bottoming either, so it cannot be used to refill an empty pile. **[RULING]** |
| **3** | Trap | conditional | If killed **while face-down**, it **returns to play face-up** with full 2 HP instead of dying — immediately, in **the same lane**. **[RULING]** It comes back fully active with no waiting period, and it returns face-up so the Trap **cannot re-trigger**. **[RULING]** A base 3 killed post-unlock triggers this too, and returns as a normal (non-base) card. **[RULING]** |
| **4** | Foresight | one-shot | **Look at any one face-down card on the board** — including base cards, yours or your opponent's. Private information. **[RAW]** |
| **5** | Flip | one-shot | **Flip all your face-down cards in its lane.** You choose the order in which their powers resolve — one at a time, seeing each result before choosing the next — and they may attack if actions remain. **[RAW]** + **[RULING]** Includes your base card in that lane **once the draw pile is empty**. **[RULING]** |
| **6** | Freeze | one-shot | **All enemy cards in the lane are frozen**: they may not attack, and **cannot be flipped at all** — not by themselves, and not by an enemy or allied 5. **Cannot freeze a 9**, ever, including a 9 already in the lane when the 6 resolves. **[RAW]** + **[RULING]** Cards that enter the lane *after* the 6 resolves are **not** frozen. **[RULING]** A frozen card can still be **reactivated by a King**, since reactivation is not a flip. **[RULING]** Duration and Queen moves: see §8. |
| **7** | Heal All | one-shot | **Heal all your damaged cards 2 HP**, in all lanes, face-up and face-down. **[RAW]** Includes base cards **once the draw pile is empty**. **[RULING]** Healing is capped at the card's maximum HP — a Jack on 1 HP heals to 3, not 5. **[RULING]** |
| **8** | Retaliate | constant | Any card that attacks this 8 **takes 1 damage** — except a 9. **[RAW]** A pair attacking an 8: both members take 1. **[RULING]** Retaliate fires **even when the attack kills the 8**. **[RULING]** |
| **9** | Nimble | constant | Cannot be frozen by a 6. Takes no damage when attacking an 8. Blocks a 10's twinstrike from splitting (see 10). **Deals 2 damage to Jacks** — to *face-up* Jacks; a face-down card is a blank 2-HP card with no Taunt to punish, so a 9 deals it the ordinary 1 (§5). **[RAW]** + **[RULING]** |
| **10** | Twinstrike | constant | When attacking, deals **1 damage each to two cards** in the opposing lane. **If an intended target is a 9 or a Jack, only that card is damaged** — both block the split, but for different reasons. **[RULING]** With **two Jacks**, taunt already confines both halves to Jacks, nothing can leak past, and the 10 deals **1 to each**. With **two 9s** it is still **1 to one 9** — Nimble dodges the spread personally, not positionally. **[RULING]** A pair of 10s splits the pair's 2 as 1 + 1, and keeps it whole at 2 whenever it cannot split (§5). **[RULING]** |
| **J** | Taunt | constant | **Must be killed before any other card in his lane can be attacked.** Has **3 HP** — face-up. Face-down he is a blank 2-HP card with no taunt (§5), so both halves of the Jack arrive on the flip together. **[RAW]** + **[RULING]** With two Jacks in a lane, the attacker chooses which one to hit. **[RULING]** |
| **Q** | Move | one-shot | **Move one allied card from another lane into the Queen's lane**, face-down or face-up. The moved card **keeps its damage**, does **not** reactivate its one-shot power, keeps constant powers, and **may attack after the move** if it has not already attacked this turn. **[RAW]** + **[RULING]** Can move a base card **once the draw pile is empty**; a moved base card **becomes a normal card** (§3). **[RULING]** Moving a paired card breaks the pair. **[RULING]** A card may be moved into a lane a Queen was herself just moved into — no restriction. **[RULING]** |
| **K** | Empower | one-shot | **All your face-up cards in this lane reactivate their powers.** Does **not** affect other Kings. Does **not** affect constant powers (8, 9, 10, J). **[RAW]** |

### King + Ace interaction **[RULING]**

A King reactivating an Ace **does grant another action** — **once**, at the moment the
King flips. It does not repeat. Because Kings cannot activate other Kings, **no infinite
loop is possible.**

The reactivated Ace also **regains its double attack** for that turn, as a **reset, not a
stack**: the Ace's attack counter returns to zero with an allowance of 2, so after the
reactivation it may attack up to twice more whatever it did earlier in the turn. **[RULING]**
An Ace that attacked once, then got Kinged, tops out at **three** attacks that turn, not
four.

The double attack attaches to the **flip**, not to who caused it: an Ace flipped by a 5 gets
it the same as one flipped by its owner's own action, and gets the +1 action too. **[RULING]**

Ranks a King can meaningfully reactivate: **A, 2, 4, 5, 6, 7, Q**.
Ranks a King cannot reactivate: **8, 9, 10, J** (constant), **K** (excluded by rule),
**3** (conditional, and only relevant face-down).

## 7. Winning

**[RULING] — this is the most important structural rule in the game.**

A lane is won when **all three** hold:

1. The opponent has **no cards remaining in that lane** (including their base card), and
2. The **draw pile is empty** — in the split-deck variant, the `base_unlocked` flag, i.e.
   **both** piles empty (§3), and
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

### Draw / stalemate **[ENGINE — accepted by the owner]**

The published rules define no draw. Once the pile and both hands are empty, it is possible
for neither player to reach two lanes. The engine declares a **draw (0.5/0.5)** after a
configurable number of consecutive turns (default 20) with no damage dealt and no kill.
This is an engine necessity for training, not a claim about the paper game.

"Turns" here means **individual player turns (plies)** — the default 20 is ten apiece. The
counter resets on damage or a kill and on nothing else. **[ENGINE — accepted by the owner]**

What the rule actually guards against is **mutual passivity**, not any kind of loop. There is
no repeatable engine in this game: powers fire on flip, a King reactivates once, and nothing
ever turns a card face-down again, so total power activations are bounded by
(cards flipped) + (King reactivations). The reachable stall is strategic — post-unlock, both
hands empty, and neither player wants to attack first because attacking exposes the attacker.
Nobody commits and the game never ends on its own.

### Terminal evaluation

- The terminal check runs **after each action fully resolves**, including every sub-decision
  that action opened. It never runs mid-resolution. **[ENGINE]**
- **A lane win cannot be undone**, so latched and live evaluation are equivalent and the
  engine simply re-checks live state. **[RULING]** The reason is worth spelling out: refilling
  an empty side requires a card from hand (empty by condition 3), a draw (pile empty by
  condition 2), or a Queen — and **a Queen only moves cards into the lane she is already in**,
  so an empty side has no Queen to pull anything back. A 3's Trap cannot help either: it
  returns immediately on death, so an already-empty lane has no 3 pending.
- A single action **may** complete the second and third lanes at once. This is a plain win —
  nothing depends on the order, and there is no need to attribute the win to one lane.
  **[ENGINE]**
- You *can* empty your **own** side of a lane and hand it to your opponent — a Queen moving
  out your last card, or your last card dying to an 8's retaliate. The one way both players
  can reach two lanes on the same check: your last card in a lane attacks the opponent's last
  card in that lane, an 8, and retaliate kills yours as your damage kills theirs. Both sides
  of that lane empty, so **both** players win it. If that leaves both at two or more lanes,
  it is a **draw (0.5/0.5)** — a symmetric outcome gets a symmetric result, with no arbitrary
  tiebreak. **[ENGINE — accepted by the owner]** Astronomically rare, but the terminal check
  has to be total.

## 8. Sequencing notes for implementers

- Flipping a card and resolving its power is a single action, but the power may open
  **sub-decisions that cost no action** (a 4's peek target, a 5's activation order, a 2's
  discard, a Queen's move source, a 10's second target, a King's reactivation order).
  These are modeled as separate decision nodes. See `DESIGN.md`.
- **Resolution order is always the acting player's choice**, and always **adaptive**: when
  one action queues several resolutions (a 5's flips, a King's reactivations, a 5 that flips
  a King which then re-empowers the lane), the player picks the next one to resolve *after*
  seeing the previous one land, rather than committing to a full order up front. **[RULING]**
  + **[ENGINE]** for the adaptivity. Each resolution completes before the next begins.
- Freeze duration: an enemy card frozen by a 6 is unfrozen at the **end of the frozen
  player's next turn** — so exactly one of their turns is lost. **[RULING]**
- Freeze is a **per-card flag** set when the 6 resolves, not a property of the lane. Cards
  entering afterwards are unfrozen, and a frozen card a Queen moves to another lane **stays
  frozen** for the remaining duration. **[RULING]** A Queen is therefore not an escape hatch
  from a 6 — she relocates the problem.
- Freeze blocks exactly two things: **attacking**, and **being flipped** — by anyone. A 5
  resolving in the lane simply **skips** frozen cards; they are untouchable, not merely
  passive. **[RULING]**
- Freeze does **not** block **reactivation**: a face-up frozen card that a King empowers fires
  its power normally. It still cannot attack. **[RULING]** So a King is the one way to get
  value out of a frozen lane, and a 6 answers 5s but not Kings.
- Retaliate (8) resolves *after* the attacker's damage is applied, and fires even if that
  damage killed the 8. **[RULING]**
- A one-shot power is **mandatory** on flip — you do not get to decline the 2's scry or the
  5's flips. **[RULING]** So flip *timing* is a real decision: you flip a 5 when you want
  everything flipped, not before.
- A power with **no legal target simply fizzles**, and the flip remains a **legal action** —
  a Queen with no allied card elsewhere, a 4 with no face-down card on the board, a 5 whose
  lane holds nothing else face-down. **[RULING]** Often that is precisely why you flip it: a
  Queen with no move available is still a body that can attack.
- The 5 is all-or-nothing: it flips **every** one of your face-down cards in the lane, and
  post-unlock that includes your base card whether you want it flipped or not. **[RULING]**
  So the 5 becomes a *committal* card once the pile empties — it forces a reveal you may not
  want, which is a real cost against holding it for the endgame. The sole exception is frozen
  cards, which the 5 cannot touch and simply skips. **[RULING]**
- A 9 attacking a **face-up** Jack deals 2 damage in a single attack action. Against a
  face-down one it deals 1, because a face-down card is blank (§5).
- Jack taunt applies to *any* attack targeting the lane, including a 10's twinstrike. Against
  a **single** Jack the 10 deals 1 to the Jack only; against **two** Jacks it deals 1 to each,
  because taunt has already confined both halves of the split to Jacks and there is nothing
  for the split to leak past. **[RULING]**
- **The 9 and the Jack block for different reasons, and it shows in the two-card case.** The
  Jack's block is about *leakage past the taunt*, so with two Jacks there is nothing to leak
  past and both take 1. The 9's block is *personal* — Nimble dodges the spread itself — so a
  lane of two 9s still yields only **1 damage to one 9**. **[RULING]** Do not unify these
  into one "blocker" concept in the engine; they are different mechanics that happen to share
  a symptom in the one-card case.

## 9. Variants — **the split-deck variant is this project's default**

The project owner and the game's Discord community commonly play a variant that differs
from the published rules. **We treat it as the primary configuration**, because it makes
the game symmetric and therefore far better behaved for equilibrium analysis.

### 9a. Split deck (red/black) **[RULING]**

The deck is split by color. Each player owns one color — 26 cards, two suits, ranks A–K
twice — and **draws only from their own deck**. Both players therefore have access to an
identical multiset of ranks.

Card counts **[RULING]**: per player, 26 − 3 base − 5 hand = 18, then remove 5 unseen → a
**13-card personal draw pile**. This preserves the base game's totals exactly (10 cards
removed overall, 26 cards of draw across both players).

Setup order, per player, from their own shuffled 26 **[RULING]** + **[ENGINE]** for the
sequence:

1. Deal the **3 base cards off the top** of that player's own color deck, one per lane.
2. Deal **5** to hand.
3. Remove **5** unseen from the remaining 18.
4. **13** remain as that player's personal draw pile.

### 9b. Mirrored removal (deterministic variant) **[RULING]**

As 9a, but both players **remove the same set of ranks**, so the two decks are
rank-identical. The removed multiset is chosen **uniformly at random**, and it is **revealed
to both players**. **[RULING]** That last point is the strategically important one: the
unseen pool collapses to "opponent's hand + the six base cards," so belief tracking is over a
known set of ranks. It is the cleanest target for equilibrium analysis.

Mirroring forces a different setup order **[RULING]**: the shared 5-rank multiset is drawn
first and stripped from **both** decks, and only then are base cards and hands dealt. Doing
it in 9a's order is not always feasible — a rank can be gone from one player's pile (in hand
or on base) while still present in the other's, leaving no mirrored set to remove. Rejection
sampling (reshuffle until a mirrorable removal exists) was rejected because it biases the
deal distribution in ways that are hard to characterise, and partial mirroring was rejected
because approximately-mirrored decks defeat the point of 9b.

### Interaction with the "draw pile empty" trigger **[RULING]**

Base cards unlock, and lane wins become possible, when **both** draw piles are empty. Since
both players draw one card per turn, the piles empty within one turn of each other, so it
rarely bites — but it is now pinned down: one global `base_unlocked` flag, set on both piles
empty, driving every gated rule (§3).

The **2's View** power is the exception: it draws from the pile you personally own, so it is
gated on **your own** pile being empty, not the global flag. **[RULING]**

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

## 10. House rules — deliberate deviations from RAW

Everything above this section either restates the published rules or disambiguates them.
This section is where we knowingly *change* the game. Each item ships behind a config flag
with the RAW behaviour still available, so its effect can be measured rather than asserted.

### 10a. The 2 bottoms instead of discarding **[HOUSE]** — flag `two_power: bottom | discard`

**RAW:** the 2 draws a card, then you discard a card from your hand.
**House rule:** the 2 draws a card, then you put a card from your hand on the **bottom of your
draw pile**. Default in every configuration, base game included.

**Why.** The RAW discard permanently removes a card from the draw pile, which changes the
pile's **parity**. With a shared 26-card pile the players alternate draws 13/13; remove one
card and someone now gets an extra draw, and the player who fires the 2 chooses who. That
turns a filtering effect into a lever on the draw count, and the lever favours whoever is
positioned to use it first — an artifact of the rule rather than a strategic dimension anyone
designed. Bottoming makes the 2 **pile-neutral and hand-neutral**: it is pure selection.

**Consequences the engine must account for:**

- **Turns-to-unlock becomes fixed.** One draw per turn and no pile shrinkage means the pile
  empties after exactly *pile size* turns, regardless of how many 2s are played. The whole
  endgame trigger is deterministic at deal time — the confound the split deck was chosen to
  remove, removed again on another axis. There is **no last-card stall**: firing a 2 when one
  card remains draws it and bottoms one back, leaving the pile exactly where it would have
  been. The engine needs no special case for an empty-after-draw pile.
- **Cards are recycled, not destroyed.** A bottomed card *will* be drawn again if the pile
  outlasts it, so the discard pile is no longer the only place cards go. Belief tracking has
  to model a bottom-of-pile position, not just "in pile somewhere."
- **The bottomer holds private information.** You know the identity and position of the card
  you bottomed; your opponent knows only that you bottomed *something*. In the split variant
  that is knowledge of your own future draw; in the base game it is knowledge of a card your
  **opponent** may draw, which is strictly more interesting.
- **In the base game you bottom into the shared pile.** So the choice carries real risk: the
  card may come back to your opponent rather than to you.

Whether the parity problem is real is an empirical question, and `two_power: discard` exists
precisely so Phase 1 can answer it. Log the comparison in `FINDINGS.md`.
