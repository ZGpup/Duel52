# Open Questions

Unresolved rules and design points. When one is settled: move the ruling into
`game_rules.md` with the appropriate marker, then delete the entry here.

Resolution route:
- **[TEST]** — settle by playing the online implementation at
  <https://www.juddmadden.com/duel52/play.html>. Do this before asking the owner.
- **[OWNER]** — needs the owner or the Discord community; the online version plays
  rules-as-written and cannot answer variant questions.
- **[ENGINE]** — no "correct" answer exists; we must pick something and document it.

---

## High priority

### Q1 — Split-deck card counts **[OWNER]**
`game_rules.md` §9a assumes: 26 per player − 3 base − 5 hand = 18, remove **5** unseen →
**13-card personal draw pile**. This preserves the base game's totals (10 removed overall,
26 total draw), which is why I think it's intended. Confirm the removal count, and whether
the 3 base cards come off the top of the player's own color deck.

### Q2 — Split-deck unlock trigger **[OWNER]**
Base cards unlock when the draw pile empties. With two piles: when **both** are empty, or
each player's own? The piles empty within a turn of each other so it rarely matters, but
the engine needs a definite rule. Currently assuming **both**.

### Q3 — Mirrored-removal specifics **[OWNER]**
In variant 9b, do both players remove the same *ranks* (chosen how — randomly? agreed?),
and is the removed set revealed to both players? If revealed, belief tracking changes
substantially: the unseen pool becomes fully known, which is a materially different game
and probably the cleanest target for equilibrium analysis.

### Q4 — Ace double-attack scope **[TEST]**
The rules say a freshly flipped Ace "may attack twice." Two sub-questions:
(a) Is this genuinely two attacks by the Ace, or just loose phrasing for "flip costs you
nothing net, so the Ace can flip and attack"?
(b) When a King reactivates an Ace, does the Ace regain the double-attack, or only the
extra action? The owner confirmed the *action* is granted once — the attack question is
still open.

### Q5 — Stalemate rule **[ENGINE]**
No draw exists in the published rules, but once piles and hands are empty neither player
may be able to reach two lanes. Currently: **draw after 20 consecutive turns with no damage
and no kill.** Needs a sanity check against measured Phase 1 data — if random play hits the
cutoff often, the threshold or the rule is wrong.

---

## Medium priority

### Q6 — Draw timing and hand limit **[TEST]**
Is the draw at the start of the turn (assumed) or the end? Is there any hand-size limit?

### Q7 — Freeze interactions **[TEST]**
(a) A 6 stops enemy cards from attacking or *flipping themselves* — can a frozen face-down
card still be flipped externally by a 5 or a King?
(b) Exact duration: assumed to expire at the end of the frozen player's next turn.
(c) Does a 6 freeze cards played into the lane after it resolves? (Rules say new cards may
be played; presumably they are unfrozen.)

### Q8 — 3-Trap details **[TEST]**
(a) Does the returned 3 arrive in the same lane, immediately?
(b) Can it attack the turn it returns?
(c) Does it trigger when killed as a base card post-unlock?
(d) It returns face-up, so it cannot re-trigger — confirm.

### Q9 — Queen move details **[TEST]**
(a) Does the moved card retain damage? (Assumed yes.)
(b) Can it attack after being moved, if it hasn't attacked this turn?
(c) Does a moved base card become a normal card, or stay flagged as a base card?
(d) Can the Queen move a card *into* a lane she was just moved into by another Queen?

### Q10 — 7-Heal cap **[TEST]**
Assumed healing is capped at max HP, so a Jack on 1 HP (2 damage) heals to 3. Confirm the
Jack case specifically.

### Q11 — Multiple Jacks **[TEST]**
Two Jacks in one lane: does the attacker choose which to hit, or must a specific one die
first?

### Q12 — 2-View with an empty pile **[TEST]**
Once the draw pile is empty, does flipping a 2 do nothing, or still force a discard?

---

## Low priority

### Q13 — Retaliate ordering **[ENGINE]**
Assumed the 8's retaliate resolves after the attacker's damage. Matters only when both
would be lethal — an attacker on 1 HP trading with an 8 on 1 HP. Either order kills both,
so this is likely cosmetic; verify it cannot change a lane outcome.

### Q14 — Pair dissolution **[TEST]**
Confirmed: broken by a Queen move. Assumed: broken when a member dies. Can a pair be
voluntarily dissolved, and can a card leave one pair to join another?

### Q15 — 5 / King resolution ordering **[TEST]**
The player chooses the order. Confirm that a 5 flipping a King (which then empowers the
lane, which may re-trigger cards the 5 just flipped) resolves the way we implement it —
this is the most complex interaction in the game and worth an explicit test regardless.

### Q16 — Simultaneous two-lane win **[ENGINE]**
Can a single action win the second and third lanes at once, and does anything depend on
the order? Almost certainly not — but the terminal check should be unambiguous.
