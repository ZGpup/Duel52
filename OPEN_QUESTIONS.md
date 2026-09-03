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

## Nothing open

As of **2026-09-03**, every rules question raised for this project has been answered and
ported into `game_rules.md`. The engine can be built against that document without a pending
decision anywhere in it.

Two things still carry an **[ASSUMED]** marker in `game_rules.md`, both inferences nobody has
contradicted and neither blocking:

- **Base cards are hidden from their owner too.** High confidence: the official text says
  base cards cannot be looked at before being flipped, and the 4's Foresight explicitly
  targets *your own* base cards, which would be pointless otherwise. Corroborated by the
  ruling that a Queen-moved base card stays hidden.
- **Hand sizes are public.** Physically forced — you can see how many cards someone holds.

## What to do when a new question appears

1. Check whether `game_rules.md` already answers it — several rulings are stated once in a
   general form (resolution ordering, mandatory powers, fizzling) rather than repeated per
   card.
2. If not, and the online implementation can settle it, settle it there.
3. Otherwise add an entry here with a route marker, implement a defensible default, mark it
   `[ASSUMED]` in `game_rules.md`, and flag it. Do not block on it.

## Rulings that reversed an earlier answer

Recorded so nobody re-derives the superseded version from an old note:

- **Frozen cards cannot be flipped at all.** An earlier answer said an external 5 could flip
  a frozen card; the final ruling is that a 5 skips them entirely. Freeze blocks attacking
  and being flipped, by anyone — but *not* reactivation by a King.
- **Pair attacks do carry rank modifiers.** The first pass assumed the flat 2 overrode
  everything. It does not: a 9-pair deals 4 to a Jack and takes no retaliate from an 8, and a
  10-pair is forced to split 1 + 1.
- **The 2 bottoms rather than discards** (`game_rules.md` §10a). This is the project's first
  deliberate deviation from RAW, marked `[HOUSE]` and config-flagged as
  `two_power: bottom | discard`.
