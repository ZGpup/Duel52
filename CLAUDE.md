# Duel 52 — Project Instructions

Goal: build a rules-exact engine for the card game **Duel 52** and train a strong
self-play agent, in order to answer a question nobody has published an answer to —
**what does optimal play actually look like?** The insight is the deliverable; the bot is
the instrument.

## Documents

| File | Purpose |
|---|---|
| `game_rules.md` | **The spec.** Canonical, disambiguated ruleset. The engine implements this. |
| `DESIGN.md` | Engine + model architecture: state, action encoding, observations, training. |
| `PLAN.md` | Phased roadmap with status. Update as phases complete. |
| `OPEN_QUESTIONS.md` | Unresolved rules and design questions. Resolve → move the ruling into `game_rules.md` and delete the entry. |
| `FINDINGS.md` | Strategy insights as they emerge. This is the actual output of the project. |

## Facts that are easy to get wrong

Read `game_rules.md` before touching engine code. These five trip people up:

1. **Base cards are hidden from their owner too**, not just the opponent. That is why the
   4's Foresight can target your own base cards.
2. **Lane wins are endgame-only.** A lane cannot be won until the draw pile *and* the
   opponent's hand are both empty. The whole draw phase is positioning.
3. **10 cards are removed unseen at setup.** Belief over hidden cards never fully resolves,
   even at the end. Do not abstract this away.
4. **Suits are mechanically irrelevant** — collapse to rank everywhere. (Color denotes deck
   ownership in the split-deck variant; suit still never matters.)
5. **The split-deck (red/black) variant is the default configuration**, not the
   rules-as-written game. See `game_rules.md` §9.

## Conventions

- **The Rust engine is the sole authority on legality.** Never reimplement rules logic in
  Python — call the engine. Python does training and analysis only.
- **Every ruling in `game_rules.md` gets a named test.** Test names reference the rule
  section, e.g. `rule_6_king_reactivates_ace_grants_one_action`.
- **Everything is seeded and deterministic.** Same seed + same config → identical game.
  Non-reproducible results are bugs.
- **Config-driven, no hardcoded constants.** Variant selection, deck composition, removal
  count, draw rules, and stalemate threshold all live in config.
- **Device-agnostic.** Code must run on MPS locally and CUDA on a rented box with no edits
  beyond a config value. That is the handoff path.

## Working agreements

- When a rules question comes up, check `OPEN_QUESTIONS.md` first. If it's not there and
  the online implementation at <https://www.juddmadden.com/duel52/play.html> can settle it,
  settle it there rather than interrupting the owner. Escalate only what testing can't answer.
- Owner has limited time on this project and is delegating implementation. Prefer making a
  defensible call, documenting it as `[ASSUMED]`, and flagging it — over blocking.
- Log measured results in `FINDINGS.md` with the config and seed range that produced them.
  An unreproducible finding is not a finding.

## Commands

Nothing is built yet. As the engine lands, record the real commands here:

```
# build engine        (TBD)
# run tests           (TBD)
# play a human game   (TBD)
# self-play benchmark (TBD)
# train               (TBD)
```
