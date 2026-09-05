"""Duel 52 — Python interface to the Rust engine.

``CLAUDE.md``: *"The Rust engine is the sole authority on legality. Never reimplement
rules logic in Python — call the engine. Python does training and analysis only."*

This package is a thin wrapper. It re-exports the compiled extension and adds convenience
that is unambiguously *not* rules logic: a random rollout, and the Phase 1 statistics
helper. Anything that decides what is legal, what a power does, or who has won lives in
Rust.

Quick start
-----------

>>> from duel52 import Game
>>> g = Game(variant="split", seed=42)      # the project default variant
>>> g.to_move
'p0'
>>> actions = g.legal_actions()             # list of plain dicts
>>> g.apply_index(0)                        # or g.apply(actions[0])
>>> g.is_over
False

Things worth knowing before you use this
----------------------------------------

* ``Game`` holds engine-side **ground truth**, including the opponent's hand and the draw
  pile order. Use :meth:`Game.observation` to get the filtered, per-player view — never
  read another player's private state into an agent.
* **Slot indices are not stable.** They compact when a card dies and shift when a Queen
  moves a card. Re-read ``legal_actions()`` after every ``apply``.
* Lane wins are endgame-only, base cards are hidden from their owner too, and ten cards
  were removed unseen at setup. See ``game_rules.md``.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from ._engine import VERSION as ENGINE_VERSION
from ._engine import (
    Agent,
    Game,
    ladder_agents,
    lane_permutations,
    power_reference,
    random_play_stats,
)

__all__ = [
    "ENGINE_VERSION",
    "Agent",
    "Game",
    "VARIANTS",
    "TWO_POWERS",
    "ladder_agents",
    "lane_permutations",
    "power_reference",
    "random_play_stats",
    "play_agent_game",
    "play_random_game",
    "rollout",
]

#: The three configurations of ``game_rules.md`` §9. ``"split"`` is the project default.
VARIANTS = ("base", "split", "mirrored")

#: Settings of the 2's power (``game_rules.md`` §10a). ``"bottom"`` is the house rule and
#: the default; ``"discard"`` is rules-as-written and exists so the parity claim behind the
#: house rule can be measured rather than assumed.
TWO_POWERS = ("bottom", "discard")


def rollout(
    game: Game,
    choose: Callable[[Game, list[dict[str, Any]]], int],
    *,
    max_decisions: int = 100_000,
) -> Game:
    """Play ``game`` to completion, asking ``choose`` for an index into the legal actions.

    ``choose`` receives the game and the current legal-action list and returns the index of
    the one to take. The game is mutated in place and returned.

    ``max_decisions`` is a guard against a policy that never terminates; the engine's own
    stalemate rule (``game_rules.md`` §7) already bounds real games far below it.
    """
    decisions = 0
    while not game.is_over:
        if decisions >= max_decisions:
            raise RuntimeError(
                f"rollout exceeded {max_decisions} decisions — the policy is not "
                f"terminating (engine state: {game!r})"
            )
        actions = game.legal_actions()
        index = choose(game, actions)
        game.apply_index(index)
        decisions += 1
    return game


def play_random_game(
    *,
    variant: str = "split",
    seed: int = 0,
    two_power: str | None = None,
) -> Game:
    """Play one uniformly random game and return the finished position.

    Uses Python's :mod:`random` seeded from ``seed``, so this is reproducible — but it is
    *not* the same stream the Rust :func:`random_play_stats` uses. For the Phase 1 numbers
    call that instead; it is both faster and the reference implementation.
    """
    import random as _random

    rng = _random.Random(seed)
    game = Game(variant=variant, seed=seed, two_power=two_power)
    return rollout(game, lambda _g, actions: rng.randrange(len(actions)))


def play_agent_game(
    *,
    p0: str = "random",
    p1: str = "random",
    variant: str = "split",
    seed: int = 0,
    two_power: str | None = None,
) -> Game:
    """Play one game between two of the frozen Phase 2 ladder agents.

    ``p0`` and ``p1`` are ladder names — see :func:`ladder_agents` for the frozen roster,
    and ``duel52 help`` for the budget syntax (``"ismcts:1600"``, ``"pimc:32x1"``).

    The agents run in Rust, so this is a thin driver rather than an implementation: Phase 3's
    gated evaluation can use it to play a checkpoint against the permanent benchmark without
    reimplementing a baseline in Python.

    Each side gets **one** :class:`Agent`, built once and carried through the game. Rebuilding
    an agent per decision restarts its random stream and measurably weakens it.
    """
    game = Game(variant=variant, seed=seed, two_power=two_power)
    # Distinct streams per seat, so the two sides never draw the same random numbers.
    agents = {"p0": Agent(p0, seed=seed * 2), "p1": Agent(p1, seed=seed * 2 + 1)}
    return rollout(game, lambda g, _actions: agents[g.to_move].choose_index(g))


def sweep(
    *,
    games: int = 2000,
    first_seed: int = 0,
    variants: Iterable[str] = VARIANTS,
    two_powers: Iterable[str] = TWO_POWERS,
) -> list[dict[str, Any]]:
    """Run the Phase 1 random-vs-random sweep across variants and both settings of the 2.

    Returns one stats dict per configuration; each carries a ready-made ``report`` string.
    """
    return [
        random_play_stats(
            variant=variant, first_seed=first_seed, games=games, two_power=two_power
        )
        for variant in variants
        for two_power in two_powers
    ]


__all__.append("sweep")
