"""Kuhn poker: a test fixture for the R-NaD learner — ``PLAN.md`` item 8.

Not a second rules engine for Duel 52, and nothing in the Duel 52 path imports it. It exists
because a learner that cannot find the equilibrium of a game whose equilibrium is known is
broken, and Duel 52 would never say so.

Three cards (J, Q, K), one each, ante 1. Player 0 acts first: pass or bet 1. Histories
``""``, ``"p"``, ``"b"``, ``"pb"``; a pass facing a bet folds, a bet facing a bet calls. The
game value for player 0 is ``−1/18``, and exploitability is computed exactly by best
response, so the test needs no sampling to judge the result.

The trajectories it produces have exactly the shape :func:`duel52.rnad.actor.play_batch`
returns, so the learner is exercised through the same code path Duel 52 uses.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from torch import nn

from .actor import Trajectories, sample_actions
from .core import legal_policy

__all__ = ["KuhnNet", "play_kuhn", "policy_table", "exploitability"]

HISTORIES = ["", "p", "b", "pb"]
OBS_DIM = 12  # one-hot (card, history)
PASS, BET = 0, 1


def _to_act(history: str) -> int:
    return len(history) % 2


def _payoff(history: str, cards: tuple[int, int]) -> float | None:
    """Player 0's payoff at a terminal history, ``None`` if the game continues."""
    showdown = 1.0 if cards[0] > cards[1] else -1.0
    if history == "pp":
        return showdown
    if history in ("bb", "pbb"):
        return 2.0 * showdown
    if history == "bp":
        return 1.0
    if history == "pbp":
        return -1.0
    return None


def _obs_index(card: int, history: str) -> int:
    return card * 4 + HISTORIES.index(history)


class KuhnNet(nn.Module):
    """A small MLP with the Duel 52 networks' contract: ``obs → (logits, value)``."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.torso = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.policy = nn.Linear(hidden, 2)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.torso(obs)
        return self.policy(h), self.value(h).squeeze(-1)


@torch.no_grad()
def play_kuhn(model: nn.Module, *, games: int, rng: np.random.Generator) -> Trajectories:
    deals = list(itertools.permutations(range(3), 2))
    cards = [deals[i] for i in rng.integers(0, len(deals), size=games)]
    histories = [""] * games
    rows: dict[str, list] = {k: [] for k in ("obs", "action", "mu", "player", "game", "t")}
    lengths = np.zeros(games, dtype=np.int64)
    returns = np.zeros((games, 2))

    live = list(range(games))
    while live:
        obs = torch.zeros(len(live), OBS_DIM)
        for row, g in enumerate(live):
            obs[row, _obs_index(cards[g][_to_act(histories[g])], histories[g])] = 1.0
        logits, _ = model(obs)
        probs = legal_policy(logits, torch.ones_like(logits, dtype=torch.bool))
        action, mu = sample_actions(probs, torch.from_numpy(rng.random(len(live))).float())
        still = []
        for row, g in enumerate(live):
            rows["obs"].append(obs[row])
            rows["action"].append(int(action[row]))
            rows["mu"].append(float(mu[row]))
            rows["player"].append(_to_act(histories[g]))
            rows["game"].append(g)
            rows["t"].append(int(lengths[g]))
            lengths[g] += 1
            histories[g] += "pb"[int(action[row])]
            payoff = _payoff(histories[g], cards[g])
            if payoff is None:
                still.append(g)
            else:
                returns[g] = (payoff, -payoff)
        live = still

    n = len(rows["action"])
    return Trajectories(
        obs=torch.stack(rows["obs"]),
        legal=torch.ones(n, 2, dtype=torch.bool),
        action=torch.tensor(rows["action"]),
        mu=torch.tensor(rows["mu"], dtype=torch.float32),
        player=torch.tensor(rows["player"]),
        game=torch.tensor(rows["game"]),
        t=torch.tensor(rows["t"]),
        lengths=lengths,
        returns=returns,
    )


@torch.no_grad()
def policy_table(model: nn.Module) -> dict[tuple[int, str], float]:
    """Probability of betting at each of the twelve information states."""
    obs = torch.eye(OBS_DIM)
    logits, _ = model(obs)
    probs = legal_policy(logits, torch.ones_like(logits, dtype=torch.bool))
    return {(card, h): float(probs[_obs_index(card, h), BET]) for card in range(3) for h in HISTORIES}


def _value(history, cards, player, br, sigma) -> float:
    """``player``'s expected payoff from ``history``: ``player`` plays ``br``, the other ``sigma``."""
    payoff = _payoff(history, cards)
    if payoff is not None:
        return payoff if player == 0 else -payoff
    actor = _to_act(history)
    key = (cards[actor], history)
    bet = (1.0 if br[key] == BET else 0.0) if actor == player else sigma[key]
    return (1 - bet) * _value(history + "p", cards, player, br, sigma) + bet * _value(
        history + "b", cards, player, br, sigma
    )


def _best_response_value(player: int, sigma: dict[tuple[int, str], float]) -> float:
    deals = list(itertools.permutations(range(3), 2))
    mine = [h for h in HISTORIES if _to_act(h) == player]
    br: dict[tuple[int, str], int] = {}
    # Deepest information states first, so a choice only ever looks at choices already made.
    for history in sorted(mine, key=len, reverse=True):
        for card in range(3):
            totals = [0.0, 0.0]
            for deal in deals:
                if deal[player] != card:
                    continue
                reach = 1.0
                for i, a in enumerate(history):
                    if _to_act(history[:i]) != player:
                        p_bet = sigma[(deal[1 - player], history[:i])]
                        reach *= p_bet if a == "b" else 1 - p_bet
                for action, suffix in ((PASS, "p"), (BET, "b")):
                    totals[action] += reach * _value(history + suffix, deal, player, br, sigma)
            br[(card, history)] = BET if totals[BET] > totals[PASS] else PASS
    return sum(_value("", deal, player, br, sigma) for deal in deals) / len(deals)


def exploitability(sigma: dict[tuple[int, str], float]) -> float:
    """``(BR_0(σ) + BR_1(σ)) / 2``: zero exactly at a Nash equilibrium."""
    return (_best_response_value(0, sigma) + _best_response_value(1, sigma)) / 2.0
