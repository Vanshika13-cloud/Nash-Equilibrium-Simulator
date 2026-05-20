"""
nash_solver.py
--------------
Module 1: Nash Equilibrium Finder

Given any payoff matrix for a 2-player game, this module computes:
  - Pure strategy Nash Equilibria (by best-response analysis)
  - Mixed strategy Nash Equilibria (via support enumeration using nashpy)

A Nash Equilibrium is a strategy profile where no player can improve
their payoff by unilaterally deviating. In mixed strategies, players
randomize over actions such that the opponent is indifferent.
"""

import numpy as np
import nashpy as nash
from itertools import product


def find_pure_nash(A: np.ndarray, B: np.ndarray) -> list[tuple]:
    """
    Find all pure strategy Nash Equilibria by checking best responses.

    A pure NE (i*, j*) exists when:
      - i* is a best response to j*  =>  A[i*, j*] >= A[i, j*] for all i
      - j* is a best response to i*  =>  B[i*, j*] >= B[i*, j] for all j

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
        Payoff matrix for Player 1. A[i, j] = payoff when P1 plays i, P2 plays j.
    B : np.ndarray, shape (m, n)
        Payoff matrix for Player 2. B[i, j] = payoff when P1 plays i, P2 plays j.

    Returns
    -------
    List of (i, j) tuples representing pure NE strategy profiles.
    """
    m, n = A.shape
    equilibria = []

    for i, j in product(range(m), range(n)):
        # Is i a best response for P1 given P2 plays j?
        p1_best_response = A[i, j] == np.max(A[:, j])
        # Is j a best response for P2 given P1 plays i?
        p2_best_response = B[i, j] == np.max(B[i, :])

        if p1_best_response and p2_best_response:
            equilibria.append((i, j))

    return equilibria


def find_mixed_nash(A: np.ndarray, B: np.ndarray) -> list[tuple]:
    """
    Find all mixed strategy Nash Equilibria using support enumeration.

    In a mixed NE, each player randomizes over a subset of strategies (their
    'support') such that every strategy in the support yields equal expected
    payoff — making the opponent indifferent between mixing and deviating.

    Uses the nashpy library's support_enumeration method.

    Parameters
    ----------
    A : np.ndarray  — Payoff matrix for Player 1
    B : np.ndarray  — Payoff matrix for Player 2

    Returns
    -------
    List of (sigma_1, sigma_2) tuples where sigma_i are probability vectors.
    """
    game = nash.Game(A, B)
    equilibria = []

    for eq in game.support_enumeration():
        sigma1, sigma2 = eq
        # Filter out near-zero probability noise from numerical solver
        sigma1 = np.round(sigma1, 6)
        sigma2 = np.round(sigma2, 6)
        equilibria.append((sigma1, sigma2))

    return equilibria


def compute_expected_payoffs(A: np.ndarray, B: np.ndarray,
                              sigma1: np.ndarray, sigma2: np.ndarray) -> tuple:
    """
    Compute expected payoffs for both players under mixed strategies.

    E[P1] = sigma1^T * A * sigma2
    E[P2] = sigma1^T * B * sigma2

    Parameters
    ----------
    A, B    : Payoff matrices
    sigma1  : Mixed strategy (probability vector) for Player 1
    sigma2  : Mixed strategy (probability vector) for Player 2

    Returns
    -------
    (expected_payoff_p1, expected_payoff_p2)
    """
    ep1 = float(sigma1 @ A @ sigma2)
    ep2 = float(sigma1 @ B @ sigma2)
    return round(ep1, 4), round(ep2, 4)


def print_game_summary(A: np.ndarray, B: np.ndarray,
                        row_labels: list = None, col_labels: list = None,
                        game_name: str = "Game"):
    """
    Pretty-print a bimatrix game and all its Nash Equilibria.
    """
    m, n = A.shape
    row_labels = row_labels or [f"R{i}" for i in range(m)]
    col_labels = col_labels or [f"C{j}" for j in range(n)]

    print(f"\n{'='*55}")
    print(f"  {game_name}")
    print(f"{'='*55}")
    print(f"  Payoff Matrix (Player1, Player2)\n")

    # Header
    header = f"{'':>8}" + "".join(f"{c:>12}" for c in col_labels)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, row in enumerate(row_labels):
        row_str = f"  {row:>6}"
        for j in range(n):
            row_str += f"  ({A[i,j]:>3}, {B[i,j]:>3})"
        print(row_str)

    # Pure NE
    pure = find_pure_nash(A, B)
    print(f"\n  Pure Strategy Nash Equilibria: {len(pure)} found")
    for (i, j) in pure:
        ep1, ep2 = compute_expected_payoffs(A, B,
            np.eye(m)[i], np.eye(n)[j])
        print(f"    → ({row_labels[i]}, {col_labels[j]})  "
              f"Payoffs: P1={ep1}, P2={ep2}")

    # Mixed NE
    mixed = find_mixed_nash(A, B)
    print(f"\n  Mixed Strategy Nash Equilibria: {len(mixed)} found")
    for idx, (s1, s2) in enumerate(mixed):
        ep1, ep2 = compute_expected_payoffs(A, B, s1, s2)
        s1_str = "  ".join(f"{row_labels[i]}:{s1[i]:.3f}" for i in range(m))
        s2_str = "  ".join(f"{col_labels[j]}:{s2[j]:.3f}" for j in range(n))
        print(f"    [{idx+1}] P1: [{s1_str}]")
        print(f"         P2: [{s2_str}]")
        print(f"         Expected Payoffs: P1={ep1}, P2={ep2}")

    print(f"{'='*55}\n")


# ── Canonical example games ──────────────────────────────────────────────────

# Prisoner's Dilemma: mutual defection is the unique pure NE,
# even though mutual cooperation is Pareto superior.
PRISONERS_DILEMMA = {
    "A": np.array([[-1, -3], [0, -2]], dtype=float),
    "B": np.array([[-1,  0], [-3, -2]], dtype=float),
    "rows": ["Cooperate", "Defect"],
    "cols": ["Cooperate", "Defect"],
    "name": "Prisoner's Dilemma"
}

# Battle of the Sexes: two pure NE (coordination games),
# plus one mixed NE where players randomize.
BATTLE_OF_SEXES = {
    "A": np.array([[3, 1], [0, 2]], dtype=float),
    "B": np.array([[2, 0], [1, 3]], dtype=float),
    "rows": ["Opera", "Football"],
    "cols": ["Opera", "Football"],
    "name": "Battle of the Sexes"
}

# Rock Paper Scissors: no pure NE, unique mixed NE at uniform (1/3, 1/3, 1/3)
ROCK_PAPER_SCISSORS = {
    "A": np.array([[ 0, -1,  1],
                   [ 1,  0, -1],
                   [-1,  1,  0]], dtype=float),
    "B": np.array([[ 0,  1, -1],
                   [-1,  0,  1],
                   [ 1, -1,  0]], dtype=float),
    "rows": ["Rock", "Paper", "Scissors"],
    "cols": ["Rock", "Paper", "Scissors"],
    "name": "Rock-Paper-Scissors"
}


if __name__ == "__main__":
    for g in [PRISONERS_DILEMMA, BATTLE_OF_SEXES, ROCK_PAPER_SCISSORS]:
        print_game_summary(g["A"], g["B"],
                           row_labels=g["rows"],
                           col_labels=g["cols"],
                           game_name=g["name"])
