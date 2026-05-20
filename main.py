"""
main.py
-------
Nash Equilibrium Finder + Evolutionary Game Simulator
======================================================

This project bridges two perspectives on Nash Equilibria:

  1. STATIC  — Given a payoff matrix, solve mathematically for all NE
               (both pure and mixed strategy) using linear programming.

  2. DYNAMIC — Simulate a population of agents playing the game repeatedly.
               Watch the replicator dynamic evolve the population distribution
               and converge (or orbit) toward the Nash Equilibrium.

The connection: Nash Equilibria are exactly the fixed points of the
replicator dynamic. A population at a NE has no evolutionary pressure
to change — every strategy is earning average fitness.

Usage
-----
    python main.py

Outputs
-------
    outputs/  — PNG plots of all simulations
    Console   — Summary of all Nash Equilibria found
"""

import os
import numpy as np

from nash_solver import (
    print_game_summary,
    PRISONERS_DILEMMA,
    BATTLE_OF_SEXES,
    ROCK_PAPER_SCISSORS,
)
from evolutionary_sim import (
    simulate,
    plot_trajectories,
    run_multi_initial_conditions,
    HAWK_DOVE,
    RPS,
    STAG_HUNT,
)

os.makedirs("outputs", exist_ok=True)


def section(title: str):
    print(f"\n{'#'*60}")
    print(f"  {title}")
    print(f"{'#'*60}")


def main():

    # ── MODULE 1: Nash Equilibrium Solver ────────────────────────────────────
    section("MODULE 1: Nash Equilibrium Finder")

    print("""
  For each game below, we compute:
    - Pure strategy NE (best-response analysis)
    - Mixed strategy NE (support enumeration / linear programming)
    - Expected payoffs at each equilibrium
    """)

    for g in [PRISONERS_DILEMMA, BATTLE_OF_SEXES, ROCK_PAPER_SCISSORS]:
        print_game_summary(
            g["A"], g["B"],
            row_labels=g["rows"],
            col_labels=g["cols"],
            game_name=g["name"]
        )

    # ── MODULE 2: Evolutionary Dynamics ──────────────────────────────────────
    section("MODULE 2: Evolutionary Replicator Dynamics")

    print("""
  For each symmetric game, we:
    1. Start from equal strategy frequencies
    2. Evolve population using the replicator equation
    3. Observe convergence toward Nash Equilibria
    4. Repeat from many random starting points to map basins of attraction
    """)

    games = [HAWK_DOVE, STAG_HUNT, RPS]

    for game in games:
        A      = game["A"]
        labels = game["labels"]
        name   = game["name"]
        n      = A.shape[0]

        print(f"\n{'─'*50}")
        print(f"  Game: {name}")
        print(f"  Strategies: {labels}")

        # Single trajectory from equal start
        x0 = np.ones(n) / n
        history = simulate(A, x0, steps=6000, dt=0.01)
        final = history[-1]

        print(f"  x0 (equal):  { {l: round(v,3) for l,v in zip(labels, x0)} }")
        print(f"  x_final:     { {l: round(v,4) for l,v in zip(labels, final)} }")

        plot_trajectories(
            history, labels,
            game_name=name,
            save_path=f"outputs/{name.replace(' ','_')}_trajectory.png"
        )

        # Multiple initial conditions
        run_multi_initial_conditions(
            A, labels,
            game_name=name,
            n_trials=10,
            steps=4000,
            save_path=f"outputs/{name.replace(' ','_')}_multi_init.png"
        )

    # ── Custom game: try your own ─────────────────────────────────────────────
    section("CUSTOM GAME: Coordination Game")

    print("""
  Define your own payoff matrices and run both modules.
  Here we analyze a simple 3-strategy coordination game.
    """)

    # Example: 3-strategy game where strategy 0 dominates
    A_custom = np.array([[3, 1, 0],
                          [1, 2, 1],
                          [0, 1, 3]], dtype=float)
    B_custom = A_custom.T   # Symmetric game

    print_game_summary(
        A_custom, B_custom,
        row_labels=["Alpha", "Beta", "Gamma"],
        col_labels=["Alpha", "Beta", "Gamma"],
        game_name="Custom 3-Strategy Game"
    )

    x0 = np.array([0.34, 0.33, 0.33])
    history = simulate(A_custom, x0, steps=6000)
    plot_trajectories(
        history,
        ["Alpha", "Beta", "Gamma"],
        game_name="Custom 3-Strategy Game",
        save_path="outputs/custom_game_trajectory.png"
    )

    run_multi_initial_conditions(
        A_custom,
        ["Alpha", "Beta", "Gamma"],
        game_name="Custom 3-Strategy Game",
        n_trials=12,
        save_path="outputs/custom_game_multi_init.png"
    )

    print("\n✓ All outputs saved to outputs/")
    print("✓ Done.\n")


if __name__ == "__main__":
    main()
