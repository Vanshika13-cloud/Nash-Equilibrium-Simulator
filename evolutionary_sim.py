"""
evolutionary_sim.py
-------------------
Module 2: Evolutionary Game Dynamics Simulator

Simulates the replicator dynamic — a model from evolutionary game theory
describing how a population's strategy distribution evolves over time.

Core idea:
  - A large population plays a symmetric game repeatedly.
  - Each player's strategy is a "type" (e.g. Cooperate, Defect).
  - Strategies that earn above-average payoff grow in frequency.
  - Strategies that earn below-average payoff shrink.
  - The system converges to a Nash Equilibrium.

The replicator equation (continuous time):
  dx_i/dt = x_i * (f_i(x) - phi(x))

Where:
  x_i   = frequency of strategy i in the population
  f_i   = expected payoff of strategy i against current population x
  phi   = average payoff of the population = sum_i x_i * f_i
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation


def replicator_step(x: np.ndarray, A: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """
    Advance the replicator dynamic by one time step (Euler method).

    f_i(x) = (A @ x)[i]        — expected payoff of strategy i
    phi(x) = x^T @ A @ x       — mean population payoff
    dx_i   = x_i * (f_i - phi) * dt

    Parameters
    ----------
    x  : Current population state (probability vector over strategies)
    A  : Symmetric payoff matrix
    dt : Time step size

    Returns
    -------
    x_new : Updated population state (renormalized to sum to 1)
    """
    f = A @ x               # Fitness of each strategy
    phi = x @ f             # Mean population fitness
    dx = x * (f - phi)      # Replicator equation
    x_new = x + dx * dt
    x_new = np.clip(x_new, 0, None)          # No negative frequencies
    x_new = x_new / x_new.sum()              # Renormalize
    return x_new


def simulate(A: np.ndarray, x0: np.ndarray,
             steps: int = 5000, dt: float = 0.01) -> np.ndarray:
    """
    Run the full replicator dynamic simulation.

    Parameters
    ----------
    A     : Symmetric payoff matrix (n x n)
    x0    : Initial population distribution (n,) — must sum to 1
    steps : Number of time steps
    dt    : Step size

    Returns
    -------
    history : np.ndarray of shape (steps+1, n)
               Population state at each time step.
    """
    n = len(x0)
    history = np.zeros((steps + 1, n))
    history[0] = x0
    x = x0.copy()

    for t in range(steps):
        x = replicator_step(x, A, dt)
        history[t + 1] = x

    return history


def plot_trajectories(history: np.ndarray, strategy_labels: list,
                      game_name: str = "Evolutionary Dynamics",
                      save_path: str = None):
    """
    Plot strategy frequencies over time.
    """
    steps, n = history.shape
    time = np.arange(steps) * 0.01

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Replicator Dynamic: {game_name}", fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, n))

    # Left: frequency over time
    ax1 = axes[0]
    for i in range(n):
        ax1.plot(time, history[:, i], label=strategy_labels[i],
                 color=colors[i], linewidth=2)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Strategy Frequency")
    ax1.set_title("Population Frequencies Over Time")
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Mark final state
    final = history[-1]
    for i in range(n):
        ax1.axhline(y=final[i], color=colors[i], linestyle='--', alpha=0.4)

    # Right: stacked area chart
    ax2 = axes[1]
    ax2.stackplot(time, history.T, labels=strategy_labels,
                  colors=colors, alpha=0.8)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Proportion of Population")
    ax2.set_title("Population Composition (Stacked)")
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    plt.close()
    return final


def run_multi_initial_conditions(A: np.ndarray, strategy_labels: list,
                                  game_name: str, n_trials: int = 8,
                                  steps: int = 5000, save_path: str = None):
    """
    Run simulation from many random starting points to visualize the
    basin of attraction — which initial conditions converge to which NE.

    Parameters
    ----------
    A            : Payoff matrix
    strategy_labels : Names of strategies
    game_name    : Title for plot
    n_trials     : Number of random starting distributions
    steps        : Simulation length
    save_path    : Optional file path to save figure
    """
    n = A.shape[0]
    np.random.seed(42)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Multiple Initial Conditions: {game_name}",
                 fontsize=13, fontweight='bold')

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_trials))

    for trial in range(n_trials):
        # Random Dirichlet start (uniform over simplex)
        x0 = np.random.dirichlet(np.ones(n))
        history = simulate(A, x0, steps=steps)

        for i in range(n):
            ax.plot(np.arange(steps + 1) * 0.01, history[:, i],
                    color=colors[trial], alpha=0.5,
                    linewidth=1.2,
                    label=f"Trial {trial+1}" if i == 0 else "")

    ax.set_xlabel("Time")
    ax.set_ylabel("Strategy Frequency")
    ax.set_title(f"Strategies: {', '.join(strategy_labels)}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right', ncol=2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ── Symmetric payoff matrices for evolutionary games ─────────────────────────

# Hawk-Dove: classic model of conflict.
# Hawk always fights; Dove always retreats.
# Mixed NE: stable coexistence at some Hawk frequency.
HAWK_DOVE = {
    "A": np.array([[0, 4],
                   [1, 2]], dtype=float),
    "labels": ["Hawk", "Dove"],
    "name": "Hawk-Dove"
}

# Rock-Paper-Scissors: symmetric zero-sum.
# Replicator orbits around the mixed NE (1/3, 1/3, 1/3) — never converges.
RPS = {
    "A": np.array([[ 0, -1,  1],
                   [ 1,  0, -1],
                   [-1,  1,  0]], dtype=float),
    "labels": ["Rock", "Paper", "Scissors"],
    "name": "Rock-Paper-Scissors"
}

# Stag Hunt: coordination game.
# Two pure NE: (Stag, Stag) and (Hare, Hare).
# Replicator converges to one depending on initial conditions.
STAG_HUNT = {
    "A": np.array([[4, 0],
                   [3, 3]], dtype=float),
    "labels": ["Stag", "Hare"],
    "name": "Stag Hunt"
}


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    for game in [HAWK_DOVE, STAG_HUNT, RPS]:
        A = game["A"]
        labels = game["labels"]
        name = game["name"]
        n = A.shape[0]

        print(f"\nSimulating: {name}")

        # Single run from equal starting frequencies
        x0 = np.ones(n) / n
        history = simulate(A, x0, steps=5000)
        final = history[-1]

        print(f"  Starting distribution: {dict(zip(labels, x0.round(3)))}")
        print(f"  Final distribution:    {dict(zip(labels, final.round(4)))}")

        plot_trajectories(history, labels, game_name=name,
                          save_path=f"outputs/{name.replace(' ', '_')}_trajectory.png")

        run_multi_initial_conditions(A, labels, game_name=name,
                          save_path=f"outputs/{name.replace(' ', '_')}_multi_init.png")
