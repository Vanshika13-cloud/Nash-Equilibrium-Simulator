# Nash Equilibrium Finder + Evolutionary Game Simulator

A Python project bridging **static game theory** (solving for Nash Equilibria mathematically) with **evolutionary dynamics** (simulating how populations converge to those equilibria over time).

---

## The Core Idea

A **Nash Equilibrium** is a strategy profile where no player can improve their outcome by unilaterally changing their strategy. This project approaches NE from two angles:

| Approach | Method | Question answered |
|---|---|---|
| **Static** | Linear programming (support enumeration) | *Where are the equilibria?* |
| **Dynamic** | Replicator equation simulation | *How does a population get there?* |

The mathematical connection: Nash Equilibria are exactly the **fixed points of the replicator dynamic**. A population at a NE has zero evolutionary pressure to change.

---

## Project Structure

```
nash_project/
│
├── nash_solver.py        # Module 1 — Nash Equilibrium Finder
│   ├── find_pure_nash()      Pure NE via best-response analysis
│   ├── find_mixed_nash()     Mixed NE via support enumeration
│   ├── compute_expected_payoffs()
│   └── print_game_summary()  Pretty-print full game analysis
│
├── evolutionary_sim.py   # Module 2 — Replicator Dynamic Simulator
│   ├── replicator_step()     One Euler step of dx_i/dt = x_i(f_i - phi)
│   ├── simulate()            Full trajectory simulation
│   ├── plot_trajectories()   Frequency over time + stacked area chart
│   └── run_multi_initial_conditions()   Basin of attraction mapping
│
├── main.py               # Runner — ties both modules together
└── outputs/              # Generated PNG plots
```

---

## Installation

```bash
pip install numpy scipy matplotlib nashpy
```

---

## Usage

```bash
python main.py
```

---

## Games Analyzed

### Module 1 — Asymmetric Games (Nash Solver)

**Prisoner's Dilemma**
- Unique pure NE: (Defect, Defect) — even though (Cooperate, Cooperate) is Pareto superior
- Illustrates why rational self-interest leads to collectively suboptimal outcomes

**Battle of the Sexes**
- Two pure NE: (Opera, Opera) and (Football, Football)
- One mixed NE: players randomize with specific probabilities
- Illustrates coordination problems with conflicting preferences

**Rock-Paper-Scissors**
- No pure NE
- Unique mixed NE: uniform distribution (1/3, 1/3, 1/3)

### Module 2 — Symmetric Games (Evolutionary)

**Hawk-Dove**
- Models animal conflict over a resource
- Mixed NE = evolutionarily stable strategy (ESS): stable coexistence of Hawks and Doves

**Stag Hunt**
- Two pure NE with different risk profiles
- Replicator converges to different equilibria depending on initial conditions — illustrates **basins of attraction**

**Rock-Paper-Scissors**
- Replicator *orbits* around the mixed NE without converging — illustrates limit cycles in evolutionary dynamics

---

## Key Math

**Replicator Equation (continuous time):**

$$\dot{x}_i = x_i \left( f_i(\mathbf{x}) - \bar{f}(\mathbf{x}) \right)$$

Where:
- $x_i$ = frequency of strategy $i$ in population
- $f_i(\mathbf{x}) = (A\mathbf{x})_i$ = expected payoff of strategy $i$
- $\bar{f}(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ = mean population payoff

**Mixed Strategy NE condition:**

A probability vector $\sigma^*$ is a NE if for all strategies $i$ in the support:

$$\sum_j \sigma^*_j \cdot A_{ij} = \text{constant}$$

i.e., all strategies in the support yield equal expected payoff.

---

## Skills Demonstrated

- Game theory (Nash Equilibria, mixed strategies, evolutionary stability)
- Linear programming via `scipy` / `nashpy`
- Numerical ODE simulation (Euler method)
- Data visualization (`matplotlib`)
- Object-oriented and modular Python design
