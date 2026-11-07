# Research Direction

## Research Problem
Reduce the CCE exploitability that distributed players reach in polymatrix games when every message they exchange is noisy and the privacy budget is fixed.

## Core Contribution
The relevant contribution space is the update rule each player applies to the noisy messages it receives. This covers how it stabilises its own strategy against the noise it injects, how it estimates its payoff gradient from noisy neighbour messages, and how it uses the interaction graph. None of this may increase how strongly its messages depend on its private payoffs.

## Approach Scope
Distributed first-order / online-learning style updates for polymatrix games under the fixed protocol:
- per-round regularisation towards the uniform strategy;
- step directions built from current and past received messages (post-processing of public information is free);
- graph- or degree-aware step rules;
- schedules over the fixed number of rounds.

Each update must stay within the per-round payoff-dependence allowance and use only the player's own view and received messages.

## Out of Bounds
- Centralised computation, or any sharing of information between players outside the broadcast messages.
- Reading other players' utilities, noiseless strategies or future noise; reading evaluator code, data or process state.
- Changing the noise, the number of rounds, the instances, or the step allowance.
- Optimising a different equilibrium notion or selecting runs by their exploitability.
