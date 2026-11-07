# Research Direction

## Research Problem
Reduce the CCE exploitability that distributed players reach in polymatrix games when every message they exchange is noisy and the privacy budget is fixed.

## Core Contribution
The relevant contribution space is the update rule each player applies to the noisy messages it receives: how it stabilises its own strategy against the noise it injects, how it estimates its payoff gradient from noisy neighbour messages, and how it uses the interaction graph -- all without increasing how strongly its messages depend on its private payoffs.

## Approach Scope
Distributed first-order / online-learning style updates for polymatrix games under the fixed protocol: per-round regularisation towards the uniform strategy, step directions built from current and past received messages (post-processing of public information is free), graph- or degree-aware step rules, and schedules over the fixed number of rounds, provided each update stays within the per-round payoff-dependence allowance.

## Out of Bounds
- Centralised computation that inspects other players' utilities or noiseless strategies.
- Changing the noise, the number of rounds, the instances, or the step allowance.
- Reporting played profiles other than the ones whose noisy versions were broadcast.
- Optimising a different equilibrium notion or selecting runs by their exploitability.
