# Research Direction

## Research Problem
Improve distributed differentially private equilibrium finding for polymatrix games. The question is how to reduce equilibrium exploitability while preserving privacy against adversaries observing limited communication channels.

## Core Contribution
The relevant contribution space is private iterative equilibrium computation using noisy local communication and graph-aware stabilization. Improvements should address the accuracy/privacy trade-off in this setting rather than switching to a centralized solver.

## Approach Scope
Work within distributed first-order or online-learning style algorithms for polymatrix games. Changes may alter update rules, regularization schedules, noise schedules, averaging, or graph-aware sensitivity control, provided the algorithm still communicates only local noisy strategy information and is evaluated under the same privacy notion.

## Out of Bounds
- Centralized algorithms that inspect all utility matrices directly.
- Methods that remove or weaken the privacy requirement.
- Methods that optimize a different equilibrium notion without reporting CCE exploitability.
- Replacing the game class with unrelated optimization or learning tasks.
