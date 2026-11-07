# Problem Definition

## Research Question

How can a distributed group of agents compute an approximate equilibrium of a polymatrix game while keeping each agent's pairwise utility matrices private from an adversary who observes a limited number of communication channels?

## Setting

- **Game.** N players on an interaction graph; each player has 5 actions. Every edge (i, j) carries utility matrices U_ij, U_ji with entries in [-1, 1]. A player's payoff is the average over its neighbours of pi_i^T U_ij pi_j. Each player knows only its own matrices.
- **Communication.** Play proceeds in T rounds. In round t every player broadcasts one message to its neighbours: the mixed strategy it is currently playing plus Gaussian noise (before round 0 every player plays the uniform strategy). An adversary may read the messages on a bounded number of channels. Two games are *adjacent* if they differ only in the matrices of one edge; privacy means the observed messages are nearly equally likely under adjacent games (Renyi differential privacy).
- **Privacy operating point (fixed).** For every instance the benchmark fixes the number of rounds T, the noise level sigma, and a step scale eta (`data/polymatrix.benchmark_hyperparameters`). The noise realisation itself is also fixed per instance (`data/polymatrix.broadcast_noise`), so all methods are compared under common noise. Within a round, how far a player's next strategy may depend on its private payoffs, beyond what the public messages already determine, is bounded in proportion to eta (see `briefing/evaluation.md`). Together these fix the privacy budget: every compliant method is equally private, and the question is how good an equilibrium it reaches.
- **Output.** The played profiles of the T rounds define an empirical distribution; its quality is its coarse-correlated-equilibrium (CCE) exploitability: the average over players of the gain from switching to the best single fixed strategy against that distribution.

## Instances

Three graph families with N in {64, 128, 256, 512} and 12 public instance seeds each (`data/polymatrix.benchmark_instances`):

- dense Erdos-Renyi graphs (edge probability 0.12),
- dense clustered graphs (8 cliques plus sparse cross-cluster edges),
- sparse geometric graphs (each player linked to its 4 nearest neighbours in the unit square).

Games are generated procedurally; nothing needs to be downloaded.

## Why It Matters

Polymatrix games model networked multi-agent settings such as trading, security and local coordination. Payoffs can reveal private valuations, so an equilibrium-finding protocol is useful only if it reaches a good equilibrium without exposing them through its messages.

## Success Criteria

Lower CCE exploitability than a fixed private comparator that runs at the same privacy budget, on every graph family and player count, while every transcript remains a valid execution of the private protocol. Improving exploitability by communicating more, adding less noise, or letting the messages depend more strongly on private payoffs is not an improvement: such transcripts fail.
