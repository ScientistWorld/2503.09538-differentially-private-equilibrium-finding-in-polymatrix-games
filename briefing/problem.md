# Problem Definition

## Research Question

How can a distributed group of agents compute an approximate equilibrium in a polymatrix game while keeping individual pairwise utility matrices private from an adversary who observes a limited number of communication channels?

## Why It Matters

Polymatrix games model networked multi-agent settings such as trading, security, and local coordination. In these settings, payoffs can reveal private preferences or valuations, so an equilibrium-finding protocol is useful only if it reaches a good equilibrium without exposing those payoffs.

## Success Criteria

A successful solution outputs a low-exploitability coarse correlated equilibrium on random polymatrix games and maintains a decreasing differential-privacy budget as the number of players grows. Evaluation must measure both equilibrium quality and privacy; improving only one side is insufficient.
