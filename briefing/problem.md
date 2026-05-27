# Problem Definition

What problem does this paper address? Write this so that someone with no knowledge of the paper can understand what needs to be solved and how success is measured.

Do NOT describe the paper's method or solution here — only the problem itself. Think about if a future scientist agent try to solve this problem from scratch without knowing what approach the paper took. If you leak the solution, you compromise the benchmark.

## Research Question

How can a distributed group of agents compute an approximate equilibrium in a polymatrix game while keeping individual pairwise utility matrices private from an adversary who observes a limited number of communication channels?

## Why It Matters

Polymatrix games model networked multi-agent settings such as trading, security, and local coordination. In these settings, payoffs can reveal private preferences or valuations, so an equilibrium-finding protocol is useful only if it reaches a good equilibrium without exposing those payoffs.

## Success Criteria

A successful solution should output a low-exploitability coarse correlated equilibrium on random polymatrix games and should maintain a decreasing differential-privacy budget as the number of players grows. The evaluation must measure both equilibrium quality and privacy; improving only one side is insufficient.
