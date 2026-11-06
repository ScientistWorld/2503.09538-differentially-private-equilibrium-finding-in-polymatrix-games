# Differentially Private Equilibrium Finding in Polymatrix Games

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2503.09538"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2503.09538-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2503.09538](https://arxiv.org/abs/2503.09538) |
| **Domain** | Game Theory / Differential Privacy / Distributed Optimization |
| **TL;DR** | The paper studies when private distributed equilibrium computation is possible and proposes a noisy, degree-regularized CCE algorithm whose accuracy and privacy improve with player count. |

## Summary

The problem is to compute approximate equilibria in polymatrix games without revealing sensitive pairwise utility matrices through the communication process. The paper proves impossibility results for overly strong observation or accuracy requirements, then gives a positive result for exploitability under limited channel access. Its algorithm adds Gaussian noise to strategy broadcasts and uses adaptive regularization based on graph degree. Experiments on random dense and sparse graph topologies evaluate exploitability and privacy-budget scaling.

## Problem

How can a distributed group of agents compute an approximate equilibrium in a polymatrix game while keeping individual pairwise utility matrices private from an adversary who observes a limited number of communication channels?

## The paper's method

The paper introduces a distributed differentially private algorithm for computing coarse correlated equilibria in polymatrix games, with guarantees that both exploitability and privacy budget can vanish as the number of players grows under limited adversarial channel access.

## Key results of the paper

- The proposed algorithm targets coarse correlated equilibrium exploitability, not Euclidean distance to the equilibrium set.
- The privacy metric is a Renyi-DP bound for adjacent games differing on one edge.
- The numerical claim is scaling: exploitability and privacy budget improve with player count in dense graphs.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-sol-xhigh` | +0.1 |
| `gpt-5.6-terra-xhigh` | +0.1 |
| `gpt-5.5-xhigh` | +0.1 |
| `gpt-5.6-luna-xhigh` | +0.1 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2503.09538 --mode innovate --agent claude-code
sw evaluate 2503.09538
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

