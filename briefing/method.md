# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

The paper introduces a distributed differentially private algorithm for computing coarse correlated equilibria in polymatrix games, with guarantees that both exploitability and privacy budget can vanish as the number of players grows under limited adversarial channel access.

## Approach

At each iteration, every player broadcasts its mixed strategy plus isotropic Gaussian noise to its neighbors. Neighbors project the noisy broadcasts back to their simplices, compute local polymatrix gradients from these noisy strategies, and update by a Euclidean proximal projected-gradient step. The regularization strength is player-specific: it is proportional to the harmonic mean of graph degrees divided by the player's own degree, so low-degree players receive stronger stabilization.

## Main Claims

- Primary: under limited channel access, the algorithm can achieve low CCE exploitability and low Renyi-DP budget simultaneously as the number of players grows.
- Secondary: this simultaneous improvement holds in dense and sparse graph regimes, with different mechanisms governing the privacy bound.
- Negative result: if an adversary observes all channels, or if Euclidean distance to equilibrium is required instead of exploitability, vanishing accuracy and privacy cannot both be guaranteed.
