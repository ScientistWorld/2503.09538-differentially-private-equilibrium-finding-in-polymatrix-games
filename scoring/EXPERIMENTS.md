# Experiment Overview

All three experiments hold the privacy side fixed: every submission runs on the same games, for the same number of communication rounds, with the same broadcast noise and the same bound on how strongly each message may depend on private payoffs. What is measured is how close to a coarse correlated equilibrium the played profiles get under that fixed privacy budget, compared with a fixed private comparator at the same budget and with uniform play, across player counts 64 to 512.

## dense_er_scaling

Random dense interaction graphs (every pair of players interacts with probability 0.12). Establishes whether a method reaches lower exploitability than the comparator at equal privacy on dense networks, and how this holds up as the number of players grows -- the setting of the paper's main experiment.

## dense_cluster_scaling

Dense graphs organised into tightly connected clusters with few links between clusters. Checks that the dense-network behaviour is not specific to homogeneous random graphs.

## sparse_knn_scaling

Sparse geometric graphs in which each player interacts with its four nearest neighbours, with only two or three communication rounds available. This is the hard regime: the paper reports that exploitability barely improves here, so it measures whether a method can make equilibrium progress when each player sees few, very noisy messages.
