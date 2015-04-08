**Basic Test bed for All kinds of ANN**
1) Examine the evolution of correlations between
   a) the input distribution of a specific node at successive recorded iterations.
   b) the correlation between one arbitrary node in the last hidden layer and the
      2nd to last layer.
   BN achieves better correlation for input distribution, hence the stable distribution.
   It is interesting to see that BN also stabilize the correlation between nodes at early
   stage, while for Baseline, there is a strong mixing, which may contribute to fluctuation
   of the percentile distribution shown by Ioffe&Szegedy. Such strong mixing may cause slow
   down of learning.