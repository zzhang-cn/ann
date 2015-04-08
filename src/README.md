##Batch Normalization and Baseline (Vanilla) Neural Net

References:
[Ioffe & Szegedy, 2015](http://arxiv.org/abs/1502.03167)
[Michael Neilsen's tutorial website](http://neuralnetworksanddeeplearning.com/)

Two versions of NN were created. Baseline and BN.
Tests are done on MNIST data set. 10000 training samples are used to train the network,
10000 test samples are used for testing. No cross validation/dropouts/momentum method
were used.

Cost function is crossEntropy, following Neilsen's tutorial.
Activation function is Sigmoid, which makes computing the growthrate at the edge easier.

First layer of NN is not used to activate inputs. Rather it either let inputs singal pass
through (Baseline) or whiten the input (BN). If the first layer's activate signal, the
performance is generally worse.

For BN, it is not clear how many *training* mini-batches need to be used for computing
population stats, such as means and variances that are important for inference. In the code,
I employed boostrap to compute the stats. The total amount of sub-sample batches covers 2
complete epochs.

Trained NN's parameters, setups, results are recorded in .json files.
---------------------------------------------------------------------
Modifications:
1) Accuracy check is commented out. Instead, the last Hidden layer and the 2nd-to-last Hidden layer
   are recorded in *.pickle* file.

   The *(bn_)corr_layernode.png* shows the evolutoin of correlation between one node in last hidden
   layer HL[-1,n] and all the nodes in the previous hidden layer HL[-2,:]. *R2* being the square of
   Pearson's correlation coefficient. The intuition origins from Q-Q plot of two samples. If two
   samples share similar distribution, the linear regression should be a good fit, hence the R2 value	will be close to 1. x-axis is number of iterations, with unit being 20 minibatches.
   Each square in a column of the plot represent a neuron.

   The *(bn_)corr_input.png* shows the evolution of correlation between the successive recorded input
   to a specific hidden layer neuron.

   To run the code, make sure ** number of hidden layer >=2 **.
   ```
   python vanilla.py
   python batchnorm.py
   python check_dist.py
   ```