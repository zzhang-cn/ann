**Batch Normalization and Baseline (Vanilla) Neural Net**

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

**Run the code**
Code is written in python 2.7.9
Numpy and Matplotlib are required to run the codes.
To examine the BN algorithm, simply open a terminal and run
```
python batchnorm.py
```
To examine the Baseline algorithm, run
```
python vanilla.py
```
The pyplot generated img will be automatically saved as .png file, while the data,
including network configuration and final results, will be saved as .json file.