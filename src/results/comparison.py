import numpy as np
import json
import matplotlib.pyplot as plt

with open("nn-vanilla-deep.json") as baseline_file:
    data=json.load(baseline_file)
accu_base=np.array(data['accuracy_epoch'])
loss_base=np.array(data['crossEntropy_epoch'])

with open("nn-batchnorm-deep.json") as bn_file:
    data=json.load(bn_file)
accu_bn=np.array(data['accuracy_epoch'])
loss_bn=np.array(data['crossEntropy_epoch'])

diff=accu_bn-accu_base
thres=np.zeros(len(diff))
plt.figure(1)
plt.plot(diff,'ro',diff,'b',thres,'r')
plt.grid(True)
plt.text(40,0.7,"learningrate:\n BN: 0.2,\n Baseline:0.02")
plt.xlabel('# of Iterations (Unit {0} batches)'.format(50))
plt.ylabel('Accuracy: BN-Baseline')
plt.savefig('DeepNet_Compare.png')
plt.show()

