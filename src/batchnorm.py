"""
Batch Normalization for Neural Network
This code follows closely with Ioffe & Szegedy's 2015 paper.
Normalization is applied at each layer, including the input layer to whiten
inputs. Parameters, such as weights, gammas, betas are trained through
BackPropagation. At inference stage, an averaged mean and var are computed
to allow transformation of test inputs at each layer to form Gaussian.
The accuracy and costs are also computed on epoch basis to examine the 
algorithm's performance. 

"""
import data_loader as dl
import network_bn as network
import pdb
import numpy as np
import json
import matplotlib.pyplot as plt

if __name__=="__main__":
    t_img,t_label=dl.training_load()
    s_img,s_label=dl.test_load()

    numoftrains=10000
    trains=zip(t_img,t_label)
    np.random.shuffle(trains)
    t_in=np.array(zip(*trains[:numoftrains])[0])
    t_la=np.array(zip(*trains[:numoftrains])[1])

    numoftests=10000
    tests=zip(s_img,s_label)
    np.random.shuffle(tests)
    s_in=np.array(zip(*tests[:numoftests])[0])
    s_la=np.array(zip(*tests[:numoftests])[1])

    #layers=[28*28,100,100,100,10]
    layers=[28*28,100,100,100,10]
    learnrate=10.
    batchsize=50
    epochs=50
    check_freq=50
    checknum=np.arange(epochs*numoftrains/batchsize/check_freq)+1
    checks_per_epoch=numoftrains/batchsize/check_freq
    
    bnn=network.bnN(layers,learnrate,batchsize,epochs,costFn='crossEntropy')
    bnn.sgd(t_in,t_la,s_in,s_la,inf_check=True,check_freq=check_freq)
    
    model_val=np.array(bnn.model_check)
    print model_val
    
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(checknum,model_val[:,0],'ro',checknum,model_val[:,0],'b',
             checknum[checks_per_epoch-1::checks_per_epoch],
             model_val[checks_per_epoch-1::checks_per_epoch,0],'go')
    plt.ylabel('Accuracy')
    plt.xlabel('# of minibatches (Unit:{0})'.format(check_freq))
    plt.title('BatchNormalization')
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(checknum,model_val[:,1],'ro',checknum,model_val[:,1],'b')
    plt.ylabel('crossEntropy loss')
    plt.grid()
    plt.savefig('nn-batchnorm-deep-long.png')
    plt.show()

    data={"number_of_trains":numoftrains,
          "number_of_tests":numoftests,
          "layers":layers,
          "learnrate":learnrate,
          "mini-batch size":batchsize,
          "epochs":epochs,
          "weights":[x.tolist() for x in bnn.weights],
          "gammas":[x.tolist() for x in bnn.gammas],
          "betas":[x.tolist() for x in bnn.betas],
          "accuracy_epoch":model_val[:,0].tolist(),
          "crossEntropy_epoch":model_val[:,1].tolist(),
    }
    with open("nn-batchnorm-deep-long.json",'w') as f0:
        json.dump(data,f0)

