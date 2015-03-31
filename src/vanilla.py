import data_loader as dl
import network as network
import numpy as np
import matplotlib.pyplot as plt
import json

if __name__=="__main__":
    t_img,t_label=dl.training_load()
    s_img,s_label=dl.test_load()

    numoftrains=10000
    trains=zip(t_img,t_label)
    #np.random.shuffle(trains)
    t_in=np.array(zip(*trains[:numoftrains])[0])
    t_la=np.array(zip(*trains[:numoftrains])[1])
    
    numoftests=10000
    tests=zip(s_img,s_label)
    #np.random.shuffle(tests)
    s_in=np.array(zip(*tests[:numoftests])[0])
    s_la=np.array(zip(*tests[:numoftests])[1])

    #layers=[28*28,100,100,100,10]
    layers=[28*28,100,100,100,10]
    learnrate=1.
    batchsize=50
    epochs=50
    check_freq=50
    checknum=np.arange(epochs*numoftrains/batchsize/check_freq)+1
    checks_per_epoch=numoftrains/batchsize/check_freq
    
    ann=network.Vnn(layers,learnrate,batchsize,epochs)
    ann.sgd(t_in,t_la,s_in,s_la,check=True,check_freq=check_freq)
    accu=np.array(ann.accuracy)
    cost=np.array(ann.cost)

    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(checknum,accu,'ro',checknum,accu,'b',
            checknum[checks_per_epoch-1::checks_per_epoch],
                     accu[checks_per_epoch-1::checks_per_epoch],'go')
    plt.ylabel('Accuracy')
    plt.xlabel('# of minibatches (Unit:{0})'.format(check_freq))
    plt.title('Baseline NN.')
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(checknum,cost,'ro',checknum,cost,'b')
    plt.ylabel('crossEntropy loss')
    plt.grid()
    plt.savefig('nn-vanilla-deep-long.png')
    plt.show()
                                    

    data={"number_of_trains":numoftrains,
          "number_of_tests":numoftests,
          "layers":layers,
          "learnrate":learnrate,
          "mini-batch size":batchsize,
          "epochs":epochs,
          "weights":[x.tolist() for x in ann.weights],
          "bias":[x.tolist() for x in ann.bias],
          "accuracy_epoch":accu.tolist(),
          "crossEntropy_epoch":cost.tolist(),
              }
    with open("nn-vanilla-deep-long.json",'w') as f0:
                json.dump(data,f0)
