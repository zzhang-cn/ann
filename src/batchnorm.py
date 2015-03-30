import data_loader as dl
import network_bn as network
import pdb
import numpy as np
#import json
import matplotlib.pyplot as plt

if __name__=="__main__":
    t_img,t_label=dl.training_load()
    s_img,s_label=dl.test_load()

    numoftrains=10000
    tr_data=t_img[0:numoftrains]
    tr_label=t_label[0:numoftrains]
    val_data=t_img[numoftrains:]
    val_label=t_label[numoftrains:]

    layers=[28*28,40,20,10]
    learnrate=0.5
    batchsize=100
    epochs=10

    numoftests=1000
    tests=zip(s_img,s_label)
    np.random.shuffle(tests)
    s_in=zip(*tests[:numoftests])[0]
    s_la=zip(*tests[:numoftests])[1]
    
    bnn=network.bnN(layers,learnrate,batchsize,epochs)
    bnn.sgd(tr_data,tr_label,s_in,s_la,inf_check=True)
    
    model_val=np.array(bnn.model_check)
    print model_val
    epochnum=np.arange(epochs*numoftrains/batchsize)
    
    plt.figure(1)
    #plt.subplot(1,3,1)
    plt.plot(epochnum,model_val[:,0],'ro',epochnum,model_val[:,0],'b')
    plt.ylabel('Accuracy')
    plt.grid()
    # plt.subplot(1,3,2)
    # plt.plot(epochnum,model_val[:,1],'ro',epochnum,model_val[:,1],'b')
    # plt.ylabel('crossEntropy')
    # plt.grid()
    # plt.subplot(1,3,3)
    # plt.plot(epochnum,model_val[:,2],'ro',epochnum,model_val[:,2],'b')
    # plt.ylabel('penalty')
    # plt.grid()
    plt.show()

    # data={"layers":layers,
    #       "learnrate":learnrate,
    #       "penalty":penalty,
    #       "mini-batch size":batchsize,
    #       "epochs":epochs,
    #       "weights":[x.tolist() for x in ann.weights],
    #       "bias":[x.tolist() for x in ann.bias],
    #       "accuracy_epoch":model_val[:,0].tolist(),
    #       "crossEntropy_epoch":model_val[:,1].tolist(),
    #       "penalty_epoch":model_val[:,2].tolist(),
    # }
    # with open("nn_regularization.json",'w') as f0:
    #     json.dump(data,f0)

    #results=ann.inference(s_img,s_label)
    #results=ann.inference(s_img,s_label)
    #zvals=(np.array(ann.zval)).reshape(epochs,numoftrains/batchsize,len(layers)-1)
    
    #pdb.set_trace()
    # print "\nCharacteristic of this NN:"
    # print " layers:{0}\n learn_rate:{1:.2f}\n mini_batch:{2}\n epochs:{3}\n".format(layers,learnrate,minibatch,epoch)
    # print "Accuracy of this NN:{0:.6f}".format(hits)
    #print results
