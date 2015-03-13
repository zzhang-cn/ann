import data_loader as dl
import network
import pdb

if __name__=="__main__":
    t_img,t_label=dl.training_load()
    s_img,s_label=dl.test_load()

    numoftrains=50000
    tr_data=t_img[0:numoftrains]
    tr_label=t_label[0:numoftrains]
    val_data=t_img[numoftrains:]
    val_label=t_label[numoftrains:]
    
    layers=[28*28,7*7,25,10]
    learnrate=4.0
    minibatch=50
    epoch=10
    vnn=network.Vnn(layers,learnrate,minibatch,epoch)
    vnn.sgd(tr_data,tr_label)
    hits=vnn.pred(s_img,s_label)
    print "\nCharacteristic of this NN:"
    print " layers:{0}\n learn_rate:{1:.2f}\n mini_batch:{2}\n epochs:{3}\n".format(layers,learnrate,minibatch,epoch)
    print "Accuracy of this NN:{0:.6f}".format(hits)
