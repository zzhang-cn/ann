"""
NN - Batch Normalization

Following Ioffe & Szegedy's 2015 paper
***
Important:
The Batch-Normalization is applied to whatever coming in 
to the hidden layer. Not the output of the layer.
u=g(y) being an activation,
BN applies to the following quantity: BN(Wu), not BN(u).

"""
import numpy as np
import copy
import time

def crossEntropy(x,y):
    return np.sum(-np.log(np.power(x,y)*np.power(1-x,1-y)))
def sigmoid(x): # due to the potential blowing up, x is clipped.
    return 1./(1.+np.exp(-np.clip(x,-100,100)))
def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
def deltaL(preds,labels):
    return preds-labels

#---------------------------------------------------------
class bnN:

    def __init__(self,layers,learnrate,batchsize,epochs,
                 costFn=crossEntropy,actFn=sigmoid,
                 deltaLFn=deltaL,dactFn=dsigmoid):

        # hyperparameters
        self.layers=layers
        self.learnrate=learnrate
        self.batchsize=batchsize
        self.epochs=epochs

        # functions of choosing: Cost, Activation, Regularization
        self.costFn=costFn
        self.actFn=actFn
        self.dactFn=dactFn
        self.deltaLFn=deltaLFn
        
        # weights and bias initialization
        self.weights=[np.random.randn(x,y)/np.sqrt(x+1)
                      for x,y in zip(self.layers[:-1],self.layers[1:])]

        # initialize gamma and beta. excluding the input layer.
        self.gamma=[np.random.randn(x) for x in layers[:-1]]
        self.beta=[np.random.randn(x) for x in layers[:-1]]
        self.invstd=[np.zeros(l) for l in layers[:-1]]
        self.batchmean=[np.zeros(l) for l in layers[:-1]]
        self.batchvar=[np.zeros(l) for l in layers[:-1]]
        # useful quantities when doing BP.
        self.xhats=[np.zeros((self.batchsize,l)) for l in layers[:-1]]
        self.ys=copy.deepcopy(self.xhats)
        self.zs=[np.zeros((self.batchsize,l)) for l in layers[1:]]

        # model_check is to keep track of NN's performance on every E epochs
        # It records accuracy, baseline cost (crossEntropy).
        self.model_check=[]

    def feedforward(self,inputs):
        eps=1.e-15
        x_in=copy.deepcopy(inputs)
        for layer in np.arange(len(self.layers)-1):
            self.invstd[layer]=1./(np.std(x_in,0)+eps)
            xhat=self.bn(x_in,layer)
            y=self.gamma[layer]*xhat+self.beta[layer]
            z=np.dot(y,self.weights[layer])
            x_in=self.actFn(z)
            self.xhats[layer]=xhat
            self.ys[layer]=y
            self.zs[layer]=z

    # Batch Normalization, return xbar.
    def bn(self,batch_inputs,layer):
        eps=1.e-15
        batch_mean=np.mean(batch_inputs,0)
        self.batchmean[layer]+=batch_mean
        batch_var=np.var(batch_inputs,0)
        self.batchvar[layer]+=batch_var
        xhat=[(x-batch_mean)/(np.sqrt(batch_var)+eps) for x in batch_inputs]
        return np.array(xhat)
    

    def sgd(self,inputs,outputs,test_input,test_labels,
            eval_timing=True,inf_check=False):
        total_num_train=len(inputs[:,0])
        num_minibatch,res=divmod(total_num_train,self.batchsize)
        
        if res!=0:
            print "Mini-batch can't be divided by total number of tasks."
            raise SystemExit
        inouts=zip(inputs,outputs)

        for p in np.arange(self.epochs):
            if eval_timing:
                tstart=time.clock()
            np.random.shuffle(inouts)
            rand_trdata=np.array(zip(*inouts)[0])
            rand_trlabel=np.array(zip(*inouts)[1])
            for q in np.arange(num_minibatch):
                batch_data=rand_trdata[q*self.batchsize:(q+1)*self.batchsize]
                batch_label=rand_trlabel[q*self.batchsize:(q+1)*self.batchsize]
                self.batch_update(batch_data,batch_label)
                total_num_batch=q+p*num_minibatch+1
                infer_mean=[mu/total_num_batch for mu in self.batchmean]
                infer_var=[self.batchsize/(self.batchsize+1)*\
                           sb/total_num_batch for sb in self.batchvar]
                # inference
                if inf_check:
                    test_results=self.inference(test_input,test_labels,
                                            infer_mean,infer_var)
                    self.model_check.append(test_results)

            if eval_timing:
                tend=time.clock()
                print "Epoch {0} completed. Time:{1}".format(p,tend-tstart)


    def batch_update(self,inputs,labels):
        self.feedforward(inputs)
        dw,dgamma,dbeta=self.bp(inputs,labels)
        self.weights=[w-self.learnrate*w1/self.batchsize
                      for w,w1 in zip(self.weights,dw)]
        self.gamma=[g-self.learnrate*g1/self.batchsize
                    for g,g1 in zip(self.gamma,dgamma)]
        self.beta=[b-self.learnrate*b1/self.batchsize
                   for b,b1 in zip(self.beta,dbeta)]
        
    def bp(self,inputs,labels):
        deltas=[np.zeros((self.batchsize,l)) for l in self.layers[1:]]
        adj_w=[np.zeros(w.shape) for w in self.weights]
        adj_gamma=[np.zeros(r.shape) for r in self.gamma]
        adj_beta=[np.zeros(b.shape) for b in self.beta]
        
        predictions=self.actFn(self.zs[-1])
        deltas[-1]=self.deltaLFn(predictions,labels)
        adj_w[-1]=sum([np.outer(yy,dd) for yy,dd in zip(self.ys[-1],deltas[-1])])
        adj_gamma[-1]=np.sum(np.dot(deltas[-1],self.weights[-1].T)*self.xhats[-1],0)
        adj_beta[-1]=np.sum(np.dot(deltas[-1],self.weights[-1].T),0)
        for i in xrange(2,len(self.layers)):
            piece1=self.gamma[-i+1]*self.invstd[-i+1]*self.dactFn(self.zs[-i])*\
                    np.dot(deltas[-i+1],self.weights[-i+1].T)
            piece2=np.dot(np.mean(deltas[-i+1],0),self.weights[-i+1].T)*\
                    self.gamma[-i+1]*self.invstd[-i+1]*self.dactFn(self.zs[-i])
            piece3_0=self.xhats[-i+1]*np.dot(deltas[-i+1],self.weights[-i+1].T)
            piece3_1=np.mean(piece3_0,0)*self.gamma[-i+1]*self.invstd[-i+1]
            piece3=piece3_1*self.dactFn(self.zs[-i])*self.xhats[-i+1]
            deltas[-i]=piece1-piece2-piece3
            adj_w[-i]=sum([np.outer(yy,zz) for yy,zz in zip(self.ys[-i],deltas[-i])])
            adj_gamma[-i]=np.sum(np.dot(deltas[-i],self.weights[-i].T)*self.xhats[-i],0)
            adj_beta[-i]=np.sum(np.dot(deltas[-i],self.weights[-i].T),0)
        return (adj_w,adj_gamma,adj_beta)
            

    # inference function return accuracy,cost,regularization costs
    def inference(self,test_input,test_label,gmean,gvar):
        results=copy.deepcopy(test_input)
        for l in xrange(len(self.layers)-1):
            y_in=self.infer_bn(results,gmean,gvar,l)
            results=self.infer_fwd(y_in,l)
            
        res_nn=np.array([np.argmax(x) for x in results])
        label=np.array([np.argmax(x) for x in test_label])
        hits=sum(res_nn==label)
        accuracy=1.0*hits/len(test_input)
        cost=np.mean([self.costFn(x,y)
                      for x,y in zip(results,test_label)])
        return accuracy,cost

    def infer_fwd(self,inputs,layer):
        zs=[np.dot(y,self.weights[layer]) for y in inputs]
        return self.actFn(zs)
        
    def infer_bn(self,inputs,gmean,gvar,l):
        eps=1.e-15
        y=[self.gamma[l]*x/(np.sqrt(gvar[l])+eps)+
               self.beta[l]-self.gamma[l]*gmean[l]/(np.sqrt(gvar[l])+eps)
               for x in inputs]
        return y
