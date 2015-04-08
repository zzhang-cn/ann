import numpy as np
import cPickle as pkl
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
#import statsmodels.api as sm

with open('vanilla-dist.pickle','rb') as fin:
    data=pkl.load(fin)

epochs=data['epoch']
num_minibatch=data['batch_per_epoch']
check_freq=data['check_freq']
layer_1=[]
layer_2=[]
for p in np.arange(epochs):
    for q in np.arange(num_minibatch):
        num_of_batches=p*num_minibatch+q+1
        if not num_of_batches%check_freq:
            key='p'+str(p)+'q'+str(q)+'hl-1'
            layer_1.append(data[key])
            key='p'+str(p)+'q'+str(q)+'hl-2'
            layer_2.append(data[key])

h_1=np.array(layer_1)
h_2=np.array(layer_2)
num_train=len(h_1[:,0,0])
num_sample=len(h_1[0,:,0])
num_nodes=len(h_1[0,0,:])

# Temporal evolution of correlation between 1 node in last layer
# to all the nodes in the 2nd-to-last layer
thenode=1
corr_layernode=[]
for nt in np.arange(num_train):
    xdata=h_1[nt,:,thenode-1]
    qq_reg_layernode=[stats.linregress(xdata,h_2[nt,:,y]) for y in np.arange(num_nodes)]
    corr_layernode.append(qq_reg_layernode)
corr_layernode=np.array(corr_layernode)
r2=corr_layernode[:,:,2]**2
stderr=corr_layernode[:,:-1]

figc=plt.figure()
x=np.arange(num_train)+1
y=np.arange(num_nodes)+1
X,Y=np.meshgrid(x,y)
plt.pcolormesh(X,Y,r2.T,cmap=mpl.cm.spectral,edgecolors='k')
plt.colorbar(orientation='vertical')
plt.xlabel('Iteration. Unit: {0} batches'.format(check_freq))
plt.ylabel('Nodes')
plt.title('Same Init. Layer:HL[-2]-Node:HL[-1,{0}] Correlation'.format(thenode))

# Temporal evolution of correlation of input distribution of last hidden layer.
# The correlation is computed between 2 successive recordings.
corr_input=[]
for nt in np.arange(num_train-1):
    qq_node_corr=[stats.linregress(h_1[nt,:,i],h_1[nt+1,:,i]) for i in np.arange(num_nodes)]
    corr_input.append(qq_node_corr)
corr_input=np.array(corr_input)
r2=corr_input[:,:,2]**2
stderr_nodes=corr_input[:,:,-1]

figc2=plt.figure()
x=np.arange(num_train-1)+2
y=np.arange(num_nodes)+1
X,Y=np.meshgrid(x,y)
plt.pcolormesh(X,Y,r2.T,cmap=mpl.cm.spectral,edgecolors='k')
plt.colorbar(orientation='vertical')
plt.xlabel('Iteration. Unit: {0} batches'.format(check_freq))
plt.ylabel('Nodes')
plt.title('Same Init. Input (HL[-1]) Correlation for successive recordings')

#-----------------------------------------------------------------------
# BN
#-----------------------------------------------------------------------
with open('bn-dist.pickle','rb') as fin:
    data=pkl.load(fin)

epochs=data['epoch']
num_minibatch=data['batch_per_epoch']
check_freq=data['check_freq']
layer_1=[]
layer_2=[]
for p in np.arange(epochs):
    for q in np.arange(num_minibatch):
        num_of_batches=p*num_minibatch+q+1
        if not num_of_batches%check_freq:
            key='p'+str(p)+'q'+str(q)+'hl-1'
            layer_1.append(data[key])
            key='p'+str(p)+'q'+str(q)+'hl-2'
            layer_2.append(data[key])

h_1=np.array(layer_1)
h_2=np.array(layer_2)
num_train=len(h_1[:,0,0])
num_sample=len(h_1[0,:,0])
num_nodes=len(h_1[0,0,:])

# Temporal evolution of correlation between 1 node in last layer
# to all the nodes in the 2nd-to-last layer
thenode=1
corr_layernode=[]
for nt in np.arange(num_train):
    xdata=h_1[nt,:,thenode-1]
    qq_reg_layernode=[stats.linregress(xdata,h_2[nt,:,y]) for y in np.arange(num_nodes)]
    corr_layernode.append(qq_reg_layernode)
corr_layernode=np.array(corr_layernode)
r2=corr_layernode[:,:,2]**2
stderr=corr_layernode[:,:-1]

figc3=plt.figure()
x=np.arange(num_train)+1
y=np.arange(num_nodes)+1
X,Y=np.meshgrid(x,y)
plt.pcolormesh(X,Y,r2.T,cmap=mpl.cm.spectral,edgecolors='k')
plt.colorbar(orientation='vertical')
plt.xlabel('Iteration. Unit: {0} batches'.format(check_freq))
plt.ylabel('Nodes')
plt.title('BN(Same Init): Layer:HL[-2]-Node:HL[-1,{0}] Correlation'.format(thenode))

# Temporal evolution of correlation of input distribution of last hidden layer.
# The correlation is computed between 2 successive recordings.
corr_input=[]
for nt in np.arange(num_train-1):
    qq_node_corr=[stats.linregress(h_1[nt,:,i],h_1[nt+1,:,i]) for i in np.arange(num_nodes)]
    corr_input.append(qq_node_corr)
corr_input=np.array(corr_input)
r2=corr_input[:,:,2]**2
stderr_nodes=corr_input[:,:,-1]

figc4=plt.figure()
x=np.arange(num_train-1)+2
y=np.arange(num_nodes)+1
X,Y=np.meshgrid(x,y)
plt.pcolormesh(X,Y,r2.T,cmap=mpl.cm.spectral,edgecolors='k')
plt.colorbar(orientation='vertical')
plt.xlabel('Iteration. Unit: {0} batches'.format(check_freq))
plt.ylabel('Nodes')
plt.title('BN(Same Init): Input (HL[-1]) Correlation for successive recordings')
plt.show()




