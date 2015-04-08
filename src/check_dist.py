import numpy as np
import cPickle as pkl
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
#import statsmodels.api as sm

# with open('vanilla-dist.pickle','rb') as fin:
#     data=pkl.load(fin)

# epochs=data['epoch']
# num_minibatch=data['batch_per_epoch']
# check_freq=data['check_freq']
# layer_1=[]
# layer_2=[]
# for p in np.arange(epochs):
#     for q in np.arange(num_minibatch):
#         num_of_batches=p*num_minibatch+q+1
#         if not num_of_batches%check_freq:
#             key='p'+str(p)+'q'+str(q)+'hl-1'
#             layer_1.append(data[key])
#             key='p'+str(p)+'q'+str(q)+'hl-2'
#             layer_2.append(data[key])

# h_1=np.array(layer_1)
# h_2=np.array(layer_2)
# num_train=len(h_1[:,0,0])
# num_sample=len(h_1[0,:,0])
# num_nodes=len(h_1[0,0,:])

# # the input distribution to a specific node. The node can be randomly chosen.
# var_input_node=[]
# thenode=100
# for nt in np.arange(num_train-1):
#     xdata=h_1[nt,:,thenode-1]
#     ydata=h_1[nt+1,:,thenode-1]
#     var_input_node.append(stats.linregress(xdata,ydata))
# var_input_node=np.array(var_input_node)
# r2=var_input_node[:,2]**2


# # Temporal evolution of correlation between 1 node in last layer
# # to all the nodes in the 2nd-to-last layer
# thenode=1
# corr_layernode=[]
# for nt in np.arange(num_train):
#     xdata=h_1[nt,:,thenode-1]
#     qq_reg_layernode=[stats.linregress(xdata,h_2[nt,:,y]) for y in np.arange(num_nodes)]
#     corr_layernode.append(qq_reg_layernode)
# corr_layernode=np.array(corr_layernode)
# r2=corr_layernode[:,:,2]**2
# stderr=corr_layernode[:,:-1]

# print r2.shape

# figc=plt.figure()
# x=np.arange(num_train)+1
# y=np.arange(num_nodes)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,r2.T,cmap=mpl.cm.spectral,edgecolors='k')
# plt.colorbar(orientation='vertical')
# plt.xlabel('Iteration')
# plt.ylabel('Nodes')
# plt.title('Layer:HL[-2]-Node:HL[-1,{0}] Correlation'.format(thenode))
# #plt.show()

# # Temporal evolution of correlation of input distribution of last hidden layer.
# # The correlation is computed between 2 successive recordings.
# corr_input=[]
# for nt in np.arange(num_train-1):
#     qq_node_corr=[stats.linregress(h_1[nt,:,i],h_1[nt+1,:,i]) for i in np.arange(num_nodes)]
#     corr_input.append(qq_node_corr)
# corr_input=np.array(corr_input)
# r2_nodes=corr_input[:,:,2]**2
# stderr_nodes=corr_input[:,:,-1]

# figc2=plt.figure()
# x=np.arange(num_train-1)+2
# y=np.arange(num_nodes)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,r2_nodes.T,cmap=mpl.cm.spectral,edgecolors='k')
# plt.colorbar(orientation='vertical')
# plt.xlabel('Iteration')
# plt.ylabel('Nodes')
# plt.title('Layer:HL[-1] Correlation between Iterations')
# plt.show()
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

# the input distribution to a specific node. The node can be randomly chosen.
var_input_node=[]
thenode=100
for nt in np.arange(num_train-1):
    xdata=h_1[nt,:,thenode-1]
    ydata=h_1[nt+1,:,thenode-1]
    var_input_node.append(stats.linregress(xdata,ydata))
var_input_node=np.array(var_input_node)
r2=var_input_node[:,2]**2


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

print r2.shape

figc=plt.figure()
x=np.arange(num_train)+1
y=np.arange(num_nodes)+1
X,Y=np.meshgrid(x,y)
plt.pcolormesh(X,Y,r2.T,cmap=mpl.cm.spectral,edgecolors='k')
plt.colorbar(orientation='vertical')
plt.xlabel('Iteration')
plt.ylabel('Nodes')
plt.title('BN: Layer:HL[-2]-Node:HL[-1,{0}] Correlation'.format(thenode))
#plt.show()

# Temporal evolution of correlation of input distribution of last hidden layer.
# The correlation is computed between 2 successive recordings.
corr_input=[]
for nt in np.arange(num_train-1):
    qq_node_corr=[stats.linregress(h_1[nt,:,i],h_1[nt+1,:,i]) for i in np.arange(num_nodes)]
    corr_input.append(qq_node_corr)
corr_input=np.array(corr_input)
r2_nodes=corr_input[:,:,2]**2
stderr_nodes=corr_input[:,:,-1]

figc2=plt.figure()
x=np.arange(num_train-1)+2
y=np.arange(num_nodes)+1
X,Y=np.meshgrid(x,y)
plt.pcolormesh(X,Y,r2_nodes.T,cmap=mpl.cm.spectral,edgecolors='k')
plt.colorbar(orientation='vertical')
plt.xlabel('Iteration')
plt.ylabel('Nodes')
plt.title('BN: Layer:HL[-1] Correlation between Iterations')
plt.show()


# Image Plot (not as nice as Pcolormesh for illustration purpose)
# ax=figc.add_subplot(111)
# ax.set_title('Layer-Node Correlation over Iteration')
# plt.imshow(r2,cmap='hot',aspect='equal',origin='lower')
# ax.set_aspect('equal')
# ax.set_xlabel('Iteration')
# ax.set_ylabel('R2')
# plt.colorbar(orientation='vertical')
# plt.show()

# fig=plt.figure()
# plt.subplot(221)
# plt.plot(r2,'ro',r2,'b')
# plt.title('Correlation of input distribution over Iteration')
# plt.xlabel('Iteration, Unit:{0} minibatch'.format(check_freq))
# plt.ylabel('R2')
# plt.grid()

# stderr_time=var_input_node[:,-1]
# plt.subplot(222)
# plt.plot(stderr_time,'ro',stderr_time,'b')
# plt.title('Correlation of input distribution over Iteration')
# plt.xlabel('Iteration, Unit:{0} minibatch'.format(check_freq))
# plt.ylabel('StdErr of QQ fit')
# plt.grid()

# the input distribution between 2 randomly picked hidden layer nodes.
# thenode1=100
# thenode2=100
# var_corr_nodes=[]
# for nt in np.arange(num_train):
#     xdata=h_1[nt,:,thenode1-1]
#     ydata=h_2[nt,:,thenode2-1]
#     var_corr_nodes.append(stats.linregress(xdata,ydata))
# var_corr_nodes=np.array(var_corr_nodes)
# r2_corr=var_corr_nodes[:,2]**2
# plt.subplot(223)
# plt.plot(r2_corr,'ro',r2_corr,'b')
# plt.title('Correlation of inputs between 2 nodes')
# plt.xlabel('Iteration, Unit:{0} minibatch'.format(check_freq))
# plt.ylabel('R2')
# plt.grid()

# stderr_node=var_corr_nodes[:,-1]
# plt.subplot(224)
# plt.plot(stderr_node,'ro',stderr_node,'b')
# plt.title('Correlation of inputs between 2 nodes')
# plt.xlabel('Iteration, Unit:{0} minibatch'.format(check_freq))
# plt.ylabel('StdErr of QQ fit')
# plt.grid()
# plt.show()



