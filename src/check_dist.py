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
r2=corr_layernode[:,:,2]
stderr=corr_layernode[:,:-1]

# figc=plt.figure()
# x=np.arange(num_train)+1
# y=np.arange(num_nodes)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,r2.T,cmap=mpl.cm.spectral)
# plt.colorbar(orientation='vertical')
# plt.xlabel('Iteration. Unit: {0} batches'.format(check_freq))
# plt.ylabel('Nodes')
# plt.title('Same Init. Layer:HL[-2]-Node:HL[-1,{0}] Correlation'.format(thenode))

# rdata=r2[10,:]
# p_rdata=np.sort(rdata[rdata>0.])
# n_rdata=np.sort(rdata[rdata<=0.])
# rdata=np.sort(rdata)
# print len(p_rdata),len(n_rdata)
figt_bl=plt.figure()
plt.suptitle('Correlations of Inputs to Sigmoid between Node[-1,1] to Node[-2,:]')
prange=np.arange(5,21,3)
plegend=[]
for i in prange:
    plegend.append('Iter={0}'.format(i))

plt.subplot(1,2,1)
for i in prange:
    plt.plot(np.sort(r2[i,:]))
plt.legend(plegend,loc='upper left')
plt.grid()
plt.xlabel('Nodes')
plt.ylabel('r')
plt.title('Baseline')
# Temporal evolution of correlation of input distribution of last hidden layer.
# The correlation is computed between 2 successive recordings.
# corr_input=[]
# for nt in np.arange(num_train-1):
#     qq_node_corr=[stats.linregress(h_1[nt,:,i],h_1[nt+1,:,i]) for i in np.arange(num_nodes)]
#     corr_input.append(qq_node_corr)
# corr_input=np.array(corr_input)
# r2=corr_input[:,:,2]
# stderr_nodes=corr_input[:,:,-1]

# figc2=plt.figure()
# x=np.arange(num_train-1)+2
# y=np.arange(num_nodes)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,r2.T,cmap=mpl.cm.spectral)
# plt.colorbar(orientation='vertical')
# plt.xlabel('Iteration. Unit: {0} batches'.format(check_freq))
# plt.ylabel('Nodes')
# plt.title('Same Init. Input (HL[-1]) Correlation for successive recordings')


# #-----------------------------------------------------------------------
# # BN
# #-----------------------------------------------------------------------
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
r2=corr_layernode[:,:,2]
stderr=corr_layernode[:,:-1]

# figc3=plt.figure()
# x=np.arange(num_train)+1
# y=np.arange(num_nodes)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,r2.T,cmap=mpl.cm.spectral,edgecolors='k')
# plt.colorbar(orientation='vertical')
# plt.xlabel('Iteration. Unit: {0} batches'.format(check_freq))
# plt.ylabel('Nodes')
# plt.title('BN(Same Init): Layer:HL[-2]-Node:HL[-1,{0}] Correlation'.format(thenode))

plt.subplot(1,2,2)
for i in prange:
    plt.plot(np.sort(r2[i,:]))
plt.legend(plegend,loc='upper left')
plt.grid()
plt.xlabel('Nodes')
plt.ylabel('r')
plt.title('BN')
# # Temporal evolution of correlation of input distribution of last hidden layer.
# # The correlation is computed between 2 successive recordings.
# corr_input=[]
# for nt in np.arange(num_train-1):
#     qq_node_corr=[stats.linregress(h_1[nt,:,i],h_1[nt+1,:,i]) for i in np.arange(num_nodes)]
#     corr_input.append(qq_node_corr)
# corr_input=np.array(corr_input)
# r2=corr_input[:,:,2]**2
# stderr_nodes=corr_input[:,:,-1]

# figc4=plt.figure()
# x=np.arange(num_train-1)+2
# y=np.arange(num_nodes)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,r2.T,cmap=mpl.cm.spectral,edgecolors='k')
# plt.colorbar(orientation='vertical')
# plt.xlabel('Iteration. Unit: {0} batches'.format(check_freq))
# plt.ylabel('Nodes')
# plt.title('BN(Same Init): Input (HL[-1]) Correlation for successive recordings')
# plt.show()

# with open('lrate_adj.pickle','rb') as fin:
#     lrate_adj=pkl.load(fin)
# hl1_adj=np.array([x[0] for x in lrate_adj])
# hl2_adj=np.array([x[1] for x in lrate_adj])
# hl3_adj=np.array([x[2] for x in lrate_adj])

# numrec=len(hl1_adj[:,0])
# numnodeshl1=len(hl1_adj[0,:])
# numnodeshl2=len(hl2_adj[0,:])
# numnodeshl3=len(hl3_adj[0,:])

# h1stats=np.array([[np.mean(x),np.std(x)] for x in hl1_adj])
# h2stats=np.array([[np.mean(x),np.std(x)] for x in hl2_adj])
# h3stats=np.array([[np.mean(x),np.std(x)] for x in hl3_adj])

# figc6=plt.figure()

# mean=h1stats[:,0]
# ubd=h1stats[:,0]+h1stats[:,1]
# lbd=h1stats[:,0]-h1stats[:,1]
# x=np.arange(len(mean))+1
# plt.subplot(131)
# plt.plot(x,mean,'r',x,ubd,'b',x,lbd,'b')
# plt.fill_between(x,ubd,lbd,color='grey',alpha='0.5')
# plt.title('HL1')
# plt.ylabel(r'mean and std of $\gamma/\sqrt{\sigma_B^2}$')
# plt.xlabel('Iter')
# plt.grid()

# mean=h2stats[:,0]
# ubd=h2stats[:,0]+h2stats[:,1]
# lbd=h2stats[:,0]-h2stats[:,1]
# x=np.arange(len(mean))+1
# plt.subplot(132)
# plt.plot(x,mean,'r',x,ubd,'b',x,lbd,'b')
# plt.fill_between(x,ubd,lbd,color='grey',alpha='0.5')
# plt.title('HL2')
# plt.ylabel(r'mean and std of $\gamma/\sqrt{\sigma_B^2}$')
# plt.xlabel('Iter')
# plt.grid()

# mean=h3stats[:,0]
# ubd=h3stats[:,0]+h3stats[:,1]
# lbd=h3stats[:,0]-h3stats[:,1]
# x=np.arange(len(mean))+1
# plt.subplot(133)
# plt.plot(x,mean,'r',x,ubd,'b',x,lbd,'b')
# plt.fill_between(x,ubd,lbd,color='grey',alpha='0.5')
# plt.title('HL3')
# plt.ylabel(r'mean and std of $\gamma/\sqrt{\sigma_B^2}$')
# plt.xlabel('Iter')
# plt.grid()

# figc7=plt.figure()
# plt.subplot(331)
# plt.plot(hl1_adj[-1,:],'ro',hl1_adj[-1,:],'b')
# plt.grid()
# plt.subplot(332)
# plt.plot(hl2_adj[-1,:],'ro',hl2_adj[-1,:],'b')
# plt.grid()
# plt.subplot(333)
# plt.plot(hl3_adj[-1,:],'ro',hl3_adj[-1,:],'b')
# plt.grid()
# plt.subplot(334)
# plt.plot(hl1_adj[-50,:],'ro',hl1_adj[-50,:],'b')
# plt.grid()
# plt.subplot(335)
# plt.plot(hl2_adj[-50,:],'ro',hl2_adj[-50,:],'b')
# plt.grid()
# plt.subplot(336)
# plt.plot(hl3_adj[-50,:],'ro',hl3_adj[-50,:],'b')
# plt.grid()
# plt.subplot(337)
# plt.plot(hl1_adj[-100,:],'ro',hl1_adj[-100,:],'b')
# plt.grid()
# plt.subplot(338)
# plt.plot(hl2_adj[-100,:],'ro',hl2_adj[-100,:],'b')
# plt.grid()
# plt.subplot(339)
# plt.plot(hl3_adj[-100,:],'ro',hl3_adj[-100,:],'b')
# plt.grid()


# plt.subplot(132)
# plt.plot(h2stats[0],'r',h2stats[0]+h2stats[1],'b',h2stats[0]-h2stats[1],'b')
# plt.grid('True')

# plt.subplot(133)
# plt.plot(h3stats[0],'r',h3stats[0]+h3stats[1],'b',h3stats[0]-h3stats[1],'b')
# plt.grid('True')

plt.show()

# figc5=plt.figure()

# plt.subplot(131)
# y=np.arange(numrec)+1
# x=np.arange(numnodeshl1)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,hl1_adj,cmap=mpl.cm.spectral,edgecolors='k')
# #ax.set_ylabel('Iteration') 
# #ax.set_xlabel('Nodes')
# plt.title('HL1')
# plt.colorbar(orientation='horizontal')
# #plt.show()

# plt.subplot(132)
# y=np.arange(numrec)+1
# x=np.arange(numnodeshl2)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,hl2_adj,cmap=mpl.cm.spectral,edgecolors='k')
# #plt.colorbar(orientation='horizontal')
# #ax.set_ylabel('Iteration') 
# #ax.set_xlabel('Nodes')
# plt.title('HL2')
# plt.colorbar(orientation='horizontal')

# plt.subplot(133)
# y=np.arange(numrec)+1
# x=np.arange(numnodeshl3)+1
# X,Y=np.meshgrid(x,y)
# plt.pcolormesh(X,Y,hl3_adj,cmap=mpl.cm.spectral,edgecolors='k')
# #plt.colorbar(orientation='horizontal')
# #ax.ylabel('Iteration') 
# #ax.set_xlabel('Nodes')
# plt.title('HL3')
# plt.colorbar(orientation='horizontal')
# plt.show()





