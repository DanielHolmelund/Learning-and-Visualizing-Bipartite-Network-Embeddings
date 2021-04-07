# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 10:16:19 2020

@author: nnak

"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
import torch
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

#torch.set_default_tensor_type('torch.cuda.FloatTensor')



# Create embeddings
X, y = make_blobs(n_samples=np.repeat(100,10), n_features=2)


# Create plot
'''plt.scatter(X[:, 0], X[:, 1], c=y, s= 30000 / len(X), cmap="tab10")
    #plt.axis([0,1,0,1]) ; plt.tight_layout()
plt.title('True latent variables')
plt.xlabel('z1')
plt.ylabel('z2')

plt.show()'''

H=pdist(X, metric='euclidean')




def generate_network_bias(H,X,graph_type='undirected'):
    ''' Generate adj matrix, Undirected case
            H: Unique Pairwise Distances
            X: Latent variables vector
    '''
    z_pdist=H
        
    triu_indices = torch.triu_indices(row=X.shape[0], col=X.shape[0], offset=1)
    
    logit_u=-z_pdist
    #Generate zero count probability
    inv_logit=torch.exp(-torch.exp(logit_u))
    #sample from Uniform [0,1]
    print(inv_logit)
    print(inv_logit.shape)
    adj_u=torch.rand(inv_logit.shape)
    #
    adj_u[adj_u<inv_logit]=0
    #Insert the links
    adj_u[adj_u!=0]=1
    adj_m = torch.zeros((X.shape[0],X.shape[0]))

    # Create Upper Triangulat Matrix
    adj_m[triu_indices[0], triu_indices[1]] = adj_u

    return adj_m
           



adj_m=generate_network_bias(torch.from_numpy(H).float().to(device),torch.from_numpy(X).float().to(device))

adj_m=adj_m.cpu().data.numpy()
full_rank=adj_m+adj_m.transpose()
edges=(full_rank==1).sum()
X=torch.from_numpy(X).float().to(device)
print(adj_m)
asa=1


