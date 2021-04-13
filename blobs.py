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

np.random.seed(0)

# Create embeddings
X1, y1 = make_blobs(n_samples=np.repeat(100,10), n_features=2)
X2, y2 = make_blobs(n_samples=np.repeat(100,5), n_features=2)

# Create plot
'''plt.scatter(X1[:, 0], X1[:, 1], c=y1, s= 30000 / len(X), cmap="tab10")
    #plt.axis([0,1,0,1]) ; plt.tight_layout()
plt.title('True latent variables')
plt.xlabel('z1')
plt.ylabel('z2')

plt.show()'''

H1=pdist(X1, metric='euclidean')
H2=pdist(X2, metric='euclidean')


def generate_network_bias(H1,H2,X1,X2,graph_type='undirected'):
    ''' Generate adj matrix, Undirected case
            H: Unique Pairwise Distances
            X: Latent variables vector
    '''
    z_pdist_1=H1
    z_pdist_2 = H2
        
    triu_indices = torch.triu_indices(row=X1.shape[0], col=X2.shape[0], offset=1)


    logit_u_1=-z_pdist_1
    logit_u_2 = -z_pdist_2
    #Generate zero count probability
    inv_logit_1=torch.exp(-torch.exp(logit_u_1))
    inv_logit_2 = torch.exp(-torch.exp(logit_u_2))
    #sample from Uniform [0,1]

    adj_u_1=torch.rand(inv_logit_1.shape)

    adj_u_2 = torch.rand(inv_logit_2.shape)
    #

    #SOMETHING BETTER THAN MEAN?
    adj_u_1[adj_u_1<torch.mean(inv_logit_2)]=0
    adj_u_2[adj_u_2<torch.mean(inv_logit_1)]=0
    #Insert the links
    #TRUE EDGES?
    adj_u_1[adj_u_1!=0]=1
    adj_u_2[adj_u_2!=0]=1
    print((adj_u_1==1).sum())
    print(adj_u_2.shape)
    print(adj_u_1.shape)

    adj_m = torch.zeros((X1.shape[0],X2.shape[0]))

    # Create Upper Triangulat Matrix
    #WHY DOESNT IT WORK?!
    adj_m[triu_indices[0],triu_indices[1]] = adj_u_1[0]
    adj_m[triu_indices[1],triu_indices[0]] = adj_u_2[0]
    print(adj_m[triu_indices[1],triu_indices[0]].shape)
    return adj_m
           



adj_m=generate_network_bias(torch.from_numpy(H1).float().to(device),
                            torch.from_numpy(H2).float().to(device),
                            torch.from_numpy(X1).float().to(device),
                            torch.from_numpy(X2).float().to(device))

adj_m=adj_m.cpu().data.numpy()
print((adj_m==1).sum())
full_rank=adj_m+adj_m.transpose()
edges=(full_rank==1).sum()
print(edges)
print(adj_m.shape)
asa=1


