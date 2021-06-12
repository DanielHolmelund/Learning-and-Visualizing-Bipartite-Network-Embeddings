
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import torch
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#torch.set_default_tensor_type('torch.cuda.FloatTensor')


np.random.seed(10)
torch.manual_seed(10)
# Create embeddings
X1, y1 = make_blobs(n_samples=np.repeat(100,10), n_features=2)
X2, y2 = make_blobs(n_samples=np.repeat(50,10), n_features=2)

#Add random effects

'''# Create plot
plt.scatter(X1[:, 0], X1[:, 1], s= 30000 / len(X1), cmap="tab10", color="b")
plt.scatter(X2[:, 0], X2[:, 1], s= 30000 / len(X1), cmap="tab10", color="r")
    #plt.axis([0,1,0,1]) ; plt.tight_layout()
plt.title('True latent variables')
plt.xlabel('z1')
plt.ylabel('z2')

plt.show()'''


def generate_network_bias(X1,X2,graph_type='undirected'):
    ''' Generate adj matrix, Undirected case
            H: Unique Pairwise Distances
            X: Latent variables vector
    '''
    #Adding node bias

    #Generate distance between nodes across partitions
    H = (((torch.unsqueeze(X1, 1) - X2) ** 2).sum(-1)) ** 0.5

    z_pdist = H

    beta = 3
    gamma = 2.5
    logit_u = beta + gamma - z_pdist

    #Get the rate for the poisson sampling
    rate = torch.exp(logit_u)


    #Sample using the rate discovered before with seed 0
    adj_m = torch.poisson(rate)

    return adj_m



adj_m=generate_network_bias(torch.from_numpy(X1).float().to(device),
                            torch.from_numpy(X2).float().to(device))
adj_m=adj_m.cpu().data.numpy()




