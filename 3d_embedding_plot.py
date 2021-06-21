import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import csv
import seaborn as sns
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

method = "torch" #set datatype for files "torch" or "csv"

#Importing gene names as a list
with open('Datasets/Single_cell/critical_period_genes.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
data = data[1:]

#Getting index for most common:
idx = []
names = [["Entpd2"], ["Gm4577"], ["Kcnip4"]]
for j in range(len(names)):
    index_active = data.index(names[j])
    idx.append(index_active)

### Load in the embeddings
embeddings_filename_i = "results/embedding_3d/latent_i_10000"
embeddings_filename_j = "results/embedding_3d/latent_j_10000"
if method == "torch":
    data = torch.load(embeddings_filename_i)
    latent_i = data.cpu().data.numpy()
    data = torch.load(embeddings_filename_j)
    latent_j = data.cpu().data.numpy()
else:
    #latent_i = np.genfromtxt(embeddings_filename_i, delimiter = "\n")
    #latent_j = np.genfromtxt(embeddings_filename_j, delimiter = " ")
    latent_i = pd.read_csv(embeddings_filename_i).to_numpy()
    latent_j = pd.read_csv(embeddings_filename_j).to_numpy()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter3D(latent_i[:, 0], latent_i[:, 1], latent_i[:, 2], s=0.2, depthshade=0, cmap="tab10", color="b")
ax.scatter3D(latent_j[:, 0], latent_j[:, 1], latent_j[:, 2], s=0.2, cmap="tab10", color="r")
plt.show()


