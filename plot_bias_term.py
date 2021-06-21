"""
PLot the latent embedding together with the bias values.
"""
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
iteration = 28000
embeddings_filename_i = f"results/embedding/latent_i_{iteration}"
embeddings_filename_j = f"results/embedding/latent_j_{iteration}"
beta_file = f"results/embedding/beta_{iteration}"
gamma_file = f"results/embedding/gamma_{iteration}"
if method == "torch":
    data = torch.load(embeddings_filename_i)
    latent_i = data.cpu().data.numpy()
    data = torch.load(embeddings_filename_j)
    latent_j = data.cpu().data.numpy()
    data = torch.load(beta_file)
    beta = data.cpu().data.numpy()
    data = torch.load(gamma_file)
    gamma = data.cpu().data.numpy()

else:
    #latent_i = np.genfromtxt(embeddings_filename_i, delimiter = "\n")
    #latent_j = np.genfromtxt(embeddings_filename_j, delimiter = " ")
    latent_i = pd.read_csv(embeddings_filename_i).to_numpy()
    latent_j = pd.read_csv(embeddings_filename_j).to_numpy()

cmap = sns.color_palette("viridis", as_cmap = True)
f, ax = plt.subplots()
points = ax.scatter(latent_i[:, 0], latent_i[:, 1], s=0.2, c = beta, cmap = cmap)
f.colorbar(points)
plt.show()

f, ax = plt.subplots()
points = ax.scatter(latent_j[:, 0], latent_j[:, 1], s=0.2, c = gamma, cmap = cmap)
f.colorbar(points)
plt.show()
