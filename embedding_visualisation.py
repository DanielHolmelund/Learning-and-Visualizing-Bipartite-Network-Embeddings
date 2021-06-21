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
embeddings_filename_i = "results/embedding/latent_i_5000"
embeddings_filename_j = "results/embedding/latent_j_5000"
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


plt.scatter(latent_i[:, 0], latent_i[:, 1], s=0.2, cmap="tab10", color="b")
for i in range(len(idx)):
    plt.annotate(names[i], (latent_i[idx[i], 0], latent_i[idx[i], 1]), fontsize = 18)
plt.scatter(latent_j[:, 0], latent_j[:, 1], s=0.2, cmap="tab10", color="r")
#plt.savefig("Scatterplot")
plt.show()

### Plot genes with color map
df = pd.read_csv("Datasets/Single_cell/dtudata/var_metadata_subdata.csv")
data_mean = df.loc[:, "vst.mean"].to_numpy()

# remove the outliers from the plot
#idx_max = data_mean.argsort()[-150:][::-1]
#data_mean = np.delete(data_mean, idx_max)
latent_i0, latent_i1 = latent_i[:, 0], latent_i[:, 1]
#latent_i0 = np.delete(latent_i0, idx_max)
#latent_i1 = np.delete(latent_i1, idx_max)
cmap = sns.color_palette("viridis", as_cmap = True)

f, ax = plt.subplots()
points = ax.scatter(latent_i0, latent_i1, s=30000 / len(latent_j), c = data_mean, cmap = cmap)
f.colorbar(points)
plt.show()


### Preprocessing of metadata
df_metadata = pd.read_csv("Datasets/Single_cell/dtudata/metadata_subdata.csv")
#cate = Categorization(df)
# Class for plotting
#class_name = "seurat_clusters"
# Get the column index for the attribute in question
#idx_attribute = df.columns.get_loc(class_name)
#meta_data = cate.encoder()
#meta_data = meta_data[:, idx_attribute]

classLabels = df_metadata["age"]  # Classlabels corresponding to each index. One for each neuron
classLabels = classLabels[:]
classLabels = classLabels[:]
classNames = sorted(set(classLabels))  # Set of sorted class names
classDict = dict(zip(classNames, range(len(classNames))))
y = np.asarray([classDict[value] for value in classLabels])


f, ax = plt.subplots()
points = ax.scatter(latent_j[:, 0], latent_j[:, 1], c = classLabels, s = 0.2, cmap = cmap)
f.colorbar(points)
plt.show()




"""
for c in range(len(classNames)):
    class_mask = (y == c)
    cmap = plt.get_cmap("viridis", len(classNames))
    plt.plot(100*latent_j[class_mask, 0], 100*latent_j[class_mask, 1], 'o', alpha=.5, markersize = .2)
#plt.xlim(-0.5,0.5)
#plt.ylim(-0.5,0.5)
cmap = plt.get_cmap("viridis", len(classNames))
sm = plt.cm.ScalarMappable(cmap = cmap)

plt.colorbar(sm, label = "age")
#plt.legend(classNames, title="age", loc="best")
plt.show()
"""

