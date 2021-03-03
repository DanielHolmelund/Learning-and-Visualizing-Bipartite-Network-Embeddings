
'''
from scipy.io import mmread
import os

direc = "C:/Users/Daniel/Documents/DTU/Bipartite_Visualizing_Embeddings/Datasets"
os.chdir(direc)

f = open("divorce_colname.txt")
g = open("divorce_rowname.txt")
rows = g.read()
cols = f.read()
f.close()
g.close()

data = mmread("divorce.mtx")
data = data.toarray()

rows = rows.split("\n")
cols = cols.split("\n")
cols = cols[0:-2]


x = 1 '''

import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
# SE https://github.com/eliorc/node2vec/blob/master/README.md for documentation af funktioner.

# Create a graph
#graph = nx.fast_gnp_random_graph(n=100, p=0.5)
graph = nx.algorithms.bipartite.generators.gnmk_random_graph(7, 9, 8, seed=1)



# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
node2vec = Node2Vec(graph, dimensions=2, walk_length=1, num_walks=200, workers=4)  # Use temp_folder for big graphs

# Embed nodes
model = node2vec.fit(window=9, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
model.wv.most_similar('2')  # Output node names are always strings

# Save embeddings for later use
#model.wv.save_word2vec_format("EMBEDDING_FILENAME")

# Save model for later use
#model.save("EMBEDDING_MODEL_FILENAME")

# Embed edges using Hadamard method
from node2vec.edges import HadamardEmbedder

edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

# Look for embeddings on the fly - here we pass normal tuples
edges_embs[('1', '2')]

for i in range(len(edges_embs)):


''' OUTPUT
array([ 5.75068220e-03, -1.10937878e-02,  3.76693785e-01,  2.69105062e-02,
       ... ... ....
       ..................................................................],
      dtype=float32)
'''

# Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
edges_kv = edges_embs.as_keyed_vectors()

# Look for most similar edges - this time tuples must be sorted and as str
edges_kv.most_similar(str(('1', '2')))

# Save embeddings for later use
#edges_kv.save_word2vec_format("EDGES_EMBEDDING_FILENAME")

x = 1

