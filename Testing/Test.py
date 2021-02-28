
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


x = 1