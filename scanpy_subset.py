"""
Use scanpy to create a subset of the data consisting of the most active genes

"""
import scanpy as sc
import pandas as pd

results_file = 'neuron_raw_subset.h5ad'  # the file that will store the analysis results
filename = "critical_period_neurons_norm_counts.mtx"  # directory and filename
adata = sc.read(filename, cache = True)

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

t=adata.raw.X.toarray()
pd.DataFrame(data=t, index=adata.obs_names, columns=adata.raw.var_names).to_csv('adata_raw_x.csv')

