"""
The following code is build around the examples from the scanpy documentation:
https://scanpy.readthedocs.io/en/stable/index.html
"""
import scanpy as sc
import pandas as pd
from matplotlib.pyplot import rc_context

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')


adata = sc.read('neur.sub.h5ad')
results_file = 'neuron_sub.h5ad'  # the file that will store the analysis results

#Show those genes that yield the highest fraction of counts in each single cell, across all cells.
sc.pl.highest_expr_genes(adata, n_top=20, save='.pdf', show = False)

### PCA
sc.tl.pca(adata, svd_solver='arpack') # Do a PCA
sc.pl.pca(adata, color = "seurat_clusters", show = False, save = "seurat_clusters.PNG") # Plot it
sc.pl.pca(adata, color = "predicted.id", show = False, save = "predicted.PNG") # Plot it
sc.pl.pca_variance_ratio(adata, show = False, save = ".PNG") # Variance explained for each PC

### UMAP
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
sc.tl.leiden(adata)
sc.tl.paga(adata)
sc.pl.paga(adata, plot = False, title='', save='.PNG', show = False)  # remove `plot=False` if you want to see the coarse-grained graph
sc.tl.umap(adata, init_pos='paga')
#subplot
#sc.pl.umap(adata, color = ["leiden", "predicted.id", "seurat_clusters"], save='.PNG', show = False)

# Individual plots
sc.pl.umap(adata, color = ["leiden"], save='leiden.PNG', show = False)
sc.pl.umap(adata, color = ["predicted.id"], save='predictedid.PNG', show = False)
sc.pl.umap(adata, color = ["seurat_clusters"], save='seaurat_clusters.PNG', show = False)
sc.pl.umap(adata, color = ["age"], save='age.PNG', show = False)
sc.pl.umap(adata, color = ["sex_call"], save='sex.PNG', show = False)
sc.pl.umap(adata, color = ["geno"], save='geno.PNG', show = False)

### Plot clusterings of cells
# compute clusters using the leiden method and store the results with the name `clusters`
sc.tl.leiden(adata, key_added='clusters', resolution=0.5)
with rc_context({'figure.figsize': (5, 5)}):
    sc.pl.umap(adata, color='clusters', add_outline=True, legend_loc='on data',
               legend_fontsize=12, legend_fontoutline=2,frameon=False,
               title='clustering of cells', palette='Set1', save='umap_clustering.PNG', show = False)

# Finding marker genes
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', save='.PNG', show = False)
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save='.PNG', show = False)

# Plot trajectory
#sc.tl.paga(adata, groups = "leiden")
#sc.pl.paga(adata, color = ["leiden"], save='.PNG', show = False)
#sc.tl.draw_graph(adata, init_pos="paga")
#sc.pl.draw_graph(adata, color = ["leiden"], save='.PNG', show = False)

# Save the result
adata.write(results_file)

# Export data
adata.write(results_file, compression='gzip')
adata.write_csvs(results_file[:-5], )
