import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler

class PCA():
    def __init__(self, show = True, nrows = None, norm = False, savefig = False, savedata = True):
        self.nrows = nrows
        self.norm = norm
        self.savefig = savefig
        self.show = show
        self.savedata = savedata

        if self.nrows == None:
            self.data_norm = np.loadtxt(file, delimiter = " ")
            self.df_metadata = pd.read_csv(meta_data_file)
        else:
            self.data_norm = np.loadtxt(file, delimiter = " ", max_rows = nrows)
            self.df_metadata = pd.read_csv(meta_data_file, nrows = nrows)

        if self.norm == True: # Normalise data if specified
            self.data_norm = StandardScaler().fit_transform(self.data_norm)
            #self.data_norm = (self.data_norm - np.mean(self.data_norm, axis=0)) / np.std(self.data_norm, axis=0)

        self.classLabels = self.df_metadata["seurat_clusters"] #Classlabels corresponding to each index. One for each neuron
        self.classNames = sorted(set(self.classLabels)) #Set of sorted class names
        self.classDict = dict(zip(self.classNames, range(len(self.classNames))))
        self.y = np.asarray([self.classDict[value] for value in self.classLabels])

        # Perform singular value decomposition (SVD)
        self.U, self.S, self.V = svd(self.data_norm, full_matrices=False)
        self.V = self.V.T
        self.Z = self.data_norm @ self.V
        self.N, self.M = self.data_norm.shape

        if self.savedata == True:
            np.savetxt("PCA_data.csv", self.V[:, :50], delimiter = ",") #Save the first 50 PC's

    def _2d_PCA(self):
        i, j = 0, 1 #Define principle components to be plotted
        for c in range(len(self.classNames)):
            class_mask = (self.y == c)
            #class_mask = np.nonzero(self.y == c)[0].tolist()[0]
            plt.plot(self.Z[class_mask,i],self.Z[class_mask,j], 'o', alpha=.5) # plot the two first principle components.

        title = "2d PCA"
        plt.legend(self.classNames,title="seurat clusters", loc = "best")
        plt.xlabel("PC{0}".format(i+1))
        plt.ylabel("PC{0}".format(j+1))
        if self.savefig == True:
            plt.savefig("2d_PCA")
        if self.show == True:
            plt.show()

    def _3d_PCA(self):
        i, j, k = 0, 1, 2 #Define principle components to be plotted
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.title('PCA')
        for c in range(len(self.classNames)):
            # select indices belonging to class c:
            class_mask = (self.y == c)
            ax.scatter(self.Z[class_mask,i], self.Z[class_mask,j], self.Z[class_mask,k], 'o', alpha=.5)

        title = "3d PCA"
        plt.legend(self.classNames, title= "seurat clusters", loc = "best") #, bbox_to_anchor=(1.01, 1), loc='upper left'
        ax.set_xlabel('PC{0}'.format(i + 1))
        ax.set_ylabel('PC{0}'.format(j + 1))
        ax.set_zlabel('PC{0}'.format(k + 1))
        if self.savefig == True:
            plt.savefig("3d_PCA")
        if self.show == True:
            plt.show()

    def variance_explained(self):
        # Compute variance explained by principal components
        rho = (self.S * self.S) / (self.S * self.S).sum()
        # Plot variance explained
        plt.figure()
        n_components = 10 #Select number of principle components to be plotted
        plt.plot(range(1, n_components + 1), rho[:n_components],'x-')
        plt.plot(range(1, n_components + 1), np.cumsum(rho[:n_components]), 'o-')
        plt.title('Variance explained by principal components')
        plt.xlabel('Principal component');
        plt.ylabel('Variance explained');
        plt.legend(['Individual','Cumulative'])
        plt.grid()
        if self.savefig == True:
            plt.savefig("Variance_explained_PCA")
        if self.show == True:
            plt.show()

if __name__ == "__main__":
    path = "Datasets/Single_cell"
    os.chdir(path)
    file = "raw_neuron_count_matrix.csv"
    meta_data_file = "critical_period_neurons_metadata.csv"

    PCA = PCA(show = True, nrows = 60, norm = False, savefig = True, savedata = True)
    PCA._2d_PCA()
    PCA._3d_PCA()
    PCA.variance_explained()
