import torch
import matplotlib.pyplot as plt


class LatentSpace():
    def __init__(self):


    def


    def embedding(self, vector, method):
        """

        :param vector:
        :param method:
        :return:
        """
        if method == "UMAP":

            #mapper = umap.UMAP(metric = "", random_state = SEED).fit(data)

            reducer = umap.UMAP(random_state = SEED)
            embedding = reducer.transform(data)

            return embedding_vector

    def y(self):
        # blackbox for now


    def poisson(self, u, v):

        # torch.logit

        #Gamma distribution
        #m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))

        # sample(sample_shape=torch.Size([]))
        # or sample_n(n)??
        #m.sample()  # Gamma distributed with concentration=1 and rate=1



        p = 2 # p-norm
        theta[i][j] = torch.exp(gamma[i] + alpha[j] - torch.dist(u[i], v[j], p))
        return



    def scatter(self, u, v):
        plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
        plt.show




