from abc import ABCMeta, abstractmethod
import pandas as pd
from DimensionalityReduction import DimensionalityReduction
import numpy as np

class PCA(DimensionalityReduction):

    def train(self, keep_n_dims = 2):
        # get eigen vectors and save in self.eigenvectors

        n_rows, m_cols = self.data.shape

        X = self.data - self.data.mean()
        cov_mat = np.cov(X, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
        index = np.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:,index]
        eigen_values = eigen_values[index]
        eigen_vectors = eigen_vectors[:, :keep_n_dims]
        self.eigen_vectors = eigen_vectors

        return True
    def convert(self, new_data):
        # Transform data
        return pd.DataFrame(self.eigen_vectors.T.dot(new_data.T).T, index = new_data.index)

    """ 
    convert the given pandas dataframe into a pandas dataframe OR series given a cluster model or dimentionality reduction
    """
