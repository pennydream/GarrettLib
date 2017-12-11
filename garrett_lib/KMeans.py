from abc import ABCMeta, abstractmethod
import pandas as pd
from Clustering import Clustering
import sys
import numpy as np

class KMeans(Clustering):

    def train(self):
        """
        Start with using self.data and k
        find k clusters in data
        return the labels for each point in data and the center points for each cluster
        """
        k = self.k
  
        data = self.data.copy()

        n = data.shape[0]
        m = data.shape[1]
    
        q = np.random.randint(1, n, k)
        data["k_labels"] = np.zeros(n)
        q = data.iloc[q].copy()

    
        for z in range(0, 2):
            for i in range(0, n):
                temp = sys.maxint
                temp_clust = -1
                for j in range(1, k+1):
                    temp_dist = self.distance(q.iloc[j-1].drop(["k_labels"], axis=0),
                                                data.iloc[i].drop(["k_labels"], axis=0))
                    if(temp_dist < temp):
                        temp = temp_dist
                        temp_clust = j

                data.iloc[i]["k_labels"] = temp_clust
            for i in range(1, k+1):
                q.iloc[i-1] = data[data["k_labels"] == i].mean()


        self.cluster_labels = data["k_labels"]
        self.cluster_centers = q

        return True 

    def convert(self, k):
        """ 
        convert the given pandas dataframe into a pandas dataframe OR series given a cluster model or dimentionality reduction
        """
        self.k = k
        if (hasattr(self, 'is_trained')):
            if(self.is_trained):
                return [self.cluster_labels, self.cluster_centers]
            else:
                self.train()
        else:
            self.train()
  
        return [self.cluster_labels, self.cluster_centers]
    
    def distance(self, x, y):
        if(len(x) != len(y)):
            print "PROBLEMS"
            return np.nan
        total = 0
        for i in range(0, len(x)):
            total += np.square(x[i]-y[i])
        return np.sqrt(total/len(x))         
