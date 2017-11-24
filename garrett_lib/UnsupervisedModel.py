from abc import ABCMeta, abstractmethod
import pandas as pd
from Model import Model

class UnsupervisedModel(Model):

    @abstractmethod
    def convert(self):pass
    """ 
    convert the given pandas dataframe into a pandas dataframe OR series given a cluster model or dimentionality reduction
    """
