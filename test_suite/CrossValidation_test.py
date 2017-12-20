import pytest
import pandas as pd
import numpy as np

# Import the class to be tested
import sys
sys.path.append("../garrett_lib/")
from CrossValidation import CrossValidation

# Get init and data stored in CrossValidation class
def test_CrossValidation_init():
    """
    Given a pandas dataframe, test the creation of a supervised model class.
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = CrossValidation(some)
    data_2 = m.getData()
    assert some.equals(data_2) 

def test_CrossValidation_dtype():
    """
    Test that the initialization of a supervised Model class throws a type error for 
    things that are not pandas dataframes
    """
    some = "A wrong data type of type string" 
    with pytest.raises(TypeError):
        CrossValidation(some)   

def test_CrossValidation_train():
    """
    Test that CrossValidation has a working train abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = CrossValidation(some)
    assert m.train()


def test_CrossValidation_test():
    """
    Test that CrossValidation has a working test abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = CrossValidation(some)
    assert m.test()

def test_CrossValidation_results():
    """
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    labels = pd.Series([
        1,2,3,4
        ])

    m = CrossValidation(some)
    from LinearRegression import LinearRegression
    accs = m.run(LinearRegression, labels,task="regression", cv=2)
    assert accs.shape[0] == 2

