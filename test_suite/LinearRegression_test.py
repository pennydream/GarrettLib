import pytest
import pandas as pd
import numpy as np

# Import the class to be tested
import sys
sys.path.append("../garrett_lib/")
from LinearRegression import LinearRegression

# Get init and data stored in LinearRegression class
def test_LinearRegression_init():
    """ Given a pandas dataframe, test the creation of a regression class.  """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = LinearRegression(some)
    data_2 = m.getData()
    assert some.equals(data_2) 

def test_LinearRegression_dtype():
    """
    Test that the initialization of a regression class throws a type error for 
    things that are not pandas dataframes
    """
    some = "A wrong data type of type string" 
    with pytest.raises(TypeError):
        LinearRegression(some)   

def test_LinearRegression_train():
    """
    Test that regression has a working train abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    some_2 = pd.DataFrame([
         1,2,3
         ])
    m = LinearRegression(some)
    assert m.train(some_2)


def test_LinearRegression_test():
    """
    Test that regression has a working test abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    some_2 = pd.DataFrame([
         1.0,2.0,3.0
         ])

    m = LinearRegression(some)
    m.train(some_2)
    test = m.test(some)

    for i in range(3):
        assert round(test[0][i], 3) == some_2[0][i]

    print test[0]
    print some_2[0]
    #print some_2[0].equals(test[0])

    #    for i in range(test.shape[0]):
    #        print test[i], some_2[i]
    
    #assert test[0].equals(some_2[0])


