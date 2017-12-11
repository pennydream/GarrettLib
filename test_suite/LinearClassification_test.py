import pytest
import pandas as pd
import numpy as np

# Import the class to be tested
import sys
sys.path.append("../garrett_lib/")
from LinearClassification import LinearClassification

# Get init and data stored in LinearClassification class
def test_LinearClassification_init():
    """ Given a pandas dataframe, test the creation of a regression class.  """
    some = pd.DataFrame([
         [1,2,3],
         [4,5,6],
         [7,8,9],
         [10,11,12]
         ])

    m = LinearClassification(some)
    data_2 = m.getData()
    assert some.equals(data_2) 

def test_LinearClassification_dtype():
    """
    Test that the initialization of a regression class throws a type error for 
    things that are not pandas dataframes
    """
    some = "A wrong data type of type string" 
    with pytest.raises(TypeError):
        LinearClassification(some)   

def test_LinearClassification_train():
    """
    Test that regression has a working train abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [1,5,6],
         [7,8,9],
         [10,11,12]
         ])

    some_2 = pd.DataFrame([
         1.0,-1,1,-1
         ])
    m = LinearClassification(some)
    assert m.train(some_2)


def test_LinearClassification_test():
    """
    Test that regression has a working test abstract method
    """
    some = pd.DataFrame([
         [1,2,3],
         [1,1,3],
         [8,8,11],
         [10,11,12]
         ])

    some_2 = pd.DataFrame([
         1.0,1,-1,-1
         ])

    m = LinearClassification(some)
    m.train(some_2, num_iters=20)
    test = m.test(some)

    acc = 0
    for i in range(3):
        acc += (test[i] == some_2[0][i])
    assert acc > 0.5

    #print test[0]
    #print some_2[0]
    #print some_2[0].equals(test[0])

    #    for i in range(test.shape[0]):
    #        print test[i], some_2[i]
    
    #assert test[0].equals(some_2[0])


