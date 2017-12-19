import pytest
import pandas as pd

# Import the class to be tested
import sys
sys.path.append("../garrett_lib/")
from ComputeAccuracy import ComputeAccuracy

# Test getAccuracy method of ComputeAccurucy class for regression tasks... returns a float
def test_ComputeAccuracy_getAccRegression():
    """
    Given two series of true values and model results, get the root mean squared error 
    """
    some_true = pd.Series([
         10, 10.5, 10.1, 10.0 
         ])

    some_test = pd.Series([ 
         10, 10, 10, 10
         ])

    m = ComputeAccuracy(some_true, some_test)

    assert round(m.getAccuracy("regression"), 3) ==  0.255 

# Test getAccuracy method of ComputeAccurucy class for classification tasks... returns a float
def test_ComputeAccuracy_getAccClasses():
    """
    Given a pandas dataframe, test the creation of a model class.
    """
    some_true = pd.Series([
         0, 1, 1, 0 
         ])

    some_test = pd.Series([ 
         0, 1, 1, 1
         ])

    m = ComputeAccuracy(some_true, some_test)

    assert m.getAccuracy("classification") == 0.75 


# Get init and data stored in ComputeAccuracy class 
def test_ComputeAccuracy_init():
    """
    Given a pandas dataframe, test the creation of a model class.
    """
    some_true = pd.Series([
         1,2,3
         ])

    some_test = pd.Series([ 
         4,5,6
         ])

    m = ComputeAccuracy(some_true, some_test)

    data_1 = m.getDataTrue()
    assert some_true.equals(data_1)
   
    data_2 = m.getDataTest()
    assert some_test.equals(data_2) 

def test_ComputeAccuracy_dtype():
    """
    Test that the initialization of a ComputeAccuracy class throws a type error for 
    things that are not pandas dataframes
    """
    some_true = "A wrong data type of type string" 
    some_test = 5 #another wrong data type
    with pytest.raises(TypeError):
        ComputeAccuracy(some_true, some_test)    
