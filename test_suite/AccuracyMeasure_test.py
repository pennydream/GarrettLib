import pytest
import pandas as pd

# Import the class to be tested
import sys
sys.path.append("../garrett_lib/")
from AccuracyMeasure import AccuracyMeasure

# Create a realization of the AccuracyMeasure class that can be used for testing
class Helper_AccuracyMeasure(AccuracyMeasure):
    def getAccuracy(self):
        return True

# Get init and data stored in AccuracyMeasure class 
def test_AccuracyMeasure_init():
    """
    Given a pandas dataframe, test the creation of a model class.
    """
    some_true = pd.Series([
         1,2,3
         ])

    some_test = pd.Series([ 
         4,5,6
         ])

    m = Helper_AccuracyMeasure(some_true, some_test)

    data_1 = m.getDataTrue()
    assert some_true.equals(data_1)
   
    data_2 = m.getDataTest()
    assert some_test.equals(data_2) 

def test_AccuracyMeasure_dtype():
    """
    Test that the initialization of a AccuracyMeasure class throws a type error for 
    things that are not pandas dataframes
    """
    some_true = "A wrong data type of type string" 
    some_test = 5 #another wrong data type
    with pytest.raises(TypeError):
        AccuracyMeasure(some_true, some_test)    
