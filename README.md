# GarrettLib
A Basic Machine Learning Library from Scratch

## Set up

GarrettLib is written in Python (2.7.13). It requires a handful of common Python libraries, which can easily be installed using pip, anaconda, or any other Python library manager. For brevity, I will assume pip is installed on your PC. If it is not, it can be installed here: https://pip.pypa.io/en/stable/installing/
  
The following libraries are used in GarrettLib:
-  Numpy 0.19.2
-  Pandas 1.12.1
-  PyTest 3.0.5
-  Jupyter 1.0.0

The latest version of these libraries can all be installed by running the following command: 

    pip install numpy
    pip install pandas
    pip install pytest
    pip install jupyter
    
Numpy and pandas are useful for linear algebra computations as well as data frame storage and manipulation. PyTest is used for testing and Jupyter is used to display tutorials and examples. 



## Testing

Testing of GarrettLib is done using PyTest. Test classes are written in GarrettLib/test_suite. Run the following commands to run all tests for GarrettLib:

    cd ../GarrettLib/test_suite
    pytest
    
When pytest is run, all tests will run. It will show you what number of tests ran, passed, failed, and why they failed. Note: the pytest command must be run inside the test_suite directory. 


## Jupyter Use & Tutorials

Jupyter is a web application that can run locally on your PC. It acts as an IDE for Python. It makes visualization of results simple and easy. In order to use Jupyter, run:

    jupyter notebook
    
from commandline. This will open Jupyter in a web browser. Jupyter will run from port 8888. If a web browser did not automatically pop up, Jupyter can be accessed by going to "localhost:8888" in any web browser. From here, Jupyter allows one to navagate to files and open them. Jupyter also introduces another type of python filetype, the iPythonNotebook (.ipynb). This file type allows Python to be run in modules. It also allows for the user to write Markdown, the same language from what this README is written in, inside the Python code. Finally, you can also display graphs and tables in a more readable format. Overall, it is very useful for tutorials and examples of Python programming. 

In order to view a tutorial, navagate to ~/GarrettLib/tutorials and open the iPythonNotebook of your choice. From here, you can run code or simply view the code that was run the last time it was open. This folder will be populated with use cases for GarrettLib, specifically those from the User Stories shown in my design documents. 

## Go to ~/GarrettLib/tutorials/get-started.ipynb to get started!
