import pytest

def some_method(x):
    if not isinstance(x,int):
        raise TypeError("Please provide an integer argument")
    return x 

def test_some_method():
    assert some_method(1) == 1

def test_some_method_raises_typeerror():
    with pytest.raises(TypeError):
        some_method("Junk");
