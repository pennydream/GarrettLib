import pytest

class SomeThing(object):

    def __init__(self, name, state):
        self.name = name
        self.state = state

    def getName(self):
        return self.name

    def getState(self):
        return self.state

    def setState(self, new_state):
        self.state = new_state
        return self.state

def some_method(x):
    if not isinstance(x,int):
        raise TypeError("Please provide an integer argument")
    return x 

@pytest.fixture
def OneObject():
    "Returns a SomeThing object with name Thing1 and state 1"
    return SomeThing("Thing1", 1)


@pytest.mark.parametrize("new_state",[
    (2),
    (3)
])
def test_SomeThing_setState(OneObject, new_state):
    OneObject.setState(new_state)
    assert OneObject.getState() == new_state

def test_SomeThing_init(OneObject):
    assert OneObject.getState() == 1
    assert OneObject.getName() == "Thing1"

def test_some_method():
    assert some_method(1) == 1

def test_some_method_raises_typeerror():
    with pytest.raises(TypeError):
        some_method("Junk");
