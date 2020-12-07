# pylint: disable=no-member
class DictAttrAccess:
    "Allows for easy access of dictionary values through an object attribute."
    def __init__(self, token_values={}):
        self.__dict__.update(token_values)
    def __setattr__(self, name, value):
        self.__dict__[name] = value

def test_DictAttrAccess():
    testobj = object()
    daa = DictAttrAccess({'a':1, 'b':2, 'c':3, 'obj':testobj})
    assert daa.a == 1
    assert daa.b == 2
    assert daa.c == 3
    assert daa.obj == testobj
    daa.x = "hello"
    assert daa.x == "hello"
