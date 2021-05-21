import pytest

class OutofRange(Exception):
    def __init__(self,message="The value is not in range"):
        self.message=message
        super().__init__(self.message)

def test_values():
    a=3
    with pytest.raises(OutofRange):
        if a not in range(5,10):
            raise OutofRange

def test_second():
    a=3
    b=5
    assert a!=b