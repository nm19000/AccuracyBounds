# tests/test_core.py

def add(a, b):
    return a + b

def test_add1():
    assert add(2, 2) == 4

def test_add2():
    assert add(2, 2) == 5
