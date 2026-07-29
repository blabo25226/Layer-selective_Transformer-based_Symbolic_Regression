import pytest
import ND2 as nd

def test_replace1():
    x, y, z = nd.variables('x y z')

    child1 = (x + y) * z
    child2 = x * z + y * z
    parent = nd.Sin(child1)

    parent = parent.replace(child1, child2)
    assert parent.to_str() == 'sin(x * z + y * z)'


def test_replace2():
    x, y, z = nd.variables('x y z')

    child1 = (x + y) * z
    child2 = x * z + y * z
    parent = nd.Sin(child1) + child1

    parent = parent.replace(child1, child2)
    assert parent.to_str() == 'sin((x + y) * z) + x * z + y * z'


def test_replace3():
    x = nd.Variable('x', nettype='edge')

    a = nd.aggr(x * x)
    b = nd.aggr(x * x)

    a = a.replace(a, b)
    assert a.to_str() == 'aggr(x * x)'

