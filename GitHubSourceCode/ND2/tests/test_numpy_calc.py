import pytest
import numpy as np
import ND2 as nd2

@pytest.mark.parametrize("calc_cls,node,flags,expected", [
    (nd2.NumpyCalc, nd2.Number(3.14), {}, 3.14),
    (nd2.NumpyCalc, nd2.Variable("x"), dict(vars={'x':[1,2,3]}), [1,2,3]),
    (nd2.NumpyCalc, nd2.Variable("x") + 1, dict(vars={'x':[1,2,3]}), [2,3,4]),
    (nd2.NumpyCalc, 2 * nd2.Variable("x"), dict(vars={'x':[1,2,3]}), [2,4,6]),
    (nd2.NumpyCalc, nd2.Aggr(nd2.Variable("x")), dict(vars={'x':1}, edge_list=[[0,0,0], [1,1,3]]), [0,2,0,1]),
    (nd2.NumpyCalc, nd2.Aggr(nd2.Variable("x", nettype='edge')), dict(vars={'x':[1,2,2]}, edge_list=[[0,0,0], [1,1,3]]), [0,3,0,2]),
    (nd2.NumpyCalc, nd2.Aggr(nd2.Variable("x")), dict(vars={'x':[[1],[2]]}, edge_list=[[0,0,0], [1,1,3]]), [[0,2,0,1],[0,4,0,2]]),
])
def test_calc_number(calc_cls, node, flags, expected):
    calc = calc_cls()
    output = calc(node, **flags)
    if isinstance(output, np.ndarray): 
        output = output.tolist()
    assert output == expected
