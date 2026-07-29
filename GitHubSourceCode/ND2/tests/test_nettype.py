import pytest
import ND2 as nd

s = nd.Variable('x', nettype='scalar')
n = nd.Variable('x', nettype='node')
e = nd.Variable('x', nettype='edge')

add = nd.Add(nettype='node')
aggr = nd.Aggr(nettype='node')

add_aggr = nd.Add(nettype='node')
add_aggr.replace(add_aggr.operands[0], nd.Aggr(nettype='node'))

@pytest.mark.parametrize("node,expected", [
    (nd.aggr(s), 'node'),
    (nd.aggr(e), 'node'),
    (nd.aggr(s).operands[0], 'scalar'),
    (nd.aggr(e).operands[0], 'edge'),
    (nd.rgga(s), 'node'),
    (nd.rgga(e), 'node'),
    (nd.rgga(s).operands[0], 'scalar'),
    (nd.rgga(e).operands[0], 'edge'),
    (nd.sour(s), 'edge'),
    (nd.sour(s).operands[0], 'scalar'),
    (nd.sour(n), 'edge'),
    (nd.sour(n).operands[0], 'node'),
    ((s + n), 'node'),
    ((s + n).operands[0], 'scalar'),
    ((s + n).operands[1], 'node'),
    ((s + e), 'edge'),
    ((s + e).operands[0], 'scalar'),
    ((s + e).operands[1], 'edge'),
    ((s + s), 'scalar'),
    ((n + n), 'node'),
    ((e + e), 'edge'),
    (add, 'node'),
    (add.operands[0], None),
    (add.operands[1], None),
    (aggr, 'node'),
    (aggr.operands[0], None),
])
def test_nettype(node, expected):
    assert node.nettype == expected


@pytest.mark.parametrize("node,expected", [
    (lambda: nd.aggr(n), (AssertionError, ValueError)),
    (lambda: nd.rgga(n), (AssertionError, ValueError)),
    (lambda: nd.sour(e), (AssertionError, ValueError)),
    (lambda: nd.targ(e), (AssertionError, ValueError)),
    (lambda: n+e, (AssertionError, ValueError)),
    (lambda: nd.Aggr(nettype='edge'), (AssertionError, ValueError)),
    (lambda: nd.Aggr(nettype='scalar'), (AssertionError, ValueError)),
])
def test_nettype_error(node, expected):
    with pytest.raises(expected):
        node()
