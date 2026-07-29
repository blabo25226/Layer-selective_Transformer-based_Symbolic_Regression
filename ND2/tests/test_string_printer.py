import pytest
import ND2 as nd2

@pytest.mark.parametrize("printer_cls,node,expected", [
    (nd2.StringPrinter, nd2.Number(3.14), "3.14"),
    (nd2.StringPrinter, nd2.Variable("x"), "x"),
    (nd2.StringPrinter, nd2.Number(1) + nd2.Number(2), "1 + 2"),
    (nd2.StringPrinter, nd2.sin(1), "sin(1)"),
    (nd2.StringPrinter, nd2.cos(1), "cos(1)"),
    (nd2.StringPrinter, nd2.tanh(1), "tanh(1)"),
    (nd2.StringPrinter, nd2.sigmoid(1), "sigmoid(1)"),
    (nd2.StringPrinter, nd2.aggr(1), "aggr(1)"),
    (nd2.StringPrinter, nd2.sour(1), "sour(1)"),
    (nd2.StringPrinter, nd2.targ(1), "targ(1)"),

])
def test_printer_outputs(printer_cls, node, expected):
    printer = printer_cls()
    assert printer(node) == expected
    assert node.to_str() == expected

def test_omit_mul_sign_and_parentheses():
    expr = nd2.Number(2) * (nd2.Variable('x') + nd2.Variable('y'))
    sp = nd2.StringPrinter()
    assert sp(expr) == "2 * (x + y)"
    assert sp(expr, omit_mul_sign=True) == "2(x + y)"

    assert expr.to_str() == "2 * (x + y)"
    assert expr.to_str(omit_mul_sign=True) == "2(x + y)"
