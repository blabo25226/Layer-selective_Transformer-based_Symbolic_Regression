import torch
import inspect
import numbers
import warnings
import numpy as np
import sympy as sp
from copy import deepcopy
from typing import List, Dict, Literal, Union, Optional, Tuple, Set
from functools import reduce, wraps
from scipy.optimize import minimize
import logging

logger = logging.getLogger('ND2.core.symbols')

"""
每个 Symbol 都有固定的 n_operands 和 nettype:[any,node,edge,scalar]
Symbol() 的 nettype 会视 operands 而定发生变化
"""


class SymbolMeta(type):
    def __repr__(cls):
        return cls.__name__


class Symbol(metaclass=SymbolMeta):
    n_operands = None

    def __init__(self, *operands, nettype:Optional[Literal['node', 'edge', 'scalar']]=None):
        operands = list(operands)
        for idx, op in enumerate(operands):
            if isinstance(op, numbers.Number):
                operands[idx] = Number(op, nettype='scalar')
            else:
                assert isinstance(op, Symbol), f"Invalid operand type: {type(op)}"

        # 如果给定了 operands，检查它所暗示的 nettype
        if len(operands) > 0:
            possible_nettype = self.possible_nettype([op.nettype for op in operands])
            assert len(possible_nettype) == 1, f"{self.__class__.__name__}.possible_nettype(...) should return a single nettype"
            possible_nettype = possible_nettype.pop()
            if nettype is not None: # 如果指定了 nettype，检查它与 operands 所暗示的 nettype 是否一致
                pass
                # assert nettype == possible_nettype, f"Invalid nettype: {nettype} for {self.__class__.__name__} ({possible_nettype})"
            else: # 如果没有指定 nettype，设置 nettype 为 operands 所暗示的 nettype
                nettype = possible_nettype
        self.nettype = nettype

        # 检查 nettype 是否符合 Symbol 的要求
        if self.nettype is None:
            raise ValueError(f"Invalid nettype: {self.nettype} for {self.__class__.__name__}")
        elif self.nettype not in self.possible_nettype():
            raise ValueError(f"Invalid nettype: {self.nettype} for {self.__class__.__name__} ({self.possible_nettype()})")

        # 检查 operands.nettype 是否符合要求
        for child_idx, op in enumerate(operands):
            if op.nettype not in self.replaceable_nettype(child_idx):
                raise ValueError(f"Invalid op.nettype: {op.nettype} for {self.__class__.__name__} ({self.replaceable_nettype(child_idx)})")

        # 如果没有给定 operands，设置为 Empty
        if len(operands) == 0:
            operands = [Empty(nettype=None) for _ in range(self.n_operands)]

        self.operands = operands
        self.parent = None
        self.child_idx = None
        for idx, op in enumerate(self.operands):
            op.parent = self
            op.child_idx = idx

    @classmethod
    def possible_nettype(cls, child_types:List[Literal['node', 'edge', 'scalar']]=None) -> Set[Literal['node', 'edge', 'scalar']]:
        """这个 Symbol 可能产生的 nettype / 在指定 child_types 下产生的 nettype"""
        if child_types is None:
            return {'node', 'edge', 'scalar'}
        elif 'node' in child_types and 'edge' in child_types:
            raise ValueError(f'Inconsistent nettype in {cls.__name__}')
        elif 'node' in child_types:
            return {'node'}
        elif 'edge' in child_types:
            return {'edge'}
        elif 'scalar' in child_types:
            return {'scalar'}
        else:
            raise ValueError(f'Unknown nettype in {cls.__name__}: {child_types}')

    def replaceable_nettype(self, child_idx:Optional[int]=None) -> Set[Literal['node', 'edge', 'scalar']]:
        """可以用于替换这个 Symbol / Symbol.operands[child_idx] 的 nettype"""
        if child_idx is not None:
            return {self.nettype, 'scalar'}
        else:
            if self.parent is not None:
                return self.parent.replaceable_nettype(self.child_idx)
            else:
                return {self.nettype}

    def __repr__(self, **kwargs):
        return self.to_str(**kwargs)
    
    def __str__(self, **kwargs):
        return self.to_str(**kwargs)

    def __len__(self):
        return 1 + sum(len(operand) for operand in self.operands)

    def to_str(self, 
               raw=False, 
               latex=False, 
               number_format='', 
               omit_mul_sign=False, 
               skeleton=False) -> str:
        """
        Args:
        - raw:bool=False, whether to return the raw format
        - number_format:str='', can be '0.2f'
        - omit_mul_sign:bool=False, whether to omit the multiplication sign
        - latex:bool=False, whether to return the latex format
        - skeleton:bool=False, whether to ignore the concrete values of Number
        """
        from .printer.string_printer import StringPrinter
        return StringPrinter()(self, raw=raw, latex=latex, number_format=number_format, 
                               omit_mul_sign=omit_mul_sign, skeleton=skeleton)

    def to_tree(self, 
                number_format='', 
                flat=False, 
                skeleton=False) -> str:
        """
        Args:
        - number_format:str='', can be '0.2f'
        - flat:bool=False, whether to flat the Add and Mul
        - omit_mul_sign:bool=False, whether to omit the multiplication sign
        """
        from .printer.tree_printer import TreePrinter
        return TreePrinter()(self, number_format=number_format, flat=flat, skeleton=skeleton)

    def eval(self,
             vars:dict = {},
             edge_list:Tuple[List[int], List[int]]=None,
             num_nodes:int = None,
             use_eps:float = 0.0):
        """
        Args:
        - vars: a dictionary of variable names and their values
        - edge_list: edges (i,j) in the graph
            i and j are the indices of the nodes (starting from 0)
        - num_nodes: the number of nodes in the graph
            if not provided, it will be inferred from edge_list
        - use_eps: a small value to avoid division by zero
        """
        from .calc.numpy_calc import NumpyCalc
        return NumpyCalc()(self, vars=vars, edge_list=edge_list, num_nodes=num_nodes, use_eps=use_eps)

    def eval_torch(self, 
                   vars:dict = {},
                   edge_list:Tuple[List[int], List[int]]=None,
                   num_nodes:int = None,
                   use_eps:float = 0.0,
                   device:str='cpu'):
        """
        Args:
        - vars: a dictionary of variable names and their values
        - edge_list: edges (i,j) in the graph
            i and j are the indices of the nodes (starting from 0)
        - num_nodes: the number of nodes in the graph
            if not provided, it will be inferred from edge_list
        - use_eps: a small value to avoid division by zero
        - device: cpu or cuda
        """
        from .calc.torch_calc import TorchCalc
        return TorchCalc()(self, vars=vars, edge_list=edge_list, num_nodes=num_nodes, use_eps=use_eps, device=device)

    @classmethod
    def create_instance(cls, *operands):
        return cls(*operands)

    def preorder(self):
        yield self
        for operand in self.operands:
            yield from operand.preorder()
    
    def postorder(self):
        for operand in self.operands:
            yield from operand.postorder()
        yield self
    
    def __add__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Add(self, other)

    def __radd__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Add(other, self)

    def __sub__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Sub(self, other)

    def __rsub__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Sub(other, self)

    def __mul__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Mul(self, other)

    def __rmul__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Mul(other, self)

    def __truediv__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Div(self, other)
    
    def __rtruediv__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Div(other, self)

    def __pow__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        if other == 2.0: return Pow2(self)
        if other == 3.0: return Pow3(self)
        if other == 0.5: return Sqrt(self)
        if other == -1.0: return Inv(self)
        return Pow(self, other)
    
    def __rpow__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Pow(other, self)
    
    def __neg__(self):
        return Neg(self)

    def is_constant(self, **kwargs):
        return all([op.is_constant(**kwargs) for op in self.operands])
    
    def replace(self, child:'Symbol', other:'Symbol'):
        if self == child: 
            assert self.parent is None, f"Cannot replace {self.__class__.__name__} with {child.__class__.__name__}"
            return other
        other.parent, child.parent = child.parent, None
        other.child_idx, child.child_idx = child.child_idx, None
        if other.parent is not None:
            other.parent.operands[other.child_idx] = other
        return self
    
    def copy(self):
        copy = self.__class__(*[op.copy() for op in self.operands], nettype=self.nettype)
        return copy
    
    # def fit(self, X:Dict[str,np.ndarray], y:np.ndarray, maxiter=30, method='BFGS'):
    #     # 对 float32 报警
    #     float32 = []
    #     if y.dtype == np.float32: float32.append('y')
    #     for key, value in X.items():
    #         if value.dtype == np.float32:
    #             float32.append(key)
    #     if len(float32):
    #         logger.warning(f'{float32} is float32, which may cause numerical instability')

    #     # 创建一个替身，对这个替身的优化更加容易，fit 它的 fitable Number 即是原来的 fitable Number
    #     sutando = self.create_sutando(**X)
    #     if isinstance(sutando, Number) and not sutando.fitable: return self # 没有 fitable Number
    #     parameters = [op for op in sutando.preorder() if isinstance(op, Number) and op.fitable]
    #     def set_params(params):
    #         p = 0
    #         for param in parameters:
    #             param.value = params[p:p+param.value.size].reshape(param.value.shape)
    #             p += param.value.size
    #     def loss(params):
    #         set_params(params)
    #         return np.mean((y - sutando.eval(**X)) ** 2)
    #     x0 = np.concatenate([param.value.flatten() for param in parameters])
    #     res = minimize(loss, x0, method=method, options={'maxiter': maxiter})
    #     set_params(res.x)
    #     return self
    
    # def set_nettype(self, nettype:Literal['node', 'edge', 'scalar']):
    #     if self.nettype == 'unknown': 
    #         self.nettype = nettype
    #         for op in self.operands:
    #             if op.nettype == 'unknown':
    #                 op.set_nettype(self.nettype)
    #     elif self.nettype == 'scalar': 
    #         for op in self.operands:
    #             op.set_nettype(self.nettype)
    #     elif self.nettype == nettype:
    #         for op in self.operands:
    #             op.set_nettype(self.nettype)
    #     else:
    #         raise ValueError(f'Inconsistent nettype in {self.__class__.__name__}')

    # def create_sutando(self, *args, **kwargs) -> 'Symbol':
    #     """ 
    #     使用启发式的方法创建一个替身，与 self 共享 fitable Number
    #     替身的形式更加简洁，能够更快速地被 fit，且 fit 过程中 self 的 fitable Number 也会被更新
    #     """
    #     if self.n_operands == 0:
    #         if isinstance(self, Number) and self.fitable: return self  # 需要拟合的量
    #         else: return Number(self.eval(*args, **kwargs), nettype=self.nettype, fitable=False)  # 不需要拟合的量
    #     sutando_operands = [op.create_sutando(*args, **kwargs) for op in self.operands]
    #     if all(isinstance(op, Number) and not op.fitable for op in sutando_operands):  # 没有 fitable Number 的子公式
    #         return Number(self.__class__(*sutando_operands).eval(*args, **kwargs), nettype=self.nettype, fitable=False)
    #     return self.__class__(*sutando_operands)  # 有 fitable Number 且难以继续简化的子公式


class Empty(Symbol):
    n_operands = 0
    def __init__(self, nettype:Optional[Literal['node', 'edge', 'scalar']]=None):
        self.nettype = nettype

    def is_constant(self, **kwargs):
        return False


class Number(Symbol):
    n_operands = 0
    def __init__(self, value, nettype:Literal['node', 'edge', 'scalar']='scalar', fitable=True):
        super().__init__(nettype=nettype)
        self.value = np.asarray(value)
        self.fitable = fitable
        
    def __eq__(self, value: Union[int, float]) -> bool:
        return self.value == value

    def possible_nettype(self, child_types:List[Literal['node', 'edge', 'scalar']]=None) -> Set[Literal['node', 'edge', 'scalar']]:
        assert child_types is None
        return {self.nettype}

    def is_constant(self, **kwargs):
        return True
    
    def copy(self):
        return self.__class__(deepcopy(self.value), nettype=self.nettype, fitable=self.fitable)


class Variable(Symbol):
    n_operands = 0
    def __init__(self, name, nettype:Literal['node', 'edge', 'scalar']='scalar'):
        super().__init__(nettype=nettype)
        self.name = name

    def possible_nettype(self, child_types:List[Literal['node', 'edge', 'scalar']]=None) -> Set[Literal['node', 'edge', 'scalar']]:
        assert child_types is None
        return {self.nettype}
    
    def is_constant(self, **kwargs):
        return self.name in kwargs
    
    def copy(self):
        return self.__class__(self.name, nettype=self.nettype)


class Add(Symbol):
    n_operands = 2
    @classmethod
    def create_instance(self, *operands):
        add = [operand for operand in operands if operand.__class__ != Neg]
        sub = [operand.operands[0] for operand in operands if operand.__class__ == Neg]
        if len(sub) == 0: 
            return reduce(lambda x, y: Add(x, y), add)
        elif len(add) == 0:
            return Neg(reduce(lambda x, y: Add(x, y), sub))
        else: 
            return Sub(reduce(lambda x, y: Add(x, y), add), reduce(lambda x, y: Add(x, y), sub))

    def as_args(self, bias_at_first=False):
        operands = []
        if isinstance(self.operands[0], Add):
            operands.extend(self.operands[0].as_args())
        else:
            operands.append(self.operands[0])
        if isinstance(self.operands[1], Add):
            operands.extend(self.operands[1].as_args())
        else:
            operands.append(self.operands[1])
        if bias_at_first:
            bias = Number(sum((op.value for op in operands if isinstance(op, Number)), 0.0))
            operands = [bias] + [op for op in operands if not isinstance(op, Number)]
        return operands

class Sub(Symbol):
    n_operands = 2


class Mul(Symbol):
    n_operands = 2
    @classmethod
    def create_instance(self, *operands):
        if operands[0] == -1: return Neg(Mul.create_instance(*operands[1:]))
        numer = [operand for operand in operands if operand.__class__ != Inv]
        denom = [operand.operands[0] for operand in operands if operand.__class__ == Inv]
        if len(denom) == 0: 
            return reduce(lambda x, y: Mul(x, y), numer)
        elif len(numer) == 0:
            return Inv(reduce(lambda x, y: Mul(x, y), denom))
        else: 
            return Div(reduce(lambda x, y: Mul(x, y), numer), reduce(lambda x, y: Mul(x, y), denom))

    def as_args(self, coeff_at_first=False):
        operands = []
        if isinstance(self.operands[0], Mul):
            operands.extend(self.operands[0].as_args())
        else:
            operands.append(self.operands[0])
        if isinstance(self.operands[1], Mul):
            operands.extend(self.operands[1].as_args())
        else:
            operands.append(self.operands[1])
        if coeff_at_first:
            coeff = Number(reduce(lambda x, y: x * y, (op.value for op in operands if isinstance(op, Number)), 1.0))
            operands = [coeff] + [op for op in operands if not isinstance(op, Number)]
        return operands


class Div(Symbol):
    n_operands = 2


class Pow(Symbol):
    n_operands = 2
    @classmethod
    def create_instance(self, *operands):
        if operands[1] == 0.5: return Sqrt(operands[0])
        if operands[1] == -1: return Inv(operands[0])
        if operands[1] == 2: return Pow2(operands[0])
        if operands[1] == 3: return Pow3(operands[0])
        return Pow(*operands)


class Max(Symbol):
    n_operands = 2
    def create_instance(self, *operands):
        return reduce(lambda x, y: Max(x, y), operands)


class Min(Symbol):
    n_operands = 2
    def create_instance(self, *operands):
        return reduce(lambda x, y: Min(x, y), operands)


class Sin(Symbol):
    n_operands = 1


class Cos(Symbol):
    n_operands = 1


class Tan(Symbol):
    n_operands = 1


class Log(Symbol):
    n_operands = 1


class LogAbs(Symbol):
    n_operands = 1


class Exp(Symbol):
    n_operands = 1


class Arcsin(Symbol):
    n_operands = 1


class Arccos(Symbol):
    n_operands = 1


class Arctan(Symbol):
    n_operands = 1


class Sqrt(Symbol):
    n_operands = 1


class SqrtAbs(Symbol):
    n_operands = 1


class Abs(Symbol):
    n_operands = 1


class Neg(Symbol):
    n_operands = 1
    

class Inv(Symbol):
    n_operands = 1


class Pow2(Symbol):
    n_operands = 1


class Pow3(Symbol):
    n_operands = 1


class Tanh(Symbol):
    n_operands = 1


class Sigmoid(Symbol):
    n_operands = 1


class Regular(Symbol):
    n_operands = 2


class Sour(Symbol):
    n_operands = 1

    @classmethod
    def possible_nettype(cls, child_types=None):
        if child_types is not None:
            assert len(child_types) == 1, child_types
            return {'edge'} if child_types[0] in {'scalar', 'node'} else set()
        else:
            return {'edge'}

    def replaceable_nettype(self, child_idx=None):
        if child_idx is None:
            return {'edge'}
        else:
            return {'node', 'scalar'}

class Targ(Sour):
    pass

class Aggr(Symbol):
    n_operands = 1

    @classmethod
    def possible_nettype(cls, child_types=None):
        if child_types is not None:
            assert len(child_types) == 1
            return {'node'} if child_types[0] in {'scalar', 'edge'} else set()
        else:
            return {'node'}

    def replaceable_nettype(self, child_idx=None):
        if child_idx is None:
            return {'node'}
        else:
            return {'edge', 'scalar'}

class Rgga(Aggr):
    pass

class Readout(Symbol):
    n_operands = 1
    @classmethod
    def possible_nettype(cls, child_types=None):
        return {'scalar'}

    def replaceable_nettype(self, child_idx=None):
        if child_idx is None:
            return {'scalar'}
        else:
            return {'edge', 'node'}
