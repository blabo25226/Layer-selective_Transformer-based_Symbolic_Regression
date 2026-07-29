import torch
import numbers
import functools
import numpy as np
from ..symbols import (
    Symbol, Empty, Number, Variable, Add, Sub, Mul, Div, Pow, 
    Max, Min, Sin, Cos, Tan, Log, LogAbs, Exp, Arcsin, Arccos, 
    Arctan, Sqrt, SqrtAbs, Abs, Neg, Inv, Pow2, Pow3, Tanh, 
    Sigmoid, Regular, Sour, Targ, Aggr, Rgga, Readout
)
from typing import List, Tuple, Dict
# from ..base_visitor import Visitor
from .numpy_calc import NumpyCalc

def unpack_operands(mask_out_nan=False, calc_invalid=False, fill_nan_input=1.0, fill_nan_output=torch.nan):
    """Decorator to unpack operands of a node and apply a function to them.
    Args:
    - mask_out_nan: if True, replace NaN values in the input with fill_nan_input
    - calc_invalid: if True, calculate the output for invalid inputs
        设置为 True 可能导致性能降低, 但对 Div, Inv 等操作有帮助 (这些 func 会将 Non-nan 映射为 nan)
    - fill_nan_input: value to replace NaN values in the input
        具体取值只要保证 func() 不会产生 nan 即可
    - fill_nan_output: value to replace NaN values in the output
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, node, *args, **kwargs):
            X = [self(op, **kwargs) for op in node.operands]
            if mask_out_nan:
                if calc_invalid:
                    y = func(self, node, *X, *args, **kwargs)
                    valid = y.isfinite()
                else:
                    valid = torch.stack([x.isfinite() for x in X], dim=0).all(dim=0)
                for idx, x in enumerate(X):
                    X[idx] = torch.where(valid, x, fill_nan_input)
                y = func(self, node, *X, *args, **kwargs)
                y = torch.where(valid, y, fill_nan_output)
                return y
            return func(self, node, *X, *args, **kwargs)
        return wrapper
    return decorator


class TorchCalc(NumpyCalc):
    def __call__(self,
                 node:Symbol,
                 vars:dict = {},
                 edge_list:Tuple[List[int], List[int]]=None,
                 num_nodes:int = None,
                 use_eps:float = 0.0,
                 device:str='cpu',
                 ):
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
        if num_nodes is None and edge_list is not None:
            nodes = np.unique(np.array(edge_list).reshape(-1))
            num_nodes = max(nodes) + 1
        
        if edge_list is not None:
            edge_list = (torch.tensor(edge_list[0], device=device),
                         torch.tensor(edge_list[1], device=device))

        return super().__call__(node,
                                vars=vars, 
                                edge_list=edge_list, 
                                num_nodes=num_nodes,
                                use_eps=use_eps,
                                device=device)

    def generic_visit(self, node:Symbol, *args, **kwargs):
        raise NotImplementedError(f'{type(self).__name__}.visit_{type(node).__name__} not implemented')

    def visit_Empty(self, node:Empty, *args, **kwargs):
        raise ValueError('Incomplete Equation Tree')
    
    def visit_Number(self, node:Number, *args, **kwargs):
        device = kwargs.get('device')
        return torch.from_numpy(np.asarray(node.value), device=device)

    def visit_Variable(self, node:Variable, *args, **kwargs):
        X = kwargs.get('vars', {})
        device = kwargs.get('device')
        return torch.from_numpy(np.asarray(X[node.name]), device=device)

    @unpack_operands(mask_out_nan=True)
    def visit_Add(self, node:Add, x1, x2, *args, **kwargs):
        return x1 + x2

    @unpack_operands(mask_out_nan=True)
    def visit_Sub(self, node:Sub, x1, x2, *args, **kwargs):
        return x1 - x2

    @unpack_operands(mask_out_nan=True)
    def visit_Mul(self, node:Mul, x1, x2, *args, **kwargs):
        return x1 * x2

    @unpack_operands(mask_out_nan=True, calc_invalid=True)
    def visit_Div(self, node:Div, x1, x2, *args, **kwargs):
        eps = kwargs.get('use_eps')
        return x1 / (x2 + eps * (x2 == 0))

    @unpack_operands(mask_out_nan=True, calc_invalid=True)
    def visit_Pow(self, node:Pow, x1, x2, *args, **kwargs):
        return x1 ** x2

    @unpack_operands(mask_out_nan=True)
    def visit_Max(self, node:Max, x1, x2, *args, **kwargs):
        return torch.max(x1, x2)
    
    @unpack_operands(mask_out_nan=True)
    def visit_Min(self, node:Min, x1, x2, *args, **kwargs):
        return torch.min(x1, x2)

    @unpack_operands(mask_out_nan=True)
    def visit_Sin(self, node:Sin, x, *args, **kwargs):
        return torch.sin(x)
    
    @unpack_operands(mask_out_nan=True)
    def visit_Cos(self, node:Cos, x, *args, **kwargs):
        return torch.cos(x)
    
    @unpack_operands(mask_out_nan=True, calc_invalid=True)
    def visit_Tan(self, node:Tan, x, *args, **kwargs):
        return torch.tan(x)
    
    @unpack_operands(mask_out_nan=True, calc_invalid=True)
    def visit_Log(self, node:Log, x, *args, **kwargs):
        return torch.log(x)
    
    @unpack_operands(mask_out_nan=True, calc_invalid=True)
    def visit_LogAbs(self, node:LogAbs, x, *args, **kwargs):
        eps = kwargs.get('use_eps')
        return torch.log(torch.abs(x + eps * (x == 0)))
    
    @unpack_operands(mask_out_nan=True, calc_invalid=True)
    def visit_Exp(self, node:Exp, x, *args, **kwargs):
        return torch.exp(x)
    
    @unpack_operands(mask_out_nan=True)
    def visit_Arcsin(self, node:Arcsin, x, *args, **kwargs):
        return torch.arcsin(x)
    
    @unpack_operands(mask_out_nan=True)
    def visit_Arccos(self, node:Arccos, x, *args, **kwargs):
        return torch.arccos(x)
    
    @unpack_operands(mask_out_nan=True)
    def visit_Arctan(self, node:Arctan, x, *args, **kwargs):
        return torch.arctan(x)
    
    @unpack_operands(mask_out_nan=True, calc_invalid=True)
    def visit_Sqrt(self, node:Sqrt, x, *args, **kwargs):
        return torch.sqrt(x)
    
    @unpack_operands(mask_out_nan=True)
    def visit_SqrtAbs(self, node:SqrtAbs, x, *args, **kwargs):
        return torch.sqrt(torch.abs(x))
    
    @unpack_operands(mask_out_nan=True)
    def visit_Abs(self, node:Abs, x, *args, **kwargs):
        return torch.abs(x)
    
    @unpack_operands(mask_out_nan=True)
    def visit_Neg(self, node:Neg, x, *args, **kwargs):
        return -x
    
    @unpack_operands(mask_out_nan=True, calc_invalid=True)
    def visit_Inv(self, node:Inv, x, *args, **kwargs):
        eps = kwargs.get('use_eps')
        return 1 / (x + eps * (x == 0))
    
    @unpack_operands(mask_out_nan=True, calc_invalid=True)
    def visit_Pow2(self, node:Pow2, x, *args, **kwargs):
        return x ** 2
    
    @unpack_operands(mask_out_nan=True, calc_invalid=True)
    def visit_Pow3(self, node:Pow3, x, *args, **kwargs):
        return x ** 3
    
    @unpack_operands(mask_out_nan=True)
    def visit_Tanh(self, node:Tanh, x, *args, **kwargs):
        return torch.tanh(x)
    
    @unpack_operands(mask_out_nan=True)
    def visit_Sigmoid(self, node:Sigmoid, x, *args, **kwargs):
        return torch.sigmoid(x)
    
    @unpack_operands(mask_out_nan=True, calc_invalid=True)
    def visit_Regular(self, node:Regular, x1, x2, *args, **kwargs):
        eps = kwargs.get('use_eps')
        return 1 / (1 + (torch.abs(x1) + eps * (x1 == 0)) ** (-x2))

    @unpack_operands()
    def visit_Sour(self, node:Sour, x, *args, **kwargs):
        """(*, n_nodes or 1) -> (*, n_edges or 1)"""
        edge_list = kwargs.get('edge_list', ([],[]))

        if isinstance(x, numbers.Number) or x.size == 1:
            return x # (1,) -> (1,)
        elif node.operands[0].nettype == 'scalar': 
            if x.shape[-1] != 1: x = x[..., None]
            return x # (*, 1) -> (*, 1)
        else:
            return x[..., edge_list[0]] # (*, V) -> (*, E)

    @unpack_operands()
    def visit_Targ(self, node:Targ, x, *args, **kwargs):
        """(*, n_nodes or 1) -> (*, n_edges or 1)"""
        edge_list = kwargs.get('edge_list', ([],[]))

        if isinstance(x, numbers.Number) or x.size == 1:
            return x # (1,) -> (1,)
        elif node.operands[0].nettype == 'scalar': 
            if x.shape[-1] != 1: x = x[..., None]
            return x # (*, 1) -> (*, 1)
        else:
            return x[..., edge_list[1]] # (*, V) -> (*, E)

    @unpack_operands()
    def visit_Aggr(self, node:Aggr, x, *args, **kwargs):
        """(*, n_edges or 1) -> (*, n_nodes)"""
        edge_list = kwargs.get('edge_list', ([],[]))
        num_nodes = kwargs.get('num_nodes')
        device = kwargs.get('device')
        
        if isinstance(x, numbers.Number) or x.size == 1:
            y = torch.zeros((num_nodes,), device=device)
            y.scatter_add_(edge_list[1], x, dim=-1)
            return y
        elif node.operands[0].nettype == 'scalar': 
            if x.shape[-1] != 1: x = x[..., None]
            y = torch.zeros((num_nodes,), device=device)
            y.scatter_add_(edge_list[1], 1, dim=-1)
            y = y * x
            return y
        else:
            y = np.zeros((*x.shape[:-1], num_nodes))
            y.scatter_add_(edge_list[1], x, dim=-1)
            return y

    @unpack_operands()
    def visit_Rgga(self, node:Rgga, x, *args, **kwargs):
        """(*, n_edges or 1) -> (*, n_nodes)"""
        edge_list = kwargs.get('edge_list', ([],[]))
        num_nodes = kwargs.get('num_nodes')
        device = kwargs.get('device')
        
        if isinstance(x, numbers.Number) or x.size == 1:
            y = torch.zeros((num_nodes,), device=device)
            y.scatter_add_(edge_list[0], x, dim=-1)
            return y
        elif node.operands[0].nettype == 'scalar': 
            if x.shape[-1] != 1: x = x[..., None]
            y = torch.zeros((num_nodes,), device=device)
            y.scatter_add_(edge_list[0], 1, dim=-1)
            y = y * x
            return y
        else:
            y = np.zeros((*x.shape[:-1], num_nodes))
            y.scatter_add_(edge_list[0], x, dim=-1)
            return y
    
    @unpack_operands()
    def visit_Readout(self, node:Readout, x, *args, **kwargs):
        """ (*, n_nodes or n_edges or 1) -> (*, 1) """
        return torch.sum(x, axis=-1, keepdim=True)
