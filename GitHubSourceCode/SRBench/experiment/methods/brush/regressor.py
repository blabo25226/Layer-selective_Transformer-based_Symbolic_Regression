from pybrush import BrushRegressor
from pybrush import individual
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
import re
import numpy as np
     
kwargs = {
    'verbosity'       : 1,
    'pop_size'        : 250, 
    'max_gens'        : 250,
    'max_depth'       : 8,  # 8
    'max_size'        : 75, # 75
    'initialization'  : 'uniform',
    'validation_size' : 0.33,
    'cx_prob'         : 1/7,
    'weights_init'    : False,
    'mutation_probs'  : {"point":1/6, "insert": 1/6, "delete":  1/6, "subtree": 1/6,
                         "toggle_weight_on": 1/6, "toggle_weight_off":1/6},
    'sel'             : 'lexicase', # tournament, e-lexicase
    'algorithm'       : 'nsga2',
    'objectives'      : ['scorer', 'complexity'],
    'bandit'          : "dynamic_thompson", # "thompson", "dynamic_thompson",
    'num_islands'     : 1,
    'use_arch'        : True,
    'shuffle_split'   : True, # True, False
    # "max_stall"       : 25,
    'functions'       : [
        # # # arithmetic (just a subset of them)
        "Add", "Sub", "Mul", "Div",  "Sin",  "Cos","Tanh", 
        "Exp", "Log", "Sqrt", "Pow",

        # # # logic operators
        "And", "Or", "Not", "Xor", "Equals", "LessThan", "GreaterThan", "Leq", "Geq",

        # # # split
        "SplitBest", "SplitOn",

        # # # terminals
        "Constant", "Terminal", # "MeanLabel", 
    ]
}


est = BrushRegressor(
    **kwargs
) 


func_dict = {
    'Mul': '*',
    'Sub': '-',
    'Add': '+',
    'Div': '/',
    'Pow': '**',
}

func_arity = { # remember to add here the functions used in the experiments
    # These can have multiple arguments
    'Mul': 2,
    'Sub': 2,
    'Div': 2,
    'Add': 2,

    'Pow': 2,
    
    'Sin'  : 1,
    'Cos'  : 1,
    'Tanh' : 1,

    'Asin' : 1,
    'Acos' : 1,
    
    'Sqrt' : 1,
    
    'Log'     : 1,
    'Exp'     : 1,
    
    'Square'  : 1,
}


def pretify_expr(string, feature_names):
    # Breaking down into a list of symbols. replace 8 with % to capture weight multiplication
    # (these are already in infix notation)
    ind = string.replace(' ', '').replace(')', '').replace('(', ',').split(',')

    new_string = ""
    stack = []
    for node in ind:
        stack.append((node, []))
        while len(stack[-1][1]) == func_arity.get(stack[-1][0], 0):

            prim, args = stack.pop()
            new_string = prim
            if prim in func_dict.keys(): # converting prefix to infix notation
                new_string = '(' + func_dict[prim].join(args) + ')'
            elif "*" in prim: # node weights already are in infix notation. handling it
                l, r = prim.split("*")
                stack.append( ("Mul", [l]) )
                stack.append( (r, []) )
                continue             
            elif prim not in feature_names:
                try:
                    float(prim)
                except:
                    new_string = prim.lower() + '(' + args[0] + ')'
                
            if len(stack) == 0:
                break

            stack[-1][1].append(new_string)

    return new_string


def model(est, X=None):
    model_str = None
    if isinstance(est, BrushRegressor):
        model_str = est.best_estimator_.get_model()
    else:
        model_str = est.model()

    # raw string, without fixing infix notation
    return model_str

    # Extra processing if need to have prettier strings 
    feature_names = [f"x_{i}" for i in range(100)]
    if X is not None:
        feature_names = X.columns

    pretty_model_str = pretify_expr(model_str, feature_names=feature_names)

    return pretty_model_str


def complexity(est):
    if isinstance(est, BrushRegressor):
        return est.best_estimator_.fitness.size
    
    ind = [i for i in est.est.archive_ if i['id']==est.id][0]
    return ind['fitness']['size']

