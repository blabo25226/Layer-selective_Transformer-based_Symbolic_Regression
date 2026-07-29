from ffx import FFXRegressor

# FFX has no parameters!
hyper_params = [
    {
    },
]

est = FFXRegressor()

def complexity(est):
    return est.model_.complexity()

def model(est, X=None):
    return est.model_.str2().replace('^', '**')
