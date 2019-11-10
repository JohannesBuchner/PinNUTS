""" Some helpers to help the usage of NUTS
This package contains:

  numerical_grad            return numerical estimate of the local gradient
_function_wrapper           hack to make partial functions pickleable
NutsSampler_fn_wrapper      combine provided lnp and grad(lnp) into one function
"""
import numpy as np

class _function_wrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    are also included.
    """
    def __init__(self, f, args=(), kwargs={}):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:
            import traceback
            print("PinNUTS: Exception while calling your likelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise


class PinNutsSampler_fn_wrapper(object):
    """ Create a function-like object that combines provided lnp and grad(lnp)
    functions into one as required by nuts6.

    Both functions are stored as partial function allowing to fix arguments if
    the gradient function is not provided, numerical gradient will be computed

    By default, arguments are assumed identical for the gradient and the
    likelihood. However you can change this behavior using set_xxxx_args.
    (keywords are also supported)

    if verbose property is set, each call will print the log-likelihood value
    and the theta point
    """
    def __init__(self, lnp_func, gradlnp_func=None, *args, **kwargs):
        self.lnp_func = _function_wrapper(lnp_func, args, kwargs)
        if gradlnp_func is not None:
            self.gradlnp_func = _function_wrapper(gradlnp_func, args, kwargs)
        else:
            self.gradlnp_func = _function_wrapper(numerical_grad, (self.lnp_func,))
        self.verbose = False

    def set_lnp_args(self, *args, **kwargs):
            self.lnp_func.args = args
            self.lnp_func.kwargs = kwargs

    def set_gradlnp_args(self, *args, **kwargs):
            self.gradlnp_func.args = args
            self.gradlnp_func.kwargs = kwargs

    def __call__(self, theta):
        r = (self.lnp_func(theta), self.gradlnp_func(theta))
        if self.verbose:
            print(r[0], theta)
        return r
