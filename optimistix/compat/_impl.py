from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp

from .._minimise import minimise
from .._misc import max_norm
from .._solution import RESULTS
from .._solver import BFGS


class OptimizeResults(NamedTuple):
    """Object holding optimization results.

    **Attributes:**

    - `x`: final solution.
    - `success`: ``True`` if optimization succeeded.
    - `status`: integer solver specific return code. 0 means converged (nominal),
        1=max BFGS iters reached, 3=other failure.
    - `fun`: final function value.
    - `jac`: final jacobian array.
    - `hess_inv`: final inverse Hessian estimate.
    - `nfev`: integer number of function calls used.
    - `njev`: integer number of gradient evaluations.
    - `nit`: integer number of iterations of the optimization algorithm.
    """

    x: jax.Array
    success: Union[bool, jax.Array]
    status: Union[int, jax.Array]
    fun: jax.Array
    jac: jax.Array
    hess_inv: Optional[jax.Array]
    nfev: Union[int, jax.Array]
    njev: Union[int, jax.Array]
    nit: Union[int, jax.Array]


def minimize(
    fun: Callable,
    x0: jax.Array,
    args: tuple = (),
    *,
    method: str,
    tol: Optional[float] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> OptimizeResults:
    """Minimization of scalar function of one or more variables.

    !!! info

        This API is intended as a backward-compatibility drop-in for the now-deprecated
        `jax.scipy.optimize.minimize`. In line with that API, only `method="bfgs"` is
        supported.

        Whilst it's the same basic algorithm, the Optimistix implementation may do
        slightly different things under-the-hood. You may obtain slightly different
        (but still correct) results.

    **Arguments:**

    - `fun`: the objective function to be minimized, `fun(x, *args) -> float`,
          where `x` is a 1-D array with shape `(n,)` and `args` is a tuple
          of the fixed parameters needed to completely specify the function.
          `fun` must support differentiation.
    - `x0`: initial guess. Array of real elements of size `(n,)`, where `n` is
          the number of independent variables.
    - `args`: extra arguments passed to the objective function.
    - `method`: solver type. Currently only `"bfgs"` is supported.
    - `tol`: tolerance for termination.
    - `options`: a dictionary of solver options. The following options are supported:
        - `maxiter` (int): Maximum number of iterations to perform. Each iteration
            performs one function evaluation. Defaults to unlimited iterations.
        - `norm`: (callable `x -> float`): the norm to use when calculating errors.
            Defaults to a max norm.

    **Returns:**

    An [`optimistix.compat.OptimizeResults`][] object.
    """
    if method.lower() != "bfgs":
        raise ValueError(f"Method {method} not recognized")
    if not eqx.is_array(x0) or x0.ndim != 1:
        raise ValueError("x0 must be a 1-dimensional array")
    if not isinstance(args, tuple):
        msg = "args argument to `optimistix.compat.minimize` must be a tuple, got {}"
        # TypeError, not ValueError, for compatibility with old
        # `jax.scipy.optimize.minimize`.
        raise TypeError(msg.format(args))
    if tol is None:
        tol = 1e-5
    if options is None:
        options = {}
    else:
        options = dict(options)
    max_steps = options.pop("maxiter", None)
    options.pop("norm", max_norm)
    if len(options) != 0:
        raise ValueError(f"Unsupported options: {set(options.keys())}")

    def wrapped_fn(y, args):
        return fun(y, *args)

    solver = BFGS(rtol=tol, atol=tol, norm=max_norm)
    sol = minimise(wrapped_fn, solver, x0, args, max_steps=max_steps, throw=False)
    status = jnp.where(
        sol.result == RESULTS.successful,
        0,
        jnp.where(sol.result == RESULTS.nonlinear_max_steps_reached, 1, 3),
    )
    return OptimizeResults(
        x=sol.value,
        success=sol.result == RESULTS.successful,
        status=status,
        fun=sol.state.f_info.f,
        jac=sol.state.f_info.grad,
        hess_inv=sol.state.f_info.hessian_inv.as_matrix(),
        nfev=sol.stats["num_steps"],
        njev=sol.state.num_accepted_steps,
        # Old JAX implementation counts each full line search as an iteration.
        nit=sol.state.num_accepted_steps,
    )
