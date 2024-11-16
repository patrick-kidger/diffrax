import abc
import functools as ft
from collections.abc import Callable
from typing import Optional

import equinox as eqx
import equinox.internal as eqxi
import lineax as lx
from jaxtyping import Array, PyTree

from ._ad import implicit_jvp


class AbstractAdjoint(eqx.Module, strict=True):
    """The abstract base class of all adjoints."""

    @abc.abstractmethod
    def apply(
        self,
        primal_fn: Callable,
        rewrite_fn: Callable,
        inputs: PyTree,
        tags: frozenset[object],
    ) -> PyTree[Array]:
        """Runs the main solver loop. Subclasses can override this to provide custom
        autodifferentiation behaviour; see for example the implementation of
        [`optimistix.ImplicitAdjoint`][].
        """


class RecursiveCheckpointAdjoint(AbstractAdjoint, strict=True):
    """Backpropagate by differentiating through the iterates directly.

    Uses a binomial checkpointing scheme to keep memory usage low.

    !!! info

        Note that this cannot be forward-mode autodifferentiated. (E.g. using
        `jax.jvp`.)

    ??? cite "References"

        Selecting which steps at which to save checkpoints (and when this is done, which
        old checkpoint to evict) is important for minimising the amount of recomputation
        performed.

        The implementation here performs "online checkpointing", as the number of steps
        is not known in advance. This was developed in:

        ```bibtex
        @article{stumm2010new,
            author = {Stumm, Philipp and Walther, Andrea},
            title = {New Algorithms for Optimal Online Checkpointing},
            journal = {SIAM Journal on Scientific Computing},
            volume = {32},
            number = {2},
            pages = {836--854},
            year = {2010},
            doi = {10.1137/080742439},
        }

        @article{wang2009minimal,
            author = {Wang, Qiqi and Moin, Parviz and Iaccarino, Gianluca},
            title = {Minimal Repetition Dynamic Checkpointing Algorithm for Unsteady
                     Adjoint Calculation},
            journal = {SIAM Journal on Scientific Computing},
            volume = {31},
            number = {4},
            pages = {2549--2567},
            year = {2009},
            doi = {10.1137/080727890},
        }
        ```

        For reference, the classical "offline checkpointing" (also known as "treeverse",
        "recursive binary checkpointing", "revolve" etc.) was developed in:

        ```bibtex
        @article{griewank1992achieving,
            author = {Griewank, Andreas},
            title = {Achieving logarithmic growth of temporal and spatial complexity in
                     reverse automatic differentiation},
            journal = {Optimization Methods and Software},
            volume = {1},
            number = {1},
            pages = {35--54},
            year  = {1992},
            publisher = {Taylor & Francis},
            doi = {10.1080/10556789208805505},
        }

        @article{griewank2000revolve,
            author = {Griewank, Andreas and Walther, Andrea},
            title = {Algorithm 799: Revolve: An Implementation of Checkpointing for the
                     Reverse or Adjoint Mode of Computational Differentiation},
            year = {2000},
            publisher = {Association for Computing Machinery},
            volume = {26},
            number = {1},
            doi = {10.1145/347837.347846},
            journal = {ACM Trans. Math. Softw.},
            pages = {19--45},
        }
        ```
    """

    checkpoints: Optional[int] = None

    def apply(self, primal_fn, rewrite_fn, inputs, tags):
        del rewrite_fn, tags
        while_loop = ft.partial(
            eqxi.while_loop, kind="checkpointed", checkpoints=self.checkpoints
        )
        return primal_fn(inputs + (while_loop,))


class ImplicitAdjoint(AbstractAdjoint, strict=True):
    r"""Backpropagate via the [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem).

    For example, using the root-finding case by way of example: suppose we find the
    root `y(θ)` for which `f(y(θ), θ) = 0`.
    Then we can skip backpropagating through the solver by computing
    $\frac{\mathrm{d}y}{\mathrm{d}\theta} = - (\frac{\mathrm{d}f}{\mathrm{d}y})^{-1}\frac{\mathrm{d}f}{\mathrm{d}\theta}$
    via the implicit function theorem.

    For most problems this is the preferred technique for backpropagating through
    a nonlinear solve.
    """  # noqa: E501

    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)

    def apply(self, primal_fn, rewrite_fn, inputs, tags):
        inputs = inputs + (ft.partial(eqxi.while_loop, kind="lax"),)
        return implicit_jvp(primal_fn, rewrite_fn, inputs, tags, self.linear_solver)


RecursiveCheckpointAdjoint.__init__.__doc__ = """**Arguments:**

- `checkpoints`: the number of checkpoints to save. The amount of memory used by the
    iterative solve will be roughly equal to the number of checkpoints
    multiplied by the size of `y0`. You can speed up backpropagation by allocating more
    checkpoints. (So it makes sense to set as many checkpoints as you have memory for.)
    This value can also be set to `None` (the default), in which case it will be set to
    `log(max_steps)`, for which a theoretical result is available guaranteeing that
    backpropagation will take `O(n log n)` time in the number of steps `n <= max_steps`.
"""

ImplicitAdjoint.__init__.__doc__ = """**Arguments:**

- `linear_solver`: the linear solver to solve the linear problem in the cotangent pass.
"""
