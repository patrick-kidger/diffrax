import abc

import equinox as eqx


class AbstractNonlinearSolver(eqx.Module):
    @abc.abstractmethod
    def __call__(self, fn, x, args):
        pass
