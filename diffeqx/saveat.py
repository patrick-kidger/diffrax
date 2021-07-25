from .custom_types import Array


class SaveAt:
    def __init__(self, *, t0: bool = False, t1: bool = False, t: Optional[Array["times"]] = None, steps: bool = False, controller_state: bool = False, solver_state: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.t0 = t0
        self.t1 = t1
        self.t = t
        self.steps = steps
        self.controller_state = controller_state
        self.solver_state = solver_state

