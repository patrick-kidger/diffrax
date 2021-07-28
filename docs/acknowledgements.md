# Acknowledgements

diffrax is built on the shoulders of giants.

First and most obviously is JAX (and XLA). This makes diffrax possible in the place.

There have been several preceding diffeq libraries -- most notably torchdiffeq, torchsde, torchcde, DifferentialEquations.jl. Much of diffrax is new, but much of it draws inspiration from time spent using or developing these.

Several specific pieces can be attributed:
- The fehlberg2/bosh3/dopri5/dopri8/heun Butcher tableaus have been copied verbatim from torchdiffeq.
- The adaptive step sizing comes from DifferentialEquations.jl (https://diffeq.sciml.ai/stable/extras/timestepping/)
- The tsit5 tableau has been copied from torchdyn.
- The BrownianInterval has been copied from torchsde.
- The interpolations have been copied from torchcde.
