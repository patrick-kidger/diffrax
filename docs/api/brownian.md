# Brownian controls

SDEs are simulated using a Brownian motion as a control. (See the neural SDE example.)

??? abstract "`diffrax.AbstractBrownianPath`"

    ::: diffrax.AbstractBrownianPath
        options:
            members:
                - evaluate

---

::: diffrax.VirtualBrownianTree
    options:
        members:
            - __init__
            - evaluate

::: diffrax.UnsafeBrownianPath
    options:
        members:
            - __init__
            - evaluate

---

## Lévy areas

Brownian controls can return certain types of Lévy areas. These are iterated integrals
of the Brownian motion, and are used by some SDE solvers. When a solver requires a 
Lévy area, it will have a `minimal_levy_area` attribute, which will always return an
abstract Lévy area type, and it can accept any subclass of that type.
The inheritance hierarchy is as follows:
```
AbstractBrownianIncrement
│   └── BrownianIncrement
└── AbstractSpaceTimeLevyArea
    │   └── SpaceTimeLevyArea
    └── AbstractSpaceTimeTimeLevyArea
            └── SpaceTimeTimeLevyArea
```
For example if `solver.minimal_levy_area` returns an `AbstractSpaceTimeLevyArea`, then
the Brownian motion (which is either an `UnsafeBrownianPath` or 
a `VirtualBrownianTree`) should be initialized with `levy_area=SpaceTimeLevyArea` or 
`levy_area=SpaceTimeTimeLevyArea`. Note that for the Brownian motion,
a concrete class must be used, not its abstract parent.

::: diffrax.AbstractBrownianIncrement
    options:
        members: false

::: diffrax.BrownianIncrement
    options:
        members: false

::: diffrax.AbstractSpaceTimeLevyArea
    options:
        members: false

::: diffrax.SpaceTimeLevyArea
    options:
        members: false

::: diffrax.AbstractSpaceTimeTimeLevyArea
    options:
        members: false

::: diffrax.SpaceTimeTimeLevyArea
    options:
        members: false