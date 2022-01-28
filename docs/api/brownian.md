# Brownian controls

SDEs are simulated using a Brownian motion as a control. (See the neural SDE example.)

??? abstract "`diffrax.AbstractBrownianPath`"

    ::: diffrax.AbstractBrownianPath
        selection:
            members:
                - evaluate

---

::: diffrax.UnsafeBrownianPath
    selection:
        members:
            - __init__
            - evaluate

::: diffrax.VirtualBrownianTree
    selection:
        members:
            - __init__
            - evaluate
