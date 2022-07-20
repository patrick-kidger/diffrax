# Events

Events allow for interrupting a differential equation solve, and changing its internal state, or terminating the solve before `t1` is reached.

At the moment a single kind of event is supported: discrete events which are checked at the end of every step, and which halt the integration once they become true.

??? abstract "`diffrax.AbstractDiscreteTerminatingEvent`"

    ::: diffrax.AbstractDiscreteTerminatingEvent
        selection:
            members:
                - __call__

---

::: diffrax.DiscreteTerminatingEvent
    selection:
        members:
            - __init__

::: diffrax.SteadyStateEvent
    selection:
        members:
            - __init__
