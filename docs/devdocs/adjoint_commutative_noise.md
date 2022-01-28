# Higher-order commutative noise produces commutative noise during BacksolveAdjoint

## Statement

Consider solving the Ito SDE

$\mathrm{d}y(t) = μ(t, y(t))\mathrm{d}t + σ(t, y(t)) \mathrm{d}w(t)$

or the Stratonovich SDE

$\mathrm{d}y(t) = μ(t, y(t))\mathrm{d}t + σ(t, y(t))\circ\mathrm{d}w(t)$

in either case assumed to satisfy the commutativity condition

$σ_{i\, j_2} \frac{\partial σ_{k\, j_1}}{\partial y_i} = σ_{i\, j_1} \frac{\partial σ_{k\, j_2}}{\partial y_i}$.

Then the backward pass solved during [`diffrax.BacksolveAdjoint`][] will also satisfy the commutativity condition if and only if the following higher-order commutativity condition is satisfied.

$σ_{i\, j_2} \frac{\partial^2 σ_{k\, j_1}}{\partial y_i \partial y_m} = σ_{i\, j_1} \frac{\partial^2 σ_{k\, j_2}}{\partial y_i \partial y_m}$

!!! note

    The commutativity condition is a common prerequisite for solving an SDE with a higher-order solver.

!!! note

    The higher-order commutativity condition is satisfied by all the dominant subclasses of commutative noise: additive noise, diagonal noise, scalar noise. It is also satisfied by noise that is affine in the state $y$. But it is not obviously satisfied by commutative noise in general?

    As far as I know the higher-order commutativity condition is new here.

## Proof

Without loss of generality we consider specifically the reverse-time adjoint SDE (formally justified using rough path theory, see [1, Appendix C.3.3])

$\mathrm{d}a_i(t) = -a_j(t) \frac{\partial μ_j}{\partial y_i}(t, y(t))\mathrm{d}t - a_j(t) \frac{\partial σ_{j\, k}}{\partial y_i}(t, y(t)) \circ \mathrm{d}w_k.$

This is without loss of generality as:

- If the SDE is Ito then we convert it to Stratonovich; this incurs a correction term in the drift but does not affect the diffusion, and it is only the diffusion we are interested in.
- We do not consider the derivatives with respect to any parameters $θ$ as these may be treated as derivatives with respect to $y(0)$ in the usual way.
- We do not consider solving the original SDE for $y$ backwards-in-time. In isolation then by assumption this already has commutative noise. Then, taking any individual path $y$, we may treat the reverse-time adjoint SDE in isolation. (Note that the coupling between $y$ and $w$ is irrelevant: by rough path theory we may place ourselves in the deterministic setting.)

Let $Σ_{i\, k}(t, a) = -a_j \frac{\partial σ_{j\, k}}{\partial y_i}(t, y(t))$.

Then

$Σ_{i\, j_2} \frac{\partial Σ_{k\, j_1}}{\partial a_i} = a_m \frac{\partial σ_{m\, j_2}}{\partial y_i} δ_{i\, n} \frac{\partial σ_{n\, j_1}}{\partial y_k} = a_m \frac{\partial σ_{m\, j_2}}{\partial y_i} \frac{\partial σ_{i\, j_1}}{\partial y_k}$

Now differentiate the commutativity condition for $σ$, with respect to $y_m$, to obtain

$\frac{\partial σ_{i\, j_2}}{\partial y_m} \frac{\partial σ_{k\, j_1}}{\partial y_i} + σ_{i\, j_2} \frac{\partial^2 σ_{k\, j_1}}{\partial y_i \partial y_m} = \frac{\partial σ_{i\, j_1}}{\partial y_m} \frac{\partial σ_{k\, j_2}}{\partial y_i} + σ_{i\, j_1} \frac{\partial^2 σ_{k\, j_2}}{\partial y_i \partial y_m}$

which may be substituted into the previous equation to obtain

$a_m \frac{\partial σ_{m\, j_1}}{\partial y_i} \frac{\partial σ_{i j_2}}{\partial y_k} + a_m \left[ σ_{i\, j_2} \frac{\partial^2 σ_{m\, j_1}}{\partial y_i \partial y_k} - σ_{i\, j_1}\frac{\partial^2 σ_{m\, j_2}}{\partial y_i \partial y_k}\right]$

We recognise the first term as the desired commutativity relation for $Σ$; that is we will satisfy the commutativity relation if and only if the second term is zero. Now $a_m$ is arbitrary so by taking it to equal every basis vector in turn, we find that the the higher-order commmutativty condition for $σ$ is precisely the condition needed for $Σ$ to satisfy the commutativity condition.

## References

[1] Kidger, *On Neural Differential Equations*, PhD Thesis, University of Oxford, 2021
