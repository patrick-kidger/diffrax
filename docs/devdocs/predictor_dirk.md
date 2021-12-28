# Predictors for Diagonal Implicit Runge--Kutta (DIRK) methods

## Linear-combination-predictors

Each stage of a diagonal implicit RK method involves solving (via Newton iterations) a nonlinear system

$f_i = f(y_0 + \sum_{j=1}^i a_{ij} f_j Δt)$

for $i = 1, \ldots, s$.

How do we initialise our guess for $f_i$? Given the information available to us at each stage, a natural choice is to take some linear combination of the $f_j$ already available to us. (This feels a bit like using an explicit RK (ERK) method, although we don't use $y_0$ and don't make any extra function evaluations.)

Making a good choice of predictor is important, and can have a strong impact on whether the Newton solver converges. One anecdotal example: for a particular stiff ODE, naively always predicting $f_i=0$ meant the Kvaerno5 solver could solve about 34% of initial conditions. (And otherwise failures of the Newton solver, even over adaptively-chosen small timesteps, meant that the ODE could not be resolved in a timely manner.) Switching to a linear-combination-predictor increased this dramatically, to 98% of initial conditions studied.

What is an appropriate choice of linear combination? This is something that seems to have frequently been left out of the literature, e.g.:

- Kvaerno's paper presents only the implicit tableaus.
- Hairer and Wanner only seems to discuss predictors for fully-implicit RK (FIRK) methods (by polynomially interpolating the stages of the previous step).
- Mistakes are made in practice, e.g. [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl/blob/6fdc99fa4da79633e0161f0bb8aaff3f39cd39bc/src/perform_step/kencarp_kvaerno_perform_step.jl#L652) use zero as the predictor for a second stage of an ESDIRK method -- when it would make sense to at least use the result of the first stage.

As a result, an additional part of each DIRK method's Butcher tableau is some collection of values $α_{ij}$ for $i > j$. At each stage we will initialise

$f_i = \sum_{j = 1}^{i-1} α_{ij} f_j$

as our prediction for the nonlinear system. Looking at this, it is natural to require that

$1 = \sum_{j=1}^{i-1} α_{ij}$.

[The value $f_i$ is likely close to each value of $f_j$; we have no reason to suppose that it is e.g. half their values.]

## Implications

Now, this implies that the choice of $α_{ij}$ is actually quite constrained.

### First stage
First, note that the entire discussion above only applies to the second stage and beyond. For the first stage, we don't have access to any $f_j$.

- If the method satifies the FSAL property then we don't care: we already have $f_1 = f(y_0)$ because we got it from the previous step.
- If the method has an explicit first stage (as with ESDIRK methods) then we also don't care: we're just going to make an explicit step anyway.
- If the method has an implicit first stage (and necessarily fails the FSAL property) then we're out of luck. Bite the bullet, make an extra function evaluation, and evaluate $f(y_0)$ as an estimate for $f_1$. [Although perhaps something could be cooked up using the evaluations of the previous step?]

### Second stage
For the second stage: our sum-to-1 property means that we necessarily have $1 = α_{21}$. [This was the mistake made in OrdinaryDiffEq.jl, above: they just take this value to be zero.]

### Third stage
For the third stage: the sum-to-1 property means that we have $1 = α_{31} + α_{32}$. Now we only have two values $f_1$ and $f_2$ available to us, so there's not a lot we can do other than assume that $t \mapsto f(y(t))$ is locally linear, and extrapolate. Letting $F(t) = ζt + η$ be this local linear approximation, we have $F(c_1 Δt) = f_1$ and $F(c_2 Δt) = f_2$, giving

$F(c_3 Δt) = f_2 + (f_2 - f_1) (c_3 - c_2) / (c_2 - c_1)$

and thus $α_{31} = (c_2 - c_3) / (c_2 - c_1)$ and $α_{32} = (c_3 - c_1) / (c_2 - c_1)$.

These may be negative: for example given an explicit first stage ($c_1 = 0$) then necessarily $α_{31} < 0$.

### Higher stages
For higher stages: now things start to get a bit more interesting / unexplored. One reasonable choice is to fit a higher-order polynomial to $t \mapsto f(y(t))$. Another reasonable choice is to keep using a linear approximation, using your two favourite stages from before. A final choice is to take $α_{ij} = a_{(i-1)j}$, although this is only appropriate for those stages for which $1 = \sum_{j=1}^{i-1} a_{(i-1)j}$; this typically true of the last few stages (which often have $\sum_{j=1}^i a_{ij} = c_i = 1$).

At time of writing (and to the best of my knowledge), the best choice for higher stages isn't really known.

## See also
See also [here](https://github.com/SciML/OrdinaryDiffEq.jl/blob/6fdc99fa4da79633e0161f0bb8aaff3f39cd39bc/src/tableaus/sdirk_tableaus.jl#L1209) (along with other similar examples in that file) who derive predictors using polynomial interpolations.

- Any time they reference "Hermite" they actually obtain the result given by the procedure we describe for the third stage.
- For their higher-order interpolations it's not 100% clear what they're doing. They appear to be extrapolating the solution $y$ rather than the vector field $f$, which is bit a confusing. (Their implementation still solves the nonlinear system for $f_i = f(y_i)$, rather than the equivalent nonlinear system for $y_i$.) Should follow up on this.

## Remark

This document isn't meant to be authoritative, and I'd welcome any comments.

