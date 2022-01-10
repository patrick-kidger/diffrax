# Interpolations

When solving controlled differential equations, it is relatively common for the control to be an interpolation of discrete data.

The following interpolation routines may be used to perform this interpolation.

!!! note

    Missing data, represented as `NaN`, can be handled here as well. (And if you are familiar with the problem of informative missingness, note that this can be handled as well: [see Sections 3.5 and 3.6 of this paper](https://arxiv.org/abs/2005.08926).)

??? cite "References"

    The main two references for using interpolation with controlled differential equations are as follows.

    Original neural CDE paper:
    ```bibtex
    @article{kidger2020neuralcde,
            author={Kidger, Patrick and Morrill, James and Foster, James and Lyons, Terry},
            title={{N}eural {C}ontrolled {D}ifferential {E}quations for {I}rregular {T}ime {S}eries},
            journal={Neural Information Processing Systems},
            year={2020},
    }
    ```

    Investigating specifically the choice of interpolation scheme for CDEs:
    ```bibtex
    @article{morrill2021cdeonline,
            title={{N}eural {C}ontrolled {D}ifferential {E}quations for {O}nline {P}rediction {T}asks},
            author={Morrill, James and Kidger, Patrick and Yang, Lingyi and Lyons, Terry},
            journal={arXiv:2106.11028},
            year={2021}
    }
    ```


??? info "How to pick an interpolation scheme"

    There are a few main types of interpolation provided here. For 99% of applications you will want either rectilinear or cubic interpolation, as follows.

    - Do you need to make online predictions at inference time?
        - Yes: Do you need to make a prediction continuously, or just every time you get the next piece of data?
            - Continuously: Use rectilinear interpolation.
            - At data: Might there be missing values in the data?
                - Yes: Use rectilinear interpolation.
                - No: Use Hermite cubic splines with backward differences.
        - No: Use Hermite cubic splines with backward differences.

    Rectilinear interpolation can be obtained by combining [`diffrax.rectilinear_interpolation`][] and [`diffrax.LinearInterpolation`][].

    Hermite cubic splines with backward differences can be obtained by combining [`diffrax.backward_hermite_coefficients`][] and [`diffrax.CubicInterpolation`][].

---

## Interpolation classes

The following are the main interpolation classes. Instances of these classes are suitable controls to pass to [`diffrax.ControlTerm`][].

::: diffrax.LinearInterpolation
    selection:
        members:
            - __init__
            - evaluate
            - derivative
            - t0
            - t1
        
::: diffrax.CubicInterpolation
    selection:
        members:
            - __init__
            - evaluate
            - derivative
            - t0
            - t1

---

## Handling missing data

We would like [`diffrax.LinearInterpolation`][] to be able to handle missing data (represented as `NaN`). The following can be used for this purpose.

::: diffrax.linear_interpolation

::: diffrax.rectilinear_interpolation

---

## Calculating coefficients

::: diffrax.backward_hermite_coefficients
