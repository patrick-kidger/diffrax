# Extending Diffrax

It's completely possible to extend Diffrax with your own custom solvers, step size controllers, and so on.

For example, a custom solver should inherit from [`diffrax.AbstractSolver`][].
