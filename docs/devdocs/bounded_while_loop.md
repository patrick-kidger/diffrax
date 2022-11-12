# Bounded while loop

Some notes on implementing a bounded while loop in JAX. (Note that the bound is required for any hope of reverse-mode autodifferentability, due to the static memory requirements imposed by XLA.)

Let $n$ be the number of steps actually taken.  
Let $m$ be the maximum number of steps allowed.  
Let $d$ be the depth of the recursive structure, when one is used.  
Let $b$ be the base of the recursive structure, when one is used.  
(So roughly $b^d = m$.)

"Forward time" will refer to the amount of work done on the forward pass.  
"Backward time" will refer to the amount of work done on the backawrd pass, including recomputing from checkpoints.  
"Compile time" will refer to the size of the jaxpr or XLA HLO. (Which we assume to be proportional, although there are a few exceptions to this.)  
"Memory usage" will refer to the maximum amount of memory needed to store an entire forward pass, if we land in the case that $n=m$.

In practice, because XLA statically allocates memory, then the value specified by "memory usage" is actually allocated when performing a backward pass. And as spatial complexity bounds temporal complexity, then the actual backward time is the maximum of "backward time" and "memory usage". (!)

We use $O(\ldots)$ to denote the costs involved, as usual. We additionally introduce $I(\ldots)$ to denote the cost of performing identity operations, which are used in some implementations instead of making a step. Identity operations are very cheap but not completely free so we count them separately.

### Implementation 1: `scan`-`cond`

This implementation just does a `scan` for m steps, checking `cond` on each one.

Forward time: $O(n) + I(m)$  
Backward time: $O(n) + I(m)$  
Compile time: $O(1)$  
Memory usage: $O(m)$

Verdict: unsuitable, because of the huge memory usage. In addition the runtime $I(m)$ is disdvantageous.

### Implementation 2: nested `scan`-`cond`

This is probably the first serious idea you come up with when trying to write a bounded while loop. Do a `scan` for $b$ steps, checking `cond` on each one. Nest that implementation recursively $d$ times, so that you make a total of $m$ steps. That is, nested `scan`-`cond`-`scan`-...-`cond` where there are $d$-many `scan`s each of length $b$.

Forward time: $O(n) + I(db)$  
Backward time: $O(n) + I(db)$  
Compile time: $O(1)$  
Memory usage: $O(m)$

This fixes the $I(m)$ runtime of the previous implementation by nesting things, so that you start making larger identity steps once you're done. Unfortunately the $O(m)$ memory usage (and thus speed on the backward pass) remains, so this is still unsuitable.

### Implementation 3: treeverse

Okay, memory usage is an issue. The obvious thing to do is to start thinking about gradient checkpointing, for which treeverse is the known optimality result. Assuming $b=2$ for simplicity/optimality, then this is arrived at by recursively calculating `fn(jax.checkpoint(fn)(x))` where the base case takes `fn` to be a `scan` over $b=2$ steps.

[Morally speaking this is taking the same tree structure as in Implementation 2 and then adding some checkpoints.]

Forward time: $O(n) + I(d)$  
Backward time: $O(n \log n) + I(d \log d)$  
Compile time: $O(m)$  
Memory usage: $O(d)$

[Assuming $b=2$ and therefore it doesn't appear in these values.]

Great, we've fixed our memory usage! Note that the additional work needing to recompute from our checkpoints increases our backward computation time slightly.

Unfortunately the compile time has exploded: every level of our recusion involves calling `fn` twice (once inside the checkpoint, once outside) and by doing so recursively we're making $2^d = m$ such calls. Both the jaxpr and the resulting XLA HLO will be of size $O(m)$, as we've basically just written out the whole loop manually! Compile times are already one of the most serious issues facing the JAX ecosystem, so this is also unacceptable.

Whilst treeverse is optimal for run time, it is maximally nonoptimal for compile time.

### Implementation 4: naive checkpointing

Next let's try naive checkpointing. This just means picking some $\sqrt{m}$ equally-spaced points between $0$ and $m$ and placing a checkpoint at each one. Unlike treeverse, this does not use any recursive checkpointing. [Note that this is the kind of checkpointing you often see used in practice with e.g. ResNets etc.]

This can be implemented very simply: nest `scan`-`cond`-`checkpoint`-`scan`-`cond`, where the length of each `scan` is $\sqrt{m}$.

Forward time: $O(n) + I(\sqrt{m})$  
Backward time: $O(n) + I(\sqrt{m})$  
Compile time: $O(1)$  
Memory usage: $O(\sqrt{m})$

Each intermediate step is re-computed from a checkpoint precisely once, so the backward pass has the same complexity as the forward pass.

This is a surprisingly decent option: $O(\sqrt{m})$ represents much worse memory usage (and therefore backward computation time) than we'd like, but this still represents a not-completely-awful trade-off compared to our previous options.

### Implementation 5: nested `checkpoint`-`scan`-`cond`

Can we combine the best pieces of implementations 2/3 and 4? In other words, nest `scan`-`checkpoint`-`scan`-`cond`-`checkpoint`-`scan`-`cond`-...`checkpoint`-`scan`-`cond`, with $d$-many `scan`s each of length $b$. As an example, in the $b=2$ case and unrolling any individual `scan` produces something a bit like implementation 3, except with `jax.checkpoint(fn)(jax.checkpoint(fn)(x))` instead.

Forward time: $O(n) + I(db)$  
Backward time: $O(dn) + I(db)$  
Compile time: $O(1)$  
Memory usage: $O(db)$

Overall we have performed $O(dn) + I(db)$ work on the backward pass. This is _liveable_... but still not stellar. That $d$ factor slows the backward pass down by a noticable factor. We see that for this to work, we must choose $b \neq 2$ (often an optimal value), as otherwise $d$ becomes large. In practice I've found that tractable values for an ODE solve are something like $d=3$ and $b=16$, for a maximum number of $16^3 = 4096$ steps.

This is at least better than implementation 4, in that the memory usage, and therefore the practical backward time, has come down from $O(\sqrt{m})$ to $O(b \log m) (=O(db))$.

Theoretical justification for these values as follows:

The forward time and compile time are both as in implementation 2. The memory usage can be found by considering backpropagating from the step just prior to the end: we have saved $b-1$ checkpoints at the top level, and we have saved $b-1$ checkpoints at the second (nested) level, etc., for $d$ levels.

Now for the runtime of the backward pass.

Suppose we take a lot of steps, so that $n \approx m$. Consider reconstructing the final iteration of the top-level `scan` from its checkpoint at the start. This takes $O(m/b)$ work (the forward evaluation over the proportion of the overall interval that that final iteration covers), and leaves us with a number of checkpoints along the second-level `scan`. The forward evaluation through each of those in turn takes $O(m/b^2)$ work -- by the same logic -- and there are $b$ many of them, once again requiring $O(m/b)$ overall work. This happens for $d$ many levels, so that the overall amount of work to backpropagate through this final iteration is actually $O(dm/b)$. Now the fact that it was the final iteration didn't actually affect this analysis (that was just for pedagogical simplicity), so we do the above procedure $b$ times, for an overall $O(dm) = O(dn)$ amount of work. Meanwhile we take very few identity steps, so the $I$ term is approximately zero.

Now suppose that we take very few steps, so that $n$ is much smaller than $m$, and in fact contained within just the first top-level iteration (i.e. $n < m/b$). Then all the latter iterations of the top-level `scan` are just the identity and do not contribute anything to our $O$ measurement, so consider just the first iteration. We are now within our top-level checkpointed region, and so need to recompute all of our checkpoints. Once again suppose $n$ is very small and contained within just the first sub-iteration (that is $n < m/b^2$). Repeat ad nauseum, so that the entirety of our $O$-measured work is contained within the very first bottom-level iteration. This bottom-level iteration takes $O(n)$ work to compute in isolation. However we have recomputed it and then discarded it many times: $d - 1$ times, to be precise. Once when computing the checkpoints for the second-level iteration; once when computing the checkpoint for the third-level iteration; etc. And thus overall we have performed $O(dn)$ work. (Meanwhile, the number of identity steps we have neglected in this analysis cost $I(db)$. Indeed they are counted in an identical manner to the forward pass.)

### Implementation 6?

Maybe there's another better way of doing it? I make no claims that the above result is as good as it gets.

## Coda

### Optimums

The theoretical optimum without checkpointing is:

Forward time: $O(n)$  
Backward time: $O(n)$  
Compile time: $O(1)$  
Memory usage: $O(n)$

and with treeverse it is:

Forward time: $O(n)$  
Backward time: $O(n \log n)$  
Compile time: $O(1)$  
Memory usage: $O(\log n)$

It is clearly impossible to obtain the non-checkpointing optimum under the JAX/XLA model of computation, due to the requirement that all memory must be statically allocated in advance. (This is a great pity, as it's also the single best option for most problems.)

- Ever-so-maybe it might be possible to achieve the treeverse optimal value by writing `bounded_while_loop` as a new primitive? This would certainly reduce the jaxpr size down to $O(1)$, but it's not clear (at least to me) what the size of the backward pass, expressed as an XLA HLO expression, must be -- and compile times are proportional to that too.
- Alternatively, a way to compile a function only once (rather than inlining everything) would also make it possible to represent treeverse, as then `jax.checkpoint(fn)` can be compiled in constant time from `fn`, without introducing an exponential explosion as depth progresses.

### Higher-order derivatives

I don't know of anything discussing the interaction between checkpointing schemes and higher-order autodifferentiation. Given that a checkpointing scheme is required for memory usage (and thus backward pass speed) to be tractable, then it's not clear to me what the best approach is when this is a concern that needs to be born in mind.

### Other implementation complexities

JAX has a variety of other limitations that must be worked around when building a bounded while loop. Most noticably:

- Handling `vmap` appropriately, as `vmap`'ing a `cond` produces a `select`. (Which would then always run the entire loop to completion.)
- Handling in-place updates. The recursively nested structures here mean that XLA:CPU is unable to optimise away in-place updates made during the body function of the while loop. (And instead makes copies.)
- Actually getting the compile time that the above asymptotics promise. In particular it is possible to get a compile time that is exponential in the size of the program when using nested `cond`s.

(See the implementation itself for further thoughts on these.)
