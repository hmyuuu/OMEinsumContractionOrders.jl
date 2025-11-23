# [Optimizer Guidelines](@id Sec_Guidelines)

This page condenses the most practical advice for working with the optimizers provided by
OMEinsumContractionOrders.jl. It does **not** replace the rest of the manual—instead, treat it as a
checklist and then dive into the background, tutorial and reference pages of this project for the
full story.

## Treewidth-family workflow

[`Treewidth`](@ref Sec_Treewidth) and [`ExactTreewidth`](@ref Sec_ExactTreewidth) convert an einsum
expression into a line graph, compute a tree decomposition, and translate that decomposition back
into a contraction tree. Keep the following points in mind:

1. **Pick an elimination algorithm before you start.**
   - Use `Treewidth(alg=AMF())` or `Treewidth(alg=LexBFS())` for very large instances where speed is
     the priority.
   - Use `Treewidth()` (which expands to `SafeRules(BT(), MMW{3}(), MF())`) when you want a solid
     quality/speed trade-off.
   - Use [`ExactTreewidth()`](@ref Sec_ExactTreewidth) only when you need the true optimum on networks
     with a few dozen tensors.
2. **Normalize size dictionaries.** Treewidth-based optimizers internally work with log₂ weights, so
   call `uniformsize` or otherwise ensure `size_dict` contains precise integer dimensions.
3. **Control binarization.** If you set `binary=false` in [`optimize_treewidth`](@ref), you receive the
   raw multi-way tree decomposition. Leave it at the default (`true`) when you want BLAS-friendly
   binary trees.
4. **Score the result immediately.** Call `contraction_complexity` on the returned `NestedEinsum` to
   record time/space/read-write metrics, then store them alongside your experiment metadata.
5. **Iterate on heuristics.** Because each elimination algorithm has different strengths, keep a list
   of candidates (e.g. `LexBFS`, `MF`, `SafeRules`) and sweep over them when investigating a new
   family of tensor networks.

> 💡 **Tip:** Always keep the tensors and index sets sorted deterministically before calling the
> optimizer. That way, regression failures are easier to reproduce when you revisit this project.

## TreeSA playbook

[`TreeSA`](@ref Sec_TreeSA) searches over binary contraction trees using simulated annealing. Use the
following checklist to configure it correctly:

- **Hyperparameters**
  - `βs`: schedule of inverse temperatures. Start with something coarse such as `range(0.1, 4.0,
    length=20)` and refine once you know how quickly your instance cools.
  - `ntrials`: number of randomized restarts. Small networks (≤ 100 tensors) often converge with
    `ntrials=4`; noisy or highly irregular networks benefit from 16 or more.
  - `niters`: annealing steps per trial. Increase this when you notice the score plateauing too early.
  - `initializer`: defaults to a randomized contraction tree; set it to `Treewidth()` or `GreedyMethod()`
    when you want TreeSA to polish an existing solution.
- **Scoring**
  - Pass a [`ScoreFunction`](@ref) with explicit time/space/read-write weights so TreeSA optimizes the
    metric that matters for your hardware budget.
- **Slicing**
  - Combine TreeSA with [`TreeSASlicer`](@ref) when space complexity dominates. First optimize the
    contraction tree, then apply slicing with a `sc_target` that matches your memory limit.
- **Diagnostics**
  - Record the best score from each trial. If the variance across trials is large, increase the number
    of trials or widen the temperature schedule.

TreeSA excels when you already have a decent tree (from `Treewidth` or `GreedyMethod`) and need extra
quality. Because its runtime scales with `ntrials × niters`, run a quick pass with tiny settings to
get a baseline, then ramp the hyperparameters only when necessary.

## Picking between Treewidth and TreeSA

| Situation | Recommended optimizer |
| --- | --- |
| Exact reference for ≤ 50 tensors | [`ExactTreewidth()`](@ref Sec_ExactTreewidth) |
| Fast baseline for large sparse networks | `Treewidth(alg=AMF())` or `Treewidth(alg=LexBFS())` |
| Balanced speed/quality without tuning | `Treewidth()` (SafeRules + BT fallback) |
| Need to squeeze out the last few bits of space/time | `TreeSA` initialized from a Treewidth result |

When in doubt, start with `Treewidth()` to obtain a contraction tree quickly. If the resulting
space/time complexity is still too high, feed that tree into `TreeSA` and let it search locally for a
better structure.

## Further reading

For the theoretical background, walk through the rest of this project:

- [`background.md`](background.md) explains tensor networks, tree decompositions and slicing.
- [`tutorial.md`](tutorial.md) demonstrates the API step by step.
- [`optimizers.md`](optimizers.md) catalogs every optimizer with detailed descriptions.
- [`ref.md`](ref.md) hosts the autodocs for every public symbol.

Keeping this guideline handy while you read the above pages will help you connect the big-picture
ideas with the practical knobs exposed by the code.
