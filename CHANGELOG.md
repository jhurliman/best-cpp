# Changelog

## 1.0.0 (prepared, not published)

- Replace speculative parallel proposals with sequential adaptive Metropolis-within-Gibbs. Remove shared random-generator races and proposal-selection bias. Remove the thread-count argument from `AMWG::Init`; callers now pass only the starting state and log-posterior callback.
- `Sample(n)` records exactly n completed sweeps; `Burn(n)` advances without retaining samples. Adaptation tracks sweeps independently of chain storage. Reinitialization resets the seeded generator and adaptation. Sampling no longer prints progress.
- Validate initialization, observations, batch sizes and statistical parameters. Callback exceptions propagate on the calling thread; a partially completed sweep may have advanced the state, but is not added to the chain.
- Enforce both sigma bounds and nu >= 1, evaluate the model in log space, retain double precision, and reuse Student-t normalization across observations.
- Own model observations to prevent dangling references. BEST cannot be copied or moved because its callback refers to the model.
- Define the empirical 95% interval as the shortest window containing ceil(0.95*n) observations, choosing the first window on ties. Reject empty/nonfinite inputs.
- Compute population standard deviation without a temporary allocation. Add numerical, lifecycle, seeded-distribution and independent-chain regression checks.

These corrections change seeded sequences, inference results and invalid-input behavior. Old results should be recomputed. Reproducibility is within the same implementation and standard library; C++ random-distribution algorithms are not portable bit-for-bit.

- Add CMake install/export targets, Bzlmod library and renamed-repository consumers, archive-based registry tests, standalone header checks, optional CLI/benchmark targets and Linux/macOS/Windows CI.
- Clamp initial group scales to both prior bounds, including highly imbalanced group sizes.
