# Performance checks

A seven-run comparison on macOS arm64 with AppleClang 21, `-O2`, measured 10,000 sweeps of a one-dimensional standard-normal posterior, batch size 50, seed 42, original implementation worker count explicitly 1. The original f69609a sampler took a median **39.93 ms**; the sequential implementation took **0.657 ms**, roughly **61× faster for this small-posterior workload**. Timing included initialization; original progress output was redirected to an in-memory stream. This is not a general speedup or effective-sample-size claim. The algorithms and random sequences differ, and the original multiworker algorithm is not a valid baseline for inference quality.

`bench/sampler.cpp` provides a repeatable current-version seven-run benchmark. Run a Release build; debug/optimization settings change timing substantially. The benchmark retains and consumes results to prevent elimination.

The BEST likelihood also computes the Student-t normalization once per group instead of calling gamma functions for every observation. Standard deviation uses a scaled accumulation without an intermediate vector. Larger-data throughput and multi-chain parallel scaling have not been quantified here.
