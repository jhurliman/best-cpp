# Sampling and model reference

- `AMWG<Real, N>::Init(start, logPosterior)` accepts a finite starting state/log density. The callback returns a **log** density. `-infinity`, NaN and positive infinity proposals are rejected. Exceptions propagate; a partially completed sweep can have advanced state but is not recorded.
- `Sample(n)` appends exactly n completed sweeps. `NextSample()` appends one and returns N proposals. `Burn(n)` advances without modifying stored draws. Repeated calls preserve adaptation; `Init` resets it and reseeds the generator.
- Updates are sequential Metropolis-within-Gibbs with diminishing batch adaptation toward 0.44 acceptance. The thread-count argument has been removed. Run separate instances with separate seeds for parallel chains. A single instance is not safe for concurrent mutation.
- BEST owns its observations and is not copyable/movable. Groups must be nonempty and finite with positive pooled population variance; extreme scales whose prior bounds cannot be represented are rejected. Constant individual groups are allowed when pooled variance is positive.
- Parameters are `(mu1, mu2, sigma1, sigma2, nu)`. For compatibility with the historical implementation, mean priors have pooled mean and **pooled SD × 1,000,000**; sigma priors are uniform from pooled SD / 1,000 to pooled SD × 1,000; `nu-1` is exponential with mean 29. The broad mean scale is a library choice, not a claim of exact equivalence to other BEST software.
- `chain()` exposes draws for external diagnostics; `LogPosterior(params)` supports inspection. `ComputeStats` requires samples and returns the mean difference and shortest interval covering ceil(0.95*n) empirical draws. This is a shortest contiguous sample interval, not a general multimodal highest-density region.
- Statistics use floating-point containers; `stdev` is the population SD. Invalid domains/empty data throw exceptions. Extreme-tail densities can round to zero; use log-density helpers for inference.


See [README.md](README.md) for a complete example and build integration.
