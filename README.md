# best-cpp

Header-only C++17 implementation of a two-group Student-t Bayesian model inspired by [Bayesian estimation supersedes the t test](https://www.krigolsonteaching.com/uploads/4/3/8/4/43848243/kruschke2012jepg.pdf). MIT licensed; no third-party C++ dependencies.

```cpp
#include <best.hpp>
#include <vector>

std::vector<double> a{1.8, 2.0, 2.2, 1.9, 2.1};
std::vector<double> b{2.8, 3.0, 3.2, 2.9, 3.1};
BEST<std::vector<double>> model(a, b, 50, 42); // batch size, seed
model.Burn(5000);
model.Sample(20000);
std::pair<double, double> interval;
double difference;
model.ComputeStats(interval, difference); // a mean minus b mean
```

The run lengths above are illustrative. Examine multiple independently seeded chains and convergence/effective sample size before interpreting results; these diagnostics are not implemented here. A fixed seed is reproducible with the same implementation and C++ standard library, not across arbitrary toolchains. Small or weakly identified data can produce very broad posteriors.

## Bazel monorepos

The public target is `@bayes//:best_cpp`. Until a release is registered in BCR, use a pinned Git commit override in the root `MODULE.bazel`:

```starlark
bazel_dep(name = "best_cpp", version = "1.0.0", repo_name = "bayes")
git_override(
    module_name = "best_cpp",
    remote = "https://github.com/jhurliman/best-cpp.git",
    commit = "<full commit containing MODULE.bazel>",
)
```

Alternatively use `local_path_override(module_name = "best_cpp", path = "/path/to/best-cpp")`. The runnable [consumer example](examples/bazel-consumer) tests this with a renamed repository. Set C++17 or later in the consuming toolchain (`--cxxopt=-std=c++17` for GCC/Clang). No repository-name assumptions or third-party numerical dependencies are imposed. Bazel 9.2 is tested; the module uses rules_cc 0.2.22. After BCR registration, the override can be removed. This prepared version is **not registered or released yet**.

## CMake

```sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --install build
```

Consumers use `find_package(best_cpp 1 CONFIG REQUIRED)` and `target_link_libraries(app PRIVATE best_cpp::best_cpp)`, with the install prefix in `CMAKE_PREFIX_PATH`. `add_subdirectory` exposes the same target. No global compiler flags are changed and tests/examples are off by default. The target preserves `#include <best.hpp>`, `<amwg.hpp>` and `<stats.hpp>`; installed files live under `include/best_cpp`. Raw-header consumers can add the source `best/` directory to their include path.

## Sampling and model contract

- `AMWG<Real, N>::Init(start, logPosterior, threads)` accepts a finite starting state/log density. The callback returns a **log** density. `-infinity`, NaN and positive infinity proposals are rejected. Exceptions propagate; a partially completed sweep can have advanced state but is not recorded.
- `Sample(n)` appends exactly n completed sweeps. `NextSample()` appends one and returns N proposals. `Burn(n)` advances without modifying stored draws. Repeated calls preserve adaptation; `Init` resets it and reseeds the generator.
- Updates are sequential Metropolis-within-Gibbs with diminishing batch adaptation toward 0.44 acceptance. The legacy threads argument is accepted but ignored. Run separate instances with separate seeds for parallel chains. A single instance is not safe for concurrent mutation.
- BEST owns its observations and is not copyable/movable. Groups must be nonempty and finite with positive pooled population variance; extreme scales whose prior bounds cannot be represented are rejected. Constant individual groups are allowed when pooled variance is positive.
- Parameters are `(mu1, mu2, sigma1, sigma2, nu)`. For compatibility with the historical implementation, mean priors have pooled mean and **pooled SD × 1,000,000**; sigma priors are uniform from pooled SD / 1,000 to pooled SD × 1,000; `nu-1` is exponential with mean 29. The broad mean scale is a library choice, not a claim of exact equivalence to other BEST software.
- `chain()` exposes draws for external diagnostics; `LogPosterior(params)` supports inspection. `ComputeStats` requires samples and returns the mean difference and shortest interval covering ceil(0.95*n) empirical draws. This is a shortest contiguous sample interval, not a general multimodal highest-density region.
- Statistics use floating-point containers; `stdev` is the population SD. Invalid domains/empty data throw exceptions. Extreme-tail densities can round to zero; use log-density helpers for inference.

## Development

```sh
cmake -S . -B build -DBEST_CPP_BUILD_TESTS=ON -DBEST_CPP_BUILD_CLI=ON -DBEST_CPP_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/best-benchmark
bazelisk test //:regression --cxxopt=-std=c++17
BAZEL=bazelisk python3 tools/test_bcr.py
```

The CLI accepts two whitespace-separated numeric files and reports malformed/missing input as errors. Existing IDE project files are historical; CMake is the supported portable build path. See [CHANGELOG.md](CHANGELOG.md), [PERFORMANCE.md](PERFORMANCE.md) and [RELEASING.md](RELEASING.md).
