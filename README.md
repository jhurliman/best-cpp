# best-cpp

[![CI](https://github.com/jhurliman/best-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/jhurliman/best-cpp/actions/workflows/ci.yml)

**Estimate the difference between two groups and its uncertainty, in C++.** `best-cpp` fits a two-group Student-t Bayesian model and samples the posterior distribution of the difference in means.

The model is inspired by [Bayesian estimation supersedes the t test](https://www.krigolsonteaching.com/uploads/4/3/8/4/43848243/kruschke2012jepg.pdf). Each group has its own mean and scale, with shared degrees of freedom that allow heavier tails than a normal distribution.

- **Inspect the estimate.** Get a mean difference, a 95% empirical interval, and the full chain for your own analysis.
- **Use it directly from C++.** Header-only, C++17, with no third-party numerical dependencies.
- **Build your own model.** The underlying adaptive Metropolis-within-Gibbs sampler accepts a log-posterior callback.
- **Choose your build system.** CMake installation and Bazel monorepo consumption have independent consumer checks.

## Compare two groups

This complete program samples the difference **A mean − B mean**:

```cpp
#include <best.hpp>
#include <iostream>
#include <vector>

int main() {
  const std::vector<double> a{1.8, 2.0, 2.2, 1.9, 2.1};
  const std::vector<double> b{2.8, 3.0, 3.2, 2.9, 3.1};
  BEST<std::vector<double>> model(a, b, 50, 42); // batch size, seed

  model.Burn(5000);
  model.Sample(20000);

  std::pair<double, double> interval;
  double difference;
  model.ComputeStats(interval, difference);
  std::cout << "Mean difference: " << difference << '\n'
            << "95% sample interval: [" << interval.first
            << ", " << interval.second << "]\n";
}
```

Run lengths are illustrative, not a convergence guarantee. Check multiple independently seeded chains and effective sample size before interpreting results; those diagnostics are not built in. Small or weakly identified datasets can produce broad posteriors. A fixed seed reproduces a run within the same implementation and C++ standard library, not bit-for-bit across all toolchains.

## CMake

No external numerical library is required. To install the headers and exported target:

```sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --install build
```

Then use the package from your application:

```cmake
find_package(best_cpp 1 CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE best_cpp::best_cpp)
```

Set `CMAKE_PREFIX_PATH` to the installation prefix. `add_subdirectory` exposes the same target. The target supplies the include directory and C++17 requirement without changing global compiler flags. Tests, CLI and benchmarks are off by default.

For a raw-header integration, add the source `best/` directory to your include path. Public headers are `<best.hpp>`, `<amwg.hpp>` and `<stats.hpp>`. See the complete [installed CMake consumer](examples/cmake-consumer).

## Bazel monorepos

Version 1.0.0 is prepared in this branch but is not yet released or registered in BCR. Use a local checkout first:

```starlark
# MODULE.bazel
bazel_dep(name = "best_cpp", version = "1.0.0", repo_name = "bayes")
local_path_override(module_name = "best_cpp", path = "third_party/best-cpp")
```

Add `@bayes//:best_cpp` to your target's `deps`. The [independent consumer](examples/bazel-consumer) demonstrates repository renaming. Configure C++17 or later in your consuming toolchain (`--cxxopt=-std=c++17` for GCC/Clang); Bazel 9.2 is tested.

For a remote dependency, replace the local override with `git_override(module_name = "best_cpp", remote = "https://github.com/jhurliman/best-cpp.git", commit = "<reviewed full commit SHA>")`. After BCR registration, the override can be removed. [RELEASING.md](RELEASING.md) covers the archive-based registry test and submission process.

## Working with samples

| Operation | Behavior |
| --- | --- |
| `Burn(n)` | Advance the sampler without retaining those draws. |
| `Sample(n)` | Append exactly `n` completed sweeps. |
| `ComputeStats(interval, mean)` | Summarize the difference in group means; requires retained samples. |
| `chain()` | Read the five-parameter draws for external diagnostics. |
| `LogPosterior(parameters)` | Evaluate the model's joint log density. |

BEST owns its observations and cannot be copied or moved. Inputs must be nonempty and finite with positive pooled variance. Draws contain `(mu1, mu2, sigma1, sigma2, nu)`.

The model retains the historical broad mean prior: pooled mean with standard deviation equal to pooled SD × 1,000,000. Group scales have uniform priors from pooled SD / 1,000 to pooled SD × 1,000; `nu − 1` has an exponential prior with mean 29. These choices matter when data is sparse. See [API.md](API.md) for bounds, exception behavior and the precise empirical-interval definition.

## A custom posterior

The sampler is also usable without the two-group model. This example draws from a standard normal target:

```cpp
#include <amwg.hpp>
#include <iostream>

int main() {
  AMWG<double, 1> sampler(50, 42);
  sampler.Init({{0.0}}, [](const std::array<double, 1>& x) {
    return -0.5 * x[0] * x[0]; // log density, up to a constant
  });
  sampler.Burn(5000);
  sampler.Sample(10000);
  std::cout << sampler.chain().size() << " draws\n";
}
```

Coordinate updates are sequential. Run independently owned instances with separate seeds for parallel chains; do not mutate one instance concurrently. `Init` takes only the starting state and callback—the old thread-count argument has been removed.

## Development and validation

```sh
cmake -S . -B build -DBEST_CPP_BUILD_TESTS=ON -DBEST_CPP_BUILD_CLI=ON -DBEST_CPP_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/best-benchmark
```

CI checks numerical reference values, known-distribution samples, lifecycle behavior, reproducibility and imbalanced inputs. It also checks sanitizers, standalone headers, Windows/Linux/macOS CMake consumers and independent Bazel archive consumption. These are regression checks, not a proof of statistical convergence.

The optional CLI compares two whitespace-separated numeric files: `./build/best group-a.txt group-b.txt`. [PERFORMANCE.md](PERFORMANCE.md) reports a narrowly scoped sampler benchmark. [CHANGELOG.md](CHANGELOG.md) explains the corrected sampling behavior and incompatible API changes; previous inference results should be recomputed.

## License

[MIT](LICENSE).
