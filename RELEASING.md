# Releasing best-cpp

Version 1.0.0 is the first versioned release. It changes historical sampling results, seeded sequences, threading behavior and error handling; read CHANGELOG.md before upgrading. Recompute previous inference results.

1. Merge the sampler-correctness PR, then its packaging follow-up. Require Linux/macOS regression and sanitizer jobs, Windows/Linux/macOS CMake consumers, and the Bazel archive-consumer job to pass.
2. Verify `CMakeLists.txt`, `MODULE.bazel`, `best/version.hpp`, the consumer module and changelog agree on 1.0.0. Run the commands in README.md from the reviewed commit. The archive test uses committed HEAD, so commit release files before running it.
3. Create tag `v1.0.0` on the reviewed commit and publish release notes describing compatibility changes and convergence limitations.
4. Download the exact public GitHub tag archive; its compressed bytes may differ from `git archive`. Generate a candidate BCR entry with `python3 tools/prepare_bcr.py best-cpp-1.0.0.tar.gz --output /path/to/bazel-central-registry`.
5. Review the generated `modules/best_cpp` files and submit a separate BCR PR. The tool preserves existing registry configuration and prior metadata versions. Until BCR accepts it, consumers need a pinned commit or local-path override.

`tools/test_bcr.py` creates an isolated temporary registry, serves an integrity-checked source archive and builds a renamed independent consumer without local or Git overrides. It does not publish anything. No Conan recipe is needed for the requested CMake/Bazel integration; there was no existing Conan package in this repository.
