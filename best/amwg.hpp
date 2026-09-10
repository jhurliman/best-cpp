#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

static const size_t kDefaultBatchSize = 50;

// Sequential adaptive Metropolis-within-Gibbs; one proposal per coordinate.
// Instances are not safe for concurrent mutation. Independent chains may run
// on separate threads with independent seeds and posterior state.
template<typename RealType, size_t NumParams>
class AMWG {
  static_assert(std::is_floating_point<RealType>::value, "floating point required");
  static_assert(NumParams > 0, "at least one parameter required");
public:
  using ParamArray = std::array<RealType, NumParams>;
  using PosteriorFunc = std::function<RealType(const ParamArray&)>;
  explicit AMWG(size_t batchSize = kDefaultBatchSize,
                uint32_t seed = std::mt19937::default_seed)
    : rng_(seed), seed_(seed), batchSize_(batchSize) {
    if (!batchSize) throw std::invalid_argument("batch size must be positive");
  }
  // The legacy threads argument is retained for source compatibility only.
  void Init(ParamArray start, PosteriorFunc posterior, uint32_t threads = 1) {
    (void)threads;
    if (!posterior) throw std::invalid_argument("posterior is required");
    for (auto x : start)
      if (!std::isfinite(x)) throw std::invalid_argument("nonfinite initial state");
    RealType density = posterior(start);
    if (!std::isfinite(density)) throw std::invalid_argument("initial log density must be finite");
    posterior_ = std::move(posterior);
    state_ = start;
    density_ = density;
    chain_.clear();
    rng_.seed(seed_);
    logSD_.fill(0);
    accepted_.fill(0);
    batchCount_ = withinBatch_ = 0;
    initialized_ = true;
  }
  size_t NextSample() {
    requireInit();
    chain_.reserve(chain_.size() + 1);
    step();
    chain_.push_back(state_);
    return NumParams;
  }
  void Sample(size_t n) {
    requireInit();
    if (n > chain_.max_size() - chain_.size()) throw std::length_error("chain too large");
    chain_.reserve(chain_.size() + n);
    for (size_t i=0; i<n; ++i) { step(); chain_.push_back(state_); }
  }
  void Burn(size_t n) { requireInit(); for (size_t i=0; i<n; ++i) step(); }
  RealType posterior_density() const { requireInit(); return density_; }
  const ParamArray& state() const { requireInit(); return state_; }
  std::vector<ParamArray>& chain() { return chain_; }
  const std::vector<ParamArray>& chain() const { return chain_; }
private:
  void requireInit() const { if (!initialized_) throw std::logic_error("Init must be called first"); }
  void step();
  std::mt19937 rng_;
  uint32_t seed_;
  size_t batchSize_, batchCount_=0, withinBatch_=0;
  bool initialized_=false;
  PosteriorFunc posterior_;
  ParamArray state_{}, logSD_{};
  std::array<size_t, NumParams> accepted_{};
  RealType density_=0;
  std::vector<ParamArray> chain_;
};
#include "amwg.inl"
