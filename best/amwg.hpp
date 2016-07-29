#pragma once

#include <array>
#include <functional>
#include <random>
#include <type_traits>
#include <vector>

static const size_t kDefaultBatchSize = 50;

/**
 * Implementation of an Adaptive Metropolis within Gibbs sampler.
 */
template<typename RealType, size_t NumParams>
class AMWG {
public:
  
  using ParamArray = std::array<RealType, NumParams>;
  using PosteriorFunc = std::function<RealType(const ParamArray&)>;
  
  AMWG(size_t batchSize = kDefaultBatchSize, uint32_t seed = std::mt19937::default_seed)
    : rng_(seed)
    , rand_(0.0, 1.0)
    , currentPosteriorDensity_(0.0)
    , batchSize_(batchSize)
    , batchCount_(0)
    , logSD_()
    , acceptanceCount_()
  {};
  
  void Init(std::array<RealType, NumParams> startValues, PosteriorFunc posteriorFunc) {
    state_ = startValues;
    posteriorFunc_ = posteriorFunc;
    chain_.clear();
    currentPosteriorDensity_ = posteriorFunc_(state_);
  };
  
  void NextSample();
  void Sample(size_t n);
  void Burn(size_t n);
  RealType posterior_density() { return currentPosteriorDensity_; };
  std::vector<ParamArray>& chain() { return chain_; };
  
private:
  
  PosteriorFunc posteriorFunc_;
  std::mt19937 rng_;
  std::uniform_real_distribution<RealType> rand_;
  std::vector<ParamArray> chain_;
  RealType currentPosteriorDensity_;
  size_t batchSize_;
  size_t batchCount_;
  ParamArray state_;
  ParamArray logSD_;
  std::array<size_t, NumParams> acceptanceCount_;
};

#include "amwg.inl"
