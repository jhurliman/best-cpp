#include <algorithm>

template<typename RealType, size_t NumParams>
void AMWG<RealType, NumParams>::NextSample() {
  static const RealType kTargetAcceptRate(0.44f);
  
  chain_.emplace_back(state_);
  
  for (size_t i = 0; i < NumParams; i++) {
    RealType prevValue = state_[i];
    
    // Modify one parameter in the current state by sampling from a Gaussian distribution
    // with mean and standard deviation of the current parameter
    RealType stdev = exp(logSD_[i]);
    std::normal_distribution<RealType> normal(prevValue, stdev);
    state_[i] = normal(rng_);
    
    // Measure the posterior density of this proposal
    RealType proposedPosteriorDensity = posteriorFunc_(state_);
    if (!isfinite(proposedPosteriorDensity)) {
      state_[i] = prevValue;
      continue;
    }
    
    // If this proposal is better than the previous, we always accept.
    // Otherwise, we accept proportional to the likelihood ratio
    RealType acceptProb = exp(proposedPosteriorDensity - currentPosteriorDensity_);
    RealType rand = rand_(rng_);
    if (acceptProb > rand) {
      acceptanceCount_[i]++;
      currentPosteriorDensity_ = proposedPosteriorDensity;
    } else {
      state_[i] = prevValue;
    }
  }
  
  // If this iteration completes a batch, update logSD and reset acceptanceCount
  if (chain_.size() % batchSize_ == 0) {
    batchCount_++;
    RealType ooBatchCount = std::min(RealType(0.01), RealType(1.0) / std::sqrt(static_cast<RealType>(batchCount_)));
    
    for (size_t i = 0; i < NumParams; i++) {
      RealType pctOfBatch = static_cast<RealType>(acceptanceCount_[i]) / static_cast<RealType>(batchSize_);
      logSD_[i] += (pctOfBatch >= kTargetAcceptRate) ? ooBatchCount : -ooBatchCount;
      acceptanceCount_[i] = 0;
    }
  }
}

template<typename T, size_t NumParams>
void AMWG<T, NumParams>::Sample(size_t n) {
  chain_.reserve(chain_.size() + n);
  
  for (size_t i = 0; i < n; i++) {
    NextSample();
  }
}

template<typename T, size_t NumParams>
void AMWG<T, NumParams>::Burn(size_t n) {
  std::vector<ParamArray> prevChain(std::move(chain_));
  Sample(n);
  chain_ = std::move(prevChain);
}
