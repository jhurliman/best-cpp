#pragma once
template<typename RealType, size_t NumParams>
void AMWG<RealType, NumParams>::step() {
  std::uniform_real_distribution<RealType> uniform(0, 1);
  for (size_t i=0; i<NumParams; ++i) {
    ParamArray proposal = state_;
    std::normal_distribution<RealType> normal(state_[i], std::exp(logSD_[i]));
    proposal[i] = normal(rng_);
    if (!std::isfinite(proposal[i])) continue;
    RealType next = posterior_(proposal);
    // Reject invalid/out-of-support proposals, including +infinity and NaN.
    RealType u = uniform(rng_);
    if (std::isfinite(next) && (next >= density_ || std::log(u) < next-density_)) {
      state_ = proposal;
      density_ = next;
      ++accepted_[i];
    }
  }
  if (++withinBatch_ == batchSize_) {
    ++batchCount_;
    RealType delta = std::min(RealType(0.01), RealType(1)/std::sqrt(RealType(batchCount_)));
    for (size_t i=0; i<NumParams; ++i) {
      logSD_[i] += RealType(accepted_[i])/RealType(batchSize_) >= RealType(0.44) ? delta : -delta;
      // Keep proposal scales representable over very long chains.
      logSD_[i] = std::max(std::log(std::numeric_limits<RealType>::min())/2,
                        std::min(std::log(std::numeric_limits<RealType>::max())/2, logSD_[i]));
      accepted_[i] = 0;
    }
    withinBatch_ = 0;
  }
}
