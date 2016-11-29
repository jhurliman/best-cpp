#include <algorithm>
#include <iomanip>
#include <iostream>

template<typename RealType, size_t NumParams>
void AMWG<RealType, NumParams>::threadWorker(uint32_t threadId) {
  int curEpoch = 0;

  while (running_) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      while (curEpoch >= epoch_) cv_.wait(lock);
    }

    if (!running_) return;

    createProposal(threadId);

    ++curEpoch;
    ++completedCount_;
  }
}

template<typename RealType, size_t NumParams>
void AMWG<RealType, NumParams>::createProposal(uint32_t threadId) {
  ParamArray localState = state_;

  // Modify one parameter in the current state by sampling from a Gaussian distribution
  // with mean and standard deviation of the current parameter
  RealType stdev = exp(logSD_[curParam_]);
  std::normal_distribution<RealType> normal(state_[curParam_], stdev);
  localState[curParam_] = normal(rng_);

  // Measure the posterior density of this proposal
  RealType proposedPosteriorDensity = posteriorFunc_(localState);
  bool accept;
  if (!isfinite(proposedPosteriorDensity)) {
    proposedPosteriorDensity = RealType(0.0);
    localState[curParam_] = state_[curParam_];
    accept = false;
  } else {
    // If this proposal is better than the previous, we always accept.
    // Otherwise, we accept proportional to the likelihood ratio
    RealType acceptProb = exp(proposedPosteriorDensity - currentPosteriorDensity_);
    RealType rand = rand_(rng_);
    accept = (acceptProb > rand);
  }

  threadStates_[threadId] = std::make_tuple(localState[curParam_], proposedPosteriorDensity, accept);
}

template<typename RealType, size_t NumParams>
size_t AMWG<RealType, NumParams>::NextSample() {
  static const RealType kTargetAcceptRate(0.44f);
  
  size_t threadCount = threads_.size();

  chain_.emplace_back(state_);

  size_t steps = 0;

  for (curParam_ = 0; curParam_ < NumParams; curParam_++) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      //std::unique_lock<std::mutex> lock(mutex_);
      ++epoch_;
      completedCount_ = 0;
    }

    cv_.notify_all();

    while (completedCount_ < threadCount) std::this_thread::yield();

    for (auto&& result : threadStates_) {
      RealType proposal = std::get<0>(result);
      RealType proposedPosteriorDensity = std::get<1>(result);
      bool accepted = std::get<2>(result);

      ++steps;

      // Iterate over each result, stopping on the first acceptance
      if (accepted) {
        acceptanceCount_[curParam_]++;
        currentPosteriorDensity_ = proposedPosteriorDensity;
        state_[curParam_] = proposal;
        break;
      }
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

  return steps;
}

template<typename T, size_t NumParams>
void AMWG<T, NumParams>::Sample(size_t n) {
  chain_.reserve(chain_.size() + n);
  
  size_t totalSteps = n * NumParams;


  for (size_t i = 0, s = 0; i < n && s < totalSteps; i++) {
    s += NextSample();
    if (i % 10 == 0) {
      double pct = (static_cast<double>(s) / static_cast<double>(totalSteps)) * 100.0;
      std::cout << std::fixed << std::setprecision(2) << pct << "%" << std::endl;
    }
  }
}

template<typename T, size_t NumParams>
void AMWG<T, NumParams>::Burn(size_t n) {
  std::vector<ParamArray> prevChain(std::move(chain_));
  Sample(n);
  chain_ = std::move(prevChain);
}
