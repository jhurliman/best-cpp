#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <random>
#include <thread>
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
    , running_(false)
    , epoch_(0)
    , completedCount_(0)
    , curParam_(0)
  {};

  ~AMWG() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_ = false;
      epoch_ = SIZE_MAX;
    }

    cv_.notify_all();

    for (auto&& t : threads_) {
      if (t.joinable()) t.join();
    }
  }
  
  void Init(std::array<RealType, NumParams> startValues, PosteriorFunc posteriorFunc, uint32_t threads = std::thread::hardware_concurrency()) {
    state_ = startValues;
    posteriorFunc_ = posteriorFunc;
    chain_.clear();
    currentPosteriorDensity_ = posteriorFunc_(state_);

    running_ = true;

    threads_.reserve(threads);
    threadStates_.resize(threads);

    for (uint32_t threadId = 0; threadId < threads; threadId++) {
      threads_.emplace_back(std::thread(&AMWG::threadWorker, this, threadId));
    }
  };
  
  size_t NextSample();
  void Sample(size_t n);
  void Burn(size_t n);
  RealType posterior_density() { return currentPosteriorDensity_; };
  std::vector<ParamArray>& chain() { return chain_; };
  
private:
  
  void threadWorker(uint32_t threadId);
  void createProposal(uint32_t threadId);

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

  std::atomic_bool running_;
  std::atomic<size_t> epoch_;
  std::atomic<size_t> completedCount_;
  std::atomic<size_t> curParam_;
  std::vector<std::thread> threads_;
  std::vector<std::tuple<RealType, RealType, bool>> threadStates_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

#include "amwg.inl"
