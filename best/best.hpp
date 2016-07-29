#pragma once

#include <cassert>
#include <type_traits>

#include "amwg.hpp"
#include "stats.hpp"

/**
 * Implementation of "Bayesian estimation supersedes the t test". Estimate the
 * difference between two sets of real values.
 */
template <typename Container>
class BEST {
  // Length of startValues array below
  static const size_t kParamCount = 5;
  
public:
  
  using RealType = typename Container::value_type;
  using ParamArray = typename AMWG<RealType, kParamCount>::ParamArray;
  
  BEST(const Container& y1, const Container& y2, size_t batchSize = kDefaultBatchSize, uint32_t seed = std::mt19937::default_seed)
    : sampler_(batchSize, seed)
    , y1_(y1)
    , y2_(y2)
  {
    size_t jointSize = y1.size() + y2.size();
    
    // Find the mean of {y1,y2}
    RealType sum = std::accumulate(y1.begin(), y1.end(), RealType(0.0)) +
                   std::accumulate(y2.begin(), y2.end(), RealType(0.0));
    RealType mean = RealType(sum / static_cast<RealType>(jointSize));
    
    // Find the standard deviation of {y1,y2}
    std::vector<RealType> diff(jointSize);
    std::transform(y1.begin(), y1.end(), diff.begin(), [mean](RealType x) { return x - mean; });
    std::transform(y2.begin(), y2.end(), diff.begin() + y1.size(), [mean](RealType x) { return x - mean; });
    RealType sqSum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    RealType stdev = std::sqrt(sqSum / jointSize);
    
    meanMu_ = mean;
    RealType sdMu = stdev;
    scaledSdMu_ = sdMu * RealType(1000000.0);
    sigmaLow_ = sdMu / RealType(1000.0);
    sigmaHigh_ = sdMu * RealType(1000.0);
    
    // Initial parameter values: mu_y1, mu_y2, sigma_y1, sigma_y2, dof
    std::array<RealType, kParamCount> startValues = {
      stats::mean(y1),
      stats::mean(y2),
      stats::stdev(y1),
      stats::stdev(y2),
      RealType(5.0)
    };
    
    sampler_.Init(startValues, std::bind(&BEST::JointPosterior, this, std::placeholders::_1));
  }
  
  void Burn(size_t n) { sampler_.Burn(n); };
  
  void Sample(size_t n) { sampler_.Sample(n); };
  
  void ComputeStats(std::pair<RealType, RealType>& hdi, RealType& mean) {
    std::vector<ParamArray>& chain = sampler_.chain();
    
    std::vector<RealType> muDiff(chain.size());
    std::transform(chain.begin(), chain.end(), muDiff.begin(), [](ParamArray params) { return params[0] - params[1]; });
    hdi = stats::highestDensityInterval(muDiff);
    mean = stats::mean(muDiff);
  }
  
private:
  
  RealType Posterior(RealType sigma, RealType mu, RealType nu, const Container& data) {
    RealType logP = log(stats::UniformPDF(sigmaLow_, sigmaHigh_));
    logP += log(stats::NormalPDF(mu, meanMu_, scaledSdMu_));
    
    RealType ooSD = RealType(1.0) / sigma;
    for (RealType val : data) {
      logP += log(ooSD * stats::StudentTPDF((val - mu) * ooSD, nu));
    }
    
    return logP;
  };
  
  RealType JointPosterior(const ParamArray& params) {
    // A trick to get an exponentially distributed prior on nu that starts at 1
    const RealType kOOTwentyNine(1.0 / 29.0);
    
    RealType mu1 = params[0];
    RealType mu2 = params[1];
    RealType sigma1 = params[2];
    RealType sigma2 = params[3];
    RealType nu = params[4];
    
    if (sigma1 < sigmaLow_ || sigma2 < sigmaLow_) return -std::numeric_limits<RealType>::infinity();
    
    RealType logP = log(stats::ExponentialPDF(nu - RealType(1.0), kOOTwentyNine));
    logP += Posterior(sigma1, mu1, nu, y1_);
    logP += Posterior(sigma2, mu2, nu, y2_);
    
    return logP;
  };
  
  const Container& y1_;
  const Container& y2_;
  AMWG<RealType, kParamCount> sampler_;
  RealType meanMu_;
  RealType scaledSdMu_;
  RealType sigmaLow_;
  RealType sigmaHigh_;
};
