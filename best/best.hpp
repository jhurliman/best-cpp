#pragma once
#include "amwg.hpp"
#include "stats.hpp"

// Two-group Student-t BEST model. Owns observations; copying/moving is disabled
// because the sampler's posterior callback refers to this model.
template<class Container> class BEST {
  static const size_t kParamCount=5;
public:
  using RealType=typename Container::value_type;
  using ParamArray=typename AMWG<RealType,kParamCount>::ParamArray;
  BEST(const Container& y1,const Container& y2,size_t batchSize=kDefaultBatchSize,
       uint32_t seed=std::mt19937::default_seed)
    : y1_(y1), y2_(y2), sampler_(batchSize,seed) {
    stats::detail::values(y1_); stats::detail::values(y2_);
    std::vector<RealType> joint(y1_.begin(),y1_.end());
    joint.insert(joint.end(),y2_.begin(),y2_.end());
    meanMu_=stats::mean(joint);
    RealType sd=stats::stdev(joint);
    if (!(sd>0) || !std::isfinite(sd)) throw std::invalid_argument("pooled variance must be positive and finite");
    scaledSdMu_=sd*RealType(1000000);
    sigmaLow_=sd/RealType(1000); sigmaHigh_=sd*RealType(1000);
    if (!(sigmaLow_>0) || !std::isfinite(scaledSdMu_) || !std::isfinite(sigmaHigh_))
      throw std::invalid_argument("data scale cannot represent prior bounds");
    ParamArray start={{stats::mean(y1_),stats::mean(y2_),
      std::min(sigmaHigh_,std::max(sigmaLow_,stats::stdev(y1_))),std::min(sigmaHigh_,std::max(sigmaLow_,stats::stdev(y2_))),RealType(5)}};
    sampler_.Init(start,[this](const ParamArray& p){return LogPosterior(p);});
  }
  BEST(const BEST&)=delete;
  BEST& operator=(const BEST&)=delete;
  BEST(BEST&&)=delete;
  BEST& operator=(BEST&&)=delete;
  void Burn(size_t n) { sampler_.Burn(n); }
  void Sample(size_t n) { sampler_.Sample(n); }
  const std::vector<ParamArray>& chain() const { return sampler_.chain(); }
  void ComputeStats(std::pair<RealType,RealType>& hdi,RealType& mean) const {
    std::vector<RealType> diff;
    diff.reserve(chain().size());
    for (const auto& p:chain()) diff.push_back(p[0]-p[1]);
    hdi=stats::highestDensityInterval(diff); mean=stats::mean(diff);
  }
  RealType LogPosterior(const ParamArray& p) const {
    for(auto x:p) if(!std::isfinite(x)) return -std::numeric_limits<RealType>::infinity();
    if(p[2]<sigmaLow_||p[2]>sigmaHigh_||p[3]<sigmaLow_||p[3]>sigmaHigh_||p[4]<1)
      return -std::numeric_limits<RealType>::infinity();
    return -std::log(RealType(29))-(p[4]-1)/RealType(29)
      +posterior(p[2],p[0],p[4],y1_)+posterior(p[3],p[1],p[4],y2_);
  }
private:
  RealType posterior(RealType sigma,RealType mu,RealType nu,const Container& data) const {
    RealType logP=-std::log(sigmaHigh_-sigmaLow_)+stats::NormalLogPDF(mu,meanMu_,scaledSdMu_);
    // The Student-t normalization is constant across all observations.
    RealType norm=stats::StudentTLogPDF(RealType(0),nu)-std::log(sigma);
    for(auto x:data) {
      RealType z=(x-mu)/sigma;
      RealType q=2*std::log(std::abs(z))-std::log(nu);
      RealType tail=q>0 ? q+std::log1p(std::exp(-q)) : std::log1p(std::exp(q));
      logP+=norm-(nu+1)/2*tail;
    }
    return logP;
  }
  Container y1_,y2_;
  AMWG<RealType,kParamCount> sampler_;
  RealType meanMu_,scaledSdMu_,sigmaLow_,sigmaHigh_;
};
