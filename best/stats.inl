#include <algorithm>

#define M_LOG_2PI 1.837877066409345483560659472811235279

template<typename RealType>
RealType stats::NormalPDF(RealType x, RealType mean, RealType std) {
  return exp(RealType(-0.5) * RealType(M_LOG_2PI) - log(std) - pow(x - mean, 2) / (RealType(2.0) * std * std));
}

template<typename RealType>
RealType stats::ExponentialPDF(RealType x, RealType rate) {
  return std::max<RealType>(RealType(0.0), rate * exp(-rate * x));
}

template<typename RealType>
RealType stats::UniformPDF(RealType a, RealType b) {
  return RealType(1.0) / (b - a);
}

template<typename RealType>
RealType stats::StudentTPDF(RealType x, RealType dof) {
  const RealType one = RealType(1.0);
  const RealType half = RealType(0.5);
  
  return one / (std::sqrt(dof) * Beta(half, dof * half)) * pow(one + x * x / dof, -((dof + one) * half));
}

template<typename RealType>
RealType stats::Beta(RealType x, RealType y) {
  return exp(std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y));
}

template <typename Container>
typename Container::value_type stats::mean(const Container& c) {
  using RealType = typename Container::value_type;
  
  RealType sum = std::accumulate(c.begin(), c.end(), RealType(0.0));
  return sum / static_cast<RealType>(c.size());
}

template <typename Container>
typename Container::value_type stats::stdev(const Container& c) {
  using RealType = typename Container::value_type;
  
  RealType m = mean(c);
  
  std::vector<RealType> diff(c.size());
  std::transform(c.begin(), c.end(), diff.begin(), [m](RealType x) { return x - m; });
  
  RealType sqSum = std::inner_product(diff.begin(), diff.end(), diff.begin(), RealType(0.0));
  return std::sqrt(sqSum / static_cast<RealType>(c.size()));
}

template <typename Container>
typename std::pair<typename Container::value_type, typename Container::value_type> stats::highestDensityInterval(const Container& c) {
  using RealType = typename Container::value_type;
  
  const RealType p = 0.95;
  
  // Build a sorted copy of the data
  std::vector<RealType> x(c);
  std::sort(x.begin(), x.end());
  
  // Choose a credible interval
  size_t ciNumPoints = std::floor(static_cast<RealType>(x.size()) * p);
  std::pair<RealType, RealType> minWidthCI = std::make_pair(x.front(), x.back());
  
  for (size_t i = 0; i < x.size() - ciNumPoints; i++) {
    RealType ciWidth = x[i + ciNumPoints] - x[i];
    if (ciWidth < minWidthCI.second - minWidthCI.first) {
      minWidthCI = std::make_pair(x[i], x[i + ciNumPoints]);
    }
  }
  
  return minWidthCI;
}
