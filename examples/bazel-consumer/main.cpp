#include <best.hpp>
#include <version.hpp>
#include <vector>
int main() {
  static_assert(BEST_CPP_VERSION_MAJOR == 1, "version");
  BEST<std::vector<double>> model({1,2,3},{3,4,5},50,42);
  model.Burn(10);model.Sample(20);
  std::pair<double,double> hdi;double mean;
  model.ComputeStats(hdi,mean);
  return model.chain().size()==20 && std::isfinite(mean) ? 0 : 1;
}
