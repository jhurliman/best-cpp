#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "amwg.hpp"
#include "best.hpp"

int main(int argc, const char * argv[]) {
  static auto y1 = std::vector<float>{1.96, 2.06, 2.03, 2.11, 1.88, 1.88, 2.08, 1.93, 2.03, 2.03, 2.03, 2.08, 2.03, 2.11, 1.93};
  static auto y2 = std::vector<float>{1.83, 1.93, 1.88, 1.85, 1.85, 1.91, 1.91, 1.85, 1.78, 1.91, 1.93, 1.80, 1.80, 1.85, 1.93,
                                      1.85, 1.83, 1.85, 1.91, 1.85, 1.91, 1.85, 1.80, 1.80, 1.85};
  
  uint32_t now = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count());
  
  BEST<std::vector<float>> best(y1, y2, kDefaultBatchSize, now);
  
  best.Burn(50000);
  best.Sample(50000);
  
  std::pair<float, float> hdi;
  float mean;
  best.ComputeStats(hdi, mean);
  
  std::cout << "hdi = " << hdi.first << "," << hdi.second << ", mean = " << mean << std::endl;
  return 0;
}
