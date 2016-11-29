#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>

#include "amwg.hpp"
#include "best.hpp"

std::vector<float> readFile(const char* filename);

int main(int argc, const char* argv[]) {
  /*static auto y1 = std::vector<float>{1.96f, 2.06f, 2.03f, 2.11f, 1.88f, 1.88f, 2.08f, 1.93f, 2.03f, 2.03f, 2.03f, 2.08f, 2.03f, 2.11f, 1.93f};
  static auto y2 = std::vector<float>{1.83f, 1.93f, 1.88f, 1.85f, 1.85f, 1.91f, 1.91f, 1.85f, 1.78f, 1.91f, 1.93f, 1.80f, 1.80f, 1.85f, 1.93f,
                                      1.85f, 1.83f, 1.85f, 1.91f, 1.85f, 1.91f, 1.85f, 1.80f, 1.80f, 1.85f};*/
  
  if (argc != 3) {
    std::cout << "Usage: best <file1> <file2>" << std::endl;
    return 1;
  }

  auto y1 = readFile(argv[1]);
  auto y2 = readFile(argv[2]);

  uint32_t now = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count());
  
  BEST<std::vector<float>> best(y1, y2, kDefaultBatchSize, now);
  
  std::cout << "Running burn-in" << std::endl;
  best.Burn(5000);
  std::cout << "Running sampler" << std::endl;
  best.Sample(5000);
  
  std::pair<float, float> hdi;
  float mean;
  best.ComputeStats(hdi, mean);
  
  std::cout << "hdi = " << hdi.first << "," << hdi.second << ", mean = " << mean << std::endl;
  return 0;
}

std::vector<float> readFile(const char* filename) {
  std::ifstream infile(filename);

  std::vector<float> values;
  float value;

  while (infile >> value) {
    values.emplace_back(value);
  }

  return values;
}
