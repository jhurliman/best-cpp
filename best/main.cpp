#include "best.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

std::vector<double> readFile(const char* filename) {
  std::ifstream input(filename);
  if(!input) throw std::runtime_error(std::string("cannot open ")+filename);
  std::vector<double> values;
  std::string token;
  while(input>>token) {
    size_t parsed=0;
    double value=std::stod(token,&parsed);
    if(parsed!=token.size() || !std::isfinite(value))
      throw std::runtime_error(std::string("invalid numeric data in ")+filename);
    values.push_back(value);
  }
  if(!input.eof()) throw std::runtime_error(std::string("invalid numeric data in ")+filename);
  if(values.empty()) throw std::runtime_error(std::string("empty input: ")+filename);
  return values;
}
int main(int argc,const char* argv[]) {
  if(argc!=3) {std::cerr<<"Usage: best <file1> <file2>\n";return 1;}
  try {
    auto y1=readFile(argv[1]), y2=readFile(argv[2]);
    uint32_t seed=static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count());
    BEST<std::vector<double>> model(y1,y2,kDefaultBatchSize,seed);
    model.Burn(5000);model.Sample(5000);
    std::pair<double,double> hdi;double mean;
    model.ComputeStats(hdi,mean);
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    std::cout<<"hdi = "<<hdi.first<<","<<hdi.second<<", mean = "<<mean<<'\n';
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
