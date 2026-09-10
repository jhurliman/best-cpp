#include <amwg.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
int main() {
  std::vector<double> timings;
  double result=0;
  for(int run=0;run<7;++run) {
    AMWG<double,1> sampler(50,42);
    sampler.Init({{0}},[](const std::array<double,1>& p){return -p[0]*p[0]/2;},1);
    auto start=std::chrono::steady_clock::now();sampler.Sample(10000);
    timings.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    result+=sampler.chain().back()[0];
  }
  std::sort(timings.begin(),timings.end());
  std::cout<<"10000 sweeps median_ms="<<timings[3]<<" checksum="<<result<<'\n';
}
