#include "best.hpp"
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <thread>

int checks=0;
void check(bool ok,const char* message) { ++checks; if(!ok) throw std::runtime_error(message); }
void near(double x,double y,double tolerance,const char* message) { check(std::abs(x-y)<tolerance,message); }
template<class F> void rejects(F f) { bool threw=false; try { f(); } catch(const std::exception&) { threw=true; } check(threw,"expected exception"); }
using V=std::vector<double>;
using S=AMWG<double,1>;
int main() { try {
  near(stats::NormalPDF(0.,0.,1.),0.3989422804014327,1e-14,"normal reference");
  near(stats::StudentTPDF(0.,1.),1/3.141592653589793,1e-14,"Cauchy reference");
  near(stats::StudentTPDF(2.,1.),1/(5*3.141592653589793),1e-14,"Cauchy tail");
  near(stats::StudentTLogPDF(0.,1e100),stats::NormalLogPDF(0.,0.,1.),1e-12,"large dof normal limit");
  near(stats::Beta(2.,3.),1./12,1e-14,"beta reference");
  check(stats::ExponentialPDF(-1.,1.)==0,"exponential support");
  near(stats::NormalLogPDF(100.,0.,1.),-5000.918938533205,1e-10,"normal log tail");
  check(std::isfinite(stats::StudentTLogPDF(1e200,3.)),"Student log tail");
  near(stats::mean(V{1,2,3}),2,1e-14,"mean");
  near(stats::stdev(V{1,2,3}),std::sqrt(2./3),1e-14,"population sd");
  near(stats::stdev(V{1e100,-1e100})/1e100,1,1e-14,"large sd");
  check(stats::highestDensityInterval(std::list<double>{4})==std::make_pair(4.,4.),"singleton HDI");
  V ordered; for(int i=0;i<20;++i) ordered.push_back(i);
  check(stats::highestDensityInterval(ordered)==std::make_pair(0.,18.),"95 percent includes 19 of 20");
  rejects([]{stats::mean(V{});}); rejects([]{stats::stdev(V{NAN});});
  rejects([]{stats::highestDensityInterval(V{});}); rejects([]{stats::highestDensityInterval(V{1},0);});
  rejects([]{stats::StudentTPDF(0.,0.);}); rejects([]{S s(0);});
  S s(50,42); rejects([&]{s.Sample(1);}); rejects([&]{s.Burn(0);});
  rejects([&]{s.Init({{0}},{});}); rejects([&]{s.Init({{0}},[](const S::ParamArray&){return NAN;});});
  auto normal=[](const S::ParamArray& p){return -p[0]*p[0]/2;};
  for(unsigned threads:{0u,1u,4u}) {
    s.Init({{0}},normal,threads); s.Burn(1000); s.Sample(101);
    check(s.chain().size()==101,"exact sample count");
    auto before=s.chain(); s.Burn(100); check(s.chain()==before,"burn preserves chain");
    s.Sample(0); check(s.chain()==before,"zero samples");
    check(s.NextSample()==1,"one proposal per coordinate");
    check(s.chain().back()==s.state(),"stores completed sweep");
  }
  S a(50,12),b(50,12); a.Init({{0}},normal); b.Init({{0}},normal,8);
  a.Burn(77); a.Sample(123); b.Burn(77); b.Sample(23); b.Sample(100);
  check(a.chain()==b.chain(),"chunking and legacy thread count reproducibility");
  a.Init({{0}},normal); a.Burn(77); a.Sample(123); check(a.chain()==b.chain(),"reinitialization resets adaptation and RNG");
  a.chain().clear(); a.Sample(50); b.Sample(50);
  check(a.state()==b.state(),"adaptation independent of stored chain");
  S bounded; bounded.Init({{0}},[](const S::ParamArray& p){return p[0]==0 ? 0. : -INFINITY;}); bounded.Sample(100);
  check(bounded.state()[0]==0,"reject out of support");
  S throwing; throwing.Init({{0}},[](const S::ParamArray& p)->double {if(p[0]!=0)throw std::runtime_error("callback");return 0;});
  rejects([&]{throwing.Sample(1);}); check(throwing.chain().empty(),"callback exception propagation");
  // Fixed seeds and generous tolerances: regression checks, not convergence proofs.
  for(unsigned seed:{7u,42u,123u}) {
    S n(50,seed); n.Init({{0}},normal); n.Burn(5000); n.Sample(60000);
    V draws; for(auto p:n.chain()) draws.push_back(p[0]);
    near(stats::mean(draws),0,0.06,"normal empirical mean"); near(stats::stdev(draws),1,0.06,"normal empirical sd");
    S e(50,seed); e.Init({{1}},[](const S::ParamArray& p){return p[0]<0?-INFINITY:-p[0];}); e.Burn(5000);e.Sample(60000);
    draws.clear();for(auto p:e.chain())draws.push_back(p[0]);
    near(stats::mean(draws),1,0.10,"exponential empirical mean");
  }
  // Independent chains can be owned and run by separate threads.
  V outcomes(2); std::thread t1([&]{S n;n.Init({{0}},normal);n.Sample(100);outcomes[0]=n.state()[0];});
  std::thread t2([&]{S n;n.Init({{0}},normal);n.Sample(100);outcomes[1]=n.state()[0];});t1.join();t2.join();check(outcomes[0]==outcomes[1],"independent chains");
  rejects([]{BEST<V> m(V{},V{1});}); rejects([]{BEST<V> m(V{1},V{1});});
  { BEST<V> imbalanced(V{-1,1},V(2000001,0));
    check(imbalanced.chain().empty(),"imbalanced groups initialize inside prior support"); }
  BEST<V> model(V{1,2,3},V{3,4,5},50,42);
  std::pair<double,double> hdi;double mean;
  rejects([&]{model.ComputeStats(hdi,mean);});
  check(!std::isfinite(model.LogPosterior({{2,4,1,1,0.5}})),"nu lower bound");
  check(!std::isfinite(model.LogPosterior({{2,4,1e10,1,5}})),"sigma upper bound");
  check(std::isfinite(model.LogPosterior({{2,4,1,1,5}})),"valid posterior");
  // Independent Cauchy (nu=1) formula for all factors of the joint model.
  double sd=std::sqrt(5./3), priorSd=sd*1e6;
  double expected=-std::log(29.)-2*std::log(sd*1000-sd/1000)
    -2*std::log(priorSd*std::sqrt(2*3.141592653589793))-1/(priorSd*priorSd)
    -6*std::log(3.141592653589793)-4*std::log(2.);
  near(model.LogPosterior({{2,4,1,1,1}}),expected,1e-12,"joint posterior reference");
  BEST<std::vector<float>> floats({1,2,3},{3,4,5});
  near(floats.LogPosterior({{2,4,1,1,1}}),expected,1e-4,"float posterior precision");
  V original{1,2,3};BEST<V> owned(original,V{3,4,5});
  auto density=owned.LogPosterior({{2,4,1,1,1}});original[0]=100;
  check(owned.LogPosterior({{2,4,1,1,1}})==density,"model owns observations");
  // A well-identified fixture avoids treating a short diffuse chain as a
  // convergence guarantee for two groups with only three observations.
  V y1,y2; for(int i=0;i<40;++i) {double x=(i%5-2)*0.3; y1.push_back(x);y2.push_back(x+2);}
  BEST<V> identified(y1,y2,50,42);
  identified.Burn(5000);identified.Sample(20000);identified.ComputeStats(hdi,mean);
  check(identified.chain().size()==20000 && std::isfinite(mean) && hdi.first<hdi.second,"BEST end-to-end");
  near(mean,-2,0.15,"BEST difference symmetry");
  check(hdi.second<0,"separated groups credible interval");
  std::cout<<checks<<" checks passed\n";
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; } }
