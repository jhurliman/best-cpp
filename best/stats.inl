#pragma once
namespace stats {
namespace detail {
template<class T> void positive(T x) {
  if (!(x > 0) || !std::isfinite(x)) throw std::invalid_argument("finite positive parameter required");
}
template<class C> void values(const C& c) {
  static_assert(std::is_floating_point<typename C::value_type>::value, "floating point required");
  if (c.empty()) throw std::invalid_argument("empty data");
  for (auto x:c) if (!std::isfinite(x)) throw std::invalid_argument("nonfinite data");
}
}
template<class T> T NormalLogPDF(T x,T m,T sd) {
  detail::positive(sd);
  T z=(x-m)/sd;
  return -T(0.91893853320467274178L)-std::log(sd)-T(0.5)*z*z;
}
template<class T> T NormalPDF(T x,T m,T sd) { return std::exp(NormalLogPDF(x,m,sd)); }
template<class T> T ExponentialPDF(T x,T rate) {
  detail::positive(rate);
  return x<0 ? T(0) : rate*std::exp(-rate*x);
}
template<class T> T UniformPDF(T a,T b) {
  if (!std::isfinite(a)||!std::isfinite(b)||!(a<b)) throw std::invalid_argument("invalid uniform bounds");
  return T(1)/(b-a);
}
template<class T> T StudentTLogPDF(T x,T dof) {
  detail::positive(dof);
  // Evaluate the quadratic in log space to avoid overflow in x*x.
  T q = T(2)*std::log(std::abs(x))-std::log(dof);
  T tail = q>0 ? q+std::log1p(std::exp(-q)) : std::log1p(std::exp(q));
  // Asymptotic gamma ratio avoids catastrophic cancellation at large dof.
  T normalizer = dof > T(1e6)
    ? -T(0.91893853320467274178L)-T(0.25)/dof
    : std::lgamma((dof+1)/2)-std::lgamma(dof/2)
      -(std::log(dof)+T(1.14472988584940017414L))/2;
  return normalizer-(dof+1)/2*tail;
}
template<class T> T StudentTPDF(T x,T dof) { return std::exp(StudentTLogPDF(x,dof)); }
template<class T> T Beta(T x,T y) {
  detail::positive(x); detail::positive(y);
  return std::exp(std::lgamma(x)+std::lgamma(y)-std::lgamma(x+y));
}
template<class C> typename C::value_type mean(const C& c) {
  detail::values(c);
  long double sum=0;
  for (auto x:c) sum+=static_cast<long double>(x)/c.size();
  return static_cast<typename C::value_type>(sum);
}
template<class C> typename C::value_type stdev(const C& c) {
  detail::values(c);
  long double m=mean(c), scale=0, ss=1;
  // Scaled sum of squares needs no temporary allocation and avoids overflow.
  for (auto x:c) {
    long double d=std::abs(static_cast<long double>(x)-m);
    if (d!=0) {
      if (scale<d) { ss=1+ss*(scale/d)*(scale/d); scale=d; }
      else ss+=(d/scale)*(d/scale);
    }
  }
  return static_cast<typename C::value_type>(scale*std::sqrt(ss/c.size()));
}
template<class C> std::pair<typename C::value_type, typename C::value_type>
highestDensityInterval(const C& c,double mass) {
  detail::values(c);
  if (!(mass>0 && mass<=1)) throw std::invalid_argument("mass must be in (0,1]");
  using T=typename C::value_type;
  std::vector<T> x(c.begin(),c.end());
  std::sort(x.begin(),x.end());
  size_t count=std::max(size_t(1),static_cast<size_t>(std::ceil(mass*x.size())));
  size_t best=0;
  for(size_t i=1;i+count<=x.size();++i)
    if(static_cast<long double>(x[i+count-1])-x[i]<static_cast<long double>(x[best+count-1])-x[best]) best=i;
  return {x[best],x[best+count-1]};
}
}
