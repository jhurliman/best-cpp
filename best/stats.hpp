#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
namespace stats {
template<class T> T NormalLogPDF(T x, T mean, T sd);
template<class T> T StudentTLogPDF(T x, T dof);
template<class T> T NormalPDF(T x, T mean, T sd);
template<class T> T ExponentialPDF(T x, T rate);
template<class T> T UniformPDF(T a, T b);
template<class T> T StudentTPDF(T x, T dof);
template<class T> T Beta(T x, T y);
template<class C> typename C::value_type mean(const C& c);
template<class C> typename C::value_type stdev(const C& c);
template<class C> std::pair<typename C::value_type, typename C::value_type>
highestDensityInterval(const C& c, double mass=0.95);
}
#include "stats.inl"
