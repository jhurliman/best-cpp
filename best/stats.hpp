#pragma once

namespace stats {

template<typename RealType>
RealType NormalPDF(RealType x, RealType mean, RealType std);

template<typename RealType>
RealType ExponentialPDF(RealType x, RealType rate);

template<typename RealType>
RealType UniformPDF(RealType a, RealType b);

template<typename RealType>
RealType StudentTPDF(RealType x, RealType dof);

template<typename RealType>
RealType Beta(RealType x, RealType y);

template <typename Container>
typename Container::value_type mean(const Container& c);

template <typename Container>
typename Container::value_type stdev(const Container& c);

template <typename Container>
typename std::pair<typename Container::value_type, typename Container::value_type> highestDensityInterval(const Container& c);

}

#include "stats.inl"
