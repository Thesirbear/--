#pragma once

#include <complex>
#include "Eigen/Core"
#include "EigenRand/EigenRand"
#include "folly/Poly.h"

namespace project_utils {

// Eigen section
template <class T>
using MatrixX = Eigen::MatrixX<T>;

template <typename T>
using ColumnVect = Eigen::Matrix<T, Eigen::Dynamic, 1>;

using Eg_Index = Eigen::Index;

// Folly section
template <std::size_t idx, class This, class... Args>
decltype(auto) poly_call(This&& self, Args&&... args) {
    return folly::poly_call<idx>(std::forward<This>(self),
                                 std::forward<Args>(args)...);
}

template <auto... T>
using PolyMembers = folly::PolyMembers<T...>;

template <class T>
using Poly = folly::Poly<T>;

template <class... T>
using PolyExtends = folly::PolyExtends<T...>;

// Eigen rand section

using rnd = Eigen::Rand::Vmt19937_64;


template <typename T>
MatrixX<T> normal_init(Eigen::Index rows, Eigen::Index cols, rnd& rng, T mean = T(0), T std = T(0.5)) {
    Eigen::Rand::NormalGen<T> gen{mean, std};
    return gen.template generate<MatrixX<T>>(rows, cols, rng);
}


template<typename T>
MatrixX<T> uniform_init(Eigen::Index rows, Eigen::Index cols, rnd& rng, T mean = T(0), T std = T(0.5)) {
    Eigen::Rand::UniformRealGen<T> gen{mean, std};
    return gen.template generate<MatrixX<T>>(rows, cols, rng);
}


template <typename T>
MatrixX<T> xavier_init(Eigen::Index rows, Eigen::Index cols, rnd& rng) {
    T std = std::sqrt(T(2) / static_cast<T>(rows + cols));
    return normal_init<T>(rows, cols, rng, T(0), std);
}

// Project defined utils section
namespace math_section {
template <typename T>
T binpow(T alpha, size_t degree) {
    T x = T(1);
    while (degree > 0) {
        if (degree & 1) {
            x = x * alpha;
        }
        alpha = alpha * alpha;
        degree >>= 1;
    }
    return x;
}

template <typename T>  // Ассимптотика: O(max_iter log(degree))
T sqrtpow(T alpha, size_t degree, size_t max_iter = 100) {
    T l = T(0), r = alpha;
    for (size_t step = 0; step < max_iter; ++step) {
        T possible_root = (l + r) / T(2);
        T root_degree = binpow(possible_root, degree);
        if (root_degree < alpha) {
            l = possible_root;
        } else {
            r = possible_root;
        }
    }
    return l;
}
}  // namespace math_section
}  // namespace project_utils
