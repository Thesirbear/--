#pragma once

#include "NeuroLib/utility.hpp"

namespace OptimizerClass::default_descends {

using namespace project_utils;
using namespace math_section;

template <typename T>
class SGD {
   public:
    MatrixX<T> optimized_grad(const MatrixX<T>& grad) { return grad; }
};

template <typename T>
class Momentum {
   public:
    Momentum(T alpha = T(0.9)) : alpha_(alpha) {}

    MatrixX<T> optimized_grad(const MatrixX<T>& grad) {
        if (v_.size() == 0) {
            v_ = MatrixX<T>::Zero(grad.rows(), grad.cols());
        }
        v_ = alpha_ * v_ + grad;
        return v_;
    }

   private:
    T alpha_;
    MatrixX<T> v_;
};

template <typename T>
class AdaGrad {
   public:
    AdaGrad(T eps = T(1e-8)) : eps_(eps) {}

    MatrixX<T> optimized_grad(const MatrixX<T>& grad) {
        if (jump_coef_.size() == 0) {
            jump_coef_ = MatrixX<T>::Zero(grad.rows(), grad.cols());
        }
        jump_coef_.array() += grad.array().square();
        return (grad.array() / (jump_coef_.array().sqrt() + eps_)).matrix();
    }

   private:
    MatrixX<T> jump_coef_;
    T eps_;  // TODO: Сделать нормальную константу для типа T
    // т.к. в зависимости от типа, у нас может быть разная точность.
};

template <typename T>
class RmsProp {
   public:
    RmsProp(T alpha = T(0.99), T eps = T(1e-8)) : alpha_(alpha), eps_(eps) {}

    MatrixX<T> optimized_grad(const MatrixX<T>& grad) {
        if (jump_coef_.size() == 0) {
            jump_coef_ = MatrixX<T>::Zero(grad.rows(), grad.cols());
        }
        jump_coef_.array() =
            alpha_ * jump_coef_.array() + (1 - alpha_) * grad.array().square();
        return (grad.array() / (jump_coef_.array().sqrt() + eps_)).matrix();
    }

   private:
    MatrixX<T> jump_coef_;
    T alpha_;
    T eps_;  // TODO: Сделать нормальную константу для типа T
    // т.к. в зависимости от типа, у нас может быть разная точность.
};

template <typename T>
class Adam {
   public:
    Adam(T alpha1 = T(0.9), T alpha2 = T(0.999), T eps = T(1e-8))
        : alpha1_(alpha1), alpha2_(alpha2), eps_(eps), t_(0) {}

    MatrixX<T> optimized_grad(const MatrixX<T>& grad) {
        if (jump_coef1_.size() == 0) {
            jump_coef1_ = MatrixX<T>::Zero(grad.rows(), grad.cols());
            jump_coef2_ = MatrixX<T>::Zero(grad.rows(), grad.cols());
        }
        ++t_;
        jump_coef1_.array() =
            alpha1_ * jump_coef1_.array() + (T(1) - alpha1_) * grad.array();
        jump_coef2_.array() = alpha2_ * jump_coef2_.array() +
                              (T(1) - alpha2_) * grad.array().square();
        return ((jump_coef1_.array() / (1 - binpow(alpha1_, t_))) /
                ((jump_coef2_.array() / (1 - binpow(alpha2_, t_)))
                     .array()
                     .sqrt() +
                 eps_))
            .matrix();
    }

   private:
    T eps_;  // TODO: Сделать нормальную константу для типа T
    // т.к. в зависимости от типа, у нас может быть разная точность.
    size_t t_;
    T alpha1_, alpha2_;
    MatrixX<T> jump_coef2_;
    MatrixX<T> jump_coef1_;
};
}  // namespace OptimizerClass::default_descends
