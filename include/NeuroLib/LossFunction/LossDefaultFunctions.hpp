#include <csignal>

#include "NeuroLib/utility.hpp"

namespace LossFunctionsClass::default_impl {

using namespace project_utils;

template <typename T>
class Loss_Mse {
   public:
    MatrixX<T> Gradient(const MatrixX<T>& X, const MatrixX<T>& Y) {
        return (T(2) / T(X.rows()) * (X - Y).array()).matrix();
    }

    T Loss(const MatrixX<T>& X, const MatrixX<T>& Y) {
        return (X - Y).array().square().sum() / T(Y.rows());
    }
};

template <typename T>
class CrossEntropy {
   public:
    MatrixX<T> Gradient(const MatrixX<T>& X, const MatrixX<T>& Y) {
        return -(Y.array() / (X.array() + eps)).matrix() / T(Y.rows());
    }

    T Loss(const MatrixX<T>& X, const MatrixX<T>& Y) {
        return -(Y.array() * (X.array() + eps).log()).sum() / T(Y.rows());
    }

   private:
    T eps = T(1e-8);
};

template <typename T>
class SoftmaxCrossEntropy {
   public:
    MatrixX<T> Gradient(const MatrixX<T>& X, const MatrixX<T>& Y) {
        if (cache_.size() == 0) {
            return cache_;
        }
        return (cache_ - Y) / T(Y.rows());
    }

    T Loss(const MatrixX<T>& X, const MatrixX<T>& Y) {
        MatrixX<T> X_max = X.rowwise().maxCoeff();
        MatrixX<T> score = (X.colwise() - X_max.col(0)).array().exp().matrix();
        MatrixX<T> sum = score.rowwise().sum();
        cache_ = (score.array().colwise() / sum.col(0).array());
        return -(Y.array() * (cache_.array() + eps).log()).sum() / T(Y.rows());
    }

   private:
    T eps = T(1e-8);
    MatrixX<T> cache_;
};
}  // namespace LossFunctionsClass::default_impl
