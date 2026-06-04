#include <cstddef>

#include "NeuroLib/utility.hpp"

namespace BlockClass_default_impl {

using namespace project_utils;
enum Rows : size_t;
enum Cols : size_t;
template <typename T>
class LinearBlock {
   public:
    LinearBlock(Rows in, Cols out, rnd& rng)
        : W_(xavier_init<T>(static_cast<Eigen::Index>(in),
                            static_cast<Eigen::Index>(out), rng)),

          b_(MatrixX<T>::Zero(1, static_cast<Eigen::Index>(out))) {}

    MatrixX<T> forward(const MatrixX<T>& X) {
        cache_ = X;
        return predict(X);
    }

    MatrixX<T> predict(const MatrixX<T>& X) const {
        return (X * W_).rowwise() + b_.row(0);
    }

    MatrixX<T> backward(const MatrixX<T>& grad) {
        dW_ = cache_.transpose() * grad;
        db_ = grad.colwise().sum();
        return grad * W_.transpose();
    }

    const MatrixX<T>& get_W() const { return W_; }

    const MatrixX<T>& get_dW() const { return dW_; }

    const MatrixX<T>& get_b() const { return b_; }

    const MatrixX<T>& get_db() const { return db_; }

    void update_params(const MatrixX<T>& dW, const MatrixX<T>& db) {
        W_.noalias() -= dW;
        b_.noalias() -= db;
    }

    void set_params(const MatrixX<T>& W, const MatrixX<T>& b) {
        W_ = W;
        b_ = b;
    }

   private:
    MatrixX<T> cache_;
    MatrixX<T> W_, b_;
    MatrixX<T> dW_, db_;
};

template <typename T>
class LRelu {
   public:
    LRelu(T alpha = 0.05) : alpha_(alpha) {}

    MatrixX<T> forward(const MatrixX<T>& X) {
        cache_ = X;
        return forward_(X);
    }

    MatrixX<T> predict(const MatrixX<T>& X) const { return forward_(X); }

    MatrixX<T> backward(const MatrixX<T>& grad) {
        return grad.cwiseProduct(backward_(cache_));
    }

    void update_params(const MatrixX<T>& dW, const MatrixX<T>& db) {}

    const MatrixX<T>& get_W() const {
        static const MatrixX<T> none;
        return none;
    }

    const MatrixX<T>& get_dW() const {
        static const MatrixX<T> none;
        return none;
    }

    const MatrixX<T>& get_b() const {
        static const MatrixX<T> none;
        return none;
    }

    const MatrixX<T>& get_db() const {
        static const MatrixX<T> none;
        return none;
    }

    void set_params(const MatrixX<T>& W, const MatrixX<T>& b) {}

   private:
    T alpha_;
    MatrixX<T> cache_;

    MatrixX<T> forward_(const MatrixX<T>& X) const {
        T al = alpha_;
        return X.unaryExpr([al](T x) { return x < T(0) ? al * x : x; });
    }

    MatrixX<T> backward_(const MatrixX<T>& X) const {
        T al = alpha_;
        return X.unaryExpr([al](T x) { return x < T(0) ? al : T(1); });
    }
};

template <typename T>
class SoftMax {
   public:
    SoftMax() {}

    MatrixX<T> forward(const MatrixX<T>& X) {
        cache_ = forward_(X);
        return cache_;
    }

    MatrixX<T> predict(const MatrixX<T>& X) const { return forward_(X); }

    MatrixX<T> backward(const MatrixX<T>& grad) { return backward_(grad); }

    void update_params(const MatrixX<T>& dW, const MatrixX<T>& db) {}

    const MatrixX<T>& get_W() const {
        static const MatrixX<T> none;
        return none;
    }

    const MatrixX<T>& get_dW() const {
        static const MatrixX<T> none;
        return none;
    }

    const MatrixX<T>& get_b() const {
        static const MatrixX<T> none;
        return none;
    }

    const MatrixX<T>& get_db() const {
        static const MatrixX<T> none;
        return none;
    }

    void set_params(const MatrixX<T>& W, const MatrixX<T>& b) {}

   private:
    MatrixX<T> cache_;

    MatrixX<T> forward_(const MatrixX<T>& X) const {
        ColumnVect<T> X_max = X.rowwise().maxCoeff();
        MatrixX<T> score = (X.colwise() - X_max).array().exp().matrix();
        ColumnVect<T> sum = score.rowwise().sum();
        MatrixX<T> prob = (score.array().colwise() / sum.array()).matrix();
        return prob;
    }

    MatrixX<T> backward_(const MatrixX<T>& X) const {
        ColumnVect<T> sum = X.cwiseProduct(cache_).rowwise().sum();
        MatrixX<T> grad = X.colwise() - sum;
        return cache_.cwiseProduct(grad);
    }
};
}  // namespace BlockClass_default_impl
