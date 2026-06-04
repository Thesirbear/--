#include <cstddef>
#include <algorithm>
#include <cmath>

namespace LrSchedulesClass::default_impl {
template <typename T>
class LrConstant {
private:
    T lr_;

public:
    LrConstant() : lr_(1) {}
    LrConstant(T lr) : lr_(lr) {}
    T Get_lr(size_t iter) { return lr_; }
};

template <typename T>
class StepLR {
    public:
    StepLR(T alpha = 0.1, T n = 100, T d = 0.1, T min = T(1e-4)) : alpha_(alpha), n_(n), d_(d), min_(min), i_(0) {}

    T Get_lr(size_t iter) {
        if (i_ == n_) {
            alpha_ *= d_;
            i_ = 0;
        }
        i_++;
        return std::max({min_, alpha_});
    }

    private:
        std::size_t n_;
        std::size_t i_;
        T alpha_;
        T min_;
        T d_;
};
}  // namespace LrSchedulesClass::default_impl
