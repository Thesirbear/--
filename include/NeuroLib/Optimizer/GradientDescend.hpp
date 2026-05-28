#include "NeuroLib/utility.hpp"

using namespace project_utils;
namespace OptimizerClass::details::BaseDescend {
template <typename T>
struct BaseGradientDescend {
    template <class Base>
    struct Interface : Base {
        MatrixX<T> optimized_grad(const MatrixX<T>& grad) {
            return poly_call<0>(*this, grad);
        }
    };
    template <typename V>
    using Members = PolyMembers<&V::optimized_grad>;
};
}  // namespace OptimizerClass::details::BaseDescend
