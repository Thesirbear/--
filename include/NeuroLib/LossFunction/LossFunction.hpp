#pragma once

#include "NeuroLib/LossFunction/LossDefaultFunctions.hpp"
#include "NeuroLib/utility.hpp"

namespace LossFunctionsClass {

using namespace project_utils;
using namespace LossFunctionsClass::default_impl;
namespace LossBaseDetails {
template <typename T>
struct LossFunctionBase {
    template <class Base>
    struct Interface : Base {
        MatrixX<T> Gradient(const MatrixX<T>& X, const MatrixX<T>& Y) {
            return poly_call<0>(*this, X, Y);
        }

        T Loss(const MatrixX<T>& X, const MatrixX<T>& Y) {
            return poly_call<1>(*this, X, Y);
        }
    };

    template <typename V>
    using Members = PolyMembers<&V::Gradient, &V::Loss>;
};
};  // namespace LossBaseDetails

template <typename T>
using LossFunction = Poly<LossBaseDetails::LossFunctionBase<T>>;
}  // namespace LossFunctionsClass
