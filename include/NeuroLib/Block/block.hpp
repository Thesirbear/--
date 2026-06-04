#pragma once

#include <cstddef>

#include "NeuroLib/Block/defaultblocks.hpp"
#include "NeuroLib/utility.hpp"

namespace BlockClass {
using namespace BlockClass_default_impl;
using namespace project_utils;
namespace BlockBaseDetails {
template <typename T>
struct BlockBase {
    template <class Base>
    struct Interface : Base {
        MatrixX<T> predict(const MatrixX<T>& X) const {
            return poly_call<0>(*this, X);
        }

        MatrixX<T> forward(const MatrixX<T>& X) {
            return poly_call<1>(*this, X);
        }

        MatrixX<T> backward(const MatrixX<T>& X) {
            return poly_call<2>(*this, X);
        }

        void update_params(const MatrixX<T>& dW, const MatrixX<T>& db) {
            poly_call<3>(*this, dW, db);
        }

        const MatrixX<T>& get_dW() const { return poly_call<4>(*this); }

        const MatrixX<T>& get_db() const { return poly_call<5>(*this); }

        const MatrixX<T>& get_W() const { return poly_call<6>(*this); }

        const MatrixX<T>& get_b() const { return poly_call<7>(*this); }

        void set_params(const MatrixX<T>& W, const MatrixX<T>& b) {
            poly_call<8>(*this, W, b);
        }
    };

    template <typename V>
    using Members = PolyMembers<&V::predict, &V::forward, &V::backward,
                                &V::update_params, &V::get_dW, &V::get_db,
                                &V::get_W, &V::get_b, &V::set_params>;
};
};  // namespace BlockBaseDetails

template <typename T>
using Layer = Poly<BlockBaseDetails::BlockBase<T>>;
};  // namespace BlockClass
