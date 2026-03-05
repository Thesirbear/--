#pragma once 

#include "eigen/Core"
#include "folly/Poly.h"

namespace LossFunctionsClass {

    namespace LossBaseDetails {
        template<typename T>
        struct LossFunctionBase {

            using Matrix = Eigen::MatrixX<T>;

            template<class Base>
            struct Interface : Base {
                Matrix Gradient(Matrix& cur, Matrix& grad, Matrix& w) {
                    return folly::poly_call<0>(*this, cur, grad, w);
                }

                Matrix Loss(Matrix& cur, Matrix& grad, Matrix& w) {
                    return folly::poly_call<1>(*this, cur, grad, w);
                }
            };

            template<typename V>
            using Members = folly::PolyMembers<&V::Gradient, &V::Loss>;
        };
    };

    template<typename T>
    class Loss_Mse {
        public:

            using matrix = Eigen::MatrixX<T>;

            matrix Gradient(matrix& X, matrix& Y, matrix& W) {
                return 2.0/static_cast<T>(Y.rows())*X.transpose()*(X*W - Y);
            }

            matrix Loss(matrix& X, matrix& Y, matrix& W) {
                return 1.0/static_cast<T>(Y.rows())*(X*W - Y).transpose()*(X*W - Y);
            }
    };

    template<typename T>
    using LossFunction = folly::Poly<LossBaseDetails::LossFunctionBase<T>>;
}
