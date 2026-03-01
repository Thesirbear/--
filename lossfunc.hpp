#pragma once 

#include "eigen/Eigen/Core"
#include <any>

namespace LossFunctions {

    class LossFunctionBase {
        public:
            virtual std::any Gradient() = 0;
            virtual std::any Loss() = 0;
            virtual ~LossFunctionBase() = default;
    };

    template<typename T>
    class Loss_Mse : public LossFunctionBase {
        public:

            using matrix = Eigen::MatrixX<T>;

            template<typename U>
            std::any Gradient(matrix& X, Eigen::MatrixBase<U>& Y, matrix& W) {
                return 2.0/static_cast<T>(Y.rows())*X.transpose()*(X*W - Y);
            }

            template<typename U>
            std::any Loss(matrix& X, Eigen::MatrixBase<U>& Y, matrix& W) {
                return 1.0/static_cast<T>(Y.rows())*(X*W - Y).transpose()*(X*W - Y);
            }
    };

}
