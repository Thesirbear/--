#pragma once 

#include "lossfunc.hpp"
#include "eigen/Eigen/Core"
#include "Schema.hpp"
#include "lossfunc.hpp"
#include "Hyperparameter.hpp"
#include <vector>
#include <cstddef>

#include <any>

namespace Optimizer {
    
    template<typename T>
    bool DoesContainNans(Eigen::MatrixX<T>& v) {
        for (size_t pos = 0; pos < v.rows(); pos++) {
            for (size_t j = 0; j < v.cols(); j++) {
                if (std::isnan(v(pos, j))) {
                    return true;
                }
            }
        }
        return false;
    }
    
    
    template<typename T>
    class BaseOptimizer {
        using matrix = Eigen::MatrixX<T>;
        using Schema = NeuralSchema::Schema<T>;
        using Adapter = std::any;
        using HParams = HyperParameter::HyperParameter<T>;
        public:
        
            BaseOptimizer(
                matrix& X_, 
                matrix& Y_,
                Schema& B_,
                HParams& param_
            ) :
                X(X_),
                Y(Y_),
                Schem(B_),
                param(param_)
            {}

            void Optimize() {

                for (auto& x : Schem.Blocks) {
                    x.forw
                }

                matrix delta = Schema.GetWeight();
                loss_history.push_back(delta.transpose()*delta.value());
                while (rep < param.iter && std::any_cast<T>(loss_history.back()) > 1e-9 && !DoesContainNans(delta)) {
                    matrix result = Step(rep);
                    loss_history.push_back((result.transpose()*result).values());
                    delta = result;
                }
            }

            std::vector<T> GetLossHistory() {
                return loss_history;
            }

        private:

            void Update_Weights_() {
                B.forward(X);
                matrix grad = B.backward(Y);
                B.update(param.lr_scheduler.Get_lr(rep));
            }
            
            void Step() {
                rep++;
                return Update_Weights_();
            }

            size_t rep = 0;
            std::vector<Adapter> loss_history;
            matrix& X, Y;
            Schema& Schem;
            HParams& param;
    };
};
