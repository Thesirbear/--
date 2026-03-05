#pragma once 

#include "LrScheduler.hpp"
#include "eigen/Core"
#include "folly/Poly.h"
#include <vector>
#include <cstddef>
#include <utility>

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
    
    using namespace LrSchedulesClass;
    namespace BaseOptimizerDetails {
        template<typename T>
        struct BaseOptimizer {
            using matrix = Eigen::MatrixX<T>;

            template<class Base>
            struct Interface : Base {
                bool DoNextStepOfOptimizer(std::vector<T>& loss_history) {
                    return folly::poly_call<0>(*this, loss_history);
                }

                void StepOver(matrix&& new_delta) {
                    folly::poly_call<1>(*this, std::forward<matrix>(new_delta));
                }

                matrix update_weight(matrix& grad) {
                    return folly::poly_call<2>(*this, grad);
                }
            };

            template<typename V>
            using Members = folly::PolyMembers<
                &V::DoNextStepOfOptimizer,
                &V::StepOver,
                &V::update_weight>;
        };
    };

    template<typename T>
    class VanillaGradient {
        using matrix = Eigen::MatrixX<T>;
    
        public:


        VanillaGradient(size_t iter, T precision, LrSchedule<T>& lrschedule, matrix& delta) 
            :
            rep_(0),
            iter_(iter),
            precision_(precision),
            lr_scheduler(lrschedule),
            delta_(delta)
        {}
        
        bool DoNextStepOfOptimizer(std::vector<T>& loss_history) {   
            return rep_ < iter_ && loss_history.back() > precision_ && !DoesContainNans(delta_);
        }

        void StepOver(matrix&& new_delta) {
            rep_++;
            delta_ = std::move(new_delta);
        }

        matrix update_weight(matrix& grad) {
            return lr_scheduler.Get_lr(rep_) * grad;
        }

        private:

            size_t iter_;
            size_t rep_;
            T precision_;
            matrix delta_;
            LrSchedule<T> lr_scheduler;
    };

    template<typename T>
    using Optimizer = folly::Poly<BaseOptimizerDetails::BaseOptimizer<float>>;
};
