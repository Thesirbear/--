#pragma once

#include <cstddef>

#include "NeuroLib/Network/Network.hpp"
#include "NeuroLib/Optimizer/DefaultGradientDescends.hpp"
#include "NeuroLib/Optimizer/GradientDescend.hpp"
#include "NeuroLib/Scheduler/LrScheduler.hpp"
#include "NeuroLib/utility.hpp"

namespace OptimizerClass {

using namespace OptimizerClass::default_descends;
using namespace LrSchedulesClass;
using namespace NetworkClass;

template <typename T>
using GradDesc =
    Poly<OptimizerClass::details::BaseDescend::BaseGradientDescend<T>>;

template <typename T>
class Optimizer {
   public:
    Optimizer(GradDesc<T> gradient_desc, LrSchedule<T> scheduler,
              Network<T>& net)
        : sched_(scheduler), step_(0) {
        gradW_.assign(net.layers.size(), gradient_desc);
        gradB_.assign(net.layers.size(), gradient_desc);
    }

    void step(Network<T>& net) {
        T lr = sched_.Get_lr(step_);
        for (size_t i = 0; i < net.layers.size(); ++i) {
            const MatrixX<T>& dW = net.layers[i].get_dW();
            const MatrixX<T>& db = net.layers[i].get_db();
            if (net.layers[i].get_W().size() == 0) continue;
            MatrixX<T> ndW = lr * gradW_[i].optimized_grad(dW);
            MatrixX<T> ndb = lr * gradB_[i].optimized_grad(db);
            net.layers[i].update_params(ndW, ndb);
        }
        step_++;
    }

   private:
    size_t step_;
    LrSchedule<T> sched_;
    std::vector<GradDesc<T>> gradW_;
    std::vector<GradDesc<T>> gradB_;
};
};  // namespace OptimizerClass
