#pragma once

#include <cstddef>
#include "LrScheduler.hpp"
#include "lossfunc.hpp"

namespace HyperParameters {

    using namespace LrSchedulesClass;
    using namespace LossFunctionsClass;
    template<typename T>
    struct HyperParameter {
        size_t iter;
        T lr;
        LossFunction<T> loss;
        LrSchedule<T> lr_scheduler;
    };

};