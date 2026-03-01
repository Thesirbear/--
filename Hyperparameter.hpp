#pragma once


#include <any>
#include "LrScheduler.hpp"
#include "lossfunc.hpp"

namespace HyperParameter {

    template<typename T>
    struct HyperParameter {
        int iter;
        T lr;
        std::unique_ptr<LossFunctions::LossFunctionBase> loss;
        std::unique_ptr<LrSchedule::LrScheduleBase> lr_scheduler;
    };

};