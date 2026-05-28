#pragma once

#include "NeuroLib/Scheduler/LrDefaultSchedulers.hpp"
#include "NeuroLib/utility.hpp"

namespace LrSchedulesClass {

using namespace project_utils;
using namespace LrSchedulesClass::default_impl;
namespace LrScheduleBaseDetails {
template <typename T>
struct LrScheduleBase {
    template <class Base>
    struct Interface : Base {
        T Get_lr(size_t iter) { return poly_call<0>(*this, iter); }
    };

    template <typename V>
    using Members = PolyMembers<&V::Get_lr>;
};
};  // namespace LrScheduleBaseDetails

template <typename T>
using LrSchedule = Poly<LrScheduleBaseDetails::LrScheduleBase<T>>;
};  // namespace LrSchedulesClass
