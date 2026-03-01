#pragma once

#include <cstddef>
#include <any>

namespace LrSchedule {
    class LrScheduleBase {
        public:
            virtual std::any Get_lr(size_t iter) = 0;
            virtual ~LrScheduleBase() = default;
    };

    template<typename T>
    class LrConstant : public LrScheduleBase {
        private:
            T lr_;

        public:

            LrConstant() : lr_(1) {}
            LrConstant(T lr) : lr_(lr) {}
            T Get_lr(size_t iter) {
                return lr_;
            }
    };

    template<typename T>
    class TimeDecayLR : public LrScheduleBase{
        private:
            T s0;
            T p;
            T lambda_;
            size_t precision_;

            T count_pow(T x) {
                T l = 0, r = 100;
                for (size_t i = 0; i < precision_; i++) {
                    T mid = (l + r) / 2;
                    if (x - mid * mid > 1e-12) {
                        l = mid;
                    } else {
                        r = mid;
                    }
                }
                return l;
            }

        public:

            TimeDecayLR() :
                s0(1),
                p(0.5),
                lambda_(1),
                precision_(100)
            {}

            TimeDecayLR(T lambda, T precision, T s0 = 1, T p = 0.5) :
                s0(s0),
                p(p),
                lambda_(lambda),
                precision_(precision)
            {}

            T Get_lr(size_t iter) override  {
                return lambda_ * count_pow(s0 / (s0 + static_cast<T>(iter)));
            }
    };
}