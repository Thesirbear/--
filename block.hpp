#pragma once

#include "eigen/Core"
#include "folly/Poly.h"
#include <random>
#include <memory>

#ifdef Debug
    #include <iostream>
#endif


namespace BlockClass {

    using Generator = std::mt19937_64;
    enum Rows : size_t;
    enum Cols : size_t;

    namespace BlockBaseDetails {
        template<typename T>
        struct BlockBase {
            using Matrix = Eigen::MatrixX<T>;
            template<class Base>
            struct Interface : Base {
                Matrix predict(Matrix& X) const {
                    return folly::poly_call<0>(*this, X);
                }

                Matrix forward(Matrix& X) const {
                    return folly::poly_call<1>(*this, X);
                }
            };

            template<typename V>
            using Members = folly::PolyMembers<&V::predict, &V::forward>;
        };

        template<typename T>
        struct BlockBaseLinear : folly::PolyExtends<BlockBase<T>> {
            using Matrix = Eigen::MatrixX<T>;
            template<class Base>
            struct Interface : Base {
                void update(Matrix& X) {
                    folly::poly_call<0>(*this, X);
                }

                Matrix GetWeight() const {
                    return folly::poly_call<1>(*this);
                }
            };

            template<typename V>
            using Members = folly::PolyMembers<&V::update, &V::GetWeight>;
        };

        template<typename T>
        struct BlockBaseNonLinear : folly::PolyExtends<BlockBase<T>> {
            using Matrix = Eigen::MatrixX<T>;
            template<class Base>
            struct Interface : Base {
                void backward(Matrix& X) {
                    folly::poly_call<0>(*this, X);
                }
            };

            template<typename V>
            using Members = folly::PolyMembers<&V::backward>;
        };
    };

    template<typename T>
    struct Linear_Block {

        using Matrix = Eigen::MatrixX<T>;
        public:

            Linear_Block() {}

            Linear_Block(Rows x, Cols y, Generator& rng) {
                assert(x != 0 && y != 0);

                weight_ = Matrix(x, y);

                for (size_t i = 0; i < x; i++) {
                    for (size_t j = 0; j < y; j++) {
                        weight_(i, j) = uni_(rng);
                    }
                }
            }

            Linear_Block& operator=(Linear_Block<T>&& other) {
                return std::move(other);
            }
            Linear_Block& operator=(const Linear_Block<T>& other) {
                return other;
            }
            Linear_Block(const Linear_Block<T>& other) = default;
            Linear_Block(Linear_Block<T>&& other) = default;
            ~Linear_Block() = default;

            Matrix predict(Matrix& X) const {
                return X * weight_;
            }

            Matrix forward(Matrix& X) const {
                return X * weight_;
            }

            void update(Matrix& grad) {
                weight_ -= grad;
            }

            Matrix GetWeight() const {
                return weight_;
            }
        
        private:
            Matrix weight_;
            std::uniform_real_distribution<float> uni_{-1, 1};
    };

    template<typename T>
    class LRelu {
        using Matrix = Eigen::MatrixX<T>;
        public:

            LRelu& operator=(LRelu<T>&& other) noexcept {
                return std::move(other);
            }

            LRelu& operator=(const LRelu<T>& other) {
                return other;
            }
            
            LRelu(const LRelu<T>& other) = default;
            LRelu(LRelu<T>&& other) = default;
            ~LRelu() = default;

            LRelu(float alpha) : alpha_(alpha) {}

            Matrix predict(Matrix& X) const {
                for (int i = 0; i < X.rows(); i++) {
                    for (int j = 0; j < X.cols(); j++) {
                        if (X(i, j) <= 0.0) {
                            X(i, j) *= alpha_;
                        }
                    }
                }
                return X;
            }
            
            Matrix forward(Matrix& X) const {
                return predict(X);
            }

            // this function change Cur_grad!!!
            void backward(Matrix& Cur_Grad) {
                for (int i = 0; i < Cur_Grad.rows(); i++) {
                    for (int j = 0; j < Cur_Grad.cols(); j++) {
                        if (Cur_Grad(i, j) <= 0.0) {
                            Cur_Grad(i, j) = alpha_;
                        } else {
                            Cur_Grad(i, j) = 1;
                        }
                    }
                }
            }
        
        private:
            float alpha_;
    };

    template<typename T>
    using BlockLinear = folly::Poly<BlockBaseDetails::BlockBaseLinear<T>>;

    template<typename T>
    using BlockNonLinear = folly::Poly<BlockBaseDetails::BlockBaseNonLinear<T>>;

};