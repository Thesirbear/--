#pragma once

#include "eigen/Eigen/Core"
#include "lossfunc.hpp"
#include <random>
#include <any>
#include <memory>

#ifdef Debug
    #include <iostream>
#endif

namespace Block {

    using Loss = LossFunctions::LossFunctionBase;
    using Generator = std::mt19937_64;
    enum Rows : size_t;
    enum Cols : size_t;

    class BlockBase {

            virtual void predict() = 0;

            virtual void forward() = 0;

            virtual void backward() = 0;

            virtual void update() = 0;

            virtual std::any GetWeight() = 0;
        
            virtual ~BlockBase() = default;
        private:
    };

    template<typename T>
    class Linear_Block : public BlockBase {
        public:

            Linear_Block(Rows x, Cols y, Generator& rng) {
                assert(x != 0 && y != 0);

                weight_ = Eigen::MatrixX<T>(x, y);

                for (size_t i = 0; i < x; i++) {
                    for (size_t j = 0; j < y; j++) {
                        weight_(i, j) = uni_(rng);
                    }
                }
            }

            // Внутри unique_ptr копировать нельзя.
            Linear_Block(Linear_Block<T>& other) = delete;


            Linear_Block operator=(Linear_Block<T>&& other) noexcept {
                return std::move(other);
            }


            Linear_Block(Linear_Block<T>&& other) {
                weight_ = std::move(other.weight_);
                Loss_ = std::move(other.Loss_);
                uni_ = std::move(other.uni_);
            }

            template<typename U>
            Eigen::MatrixX<T> predict(Eigen::MatrixBase<U>& X) {
                return X * weight_;
            }

            template<typename U>
            Eigen::MatrixX<T> forward(Eigen::MatrixBase<U>& X) {
                cache_ = X;
                return X*weight_;
            }

            template<typename U>
            Eigen::MatrixX<T> backward(Eigen::MatrixBase<U>& Cur_Grad) {
                return Loss_->Gradient(cache_, Cur_Grad, weight_);
            }

            template<typename U>
            void update(T lr, Eigen::MatrixBase<U>& grad) {
                weight_ -= lr * grad;
            }

            Eigen::MatrixX<T> GetWeight() {
                return weight_;
            }
        
        private:
            Eigen::MatrixX<T> weight_;
            Eigen::MatrixX<T> cache_;
            std::unique_ptr<Loss> Loss_;
            std::uniform_real_distribution<float> uni_{-1, 1};
    };


    template<typename T>
    class Relu : public BlockBase {
        public:

            Relu(float alpha) : alpha_(alpha) {}

            template<typename U>
            Eigen::MatrixX<T> predict(Eigen::MatrixX<U>& X) {
                for (int i = 0; i < X.rows(); i++) {
                    for (int j = 0; j < X.cols(); j++) {
                        if (X(i, j) <= 0.0) {
                            X(i, j) *= alpha_;
                        }
                    }
                }
                return X;
            }
            
            template<typename U>
            Eigen::MatrixX<T> forward(Eigen::MatrixX<U>& X) {
                return cache_ = predict(X);
            }

            template<typename U>
            Eigen::MatrixX<T> backward(Eigen::MatrixBase<U>& Cur_Grad) {
                // magic with loss function
                for (int i = 0; i < Cur_Grad.rows(); i++) {
                    for (int j = 0; j < Cur_Grad.cols(); j++) {
                        if (Cur_Grad(i, j) <= 0.0) {
                            Cur_Grad(i, j) = alpha_;
                        } else {
                            Cur_Grad(i, j) = 1;
                        }
                    }
                }
                return cache_.transpose()*Cur_Grad;
            }
        
        private:
            float alpha_;
            Eigen::MatrixX<T> cache_;
    };
}