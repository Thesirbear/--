#pragma once
#include <random>

#include "NeuroLib/Dataloader/DataUtils.hpp"
#include "NeuroLib/utility.hpp"

namespace DatasetClass {

using namespace project_utils;

template <typename T>
class Dataset {
   public:
    struct Batch {
        MatrixX<T> x, y;
    };

    Dataset(MatrixX<T>&& X, MatrixX<T>&& Y, std::mt19937_64 rng,
            std::size_t batch = 64, bool doshuffle = false)
        : X_(std::move(X)),
          Y_(std::move(Y)),
          batch_(batch),
          rng_(rng),
          doshuffle_(doshuffle) {}

    class Iterator {
       public:
        Iterator(Dataset* dptr, std::size_t pos) : dptr_(dptr), pos_(pos) {}

        Batch operator*() const {
            const std::size_t batch =
                std::min(dptr_->batch_, dptr_->X_.cols() - pos_);
            return Batch{dptr_->X_.middleCols(pos_, batch).transpose(),
                         dptr_->Y_.middleCols(pos_, batch).transpose()};
        }

        Iterator& operator++() {
            pos_ += Curbatch_();
            return *this;
        }

        bool operator!=(const Iterator& o) const { return pos_ < o.pos_; }

       private:
        std::size_t Curbatch_() {
            return std::min(dptr_->batch_, dptr_->X_.cols() - pos_);
        }

        Dataset<T>* dptr_;
        std::size_t pos_;
    };

    Iterator begin() {
        if (doshuffle_) {
            shuffle_();
        }
        return Iterator{this, 0};
    }

    Iterator end() {
        return Iterator{this, static_cast<std::size_t>(X_.cols())};
    }

    std::size_t size() {
        return X_.rows();
    }

   private:
    void shuffle_() {
        for (std::size_t i = X_.cols() - 1; i > 0; --i) {
            std::size_t j = rng_() % i;

            X_.col(i).swap(X_.col(j));
            Y_.col(i).swap(Y_.col(j));
        }
    }

    std::mt19937_64 rng_;
    bool doshuffle_;
    std::size_t batch_;
    MatrixX<T> X_, Y_;
};
}  // namespace DatasetClass
