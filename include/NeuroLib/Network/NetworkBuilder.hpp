#pragma once
#include <cstddef>

#include "NeuroLib/Block/block.hpp"
#include "NeuroLib/utility.hpp"

namespace NetworkClass {
template <typename T>
class Network;
}

namespace NetworkClass::Builder_details {

using namespace BlockClass;

template <typename T>
class LinearLayer;
template <typename T>
class NonlinearLayer;

template <typename T>
class NetworkBuilder {
   public:
    explicit NetworkBuilder(uint64_t seed = 42) : rng_(seed) {}

    template <typename L>
    void add(L&& layer) {
        layers_.emplace_back(std::forward<L>(layer));
    }

    rnd& rng() { return rng_; }

    Network<T> build() { return Network<T>(std::move(layers_)); }

   private:
    std::vector<Layer<T>> layers_;
    rnd rng_;
};

template <typename T>
class NonlinearLayer {
   public:
    explicit NonlinearLayer(NetworkBuilder<T> builder)
        : builder_(std::move(builder)) {}

    LinearLayer<T> linear(size_t n, size_t m) && {
        builder_.add(LinearBlock<T>(Rows{n}, Cols{m}, builder_.rng()));
        return LinearLayer<T>(std::move(builder_));
    }

    Network<T> build() && { return builder_.build(); }

   private:
    NetworkBuilder<T> builder_;
};

template <typename T>
class LinearLayer {
   public:
    explicit LinearLayer(NetworkBuilder<T> builder)
        : builder_(std::move(builder)) {}

    NonlinearLayer<T> activation(Layer<T> Activate) && {
        builder_.add(Activate);
        return NonlinearLayer<T>(std::move(builder_));
    }

    Network<T> build() && { return builder_.build(); }

   private:
    NetworkBuilder<T> builder_;
};

template <typename T>
NonlinearLayer<T> create_network(uint64_t seed = 42) {
    return NonlinearLayer<T>(NetworkBuilder<T>(seed));
}
}  // namespace NetworkClass::Builder_details
