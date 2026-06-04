#pragma once

#include <vector>

#include "NeuroLib/Block/block.hpp"
#include "NeuroLib/utility.hpp"

namespace NetworkClass {

// using namespace NetworkClass::NetworkBuilder
using namespace project_utils;
using namespace BlockClass;

template <typename T>
class Network {
   public:
    explicit Network(std::vector<Layer<T>> layers)
        : layers(std::move(layers)) {}

    MatrixX<T> forward(const MatrixX<T>& X) {
        MatrixX<T> out = X;
        for (auto& layer : layers) out = layer.forward(out);
        return out;
    }

    MatrixX<T> predict(const MatrixX<T>& X) {
        MatrixX<T> out = X;
        for (auto& layer : layers) out = layer.predict(out);
        return out;
    }

    void backward(const MatrixX<T>& loss) {
        MatrixX<T> grad = loss;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad = it->backward(grad);
        }
    }

    std::vector<Layer<T>> layers;
};

namespace Builder_details {

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
}  // namespace Builder_details

template <typename T>
decltype(auto) create_network(uint64_t seed = 42) {
    return Builder_details::create_network<T>();
}
}  // namespace NetworkClass
