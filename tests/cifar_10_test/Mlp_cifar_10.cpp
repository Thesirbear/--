#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "NeuroLib/Core.hpp"
using namespace NeuroCore;
using namespace DatasetClass::data_utils;

int main(int argc, char* argv[]) {
    std::mt19937_64 rng(42);

    std::ofstream losses(std::filesystem::absolute(argv[0]).parent_path() /
                         "losses_history.csv");
    std::ofstream accuracy(std::filesystem::absolute(argv[0]).parent_path() /
                           "losses_results.csv");
    accuracy << std::fixed;
    losses << std::fixed;

    if (!losses) {
        accuracy << "Не удаётся открыть файл для записи лоссов!";
        return 0;
    }

    if (!accuracy) {
        accuracy << "Не удаётся открыть файл для записи результатов!";
        return 0;
    }

    std::vector<std::string> train = {
        "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
        "data_batch_4.bin", "data_batch_5.bin",
    };

    std::vector<std::string> test = {"test_batch.bin"};

    using scalar = float;
    auto [X, Y] = ReadCifar10<scalar>(
        std::filesystem::absolute(argv[0]).parent_path(), train);

    auto [X_t, Y_t] = ReadCifar10<scalar>(
        std::filesystem::absolute(argv[0]).parent_path(), test);

    Dataset<scalar> Cifar(std::move(X), std::move(Y), rng, 128, true);
    Dataset<scalar> Cifar_test(std::move(X_t), std::move(Y_t), rng, 128);

    Network<scalar> net = create_network<scalar>(42)
                              .linear(3072, 128)
                              .activation(LRelu<scalar>())
                              .linear(128, 10)
                              .build();

    LossFunction<scalar> loss = SoftmaxCrossEntropy<scalar>();
    Optimizer<scalar> optim =
        Optimizer<scalar>(Adam<scalar>(), TimeDecayLR<scalar>(), net);

    size_t Epochs = 100;
    std::size_t corr = 0, all = 0;
    for (size_t epoch = 0; epoch < Epochs; epoch++) {
        corr = 0, all = 0;
        for (auto&& [x, y] : Cifar) {
            MatrixX<scalar> predict = net.forward(x);
            scalar L = loss.Loss(predict, y);
            losses << L << ";";
            auto [correct, count] = DataArgmax(predict, y);
            corr += correct;
            all += count;
            MatrixX<scalar> g = loss.Gradient(predict, y);
            net.backward(g);
            optim.step(net);
        }
        losses << "\n";
        accuracy << scalar(corr) / scalar(all) * scalar(100) << ";";
    }

    for (auto&& [x, y] : Cifar_test) {
        MatrixX<scalar> predict = net.predict(x);
        scalar L = loss.Loss(predict, y);
        losses << L << ";";
        auto [correct, count] = DataArgmax(predict, y);
        corr += correct;
        all += count;
    }
    losses << "\n";
    accuracy << scalar(corr) / scalar(all) * scalar(100) << ";";

    std::ofstream model_weights(
        std::filesystem::absolute(argv[0]).parent_path() / "model.weights");

    if (!model_weights) {
        accuracy << "Не удаётся записать веса модели!";
        return 0;
    }

    for (std::size_t i = 0; i < net.layers.size(); i++) {
        model_weights << "layer " << i + 1 << "\n";
        model_weights << net.layers[i].get_W() << "\n";
        model_weights << net.layers[i].get_b() << "\n";
    }

    return 0;
}
