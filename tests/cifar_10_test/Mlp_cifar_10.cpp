#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <omp.h>

#include "NeuroLib/Core.hpp"
#include "NeuroLib/Scheduler/LrScheduler.hpp"
using namespace NeuroCore;
using namespace DatasetClass::data_utils;

#define EIGEN_USE_BLAS
int main(int argc, char* argv[]) {

    const std::size_t batch = 512;
    const std::size_t hidden = 128;
    const std::size_t seed = 42;
    const std::size_t Epochs = 1000;

    Eigen::setNbThreads(omp_get_max_threads());

    std::mt19937_64 rng(seed);

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

    Dataset<scalar> Cifar(std::move(X), std::move(Y), rng, batch, true);
    Dataset<scalar> Cifar_test(std::move(X_t), std::move(Y_t), rng, batch);

    std::cout << "Dataset is ready!\n";
    Network<scalar> net = create_network<scalar>(seed)
                              .linear(3072, hidden)
                              .activation(LRelu<scalar>())
                              .linear(hidden, hidden)
                              .activation(LRelu<scalar>())
                              .linear(hidden, hidden)
                              .activation(LRelu<scalar>())
                              .linear(hidden, hidden / 2)
                              .activation(LRelu<scalar>())
                              .linear(hidden / 2, hidden / 4)
                              .activation(LRelu<scalar>())
                              .linear(hidden / 4, 10)
                              .build();

    LossFunction<scalar> loss = SoftmaxCrossEntropy<scalar>();
    Optimizer<scalar> optim =
    Optimizer<scalar>(Adam<scalar>(), StepLR<scalar>(scalar(1e-3), batch * 100, scalar(0.5), scalar(1e-5)), net);
    
    std::size_t corr = 0, all = 0;

    for (size_t epoch = 0; epoch < Epochs; epoch++) {
        corr = 0, all = 0;
        std::size_t pos = 0;
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
            ++pos;
        }
        losses << "\n";
        std::cout << epoch << "\n";

        accuracy << scalar(corr) / scalar(all) * scalar(100) << ";";
    }
    
    corr = 0, all = 0;
    for (auto&& [x, y] : Cifar_test) {
        MatrixX<scalar> predict = net.predict(x);
        auto [correct, count] = DataArgmax(predict, y);
        corr += correct;
        all += count;
    }

    accuracy << "\n" << scalar(corr) / scalar(all) * scalar(100);
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
