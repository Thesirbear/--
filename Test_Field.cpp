#include "Schema.hpp"
#include "eigen/Eigen/Core"

#include "Hyperparameter.hpp"
#include "LrScheduler.hpp"
#include "Optimizer.hpp"
#include "block.hpp"
#include "lossfunc.hpp"
#include "Schema.hpp"

#include <ctime>
#include <iomanip>
#include <iostream>

#include <random>
#include <chrono>
#include "vector.hpp"
#include <cmath>
#include <any>
#include <memory>

#ifdef Debug
    #define Line std::cout << __LINE__ << "\n"
#else
    #define Line
#endif

std::uniform_real_distribution<float> distribution{0.0, 1.0};
std::normal_distribution<float> distr_norma;
std::mt19937_64 rnd;

bool CheckLinearRegression() {
    // Main variables
    using skalar = float;
    using Hparam = HyperParameter::HyperParameter<skalar>;

    size_t train_size = 100;
    size_t poly = 7;

    Hparam param;
    param.lr = 3e-4;
    param.iter = 300;
    param.lr_scheduler = std::make_unique<LrSchedule::TimeDecayLR<float>>();
    param.loss = std::make_unique(LossFunctions::Loss_Mse<float> mse);
    std::vector<std::pair<skalar, skalar>> GenData(train_size);

    auto func = [&](skalar x)  {
        return std::cos(1.5*M_PI*x);
    };
    
    for (size_t i = 0; i < GenData.size(); i++) {
        skalar val = distribution(rnd);
        skalar error = distr_norma(rnd)*0.01;
        GenData[i] = {val, func(val) + error};
    }

// preparing GenData
    using MatrixXf = Eigen::MatrixXf;
    Eigen::MatrixXf X(GenData.size(), poly + 1);
    Eigen::MatrixXf Y(GenData.size());

    for (size_t j = 0; j < GenData.size(); j++) {
        Y(j) = GenData[j].second;
        for (size_t k = 0; k <= poly; k++) {
            if (k == 0) {
                X(j, k) = GenData[j].first;
            } else {
                X(j, k) = X(j, k - 1) * GenData[j].first;
            }
        }
        X(j, 7) = 1;
    }

    using Block::Rows;
    using Block::Cols;

    NeuralSchema::Schema<float> Schematic (NeuralSchema::SchemaBuilder<float>().LinearLayer(Rows{poly + 1}, Cols{1}, rnd));

    Optimizer::BaseOptimizer<skalar> optim(X, Y, Schematic, mse, param);

    Eigen::MatrixXf y(1, poly + 1);
    y(0, 0) = 0.5;
    for (size_t pos = 1; pos <= poly; pos++) y(0, pos) = y(0, pos - 1) * 0.5; 
    y(0, poly) = 1;

    std::cout << B.GetWeight() << "\n";
    Eigen::MatrixXf ans = B.predict(y);
    std::cout << ans << " " << std::cos(1.5*M_PI*0.5) << "\n";
    return true;
}

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(0);
    std::cout.tie(0);

    if (argc == 2) {
        uint64_t seed = 0;
        size_t cnt = 0;
        while (argv[1][cnt] != '\0') {
            seed += argv[1][cnt] - '0';
            seed *= 10;
            cnt++;
        }
    } else {
        uint64_t seed = std::chrono::steady_clock::now().time_since_epoch().count();
        std::cout << seed << "\n";
    }

    std::cout << std::fixed << std::setprecision(16);


    return 0;
}