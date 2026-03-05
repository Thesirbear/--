#include "block.hpp"
#include "eigen/Core"

#include "folly/Poly.h"

#include "Optimizer.hpp"
#include "lossfunc.hpp"
#include "Hyperparameter.hpp"
#include "LrScheduler.hpp"

// Tests
#include "TestBlock.cpp"

#include <ctime>
#include <iostream>

#include <random>
#include "folly/Poly.h"


#ifdef Debug
    #define Line std::cout << __LINE__ << "\n"
#else
    #define Line
#endif

std::uniform_real_distribution<float> distribution{0.0, 1.0};
std::normal_distribution<float> distr_norma;
std::mt19937_64 rnd;



bool CheckLinearRegression() {
    using namespace BlockClass;
    // Main variables
    using skalar = float;
    using Hparam = HyperParameters::HyperParameter<skalar>;

    size_t train_size = 200;
    size_t poly = 7;

    Hparam param;
    param.lr = 2;
    param.iter = 1000;
    param.lr_scheduler = LrSchedulesClass::TimeDecayLR<float>(param.lr, 300);
    param.loss = LossFunctionsClass::Loss_Mse<float>();

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
    using Matrix = Eigen::MatrixX<skalar>;
    Matrix X(GenData.size(), poly + 1);
    Matrix Y(GenData.size(), 1);

    for (size_t j = 0; j < GenData.size(); j++) {
        Y(j) = GenData[j].second;
        for (size_t k = 0; k <= poly; k++) {
            if (k == 0) {
                X(j, k) = GenData[j].first;
            } else {
                X(j, k) = X(j, k - 1) * GenData[j].first;
            }
        }
        X(j, poly) = 1;
    }

    BlockLinear<skalar> B = Linear_Block<skalar>(Rows{poly + 1}, Cols{1}, rnd);

    
    std::vector<skalar> loss_history;
    loss_history.push_back((B.GetWeight().transpose()*B.GetWeight()).value());
    Optimizer::Optimizer<skalar> optim;
    {
        Matrix state = B.GetWeight();
        optim = Optimizer::VanillaGradient<skalar>(param.iter, 1e-12, param.lr_scheduler, state);
    }
    while (optim.DoNextStepOfOptimizer(loss_history)) {
        Matrix cur_weight = B.GetWeight();
        Matrix grad = param.loss.Gradient(X, Y, cur_weight);
        Matrix modifi_grad = optim.update_weight(grad);
        B.update(modifi_grad);
        loss_history.push_back((grad.transpose()*grad).value());
        // std::cout << loss_history.back() << "\n";
        optim.StepOver(std::move(grad));
    }

    Matrix y(1, poly + 1);
    y(0, 0) = 0.5;
    for (size_t pos = 1; pos <= poly; pos++) y(0, pos) = y(0, pos - 1) * 0.5; 
    y(0, poly) = 1;

    std::cout << B.GetWeight() << "\n";
    Matrix ans = B.predict(y);
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

    std::cout << std::fixed;

    TestBlock();

    CheckLinearRegression();

    return 0;
}