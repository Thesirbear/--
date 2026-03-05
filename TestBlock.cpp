#include "eigen/Core"
#include "folly/Poly.h"

#include <iostream>

#include <random>
#include "vector.hpp"

#include "block.hpp"

void TestBlockLinear() {
    std::cout << "Linear: ";
    std::mt19937_64 rnd;
    using namespace BlockClass;

    Eigen::MatrixXf g(3, 3);

    BlockLinear<float> b = Linear_Block<float>(Rows{3}, Cols{3}, rnd);

    g = b.GetWeight();

    if (!(g * b.forward(g)).isApprox((g * g) * g)) {
        std::cout << "forward is broken.\n";
        return;
    }

    if (!(g * b.predict(g)).isApprox(g * g * g)) {
        std::cout << "predict is broken.\n";
        return;
    }

    g *= 0.9;
    b.update(g);
    g *= 0.1/0.9;
    if (!b.GetWeight().isApprox(g)) {
        std::cout << "update is broken.\n";
        return;
    }
    std::cout << "Passed\n";
}

void TestBlockNonlinear() {
    std::cout << "NonLinear: ";
    using namespace BlockClass;

    BlockClass::BlockNonLinear<float> b = LRelu<float>(0.9);

    Eigen::MatrixX<float> G(2, 2);

    G(0, 0) = 1;
    G(1, 0) = -0.8;
    G(0, 1) = -0.7;
    G(1, 1) = -0.4;

    Eigen::MatrixX<float> G1(2, 2);

    G1(0, 0) = 1;
    G1(1, 0) = 0.9;
    G1(0, 1) = 0.9;
    G1(1, 1) = 0.9;

    Eigen::MatrixX<float> G2(2, 2);

    G2(0, 0) = 1;
    G2(1, 0) = -0.72;
    G2(0, 1) = -0.63;
    G2(1, 1) = -0.36;


    if (!b.forward(G).isApprox(G2)) {
        std::cout << "forward is broken.\n";
        return;
    }

    b.backward(G);
    if (!G1.isApprox(G)) {
        std::cout << "backward is broken.\n";
        return;
    }

    std::cout << "Passed\n";
}


void TestBlock() {
    TestBlockLinear();
    TestBlockNonlinear();
}