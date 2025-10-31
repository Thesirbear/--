#include <cassert>
#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vector.hpp"

namespace Vector_testing {

// const unsigned int seed = 1720135165;
const unsigned int seed =
    std::chrono::steady_clock().now().time_since_epoch().count();
std::mt19937 rnd(seed);

void Simple_Constructors() {
    std::cout << "Simple_Constructors\n";
    Vector<int> v(10);
    Vector<int> h(100);
    Vector<double> gt(1000);

    v = std::move(h);  // move operator=

    Vector<int> p(std::move(v));  // move constructor
    std::cout << "Passed\n";
}

void Default_types_stress() {
    std::cout << "Default_types_stress\n";
    Vector<int> v(1000000);
    std::vector<int> gurantee(1000000);
    for (int i = 0; i < 1000000; i++) {
        gurantee.push_back(rnd() % 100000);
        v.Push_back(gurantee.back());
    }
    auto check_by_elem = [&]() {
        for (int i = 0; i < gurantee.size(); i++) {
            if (gurantee[i] != v[i]) {
                std::cout << "Broken\n";
                exit(-1);
            }
        }
    };
    for (int i = 0; i < 100000; i++) {
        int coin = rnd() % 7;
        if (coin == 0) {
            int x = rnd() % v.Size() + v.Size();
            if (x > 1000000 || x < 0) x = 0;
            v.Resize(x);
            gurantee.resize(x);
        } else if (coin == 1) {
            int load = rnd() % 10;
            v.Change_load_factor(load);
        } else if (coin == 2) {
            Vector<int> Gt(rnd() % 100);
            std::vector<int> Gt1(Gt.Size());
            v = std::move(Gt);
            gurantee = std::move(Gt1);
        } else if (coin == 3) {
            int x = rnd() % (v.Size() * 2);
            if (x > 10000000 || x < 0) x = rnd() % 100;
            v.Reserve(x);
            gurantee.reserve(x);
        } else if (coin == 4 && v.Size() > 0) {
            int val = rnd() % 100000;
            gurantee.push_back(val);
            v.Push_back(gurantee.back());
        } else if (coin == 5 && v.Size() > 0) {
            gurantee.pop_back();
            v.Pop_back();
        } else if (coin == 6 && v.Size() > 0) {
            int pos = rnd() % gurantee.size();
            int val = rnd() % 100000;
            gurantee.insert(gurantee.begin() + pos, val);
            v.Insert(pos, val);
        } else if (v.Size() > 0) {
            int pos = rnd() % gurantee.size();
            gurantee.erase(gurantee.begin() + pos);
            v.Erase(pos);
        }
        check_by_elem();
    }
    std::cout << "Passed\n";
}

void Catch_User_Faults() {
    std::cout << "Catch_User_Faults\n";
    try {
        Vector<bool> v(1);
        v.Pop_back();
        v.Pop_back();
    } catch (const std::exception& e) {
        if (std::string(e.what()) != "Can't erase, vector already empty.") {
            std::cout << "Empty(): error is uncatchable\n";
        }
    }

    try {
        Vector<bool> v(rnd() % 100000);
        int n = v.Size();
        for (int i = 0; i < n; i++) {
            size_t id = rnd() % (n + 100);
            v.Erase(id);
        }
        v.Erase(0);
    } catch (const std::exception& e) {
        if (std::string(e.what()) != "Can't erase, position unexist.") {
            std::cout << "Out of range: error is uncatchable\n";
        }
    }

    try {
        Vector<int> g(100);
        g[101] = 50;
    } catch (const std::exception& e) {
        if (std::string(e.what()) != "Position is out of bounds.") {
            std::cout << "UB, we can't catch out of bounds.";
        }
    }

    try {
        Vector<int> g;
        int x = g.Front();
    } catch (const std::exception& e) {
        if (std::string(e.what()) != "Vector is empty.") {
            std::cout << "UB, we can't catch front of empty vector error.";
        }
    }

    try {
        Vector<int> g;
        int x = g.Back();
    } catch (const std::exception& e) {
        if (std::string(e.what()) != "Vector is empty.") {
            std::cout << "UB, we can't catch back of empty vector error.";
        }
    }

    std::cout << "Passed\n";
}

void Struct_Constructors() {
    class Knowledge {
       public:
        Knowledge() = default;
        virtual void quali() const { std::cout << "I'm trancendet\n"; }
        virtual ~Knowledge() = default;
    };
    class Truth : public Knowledge {
        int magic = 0;

       public:
        Truth() = default;
        Truth(int superstition) : magic(superstition) {}
        void quali() const override { std::cout << "I'm unbreakable\n"; }
    };

    // Upcast Check
    Vector<std::unique_ptr<Knowledge>> t(50);
    Vector<std::unique_ptr<Truth>> gt(100);
    gt.Emplace_back(std::make_unique<Truth>());
}

void Multidimensional_Test() {
    std::cout << "Multidimensional_Test" << std::endl;
    Vector<Vector<int>> v(50, Vector<int>(50));
    v[5] = Vector<int>(3, 10);
    assert(v[5][0] == 10);
    v[6] = std::move(v[5]);
    assert(v[5].Empty());
    assert(v[6][0] == 10);
    v[5] = std::move(v[5]);
    v.Push_back(Vector<int>(10, 5));
    v.Pop_back();
    v.Clear();
    v.Push_back(Vector<int>(25, 10));
    assert(v[0][1] == 10);
    std::cout << "Passed\n";
}

void TestAll() {
    std::cout << "Number selected for seed: " << seed << "\n";

    Simple_Constructors();

    try {
        Default_types_stress();
    } catch (const std::exception& e) {
        std::cout << e.what() << "\n";
    }

    Catch_User_Faults();

    Multidimensional_Test();

    Struct_Constructors();
}

}  // namespace Vector_testing

int main() {
    Vector_testing::TestAll();
    return 0;
}