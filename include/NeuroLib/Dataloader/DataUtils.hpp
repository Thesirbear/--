#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "NeuroLib/utility.hpp"

namespace DatasetClass::data_utils {
using namespace project_utils;

template <typename T>
void ReadCifar10Bin(const std::filesystem::path& path, MatrixX<T>& X,
                    MatrixX<T>& Y, std::size_t& cnt) {
    std::ifstream Bin(path, std::ios::binary);

    if (!Bin) {
        std::cerr << "Файл не открылся!\n";
    }

    std::array<std::uint8_t, 3073> buf;
    while (Bin.read(reinterpret_cast<char*>(buf.data()), buf.size())) {
        Y(buf[0], cnt) = T(1);
        for (std::size_t i = 0; i < 3072; ++i) {
            X(i, cnt) = T(buf[i + 1]) / T(255);
        }
        cnt++;
    }
}

std::size_t CountCifarSize(std::filesystem::path path) {
    std::ifstream Bin(path, std::ios::binary);
    if (!Bin) {
        return 0;
    }
    std::error_code ec;
    return static_cast<std::size_t>(std::filesystem::file_size(path, ec)) /
           3073;
}

template <typename T>
std::pair<MatrixX<T>, MatrixX<T>> ReadCifar10(
    const std::filesystem::path& path, const std::vector<std::string>& s) {
    std::size_t total = 0;
    for (const auto& x : s) total += CountCifarSize(path / x);

    MatrixX<T> X(3072, total), Y(10, total);

    std::size_t cnt = 0;
    for (auto& x : s) {
        ReadCifar10Bin<T>(path / x, X, Y, cnt);
    }
    return std::make_pair(std::move(X), std::move(Y));
}

template <typename T>
std::pair<std::size_t, std::size_t> DataArgmax(const MatrixX<T>& pred,
                                               const MatrixX<T>& y_true) {
    std::size_t count = 0;
    for (Eg_Index i = 0; i < pred.rows(); i++) {
        Eg_Index id;
        pred.row(i).maxCoeff(&id);
        count += (y_true(i, id) == 1);
    }
    return {count, pred.rows()};
}
}  // namespace DatasetClass::data_utils
