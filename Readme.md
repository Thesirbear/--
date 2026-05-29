
Чтобы всё заработало нужно поставить:

На макос для работы vcpkg:
brew install pkg-config

Сам проект:
git submodule update --init EigenRand
git submodule update --init vcpkg
cd ./vcpkg
sh bootstrap-vcpkg.sh

Тестирование:

Сборка теста:
cmake -B build
cmake -build build

Нужно поставить:
Python3
Также пакеты:
matplotlib
numpy

(Нужно предварительно скачать Cifar10 с сайта в формате bin и разархивировать в build папке), иначе не заработает.

Запуск Модели:
(Модель учиться долго без OpenMP подключить можно отдельно):
./build/Сifar10_test
Внутри в файле tests/cifar_10_test/Mlp_cifar_10.cpp можно двигать константы и подставлять другие оптимизаторы, шедулеры и т.д.

python3 cifar_runner.py
Выдаст картиночки на обученной сети.
