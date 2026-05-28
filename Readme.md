
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
cmake --build build -j8

Нужно поставить:
Python3
Также пакеты:
matplotlib
numpy

Запуск(Модель учиться долго без OpenMP подключить можно отдельно):
./build/Сifar10_test

Либо через python скрипт в tests/cifar_10_test

python3 cifar_runner.py

Выдаст картиночки и прогонит сеть.
