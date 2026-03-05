
// by yourself on macos
brew install pkg-config


##Installation of dependencies##
git submodule update --init vcpkg
cd ./vcpkg
sh bootstrap-vcpkg.sh

##Building##
cmake --build build -j8

##Running##
./build/Test 1020 > .build/out
