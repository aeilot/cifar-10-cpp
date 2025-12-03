cmake_minimum_required(VERSION 4.0)
project(cifar-10-cpp)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cifar-10-cpp main.cpp
        include/cifar-10.h
)

add_executable(cifar-10-cpp_test test.cpp
        include/cifar-10.h
)

target_link_libraries(cifar-10-cpp ${OpenCV_LIBS})
target_link_libraries(cifar-10-cpp_test ${OpenCV_LIBS}  GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(cifar-10-cpp_test)
