# cifar-10-cpp

A C++ implementation for CIFAR-10 image classification using OpenCV's Artificial Neural Network (ANN) module.

## Overview

cifar-10-cpp provides a simple and efficient way to load the CIFAR-10 dataset and train a Multi-Layer Perceptron (MLP) neural network for image classification. The project includes a custom CIFAR-10 data loader and an ANN implementation using OpenCV's machine learning library.

## Features

- **CIFAR-10 Data Loader**: Read and parse CIFAR-10 binary format dataset
- **Neural Network Training**: Train an MLP with configurable architecture
- **Prediction & Evaluation**: Test model accuracy on the CIFAR-10 test set
- **OpenCV Integration**: Leverages OpenCV for image processing and machine learning
- **Unit Testing**: Includes Google Test framework for testing data loading

## Requirements

- CMake 4.0+
- C++20 compatible compiler
- OpenCV (with ML module)
- Google Test (automatically fetched by CMake)
- CIFAR-10 dataset (binary version)

## Dataset

Download the CIFAR-10 binary version from the [official website](https://www.cs.toronto.edu/~kriz/cifar.html) and extract it to the `cifar-10-batches-bin/` directory in the project root.

The dataset should contain:
- `data_batch_1.bin` through `data_batch_5.bin` (training data)
- `test_batch.bin` (test data)

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

### Basic Example

```cpp
#include "include/cifar-10.h"

// Define paths to dataset files
std::vector<std::string> train_files = {
    "cifar-10-batches-bin/data_batch_1.bin",
    "cifar-10-batches-bin/data_batch_2.bin",
    "cifar-10-batches-bin/data_batch_3.bin",
    "cifar-10-batches-bin/data_batch_4.bin",
    "cifar-10-batches-bin/data_batch_5.bin"
};
std::string test_file = "cifar-10-batches-bin/test_batch.bin";

// Load dataset
CIFAR_10::DataSet dataset(train_files, test_file);

// Create and train ANN
CIFAR_10::CIFAR10_ANN ann(dataset);
ann.train(epochs=100, eps=1e-6);

// Evaluate on test set
ann.predict();

// Save trained model
ann.save("./");
```

## Architecture

The default neural network architecture consists of:
- **Input Layer**: 3072 neurons (32×32×3 RGB image)
- **Hidden Layer 1**: 512 neurons
- **Hidden Layer 2**: 256 neurons
- **Output Layer**: 10 neurons (one for each CIFAR-10 class)

Activation function: Sigmoid Symmetric  
Training method: RPROP (Resilient Backpropagation)

## CIFAR-10 Classes

The dataset includes 10 classes:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## Running Tests

```bash
cd build
./cifar-10-cpp_test
```

Or using CTest:
```bash
ctest --output-on-failure
```

## Project Structure

```
cifar-10-cpp/
├── include/
│   └── cifar-10.h          # CIFAR-10 data loader and ANN implementation
├── cifar-10-batches-bin/   # Dataset directory
├── main.cpp                 # Example usage
├── test.cpp                 # Unit tests
├── CMakeLists.txt          # Build configuration
└── README.md               # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Louis Chenluo Deng

## Acknowledgments

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) by Alex Krizhevsky
- [OpenCV](https://opencv.org/) for computer vision and machine learning tools
- [Google Test](https://github.com/google/googletest) for unit testing framework
