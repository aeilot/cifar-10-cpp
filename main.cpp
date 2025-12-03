#include <iostream>
#include "include/cifar-10.h"

const std::string bathPath = "/Users/aeilot/Developer/CVNet/";

int main() {
	try {
		std::vector<std::string> train_files = {
			"cifar-10-batches-bin/data_batch_1.bin",
			"cifar-10-batches-bin/data_batch_2.bin",
			"cifar-10-batches-bin/data_batch_3.bin",
			"cifar-10-batches-bin/data_batch_4.bin",
			"cifar-10-batches-bin/data_batch_5.bin"
		};
		std::string test_file = "cifar-10-batches-bin/test_batch.bin";

		for (auto &file : train_files) {
			file = bathPath + file;
		}
		test_file = bathPath + test_file;

		CIFAR_10::DataSet dataset(train_files, test_file);

	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
	}
	return 0;
}
