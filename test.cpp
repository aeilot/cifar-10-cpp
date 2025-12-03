//
// Created by Chenluo Deng on 12/3/25.
//

#include <gtest/gtest.h>
#include "include/cifar-10.h"

const std::string bathPath = "/Users/aeilot/Developer/CVNet/";

TEST(DATA_LOADER_TEST, LoadCIFAR10Dataset) {
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

		// Check the number of training batches
		EXPECT_EQ(dataset.trainBatches.size(), 5);

		// Check the number of images and labels in the first training batch
		EXPECT_EQ(dataset.trainBatches[0].images.size(), CIFAR_10::SAMPLES_PER_BATCH);
		EXPECT_EQ(dataset.trainBatches[0].labels.size(), CIFAR_10::SAMPLES_PER_BATCH);

		// Check the number of images and labels in the test batch
		EXPECT_EQ(dataset.testBatch.images.size(), CIFAR_10::SAMPLES_PER_BATCH);
		EXPECT_EQ(dataset.testBatch.labels.size(), CIFAR_10::SAMPLES_PER_BATCH);

	} catch (const std::exception& e) {
		FAIL() << "Exception thrown during dataset loading: " << e.what();
	}
}