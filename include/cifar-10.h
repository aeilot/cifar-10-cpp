//
// Created by Chenluo Deng on 12/3/25.
//

#ifndef CVNET_CIFAR_10_READER_H
#define CVNET_CIFAR_10_READER_H

#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <opencv2/opencv.hpp>

namespace CIFAR_10 {
    const int IMG_WIDTH = 32;
    const int IMG_HEIGHT = 32;
    const int IMG_CHANNELS = 3;
    const int CHANNEL_SIZE = IMG_WIDTH * IMG_HEIGHT;
    const int IMG_SIZE = CHANNEL_SIZE * IMG_CHANNELS;

    const int NUM_CLASSES = 10;
    const int SAMPLES_PER_BATCH = 10000;
    const int ENTRY_SIZE = IMG_SIZE + 1;

    struct CIFAR_10_BATCH {
        std::vector<cv::Mat> images;
        std::vector<int> labels;
    };

	static cv::Mat convert_to_one_hot(const std::vector<int>& labels, int num_classes=NUM_CLASSES) {
		int num_samples = labels.size();
		cv::Mat one_hot = cv::Mat::zeros(num_samples, num_classes, CV_32F);
		for (int i = 0; i < num_samples; ++i) {
			int label_id = labels[i];
			if (label_id >= 0 && label_id < num_classes) {
				one_hot.at<float>(i, label_id) = 1.0f;
			}
		}

		return one_hot;
	}


    class DataSet {
    public:
        std::vector<CIFAR_10_BATCH> trainBatches;
        CIFAR_10_BATCH testBatch;

        std::vector<std::string> labelNames = {
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        };

        DataSet(const std::vector<std::string>& trainingPaths, const std::string& testPath) {

            trainBatches.reserve(trainingPaths.size());
            for (const auto& path : trainingPaths) {
                trainBatches.push_back(read_file_to_batch(path));
            }

            testBatch = read_file_to_batch(testPath);
        }

    private:
        CIFAR_10_BATCH read_file_to_batch(const std::string& path) {
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Can't open file: " + path);
            }

            CIFAR_10_BATCH batch;
            batch.images.reserve(SAMPLES_PER_BATCH);
            batch.labels.reserve(SAMPLES_PER_BATCH);

            std::vector<uint8_t> buffer(ENTRY_SIZE);

            for (int i = 0; i < SAMPLES_PER_BATCH; ++i) {
                if (!file.read(reinterpret_cast<char*>(buffer.data()), ENTRY_SIZE)) {
                    break;
                }

                batch.labels.push_back(buffer[0]);

                uint8_t *r_ptr = buffer.data() + 1;
                uint8_t *g_ptr = buffer.data() + 1 + CHANNEL_SIZE;
                uint8_t *b_ptr = buffer.data() + 1 + 2 * CHANNEL_SIZE;

                cv::Mat r_channel(IMG_HEIGHT, IMG_WIDTH, CV_8UC1, r_ptr);
                cv::Mat g_channel(IMG_HEIGHT, IMG_WIDTH, CV_8UC1, g_ptr);
                cv::Mat b_channel(IMG_HEIGHT, IMG_WIDTH, CV_8UC1, b_ptr);

                std::vector<cv::Mat> channels = { b_channel, g_channel, r_channel };

                cv::Mat sample;
                cv::merge(channels, sample);

                batch.images.push_back(sample);
            }
            return batch;
        }

		
    };

	class CIFAR10_ANN {
	private:
		DataSet dataset;
		cv::Ptr<cv::ml::ANN_MLP> ann;
	public:
		explicit CIFAR10_ANN(DataSet& dataset) : dataset(dataset) {
			this->ann = cv::ml::ANN_MLP::create();
		}

		CIFAR10_ANN() = delete;
		~CIFAR10_ANN() = default;

		void train(int epochs = 10, double eps=1e-6) {
			int total_train_samples = 0;
			for (const auto& batch : dataset.trainBatches) {
				total_train_samples += batch.images.size();
			}

			std::cout << "Allocating memory for " << total_train_samples << " samples..." << std::endl;

			cv::Mat train_data(total_train_samples, IMG_SIZE, CV_32F);
			cv::Mat train_labels(total_train_samples, NUM_CLASSES, CV_32F);

			int current_idx = 0;
			for (const auto& batch : dataset.trainBatches) {
				cv::Mat batch_one_hot = convert_to_one_hot(batch.labels);

				for (size_t i = 0; i < batch.images.size(); ++i) {
					cv::Mat img_float;
					batch.images[i].convertTo(img_float, CV_32F, 1.0 / 255.0);

					cv::Mat flattened = img_float.reshape(1, 1);

					flattened.copyTo(train_data.row(current_idx));

					batch_one_hot.row(i).copyTo(train_labels.row(current_idx));

					current_idx++;
				}
			}

			cv::Mat layer_sizes = (cv::Mat_<int>(1, 4) << IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS, 1024, 256, NUM_CLASSES);

			ann->setLayerSizes(layer_sizes);
			ann->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1.0, 1.0);

			ann->setTrainMethod(cv::ml::ANN_MLP::RPROP, 0.1);
			ann->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, epochs, eps));

			cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(train_data, cv::ml::ROW_SAMPLE, train_labels);

			std::cout << "Starting training" << std::endl;
			ann->train(tdata);
			std::cout << "Training completed" << std::endl;
		}

		void predict() {
			int num_test = dataset.testBatch.images.size();
			int feature_dim = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS;

			cv::Mat test_data(num_test, feature_dim, CV_32F);

			for (int i = 0; i < num_test; ++i) {
				cv::Mat img_float;
				dataset.testBatch.images[i].convertTo(img_float, CV_32F, 1.0 / 255.0);
				cv::Mat flattened = img_float.reshape(1, 1);
				flattened.copyTo(test_data.row(i));
			}

			cv::Mat results;
			std::cout << "Predicting..." << std::endl;
			ann->predict(test_data, results);

			int correct = 0;
			for (int i = 0; i < num_test; ++i) {
				cv::Point maxLoc;
				cv::minMaxLoc(results.row(i), nullptr, nullptr, nullptr, &maxLoc);

				int predicted_label = maxLoc.x;
				if (predicted_label == dataset.testBatch.labels[i]) {
					correct++;
				}
			}

			double accuracy = static_cast<double>(correct) / num_test * 100.0;
			std::cout << "Accuracy: " << accuracy << "%" << std::endl;
		}

		void save(std::string basePath) {
			ann->save(basePath + "cifar10_ann_model.xml");
		}
	};
}

#endif //CVNET_CIFAR_10_READER_H