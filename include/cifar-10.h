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
    const int IMG_SIZE_BYTES = CHANNEL_SIZE * IMG_CHANNELS;

    const int NUM_CLASSES = 10;
    const int SAMPLES_PER_BATCH = 10000;
    const int ENTRY_SIZE = IMG_SIZE_BYTES + 1;

    struct CIFAR_10_BATCH {
        std::vector<cv::Mat> images;
        std::vector<int> labels;
    };

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
}

#endif //CVNET_CIFAR_10_READER_H