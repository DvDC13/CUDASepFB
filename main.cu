#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "cpu/choquet.h"
#include "gpu/choquet.cuh"

int main() {

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "No camera found. Exiting..." << std::endl;
        return 1;
    }

    std::vector<std::shared_ptr<Image<Pixel>>> capturedImages;
    int numFramesToCapture = 100;

    for (int i = 0; i < numFramesToCapture; i++)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        std::shared_ptr<Image<Pixel>> image = std::make_shared<Image<Pixel>>(frame.cols, frame.rows);
        for (int y = 0; y < frame.rows; y++)
        {
            for (int x = 0; x < frame.cols; x++)
            {
                cv::Vec3b pixel = frame.at<cv::Vec3b>(y, x);
                image->set(x, y, {static_cast<float>(pixel[2]), static_cast<float>(pixel[1]), static_cast<float>(pixel[0])});
            }
        }

        capturedImages.push_back(image);
        cv::imshow("frame", frame);

        if (cv::waitKey(1) == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    // make a video of the results
    cv::VideoWriter writer;
    writer.open("camera_test/camera_test.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, cv::Size(640, 480), true);

    for (int i = 1; i < capturedImages.size(); i++)
    {
        cv::Mat frame;
        std::shared_ptr<Image<bool>> result_first_pass = computeChoquet_gpu(capturedImages[0], capturedImages[i]);
        result_first_pass->save("camera_test/gpu/" + std::to_string(i) + "_first_pass.ppm");
        cv::imread("camera_test/gpu/" + std::to_string(i) + "_first_pass.ppm").copyTo(frame);
        writer << frame;
    }

    writer.release();

    return 0;
}
