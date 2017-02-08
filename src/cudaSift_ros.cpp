/*
CUDA SIFT extractor by Marten Björkman aka Celebrandil
            celle @ csc.kth.se
MIT License

Copyright (c) 2017 Mårten Björkman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Adapted for ROS by Sai Vemprala.
*/

#include <iostream>
#include <cmath>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include "cudaImage.h"
#include "cudaSift.h"

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img);
void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);

double ScaleUp(CudaImage &res, CudaImage &src);

void cudaExtractFeatures(cv::Mat img)
{
    unsigned int w = img.cols;
    unsigned int h = img.rows;

    std::cout << "Initializing data..." << std::endl;

    CudaImage img1;
    img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)img.data);
    img1.Download();

  // Extract Sift features from images
    SiftData siftData1;
    float initBlur = 1.0f;
    float thresh = 3.5f;
    InitSiftData(siftData1, 32768, true, true);
    ExtractSift(siftData1, img1, 5, initBlur, thresh, 0.0f, false);

    ROS_INFO("Number of original features: %d", siftData1.numPts);
    FreeSiftData(siftData1);
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }

    if(!cv_ptr->image.empty())
        cudaExtractFeatures(cv_ptr->image);
}

int main(int argc, char **argv)
{
    int devNum = 0;
    if (argc>1)
        devNum = std::atoi(argv[1]);
    InitCuda(devNum);

    ros::init(argc, argv, "cudasift_ros");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("/camera/image_raw", 1, imageCallback);
    ros::spin();
}

