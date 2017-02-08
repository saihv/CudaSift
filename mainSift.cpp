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
*/

#include <iostream>
#include <cmath>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cudaImage.h"
#include "cudaSift.h"

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img);
void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);

double ScaleUp(CudaImage &res, CudaImage &src);

int main(int argc, char **argv)
{
    int devNum = 0;
    if (argc>1)
    devNum = std::atoi(argv[1]);

    // Read images using OpenCV
    cv::Mat limg, rimg;
    cv::imread("../data/img1.png", 0).convertTo(limg, CV_32FC1);
    if (limg.empty())
        std::cout << "!!! Failed imread(): image not found" << std::endl;
    cv::imread("../data/img2.png", 0).convertTo(rimg, CV_32FC1);
    if (rimg.empty())
        std::cout << "!!! Failed imread(): image not found" << std::endl;

    unsigned int w = limg.cols;
    unsigned int h = limg.rows;
    std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

    std::cout << "Initializing data..." << std::endl;
    InitCuda(devNum);
    CudaImage img1, img2;
    img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)limg.data);
    img2.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)rimg.data);
    img1.Download();
    img2.Download();

  // Extract Sift features from images
    SiftData siftData1, siftData2;
    float initBlur = 1.0f;
    float thresh = 3.5f;
    InitSiftData(siftData1, 32768, true, true);
    InitSiftData(siftData2, 32768, true, true);

    // A bit of benchmarking
    ExtractSift(siftData1, img1, 5, initBlur, thresh, 0.0f, false);
    ExtractSift(siftData2, img2, 5, initBlur, thresh, 0.0f, false);

    MatchSiftData(siftData1, siftData2);
    float homography[9];
    int numMatches;
    FindHomography(siftData1, homography, &numMatches, 10000, 0.00f, 0.80f, 5.0);
    int numFit = ImproveHomography(siftData1, homography, 5, 0.00f, 0.80f, 3.0);

    std::cout << "Number of original features: " <<  siftData1.numPts << " " << siftData2.numPts << std::endl;
    std::cout << "Number of matching features: " << numFit << " " << numMatches << " " << 100.0f*numFit/std::min(siftData1.numPts, siftData2.numPts) << "% " << initBlur << " " << thresh << std::endl;

    // Print out and store summary data
    PrintMatchData(siftData1, siftData2, img1);
    cv::imwrite("../data/limg_pts.pgm", limg);

    // Free Sift data from device
    FreeSiftData(siftData1);
    FreeSiftData(siftData2);
}

void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography)
{
#ifdef MANAGEDMEM
    SiftPoint *sift1 = siftData1.m_data;
    SiftPoint *sift2 = siftData2.m_data;
#else
    SiftPoint *sift1 = siftData1.h_data;
    SiftPoint *sift2 = siftData2.h_data;
#endif
    int numPts1 = siftData1.numPts;
    int numPts2 = siftData2.numPts;
    int numFound = 0;
    for (int i=0;i<numPts1;i++)
    {
        float *data1 = sift1[i].data;
        std::cout << i << ":" << sift1[i].scale << ":" << (int)sift1[i].orientation << std::endl;
        bool found = false;
        for (int j=0;j<numPts2;j++)
        {
            float *data2 = sift2[j].data;
            float sum = 0.0f;
            for (int k=0;k<128;k++)
	           sum += data1[k]*data2[k];
            float den = homography[6]*sift1[i].xpos + homography[7]*sift1[i].ypos + homography[8];
            float dx = (homography[0]*sift1[i].xpos + homography[1]*sift1[i].ypos + homography[2]) / den - sift2[j].xpos;
            float dy = (homography[3]*sift1[i].xpos + homography[4]*sift1[i].ypos + homography[5]) / den - sift2[j].ypos;
            float err = dx*dx + dy*dy;
            if (err<100.0f)
	           found = true;
            if (err<100.0f || j==sift1[i].match)
            {
	            if (j==sift1[i].match && err<100.0f)
	               std::cout << " *";
	            else if (j==sift1[i].match)
	               std::cout << " -";
	            else if (err<100.0f)
	               std::cout << " +";
	            else
	               std::cout << "  ";
	            std::cout << j << ":" << sum << ":" << (int)sqrt(err) << ":" << sift2[j].scale << ":" << (int)sift2[j].orientation << std::endl;
            }
        }
        std::cout << std::endl;
        if (found)
            numFound++;
    }
    std::cout << "Number of founds: " << numFound << std::endl;
}

void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img)
{
    int numPts = siftData1.numPts;
#ifdef MANAGEDMEM
    SiftPoint *sift1 = siftData1.m_data;
    SiftPoint *sift2 = siftData2.m_data;
#else
    SiftPoint *sift1 = siftData1.h_data;
    SiftPoint *sift2 = siftData2.h_data;
#endif
    float *h_img = img.h_data;
    int w = img.width;
    int h = img.height;
    std::cout << std::setprecision(3);
    for (int j=0;j<numPts;j++)
    {
        int k = sift1[j].match;
        if (sift1[j].match_error<5)
        {
            float dx = sift2[k].xpos - sift1[j].xpos;
            float dy = sift2[k].ypos - sift1[j].ypos;
            int len = (int)(fabs(dx)>fabs(dy) ? fabs(dx) : fabs(dy));
            for (int l = 0; l < len; l++)
            {
	            int x = (int)(sift1[j].xpos + dx*l/len);
	            int y = (int)(sift1[j].ypos + dy*l/len);
	            h_img[y*w+x] = 255.0f;
            }
        }
        int x = (int)(sift1[j].xpos+0.5);
        int y = (int)(sift1[j].ypos+0.5);
        int s = std::min(x, std::min(y, std::min(w-x-2, std::min(h-y-2, (int)(1.41*sift1[j].scale)))));
        int p = y*w + x;
        p += (w + 1);
        for (int k = 0; k < s; k++)
            h_img[p-k] = h_img[p+k] = h_img[p-k*w] = h_img[p+k*w] = 0.0f;
        p -= (w + 1);
        for (int k = 0; k < s; k++)
            h_img[p-k] = h_img[p+k] = h_img[p-k*w] =h_img[p+k*w] = 255.0f;
    }
    std::cout << std::setprecision(6);
}
