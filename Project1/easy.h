#pragma once
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h> 
#include <Eigen/Sparse>
#include <Eigen/Dense>


#define elif else if
#define ATD at<double>

using namespace cv;
using namespace std;
// 图像水平梯度的正向计算，即img（i，j+1）-img（i，j）
// img：要计算其梯度的图像
Mat getGradientXp(Mat &img);

// 图像垂直梯度的正向计算，即img（i+1，j）-img（i，j）
// img：要计算其梯度的图像
Mat getGradientYp(Mat &img);

// 图像水平梯度的负向计算，即img（i，j-1）-img（i，j）
// img：要计算其梯度的图像
Mat getGradientXn(Mat &img);

// 图像垂直梯度的负向计算，即img（i-1，j）-img（i，j）
// img：要计算其梯度的图像
Mat getGradientYn(Mat &img);

int getLabel(int i, int j, int height, int width);

// 得到矩阵A
Mat getA(int height, int width);

// 计算b
// 使用getGradient函数。
Mat getB2(Mat &img1, Mat &img2, int posX, int posY, Rect ROI);



// 求解方程，并将其重塑为正确的高度和宽度。
// Solve equation and reshape it back to the right height and width.
Mat getResult(Mat &A, Mat &B, Rect &ROI);


/// 泊松混合
// img1: 3-channel image, we wanna move something in it into img2.
// img2: 3-channel image, dst image.
// ROI: the position and size of the block we want to move in img1.
// posX, posY: where we want to move the block to in img2
Mat poisson_blending(Mat &img1, Mat &img2, Rect ROI, int posX, int posY);

