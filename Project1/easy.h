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
// ͼ��ˮƽ�ݶȵ�������㣬��img��i��j+1��-img��i��j��
// img��Ҫ�������ݶȵ�ͼ��
Mat getGradientXp(Mat &img);

// ͼ��ֱ�ݶȵ�������㣬��img��i+1��j��-img��i��j��
// img��Ҫ�������ݶȵ�ͼ��
Mat getGradientYp(Mat &img);

// ͼ��ˮƽ�ݶȵĸ�����㣬��img��i��j-1��-img��i��j��
// img��Ҫ�������ݶȵ�ͼ��
Mat getGradientXn(Mat &img);

// ͼ��ֱ�ݶȵĸ�����㣬��img��i-1��j��-img��i��j��
// img��Ҫ�������ݶȵ�ͼ��
Mat getGradientYn(Mat &img);

int getLabel(int i, int j, int height, int width);

// �õ�����A
Mat getA(int height, int width);

// ����b
// ʹ��getGradient������
Mat getB2(Mat &img1, Mat &img2, int posX, int posY, Rect ROI);



// ��ⷽ�̣�����������Ϊ��ȷ�ĸ߶ȺͿ�ȡ�
// Solve equation and reshape it back to the right height and width.
Mat getResult(Mat &A, Mat &B, Rect &ROI);


/// ���ɻ��
// img1: 3-channel image, we wanna move something in it into img2.
// img2: 3-channel image, dst image.
// ROI: the position and size of the block we want to move in img1.
// posX, posY: where we want to move the block to in img2
Mat poisson_blending(Mat &img1, Mat &img2, Rect ROI, int posX, int posY);

