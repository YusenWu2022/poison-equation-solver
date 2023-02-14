#include <easy.h>
#include <iostream>
#include <math.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h> 
#include "constant.h"

//参考并模仿opencv中代码风格的一个简单版本，效率较低，主要问题在ROI以外区域的梯度计算和解泊松方程中庞大的线性矩阵计算
#define elif else if
#define ATD at<double>

//计算正向水平梯度，直接使用Img(i,j+1)-img(i,j)
//具体计算思路就是先得到循环平移后的矩阵，再直接相减得到梯度，而不使用卷积
cv::Mat getGradientX(cv::Mat &img)
{
	int height = img.rows;
	int width = img.cols;
	//经典操作：复制后截取一部分交错相减
	cv::Mat store = repeat(img, 1, 2);
	cv::Rect roi = cv::Rect(1, 0, width, height);
	cv::Mat roimat = store(roi);
	return roimat - img;
}

// 正向垂直梯度，使用Img(i+1,j)-img(i,j)
cv::Mat getGradientY(cv::Mat &img)
{
	int height = img.rows;
	int width = img.cols;
	cv::Mat store = repeat(img, 2, 1);
	cv::Rect roi = cv::Rect(0, 1, width, height);
	cv::Mat roimat = store(roi);
	return roimat - img;
}

// 负向水平梯度，img（i，j-1）-img（i，j）
cv::Mat getGradientXt(cv::Mat &img) {
	int height = img.rows;
	int width = img.cols;
	cv::Mat store = repeat(img, 1, 2);

	cv::Rect roi = cv::Rect(width - 1, 0, width, height);
	cv::Mat roimat = store(roi);
	return roimat - img;
}

// 负向垂直梯度，img（i-1，j）-img（i，j）
cv::Mat getGradientYt(cv::Mat &img)
{
	int height = img.rows;
	int width = img.cols;
	cv::Mat store = repeat(img, 2, 1);

	cv::Rect roi = cv::Rect(0, height - 1, width, height);
	cv::Mat roimat = store(roi);
	return roimat - img;
}

int calLable(int i, int j, int height, int width)
{
	return i * width + j;			// i is in height axis; j is in width axis.
}

// 计算系数矩阵A
cv::Mat calA(int height, int width)
{
	cv::Mat A = cv::Mat::eye(height * width, height * width, CV_64FC1);
	A *= -4;

	// M: the label matrix of roi from im2; divede M into three parts, 0,1,2, respectively.
	// 0: the corners of roi
	// 1: boundaries but not corners
	// 2: inner part of roi
	// different parts represent different methods to asssign A values
	cv::Mat M = cv::Mat::zeros(height, width, CV_64FC1);
	cv::Mat tmp = cv::Mat::ones(height, width - 2, CV_64FC1);
	cv::Rect roi = cv::Rect(1, 0, width - 2, height);
	cv::Mat roimat = M(roi);
	tmp.copyTo(roimat);
	tmp = cv::Mat::ones(height - 2, width, CV_64FC1);
	roi = cv::Rect(0, 1, width, height - 2);
	roimat = M(roi);
	tmp.copyTo(roimat);
	tmp = cv::Mat::ones(height - 2, width - 2, CV_64FC1);
	tmp *= 2;
	roi = cv::Rect(1, 1, width - 2, height - 2);
	roimat = M(roi);
	tmp.copyTo(roimat);

	//遍历计算ROI区域梯度关系，根据划分label区域的分布和性质
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			//M（i,j)区域的label
			int label = calLable(i, j, height, width);
			//如果该点标号类型是外部边界（矩形边界上的角落），进一步判断具体方位
			if (M.ATD(i, j) == 0)
			{
				if (i == 0)  A.ATD(calLable(i + 1, j, height, width), label) = 1;
				elif(i == height - 1)   A.ATD(calLable(i - 1, j, height, width), label) = 1;
				if (j == 0)  A.ATD(calLable(i, j + 1, height, width), label) = 1;
				elif(j == width - 1)   A.ATD(calLable(i, j - 1, height, width), label) = 1;
			}
			//如果该点在ROI不规则区域画出的边界上，标记相关的label，记录这个点梯度的计算涉及到周围哪些位置的数据，注意分别有且只有一个不在需要的范围内
			elif(M.ATD(i, j) == 1)
			{
				if (i == 0) {
					A.ATD(calLable(i + 1, j, height, width), label) = 1;
					A.ATD(calLable(i, j - 1, height, width), label) = 1;
					A.ATD(calLable(i, j + 1, height, width), label) = 1;
				}elif(i == height - 1) {
					A.ATD(calLable(i - 1, j, height, width), label) = 1;
					A.ATD(calLable(i, j - 1, height, width), label) = 1;
					A.ATD(calLable(i, j + 1, height, width), label) = 1;
				}
				if (j == 0) {
					A.ATD(calLable(i, j + 1, height, width), label) = 1;
					A.ATD(calLable(i - 1, j, height, width), label) = 1;
					A.ATD(calLable(i + 1, j, height, width), label) = 1;
				}elif(j == width - 1) {
					A.ATD(calLable(i, j - 1, height, width), label) = 1;
					A.ATD(calLable(i - 1, j, height, width), label) = 1;
					A.ATD(calLable(i + 1, j, height, width), label) = 1;
				}
			}
			//如果这个点在ROI内部，则直接把周围四个相邻点都纳入梯度计算中
			else {
				A.ATD(calLable(i, j - 1, height, width), label) = 1;
				A.ATD(calLable(i, j + 1, height, width), label) = 1;
				A.ATD(calLable(i - 1, j, height, width), label) = 1;
				A.ATD(calLable(i + 1, j, height, width), label) = 1;
			}
		}
	}
	return A;
}

//计算矩阵B（具体的复合梯度场）
cv::Mat calB(cv::Mat &img1, cv::Mat &img2, int posX, int posY, cv::Rect ROI)
{
	//计算复合正向水平梯度，其中目标区域（ROI）的梯度用src图像梯度取代，
	//这个计算需要完成两个图像所有的梯度计算，造成了时间的浪费
	cv::Mat merged_gradX = getGradientX(img1);
	cv::Mat gradXt = getGradientX(img2);
	cv::Mat merged_gradXtROI = merged_gradX(ROI);
	gradXt.copyTo(merged_gradXtROI);
	//复合反向水平梯度
	cv::Mat merged_gradXt = getGradientXt(img1);
	cv::Mat gradX = getGradientXt(img2);
	cv::Mat merged_gradXROI = merged_gradXt(ROI);
	gradX.copyTo(merged_gradXROI);
	//复合纵向梯度
	cv::Mat MergeGradYt = getGradientY(img1);
	cv::Mat GradYt = getGradientY(img2);
	cv::Mat MergeGradYpROI = MergeGradYt(ROI);
	GradYt.copyTo(MergeGradYpROI);
	//复合反向纵向梯度
	cv::Mat MergeGradY = getGradientYt(img1);
	cv::Mat GradY = getGradientYt(img2);
	cv::Mat MergeGradYnROI = MergeGradY(ROI);
	GradY.copyTo(MergeGradYnROI);
	//得到总梯度，直接把四个梯度相加，因为定义是两个方向梯度各两个
	cv::Mat grad = merged_gradX + merged_gradXt + MergeGradYt + MergeGradY;
	//处理ROI区域的具体数据
	int roi_height = ROI.height;
	int roi_width = ROI.width;
	cv::Mat B = cv::Mat::zeros(roi_height * roi_width, 1, CV_64FC1);
	for (int i = 0; i < roi_height; i++) {
		for (int j = 0; j < roi_width; j++) {
			double tmp = 0.0;
			tmp += grad.ATD(i + ROI.y, j + ROI.x);
			if (i == 0)              tmp -= img2.ATD(i - 1 + posY, j + posX);
			if (i == roi_height - 1)  tmp -= img2.ATD(i + 1 + posY, j + posX);
			if (j == 0)              tmp -= img2.ATD(i + posY, j - 1 + posX);
			if (j == roi_width - 1)   tmp -= img2.ATD(i + posY, j + 1 + posX);
			B.ATD(calLable(i, j, roi_height, roi_width), 0) = tmp;
		}
	}
	return B;
}

// 调用cv::solve方法求解上面所得到的Ax=B方程并reshape形状
cv::Mat getResult(cv::Mat &A, cv::Mat &B, cv::Rect &ROI) {
	cv::Mat result;
	solve(A, B, result);
	result = result.reshape(0, ROI.height);
	return  result;
}


// 泊松融合
//传入img1作为src图像数据，用ROI矩形大致框出想要融合的范围，img2是目标图像，融合指定发生在posX和posY的偏置位置上
cv::Mat poisson_blending(cv::Mat &img1, cv::Mat &img2, cv::Rect ROI, int posX, int posY)
{
	cv::Mat copy1, copy2;
	copy1 = img1;
	copy2 = img2;
	int roi_height = ROI.height;
	int roi_width = ROI.width;
	cv::Mat A = calA(roi_height, roi_width);
	cv::Mat imgData1, imgData2;
	img1.convertTo(imgData1, CV_64FC3);
	img2.convertTo(imgData2, CV_64FC3);
	//拆分三通道
	vector<cv::Mat> rgb1;
	split(imgData1, rgb1);
	vector<cv::Mat> rgb2;
	split(imgData2, rgb2);
	vector<cv::Mat> result;
	cv::Mat merged, res, r, g, b;
	//三通道分别完成
	r = calB(rgb1[0], rgb2[0], posX, posY, ROI);
	res = getResult(A, r, ROI);
	result.push_back(res);
	cout << "R channel finished..." << endl;
	g = calB(rgb1[1], rgb2[1], posX, posY, ROI);
	res = getResult(A, g, ROI);
	result.push_back(res);
	cout << "G channel finished..." << endl;
	b = calB(rgb1[2], rgb2[2], posX, posY, ROI);
	res = getResult(A, b, ROI);
	result.push_back(res);
	cout << "B channel finished..." << endl;
	//最后对三通道进行合并，把对应ROI区域的内容复制到上面并输出
	merge(result, merged);
	merged.convertTo(merged, CV_8UC1);
	cv::Rect r2 = cv::Rect(100, 100, 40, 40);
	cv::Mat replace = img2(r2);
	merged.copyTo(replace);
	cv::Mat final_result = img2;
	img2 = copy2;
	img1 = copy1;
	return final_result;
}

