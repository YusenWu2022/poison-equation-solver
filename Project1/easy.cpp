#include <easy.h>
#include <iostream>
#include <math.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h> 
#include "constant.h"

//�ο���ģ��opencv�д������һ���򵥰汾��Ч�ʽϵͣ���Ҫ������ROI����������ݶȼ���ͽⲴ�ɷ������Ӵ�����Ծ������
#define elif else if
#define ATD at<double>

//��������ˮƽ�ݶȣ�ֱ��ʹ��Img(i,j+1)-img(i,j)
//�������˼·�����ȵõ�ѭ��ƽ�ƺ�ľ�����ֱ������õ��ݶȣ�����ʹ�þ��
cv::Mat getGradientX(cv::Mat &img)
{
	int height = img.rows;
	int width = img.cols;
	//������������ƺ��ȡһ���ֽ������
	cv::Mat store = repeat(img, 1, 2);
	cv::Rect roi = cv::Rect(1, 0, width, height);
	cv::Mat roimat = store(roi);
	return roimat - img;
}

// ����ֱ�ݶȣ�ʹ��Img(i+1,j)-img(i,j)
cv::Mat getGradientY(cv::Mat &img)
{
	int height = img.rows;
	int width = img.cols;
	cv::Mat store = repeat(img, 2, 1);
	cv::Rect roi = cv::Rect(0, 1, width, height);
	cv::Mat roimat = store(roi);
	return roimat - img;
}

// ����ˮƽ�ݶȣ�img��i��j-1��-img��i��j��
cv::Mat getGradientXt(cv::Mat &img) {
	int height = img.rows;
	int width = img.cols;
	cv::Mat store = repeat(img, 1, 2);

	cv::Rect roi = cv::Rect(width - 1, 0, width, height);
	cv::Mat roimat = store(roi);
	return roimat - img;
}

// ����ֱ�ݶȣ�img��i-1��j��-img��i��j��
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

// ����ϵ������A
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

	//��������ROI�����ݶȹ�ϵ�����ݻ���label����ķֲ�������
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			//M��i,j)�����label
			int label = calLable(i, j, height, width);
			//����õ����������ⲿ�߽磨���α߽��ϵĽ��䣩����һ���жϾ��巽λ
			if (M.ATD(i, j) == 0)
			{
				if (i == 0)  A.ATD(calLable(i + 1, j, height, width), label) = 1;
				elif(i == height - 1)   A.ATD(calLable(i - 1, j, height, width), label) = 1;
				if (j == 0)  A.ATD(calLable(i, j + 1, height, width), label) = 1;
				elif(j == width - 1)   A.ATD(calLable(i, j - 1, height, width), label) = 1;
			}
			//����õ���ROI���������򻭳��ı߽��ϣ������ص�label����¼������ݶȵļ����漰����Χ��Щλ�õ����ݣ�ע��ֱ�����ֻ��һ��������Ҫ�ķ�Χ��
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
			//����������ROI�ڲ�����ֱ�Ӱ���Χ�ĸ����ڵ㶼�����ݶȼ�����
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

//�������B������ĸ����ݶȳ���
cv::Mat calB(cv::Mat &img1, cv::Mat &img2, int posX, int posY, cv::Rect ROI)
{
	//���㸴������ˮƽ�ݶȣ�����Ŀ������ROI�����ݶ���srcͼ���ݶ�ȡ����
	//���������Ҫ�������ͼ�����е��ݶȼ��㣬�����ʱ����˷�
	cv::Mat merged_gradX = getGradientX(img1);
	cv::Mat gradXt = getGradientX(img2);
	cv::Mat merged_gradXtROI = merged_gradX(ROI);
	gradXt.copyTo(merged_gradXtROI);
	//���Ϸ���ˮƽ�ݶ�
	cv::Mat merged_gradXt = getGradientXt(img1);
	cv::Mat gradX = getGradientXt(img2);
	cv::Mat merged_gradXROI = merged_gradXt(ROI);
	gradX.copyTo(merged_gradXROI);
	//���������ݶ�
	cv::Mat MergeGradYt = getGradientY(img1);
	cv::Mat GradYt = getGradientY(img2);
	cv::Mat MergeGradYpROI = MergeGradYt(ROI);
	GradYt.copyTo(MergeGradYpROI);
	//���Ϸ��������ݶ�
	cv::Mat MergeGradY = getGradientYt(img1);
	cv::Mat GradY = getGradientYt(img2);
	cv::Mat MergeGradYnROI = MergeGradY(ROI);
	GradY.copyTo(MergeGradYnROI);
	//�õ����ݶȣ�ֱ�Ӱ��ĸ��ݶ���ӣ���Ϊ���������������ݶȸ�����
	cv::Mat grad = merged_gradX + merged_gradXt + MergeGradYt + MergeGradY;
	//����ROI����ľ�������
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

// ����cv::solve��������������õ���Ax=B���̲�reshape��״
cv::Mat getResult(cv::Mat &A, cv::Mat &B, cv::Rect &ROI) {
	cv::Mat result;
	solve(A, B, result);
	result = result.reshape(0, ROI.height);
	return  result;
}


// �����ں�
//����img1��Ϊsrcͼ�����ݣ���ROI���δ��¿����Ҫ�ںϵķ�Χ��img2��Ŀ��ͼ���ں�ָ��������posX��posY��ƫ��λ����
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
	//�����ͨ��
	vector<cv::Mat> rgb1;
	split(imgData1, rgb1);
	vector<cv::Mat> rgb2;
	split(imgData2, rgb2);
	vector<cv::Mat> result;
	cv::Mat merged, res, r, g, b;
	//��ͨ���ֱ����
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
	//������ͨ�����кϲ����Ѷ�ӦROI��������ݸ��Ƶ����沢���
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

