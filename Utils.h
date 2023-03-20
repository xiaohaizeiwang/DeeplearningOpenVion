#pragma once
#include "Halcon.h"
#include "HalconCpp.h"
#include <opencv2/opencv.hpp>
#include "half.hpp"

struct Detection
{
	cv::Rect bbox;
	float score;
	int class_idx;
};

enum Det
{
	tl_x = 0,
	tl_y = 1,
	br_x = 2,
	br_y = 3,
	score = 4,
	class_idx = 5
};

int hobj2Mat(HalconCpp::HObject& hImage, cv::Mat& cv_Image);
int Mat2Hobj(cv::Mat& img, HalconCpp::HObject& hv_Img);
std::vector<std::string> split(const  std::string& s, const std::string& delim);
float f16_to_f32(int16_t* pn16);