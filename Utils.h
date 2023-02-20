#pragma once
#include "Halcon.h"
#include "HalconCpp.h"
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include "half.hpp"


int hobj2Mat(HalconCpp::HObject& hImage, cv::Mat& cv_Image);
int Mat2Hobj(cv::Mat& img, HalconCpp::HObject& hv_Img);
std::vector<std::string> split(const  std::string& s, const std::string& delim);
float f16_to_f32(int16_t* pn16);