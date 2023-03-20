#pragma once
#include "Utils.h"
#include "Halcon.h"
#include "HalconCpp.h"
#include "DeepLearning.h"

using namespace HalconCpp;

class BaseNet
{
public:
	int m_ImgWidth;
	int m_ImgHeight;
	int m_ImgChannels;
	int m_ClassNum;
	int m_iNetType;
	cv::Mat m_pStrides;
	std::vector<std::string> m_VecClassName;
	virtual s_NetStatus InitNet(char* cCfgPath, char* cWeightsPath, int* nOriImgWidth, int* nOriImgHeight, HTuple* hv_htClassNames) = 0;
	virtual s_NetStatus InitNet(char* cCfgPath, char* cWeightsPath, int *nOriImgWidth, int *nOriImgHeight, HTuple* hv_htClassNames, int nMode) = 0;
	virtual int DoInfer(cv::Mat& img, std::vector<cv::Mat>& results, float score) = 0;
};