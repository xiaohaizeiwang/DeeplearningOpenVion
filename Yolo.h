#pragma once
#include "BaseNet.h"

class Yolo : public BaseNet
{
public:

	cv::Mat m_pStrides;
	virtual s_NetStatus InitNet(char* cCfgPath, char* cWeightsPath, int* nOriImgWidth, int* nOriImgHeight, int* nChannels, HTuple* hv_htClassNames);
	virtual int DoInfer(cv::Mat& img, std::vector<cv::Mat>& results, float score)=0;

};