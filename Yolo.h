#pragma once
#include "BaseNet.h"
#include"Utils.h"

class Yolo :public BaseNet
{
public:

	virtual s_NetStatus InitNet(char* cCfgPath, char* cWeightsPath, int* nOriImgWidth, int* nOriImgHeight, HTuple* hv_htClassNames);
	virtual s_NetStatus InitNet(char* cCfgPath, char* cWeightsPath, int* nOriImgWidth, int* nOriImgHeight, HTuple* hv_htClassNames, int nMode)=0;
	virtual int DoInfer(cv::Mat& img, std::vector<cv::Mat>& results, float score)=0;
};