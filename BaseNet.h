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
	InferenceEngine::Precision m_Prc;
	std::string m_inputName;
	std::string m_outputName;
	std::vector<std::string> m_VecClassName;

	virtual s_NetStatus InitNet(char* cCfgPath, char* cWeightsPath, int *nOriImgWidth, int *nOriImgHeight, int *nChannels, HTuple* hv_htClassNames) = 0;

protected:
	InferenceEngine::InferRequest::Ptr m_inferRequest;

};