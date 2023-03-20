#pragma once
#include "Yolo.h"
#include <inference_engine.hpp>


class YoloBin : public Yolo
{

public:
	YoloBin(int nNetType);
	~YoloBin();
	s_NetStatus InitNet(char* cCfgPath, char* cWeightsPath, int* nOriImgWidth, int* nOriImgHeight, HTuple* hv_htClassNames, int nMode);
	int DoInfer(cv::Mat& img, std::vector<cv::Mat>& results, float score);
	std::vector<float> LetterboxImage(const cv::Mat& src, InferenceEngine::Blob::Ptr pVnBlob, const cv::Size& out_size);

public:
	
	std::string m_inputName;
	std::string m_outputName;
	InferenceEngine::Precision m_Prc;
	InferenceEngine::InferRequest::Ptr m_inferRequest;

};