#pragma once
#include "Yolo.h"
#include "Utils.h"
#include <openvino/openvino.hpp>


class YoloOnnx : public Yolo
{
public:
	YoloOnnx(int nNetType);
	~YoloOnnx();
	s_NetStatus InitNet(char* cCfgPath, char* cWeightsPath, int* nOriImgWidth, int* nOriImgHeight, HTuple* hv_htClassNames, int nMode);
	int DoInfer(cv::Mat& img, std::vector<cv::Mat>& results, float score);
	std::vector<float> LetterboxImage(const cv::Mat& src, ov::Tensor *pTensor, const cv::Size& out_size);

public:
	ov::InferRequest m_inferRequest;

};