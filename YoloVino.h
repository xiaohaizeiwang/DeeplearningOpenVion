#pragma once
#include "Yolo.h"

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

class YoloVino :public Yolo
{
public:
	YoloVino(int nNetType);
	~YoloVino();

	int DoInfer(cv::Mat& img, std::vector<cv::Mat>& results, float score);

	std::vector<float> LetterboxImage(const cv::Mat& src, InferenceEngine::Blob::Ptr pVnBlob, const cv::Size& out_size);

	/*void Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes, const at::TensorAccessor<float, 2>& det, std::vector<cv::Rect>& offset_box_vec,
		std::vector<float>& score_vec);

	void ScaleCoordinates(std::vector<Detection>& data, float pad_w, float pad_h,
		float scale, const cv::Size& img_shape);
	torch::Tensor xywh2xyxy(const torch::Tensor& x);

	std::vector<std::vector<Detection>> PostProcessing(const torch::Tensor& detections,
		float pad_w, float pad_h, float scale, const cv::Size& img_shape,
		float conf_thres, float iou_thres);*/


};