#include "YoloVino.h"


using namespace std;
using namespace cv;
using namespace dnn;
using namespace InferenceEngine;

YoloVino::YoloVino(int nNetType)
{
	m_ImgWidth = -1;
	m_ImgHeight = -1;
	m_ImgChannels = -1;
	m_ClassNum = -1;
	m_iNetType = nNetType;
	m_inputName = "";
	m_outputName = "";
	m_Prc = Precision::FP16;
}

YoloVino::~YoloVino()
{

}

int YoloVino::DoInfer(cv::Mat& img, vector<Mat>& results, float score)
{
	try
	{

		std::vector<float> pad_info = LetterboxImage(img, m_inferRequest->GetBlob(m_inputName), cv::Size(m_ImgWidth, m_ImgHeight));
		m_inferRequest->Infer();

		auto output = m_inferRequest->GetBlob(m_outputName);
		const SizeVector outputDims = output->getTensorDesc().getDims();

		const int dimensions = outputDims[2];
		const int rows = outputDims[1];
		vector<int> class_ids;
		vector<float> confidences;
		vector<Rect> boxes;
		float fScale = pad_info[2];
		float fLPad = pad_info[0];
		float fTPad = pad_info[1];
		int nClsNum = m_VecClassName.size();

		if (m_Prc == Precision::FP16)
		{
			int16_t*dataout = output->buffer().as<PrecisionTrait<Precision::FP16>::value_type*>();
			//解析顺序，维度为1的可直接略掉。
			for (int i = 0; i < rows; ++i)
			{
				float confidence = f16_to_f32(dataout+4);
				if (confidence >= score)
				{
					float* classes_scores = new float[nClsNum];
					for (int j = 0; j < nClsNum;j++)
					{
						classes_scores[j] = f16_to_f32(dataout + 5 + j);
					}
					Mat scores(1, m_VecClassName.size(), CV_32FC1, classes_scores);
					Point class_id;
					double max_class_score;
					minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
					if (max_class_score > score)
					{
						confidences.push_back(confidence);
						class_ids.push_back(class_id.x);

						float x = f16_to_f32(dataout + 0);
						float y = f16_to_f32(dataout + 1);
						float w = f16_to_f32(dataout + 2);
						float h = f16_to_f32(dataout + 3);
						int left = int((x - 0.5 * w - fLPad) / fScale);
						int top = int((y - 0.5 * h - fTPad) / fScale);
						int width = int(w / fScale);
						int height = int(h / fScale);
						boxes.push_back(Rect(left, top, width, height));
					}
					delete classes_scores;
				}
				dataout += dimensions;
			}
		}
		else
		{
			float * dataout= output->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
			//解析顺序，维度为1的可直接略掉。
			for (int i = 0; i < rows; ++i)
			{
				float confidence = dataout[4];
				if (confidence >= score)
				{
					float *classes_scores = dataout+5;
					Mat scores(1, m_VecClassName.size(), CV_32FC1, classes_scores);
					Point class_id;
					double max_class_score;
					minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
					if (max_class_score > score)
					{
						confidences.push_back(confidence);
						class_ids.push_back(class_id.x);

						float x = dataout[0];
						float y = dataout[1];
						float w = dataout[2];
						float h = dataout[3];
						int left = int((x - 0.5 * w - fLPad) / fScale);
						int top = int((y - 0.5 * h - fTPad) / fScale);
						int width = int(w / fScale);
						int height = int(h / fScale);
						boxes.push_back(Rect(left, top, width, height));
					}
				}
				dataout += dimensions;
			}
		}
		
		
		
		vector<int> nms_result;
		dnn::NMSBoxes(boxes, confidences, score, 0.45, nms_result);
		for (int i = 0; i < nms_result.size(); i++)
		{
			int idx = nms_result[i];
			float fArr[6] = { boxes[idx].tl().x, boxes[idx].tl().y, \
				boxes[idx].br().x, boxes[idx].br().y, class_ids[idx], confidences[idx] };
			Mat bbox(1, 6, CV_32FC1, fArr);
			results.push_back(bbox.clone());
		}
	}
	catch (const std::exception& exce)
	{
		//string sError = torch::GetExceptionString(exce);
		return -1;
	}
}

std::vector<float> YoloVino::LetterboxImage(const cv::Mat& src, Blob::Ptr pVnBlob, const cv::Size& out_size)
{
	auto in_h = static_cast<float>(src.rows);
	auto in_w = static_cast<float>(src.cols);
	float out_h = out_size.height;
	float out_w = out_size.width;

	float scale = std::min(out_w / in_w, out_h / in_h);

	int mid_h = static_cast<int>(in_h * scale);
	int mid_w = static_cast<int>(in_w * scale);

	Mat dst;
	cv::resize(src, dst, cv::Size(mid_w, mid_h));

	int top = (static_cast<int>(out_h) - mid_h) / 2;
	int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
	int left = (static_cast<int>(out_w) - mid_w) / 2;
	int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

	cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114)); 
	dst.convertTo(dst, CV_16FC(dst.channels()), 1.0f / 255.0f);

	InferenceEngine::MemoryBlob::Ptr pBlob = InferenceEngine::as<InferenceEngine::MemoryBlob>(pVnBlob);
	if (!pBlob) {
		IE_THROW() << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
			<< "but by fact we were not able to cast inputBlob to MemoryBlob";
	}

	auto BlobHolder = pBlob->wmap();
	unsigned short* blob_data = BlobHolder.as<unsigned short*>();

	vector<Mat> vecChannels;
	split(dst, vecChannels);

	int nSize = m_ImgWidth * m_ImgHeight;//sizeof(float)*
	memcpy(blob_data, vecChannels[0].data,  nSize);
	memcpy(blob_data+ nSize, vecChannels[1].data, nSize);
	memcpy(blob_data+ 2*nSize, vecChannels[2].data, nSize);

	std::vector<float> pad_info{ static_cast<float>(left), static_cast<float>(top), scale };
	return pad_info;
}

//void YoloVino::Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes,
//	const at::TensorAccessor<float, 2>& det,
//	std::vector<cv::Rect>& offset_box_vec,
//	std::vector<float>& score_vec)
//{
//
//	for (int i = 0; i < offset_boxes.size(0); i++) {
//		offset_box_vec.emplace_back(
//			cv::Rect(cv::Point(offset_boxes[i][Det::tl_x], offset_boxes[i][Det::tl_y]),
//				cv::Point(offset_boxes[i][Det::br_x], offset_boxes[i][Det::br_y]))
//		);
//		score_vec.emplace_back(det[i][Det::score]);
//	}
//}
//
//void YoloVino::ScaleCoordinates(std::vector<Detection>& data, float pad_w, float pad_h,
//	float scale, const cv::Size& img_shape)
//{
//	auto clip = [](float n, float lower, float upper)
//	{
//		return std::max(lower, std::min(n, upper));
//	};
//
//	std::vector<Detection> detections;
//	for (auto& i : data) {
//		float x1 = (i.bbox.tl().x - pad_w) / scale;  // x padding
//		float y1 = (i.bbox.tl().y - pad_h) / scale;  // y padding
//		float x2 = (i.bbox.br().x - pad_w) / scale;  // x padding
//		float y2 = (i.bbox.br().y - pad_h) / scale;  // y padding
//
//		x1 = clip(x1, 0, img_shape.width);
//		y1 = clip(y1, 0, img_shape.height);
//		x2 = clip(x2, 0, img_shape.width);
//		y2 = clip(y2, 0, img_shape.height);
//
//		i.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
//	}
//}
//
//
//torch::Tensor YoloVino::xywh2xyxy(const torch::Tensor& x)
//{
//	auto y = torch::zeros_like(x);
//	// convert bounding box format from (center x, center y, width, height) to (x1, y1, x2, y2)
//	y.select(1, Det::tl_x) = x.select(1, 0) - x.select(1, 2).div(2);
//	y.select(1, Det::tl_y) = x.select(1, 1) - x.select(1, 3).div(2);
//	y.select(1, Det::br_x) = x.select(1, 0) + x.select(1, 2).div(2);
//	y.select(1, Det::br_y) = x.select(1, 1) + x.select(1, 3).div(2);
//	return y;
//}
//
//std::vector<std::vector<Detection>> YoloVino::PostProcessing(const torch::Tensor& detections,
//	float pad_w, float pad_h, float scale, const cv::Size& img_shape,
//	float conf_thres, float iou_thres)
//{
//	/***
//	 * 结果纬度为batch index(0), top-left x/y (1,2), bottom-right x/y (3,4), score(5), class id(6)
//	 * 13*13*3*(1+4)*80
//	 */
//	constexpr int item_attr_size = 5;
//	int batch_size = detections.size(0);
//	// number of classes, e.g. 80 for coco dataset
//	auto num_classes = detections.size(2) - item_attr_size;
//
//	// get candidates which object confidence > threshold
//	auto conf_mask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);
//
//	std::vector<std::vector<Detection>> output;
//	output.reserve(batch_size);
//
//	// iterating all images in the batch
//	for (int batch_i = 0; batch_i < batch_size; batch_i++) {
//		// apply constrains to get filtered detections for current image
//		auto det = torch::masked_select(detections[batch_i], conf_mask[batch_i]).view({ -1, num_classes + item_attr_size });
//
//		// if none detections remain then skip and start to process next image
//		if (0 == det.size(0)) {
//			continue;
//		}
//
//		// compute overall score = obj_conf * cls_conf, similar to x[:, 5:] *= x[:, 4:5]
//		det.slice(1, item_attr_size, item_attr_size + num_classes) *= det.select(1, 4).unsqueeze(1);
//
//		// box (center x, center y, width, height) to (x1, y1, x2, y2)
//		torch::Tensor box = xywh2xyxy(det.slice(1, 0, 4));
//
//		// [best class only] get the max classes score at each result (e.g. elements 5-84)
//		std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(det.slice(1, item_attr_size, item_attr_size + num_classes), 1);
//
//		// class score
//		auto max_conf_score = std::get<0>(max_classes);
//		// index
//		auto max_conf_index = std::get<1>(max_classes);
//
//		max_conf_score = max_conf_score.to(torch::kFloat).unsqueeze(1);
//		max_conf_index = max_conf_index.to(torch::kFloat).unsqueeze(1);
//
//		// shape: n * 6, top-left x/y (0,1), bottom-right x/y (2,3), score(4), class index(5)
//		det = torch::cat({ box.slice(1, 0, 4), max_conf_score, max_conf_index }, 1);
//
//		// for batched NMS
//		constexpr int max_wh = 4096;
//		auto c = det.slice(1, item_attr_size, item_attr_size + 1) * max_wh;
//		auto offset_box = det.slice(1, 0, 4) + c;
//
//		std::vector<cv::Rect> offset_box_vec;
//		std::vector<float> score_vec;
//
//		// copy data back to cpu
//		auto offset_boxes_cpu = offset_box.cpu();
//		auto det_cpu = det.cpu();
//		const auto& det_cpu_array = det_cpu.accessor<float, 2>();
//
//		// use accessor to access tensor elements efficiently
//		Tensor2Detection(offset_boxes_cpu.accessor<float, 2>(), det_cpu_array, offset_box_vec, score_vec);
//
//		// run NMS
//		std::vector<int> nms_indices;
//		cv::dnn::NMSBoxes(offset_box_vec, score_vec, conf_thres, iou_thres, nms_indices);
//
//		std::vector<Detection> det_vec;
//		for (int index : nms_indices) {
//			Detection t;
//			const auto& b = det_cpu_array[index];
//			t.bbox =
//				cv::Rect(cv::Point(b[Det::tl_x], b[Det::tl_y]),
//					cv::Point(b[Det::br_x], b[Det::br_y]));
//			t.score = det_cpu_array[index][Det::score];
//			t.class_idx = det_cpu_array[index][Det::class_idx];
//			det_vec.emplace_back(t);
//		}
//
//		ScaleCoordinates(det_vec, pad_w, pad_h, scale, img_shape);
//
//		// save final detection for the current image
//		output.emplace_back(det_vec);
//	} // end of batch iterating
//
//	return output;
//}