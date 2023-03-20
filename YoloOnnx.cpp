#include "YoloOnnx.h"
#include <io.h>


using namespace std;
using namespace cv;
using namespace ov;
using namespace dnn;
using namespace InferenceEngine;

YoloOnnx::YoloOnnx(int nNetType)
{
	m_ImgWidth = -1;
	m_ImgHeight = -1;
	m_ImgChannels = -1;
	m_ClassNum = -1;
	m_iNetType = nNetType;

}

s_NetStatus YoloOnnx::InitNet(char* cCfgPath, char* cWeightsPath, int* nOriImgWidth, int* nOriImgHeight, HTuple* hv_htClassNames, int nMode)
{
	s_NetStatus sNetStatus;

	sNetStatus = Yolo::InitNet(cCfgPath, cWeightsPath, nOriImgWidth, nOriImgHeight, hv_htClassNames);
	if (sNetStatus.nErrorType)
	{
		return sNetStatus;
	}

	string sDevice = "GPU";
	switch (nMode)
	{
	case 0:
		sDevice = "CPU";
		break;
	case 1:
	case 2:
		sDevice = "GPU";
		break;
	default:
		throw exception("运行方式有误");
	}

	ov::Core ie;
	ov::CompiledModel cm = ie.compile_model(cWeightsPath, sDevice);
	m_inferRequest = cm.create_infer_request();
		
	return sNetStatus;
}

int YoloOnnx::DoInfer(cv::Mat& img, vector<Mat>& results, float score)
{
	try
	{
		
		std::vector<float> pad_info = LetterboxImage(img, &m_inferRequest.get_input_tensor(), cv::Size(m_ImgWidth, m_ImgHeight));
		m_inferRequest.infer();

		Tensor tOutput = m_inferRequest.get_output_tensor();
		
		const int dimensions = tOutput.get_shape()[2];
		const int rows = tOutput.get_shape()[1];
		vector<int> class_ids;
		vector<float> confidences;
		vector<Rect> boxes;
		float fScale = pad_info[2];
		float fLPad = pad_info[0];
		float fTPad = pad_info[1];
		int nClsNum = m_VecClassName.size();

		float* dataout = tOutput.data<float>();
			//解析顺序，维度为1的可直接略掉。
		for (int i = 0; i < rows; ++i)
		{
			float confidence = dataout[4];
			if (confidence >= score)
			{
				float* classes_scores = dataout + 5;
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

std::vector<float> YoloOnnx::LetterboxImage(const cv::Mat& src, ov::Tensor* pTensor, const cv::Size& out_size)
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

	if (dst.channels() == 1 && m_ImgChannels == 3)
	{
		cvtColor(dst, dst, COLOR_GRAY2RGB);
	}

	int top = (static_cast<int>(out_h) - mid_h) / 2;
	int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
	int left = (static_cast<int>(out_w) - mid_w) / 2;
	int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

	cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

	dst.convertTo(dst, CV_32FC(dst.channels()), 1.0f / 255.0f);

	vector<Mat> vecChannels;
	split(dst, vecChannels);
	int nSize = m_ImgWidth * m_ImgHeight;

	float* blob_data = pTensor->data<float>();

	memcpy(blob_data, vecChannels[0].data, nSize * sizeof(float));
	memcpy(blob_data + nSize, vecChannels[1].data, nSize * sizeof(float));
	memcpy(blob_data + 2 * nSize, vecChannels[2].data, nSize * sizeof(float));

	std::vector<float> pad_info{ static_cast<float>(left), static_cast<float>(top), scale };
	return pad_info;
}

