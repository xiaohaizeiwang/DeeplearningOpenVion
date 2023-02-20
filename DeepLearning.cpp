#include "DeepLearning.h"
#include "YoloVino.h"
#include <string>
#include "Utils.h"
#include "io.h"

using namespace cv;
using namespace std;

void ODRetMat2DLRet(vector<Mat>& results, s_DLDetectResult& sDLDetectResult)
{
	for (int n = 0; n < results.size(); n++)
	{
		Mat boxesInfo = results[n];
		float* ptr = (float*)boxesInfo.ptr(0);
		sDLDetectResult.htLeft.Append(ptr[0]);
		sDLDetectResult.htTop.Append(ptr[1]);
		sDLDetectResult.htRight.Append(ptr[2]);
		sDLDetectResult.htBottom.Append(ptr[3]);
		sDLDetectResult.htClassID.Append(int(ptr[4]));
		sDLDetectResult.htConfidence.Append(ptr[5]);
	}
}

//Instance Segment
void ISRetMat2DLRet(Mat& results, Mat& masks, s_DLDetectResult& sDLDetectResult)
{
	HObject ho_hMaskImg,ho_hMaskReg;
	vector<Mat> channels;
	split(masks, channels);
	for (int n = 0; n < channels.size(); n++)
	{
		float* ptr = (float*)results.ptr(n);
		sDLDetectResult.htLeft.Append(ptr[0]);
		sDLDetectResult.htTop.Append(ptr[1]);
		sDLDetectResult.htRight.Append(ptr[2]);
		sDLDetectResult.htBottom.Append(ptr[3]);
		sDLDetectResult.htClassID.Append(ptr[4]);
		sDLDetectResult.htConfidence.Append(ptr[5]);
		Mat2Hobj(channels[n],ho_hMaskImg);
		Threshold(ho_hMaskImg, &ho_hMaskReg, 0, 255);
		ConcatObj(sDLDetectResult.hoMasks, ho_hMaskReg, &sDLDetectResult.hoMasks);
	}


}

s_NetStatus InitNet(void** net, char* cCfgPath, char* cWeightsPath, HTuple* hv_htClassNames, int* nOriImgWidth, int* nOriImgHeight, int* nChannels, int iNetType, bool bUseGPU)
{
	s_NetStatus sNetStatus;

	

	try
	{
		int nNetType = -1;
		if (access(cCfgPath, 0) == -1)
		{
			sNetStatus.nErrorType = 2;
			strcpy_s(sNetStatus.pcErrorInfo, "网络配置文件不存在");
			return sNetStatus;
		}

		FileStorage fs(cCfgPath, FileStorage::READ, "UTF-8");
		if (fs["Net_Type"].empty())
		{
			sNetStatus.nErrorType = 4;
			strcpy_s(sNetStatus.pcErrorInfo, "参数-Net_Type不存在, 参数文件有误");
			return sNetStatus;
		}
		else
		{
			fs["Net_Type"] >> nNetType;
			if (nNetType < 0 || nNetType>6)
			{
				sNetStatus.nErrorType = 4;
				strcpy_s(sNetStatus.pcErrorInfo, "参数-Net_Type有误");
				return sNetStatus;
			}
		}
		fs.release();

		switch (nNetType)
		{
			case 0:
			case 1:
			/*{
				throw exception("未实现的网络");
				break;
			}*/
			case 2:
			case 3:
			case 4:
			case 5:
			case 6:
			{
				*net = new YoloVino(nNetType);
				break;
			}
			default:
				throw exception("未实现的网络");
		}
		
		
		sNetStatus = ((BaseNet*)(*net))->InitNet(cCfgPath, cWeightsPath, nOriImgWidth, nOriImgHeight, nChannels, hv_htClassNames);
		if (sNetStatus.nErrorType)return sNetStatus;

		s_DLDetectResult sDLDetectResult;
		HObject ho_hImage;
		if (((BaseNet*)(*net))->m_ImgChannels == 1)
		{
			GenImageConst(&ho_hImage, "byte", *nOriImgWidth, *nOriImgHeight);
		}
		else if (((BaseNet*)(*net))->m_ImgChannels == 3)
		{
			HObject ho_hImage1, ho_hImage2, ho_hImage3;
			GenImageConst(&ho_hImage1, "byte", *nOriImgWidth, *nOriImgHeight);
			GenImageConst(&ho_hImage2, "byte", *nOriImgWidth, *nOriImgHeight);
			GenImageConst(&ho_hImage3, "byte", *nOriImgWidth, *nOriImgHeight);
			//GenImageProto(ho_hImage, &ho_hImage, 0);
			Compose3(ho_hImage1, ho_hImage2, ho_hImage3, &ho_hImage);
		}
		
		DoInference(*net, ho_hImage, sDLDetectResult, 0.3);
	}
	catch (HException& e)
	{
		sNetStatus.nErrorType = 1;
		memcpy(sNetStatus.pcErrorInfo, e.ErrorMessage().Text(), 512);
	}
	catch (std::exception &e)
	{
		sNetStatus.nErrorType = 1;
		memcpy(sNetStatus.pcErrorInfo, e.what(),512);
	}
	return sNetStatus;
}



s_NetStatus DoInference(void* net, HObject ho_hImg, s_DLDetectResult& sDLDetectResult, float score)
{
	s_NetStatus sNetStatus;
	try
	{
		Mat cvImage;
		hobj2Mat(ho_hImg, cvImage);
		int iNetType = ((BaseNet*)net)->m_iNetType;
		switch (iNetType)
		{
			case 0:
			case 1:
			case 2:
			case 3:
			case 4:
			case 5:
			case 6:
			{
				std::vector<cv::Mat> results;
				((Yolo*)net)->DoInfer(cvImage, results, score);
				ODRetMat2DLRet(results, sDLDetectResult);
				break;
			}
			default:
				sNetStatus.nErrorType = 1;
				memcpy(sNetStatus.pcErrorInfo, "程序内部错误4", 512);
				break;
		}
		
	}
	catch (HException& e)
	{
		sNetStatus.nErrorType = 1;
		memcpy(sNetStatus.pcErrorInfo, e.ErrorMessage().Text(), 512);
	}
	catch (std::exception& e)
	{
		sNetStatus.nErrorType = 1;
		memcpy(sNetStatus.pcErrorInfo, e.what(), 512);
	}
	return sNetStatus;
}


//s_NetStatus DoInferenceInstance(void* net, HObject ho_hImg, s_DLDetectResult& sDLDetectResult, float score)
//{
//	s_NetStatus sNetStatus;
//	try
//	{
//		Mat cvImage;
//		hobj2Mat(ho_hImg, cvImage);
		
		


		/*for (int n = 0; n < results.size(); n++)
		{
			Mat boxesInfo = results[n];
			float* ptr = (float*)boxesInfo.ptr(0);
			sDLDetectResult.htLeft.Append(ptr[0]);
			sDLDetectResult.htTop.Append(ptr[1]);
			sDLDetectResult.htRight.Append(ptr[2]);
			sDLDetectResult.htBottom.Append(ptr[3]);
			sDLDetectResult.htClassID.Append(ptr[4]);
			sDLDetectResult.htConfidence.Append(ptr[5]);
		}*/
	/*}

	catch (HException& e)
	{
		memcpy(sNetStatus.pcErrorInfo, e.ErrorMessage().TextA(),512);
	}
	catch (c10::Error& e)
	{
		memcpy(sNetStatus.pcErrorInfo, e.msg().c_str(),512);
	}
	catch (std::exception& e)
	{
		memcpy(sNetStatus.pcErrorInfo, e.what(),512);
	}
	return sNetStatus;
}*/


void FreeNet(void* net)
{

	if (net != NULL)
	{
		int iNetType = ((BaseNet*)net)->m_iNetType;
		switch (iNetType)
		{
			case 0:
			{
				Yolo *pNet = (Yolo *)net;
				delete pNet;
				net = NULL;
				break;
			}
		}
	}


}