#pragma once

#ifdef DEEPLEARNING_EXPORTS
#define DEEPLEARNING_API __declspec(dllexport)
#else
#define DEEPLEARNING_API __declspec(dllimport)
#endif

#pragma warning(disable:4005) //不显示宏重定义警告

#include "Halcon.h"
#include "HalconCpp.h"

using namespace HalconCpp;

typedef  struct _DL_DETECT_RESULT
{
	HTuple  htTop;         //左上角行坐标
	HTuple  htLeft;        //左上角列坐标
	HTuple  htBottom;      //右下角行坐标
	HTuple  htRight;       //右下角列坐标
	HTuple  htClassID;       //目标类别
	HTuple  htConfidence;  //目标置信度
	HObject hoMasks;

	_DL_DETECT_RESULT()
	{
		TupleGenConst(0, 0, &htTop);
		TupleGenConst(0, 0, &htLeft);
		TupleGenConst(0, 0, &htBottom);
		TupleGenConst(0, 0, &htRight);
		TupleGenConst(0, 0, &htClassID);
		TupleGenConst(0, 0.0, &htConfidence);
		GenEmptyObj(&hoMasks);

	}
	_DL_DETECT_RESULT(const _DL_DETECT_RESULT& sDLDetectResult)
	{
		*this = sDLDetectResult;
	}
	_DL_DETECT_RESULT& operator=(const _DL_DETECT_RESULT& sDLDetectResult)
	{
		htTop = sDLDetectResult.htTop;
		htLeft = sDLDetectResult.htLeft;
		htBottom = sDLDetectResult.htBottom;
		htRight = sDLDetectResult.htRight;
		htClassID = sDLDetectResult.htClassID;
		htConfidence = sDLDetectResult.htConfidence;
		CopyObj(hoMasks, &hoMasks, 1, -1);
		return (*this);
	}

	void clear()
	{
		TupleGenConst(0, 0, &htTop);
		TupleGenConst(0, 0, &htLeft);
		TupleGenConst(0, 0, &htBottom);
		TupleGenConst(0, 0, &htRight);
		TupleGenConst(0, 0, &htClassID);
		TupleGenConst(0, 0.0, &htConfidence);
		GenEmptyObj(&hoMasks);
	}

}s_DLDetectResult;

//函数返回类型  0为正常，1－－N为自己定义的类型
typedef struct _NET_STATUS
{
	UINT nErrorType;		              //返回函数状态 0:正常 1:异常
	char pcErrorInfo[512];	              //状态描述
	//////////////////////////////////////////////////////////////////////////
	_NET_STATUS()
	{
		nErrorType = 0;
		memset(pcErrorInfo, 0, 512);
	}
	_NET_STATUS(const _NET_STATUS& sS)
	{
		*this = sS;
	}
	_NET_STATUS& operator=(const _NET_STATUS& sStatus)
	{
		nErrorType = sStatus.nErrorType;
		memcpy(pcErrorInfo, sStatus.pcErrorInfo, 512);
		return (*this);
	}
}s_NetStatus;


DEEPLEARNING_API s_NetStatus InitNet(void** net, char* cCfgPath, char* cWeightsPath, HTuple *hv_htClassNames, int *nOriImgWidth, int *nOriImgHeight, int *nChannels, int iNetType, bool bUseGPU);

DEEPLEARNING_API s_NetStatus DoInference(void* net, HObject ho_hImg, s_DLDetectResult& sDLDetectResult, float score);

//DEEPLEARNING_API s_NetStatus DoInferenceInstance(void* net, HObject ho_hImg, s_DLDetectResult& sDLDetectResult, float score);
//
DEEPLEARNING_API void FreeNet(void* net);