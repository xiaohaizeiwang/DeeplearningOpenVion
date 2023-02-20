#pragma once

#ifdef DEEPLEARNING_EXPORTS
#define DEEPLEARNING_API __declspec(dllexport)
#else
#define DEEPLEARNING_API __declspec(dllimport)
#endif

#pragma warning(disable:4005) //����ʾ���ض��徯��

#include "Halcon.h"
#include "HalconCpp.h"

using namespace HalconCpp;

typedef  struct _DL_DETECT_RESULT
{
	HTuple  htTop;         //���Ͻ�������
	HTuple  htLeft;        //���Ͻ�������
	HTuple  htBottom;      //���½�������
	HTuple  htRight;       //���½�������
	HTuple  htClassID;       //Ŀ�����
	HTuple  htConfidence;  //Ŀ�����Ŷ�
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

//������������  0Ϊ������1����NΪ�Լ����������
typedef struct _NET_STATUS
{
	UINT nErrorType;		              //���غ���״̬ 0:���� 1:�쳣
	char pcErrorInfo[512];	              //״̬����
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