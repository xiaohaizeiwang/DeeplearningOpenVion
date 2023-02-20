#include "Yolo.h"
#include <io.h>
#include <fstream>

using namespace cv;
using namespace std;
using namespace InferenceEngine;

s_NetStatus Yolo::InitNet(char* cCfgPath, char* cWeightsPath, int* nOriImgWidth, int* nOriImgHeight, int* nChannels, HTuple* hv_htClassNames)
{
	s_NetStatus sNetStatus;

	string sParamPath(cCfgPath), sClsFilePath(cWeightsPath) , sXmlPath(cWeightsPath);
	
	sClsFilePath = sClsFilePath.substr(0, sClsFilePath.find_last_of("\\"))+"/names";
	if (access(cWeightsPath, 0) == -1)
	{
		sNetStatus.nErrorType = 2;
		strcpy_s(sNetStatus.pcErrorInfo, "Ȩ���ļ�������");
		return sNetStatus;
	}
	else if (access(sParamPath.c_str(), 0) == -1)
	{
		sNetStatus.nErrorType = 2;
		strcpy_s(sNetStatus.pcErrorInfo, "���������ļ�������");
		return sNetStatus;
	}
	else if (access(sClsFilePath.c_str(), 0) == -1)
	{
		sNetStatus.nErrorType = 2;
		strcpy_s(sNetStatus.pcErrorInfo, "����ļ�������");
		return sNetStatus;
	}
	sXmlPath = sXmlPath.substr(0, sXmlPath.find_last_of(".")) + ".xml";
	if (access(sXmlPath.c_str(), 0) == -1)
	{
		sNetStatus.nErrorType = 2;
		strcpy_s(sNetStatus.pcErrorInfo, "xml�ļ�������");
		return sNetStatus;
	}
	//��ȡ�����ļ�
	FileStorage fs(sParamPath, FileStorage::READ, "UTF-8");
	if (fs["Width"].empty())
	{
		sNetStatus.nErrorType = 4;
		strcpy_s(sNetStatus.pcErrorInfo, "����-Width������, �����ļ�����");
		return sNetStatus;
	}
	if (fs["Height"].empty())
	{
		sNetStatus.nErrorType = 4;
		strcpy_s(sNetStatus.pcErrorInfo, "����-Height������, �����ļ�����");
		return sNetStatus;
	}
	if (fs["Channels"].empty())
	{
		sNetStatus.nErrorType = 4;
		strcpy_s(sNetStatus.pcErrorInfo, "����-Channels������, �����ļ�����");
		return sNetStatus;
	}
	if (fs["ImgOriWidth"].empty())
	{
		sNetStatus.nErrorType = 4;
		strcpy_s(sNetStatus.pcErrorInfo, "����-ImgOriWidth������, �����ļ�����");
		return sNetStatus;
	}
	if (fs["ImgOriHeight"].empty())
	{
		sNetStatus.nErrorType = 4;
		strcpy_s(sNetStatus.pcErrorInfo, "����-ImgOriHeight������, �����ļ�����");
		return sNetStatus;
	}
	if (fs["Strides"].empty())
	{
		sNetStatus.nErrorType = 4;
		strcpy_s(sNetStatus.pcErrorInfo, "����-Strides������, �����ļ�����");
		return sNetStatus;
	}

	fs["Width"] >> m_ImgWidth;
	fs["Height"] >> m_ImgHeight;
	fs["Channels"] >> m_ImgChannels;
	*nChannels = m_ImgChannels;
	fs["ImgOriWidth"] >> *nOriImgWidth;
	fs["ImgOriHeight"] >> *nOriImgHeight;
	fs["Strides"] >> m_pStrides;
	/*FileNode fn = fs["ClassNames"];

	((BaseNet*)(*net))->m_ClassNum = fn.size();
	string str = fn[0].string();
	for (int i = 0; i < fn.size(); i++)
	{
		((BaseNet*)(*net))->m_VecClassName.push_back(fn[i].string());
		hv_htClassNames->Append(fn[i].string().c_str());
	}*/
	ifstream ifs(sClsFilePath, ios::in);
	if (ifs.is_open())
	{
		m_VecClassName.clear();
		string className;
		while (getline(ifs, className))
		{
			if (string::npos == className.find("���"))
			{
				m_VecClassName.push_back(className);
			}
		}
		//dlgInitParamLeng += m_vecClassNames.size();
	}
	else
	{
		sNetStatus.nErrorType = 2;
		strcpy_s(sNetStatus.pcErrorInfo, "����ļ���ʧ��");
		return sNetStatus;
	}
	
	ifs.close();
	
	m_ClassNum = m_VecClassName.size();
	for (int i = 0; i < m_ClassNum; i++)
	{
		hv_htClassNames->Append(m_VecClassName[i].c_str());
	}

	InferenceEngine::Core ie;
	//���ּ�����ʽ��ֱ�Ӽ���onnx�����߼���
	// ����������ֱ�Ӽ���onnx:
	//InferenceEngine::CNNNetwork network = ie.ReadNetwork(m_modelFilename);
	InferenceEngine::CNNNetwork network = ie.ReadNetwork(sXmlPath,cWeightsPath);
	//������������ָ�������
	InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
	InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();

	m_inputName = inputs.begin()->first;
	m_outputName = outputs.begin()->first;

	InputInfo::Ptr pInputData = inputs.begin()->second;
	DataPtr pOutputData = outputs.begin()->second;

	pInputData->setPrecision(m_Prc);
	pInputData->setLayout(Layout::NCHW);
	pInputData->getPreProcess().setColorFormat(ColorFormat::RGB);

	pOutputData->setPrecision(m_Prc);
	

	//��������Ϊ��ִ������
	auto executable_network = ie.LoadNetwork(network, "GPU");
	m_inferRequest = executable_network.CreateInferRequestPtr();

	return sNetStatus;
}


