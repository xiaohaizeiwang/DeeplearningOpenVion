#include "Yolo.h"
#include <fstream>
#include <string>
#include <io.h>

using namespace std;
using namespace cv;

s_NetStatus Yolo::InitNet(char* cCfgPath, char* cWeightsPath, int* nOriImgWidth, int* nOriImgHeight, HTuple* hv_htClassNames)
{
	s_NetStatus sNetStatus;

	string sParamPath(cCfgPath), sClsFilePath(cWeightsPath), sWeightPath(cWeightsPath);

	sClsFilePath = sClsFilePath.substr(0, sClsFilePath.find_last_of("\\")) + "/names";
	if (access(cWeightsPath, 0) == -1)
	{
		sNetStatus.nErrorType = 2;
		strcpy_s(sNetStatus.pcErrorInfo, "权重文件不存在");
		return sNetStatus;
	}
	else if (access(sParamPath.c_str(), 0) == -1)
	{
		sNetStatus.nErrorType = 2;
		strcpy_s(sNetStatus.pcErrorInfo, "网络配置文件不存在");
		return sNetStatus;
	}
	else if (access(sClsFilePath.c_str(), 0) == -1)
	{
		sNetStatus.nErrorType = 2;
		strcpy_s(sNetStatus.pcErrorInfo, "类别文件不存在");
		return sNetStatus;
	}

	//读取配置文件
	FileStorage fs(sParamPath, FileStorage::READ, "UTF-8");
	if (fs["Width"].empty())
	{
		sNetStatus.nErrorType = 4;
		strcpy_s(sNetStatus.pcErrorInfo, "参数-Width不存在, 参数文件有误");
		return sNetStatus;
	}
	if (fs["Height"].empty())
	{
		sNetStatus.nErrorType = 4;
		strcpy_s(sNetStatus.pcErrorInfo, "参数-Height不存在, 参数文件有误");
		return sNetStatus;
	}
	if (fs["Channels"].empty())
	{
		sNetStatus.nErrorType = 4;
		strcpy_s(sNetStatus.pcErrorInfo, "参数-Channels不存在, 参数文件有误");
		return sNetStatus;
	}
	if (fs["ImgOriWidth"].empty())
	{
		sNetStatus.nErrorType = 4;
		strcpy_s(sNetStatus.pcErrorInfo, "参数-ImgOriWidth不存在, 参数文件有误");
		return sNetStatus;
	}
	if (fs["ImgOriHeight"].empty())
	{
		sNetStatus.nErrorType = 4;
		strcpy_s(sNetStatus.pcErrorInfo, "参数-ImgOriHeight不存在, 参数文件有误");
		return sNetStatus;
	}
	if (fs["Strides"].empty())
	{
		sNetStatus.nErrorType = 4;
		strcpy_s(sNetStatus.pcErrorInfo, "参数-Strides不存在, 参数文件有误");
		return sNetStatus;
	}

	fs["Width"] >> m_ImgWidth;
	fs["Height"] >> m_ImgHeight;
	fs["Channels"] >> m_ImgChannels;
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
			if (string::npos == className.find("类别"))
			{
				m_VecClassName.push_back(className);
			}
		}
		//dlgInitParamLeng += m_vecClassNames.size();
	}
	else
	{
		sNetStatus.nErrorType = 2;
		strcpy_s(sNetStatus.pcErrorInfo, "类别文件打开失败");
		return sNetStatus;
	}

	ifs.close();

	m_ClassNum = m_VecClassName.size();
	for (int i = 0; i < m_ClassNum; i++)
	{
		hv_htClassNames->Append(m_VecClassName[i].c_str());
	}

	return sNetStatus;
}