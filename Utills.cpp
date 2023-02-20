#include "Utils.h"

using namespace HalconCpp;
using namespace cv;
using namespace std;
using namespace half_float;

int hobj2Mat(HObject& hImage, cv::Mat& cv_Image)
{
	try
	{
		HObject* hv_Image = &hImage;
		HObject hv_ImageCrop;
		HTuple hv_Channels, hv_Types, hv_Width, hv_Height;

		//ReduceDomain(*hv_Image, reg, &hv_ImageCrop);
		//CropDomain(hv_ImageCrop, &hv_ImageCrop);
		CountChannels(*hv_Image, &hv_Channels);
		GetImageType(*hv_Image, &hv_Types);
		GetImageSize(*hv_Image, &hv_Width, &hv_Height);

		if (1 == hv_Channels && !strcmp(hv_Types[0].S(), "byte"))
		{
			HTuple hv_Pointer;
			cv_Image = cv::Mat(hv_Height.I(), hv_Width.I(), CV_8UC1);
			GetImagePointer1(*hv_Image, &hv_Pointer, NULL, NULL, NULL);
			//GetImagePointer1(hv_ImageCrop, &hv_Pointer, NULL, NULL, NULL);
			uchar* pSrc = reinterpret_cast<uchar*>(hv_Pointer[0].L());

			uchar* pDst = cv_Image.data;
			memcpy(pDst, pSrc, hv_Height.I() * hv_Width.I());
		}
		else if (3 == hv_Channels && !strcmp(hv_Types[0].S(), "byte"))
		{
			int pixCount = hv_Height.I() * hv_Width.I();

			HTuple hv_R, hv_G, hv_B;
			GetImagePointer3(*hv_Image, &hv_R, &hv_G, &hv_B, NULL, NULL, NULL);
			//GetImagePointer3(hv_ImageCrop, &hv_R, &hv_G, &hv_B, NULL, NULL, NULL);
			cv::Mat r = cv::Mat(hv_Height.I(), hv_Width.I(), CV_8UC1);
			cv::Mat g = cv::Mat(hv_Height.I(), hv_Width.I(), CV_8UC1);
			cv::Mat b = cv::Mat(hv_Height.I(), hv_Width.I(), CV_8UC1);

			uchar* pSrc, * pDst;

			//R通道
			pSrc = reinterpret_cast<uchar*>(hv_R[0].L());
			pDst = r.data;
			memcpy(pDst, pSrc, pixCount);
			//G通道
			pSrc = reinterpret_cast<uchar*>(hv_G[0].L());
			pDst = g.data;
			memcpy(pDst, pSrc, pixCount);
			//R通道
			pSrc = reinterpret_cast<uchar*>(hv_B[0].L());
			pDst = b.data;
			memcpy(pDst, pSrc, pixCount);

			//得到合成图像
			vector<cv::Mat> channels = { r,g,b };
			merge(channels, cv_Image);

		}
		else
		{
			//图像类型有误
			return 2;
		}

	}
	catch (exception* e)
	{
		return 1;
	}
	//flip(cv_Image, cv_Image, ROTATE_90_CLOCKWISE);
	return 0;
}

int Mat2Hobj(cv::Mat& img, HObject& hv_Img)
{
	//flip(img, img, ROTATE_90_CLOCKWISE);
	try
	{
		HObject hv_ImageCrop;

		int type = img.type();
		int height = img.rows;
		int width = img.cols;

		if (type == CV_32FC1)
		{
			float* pDst = new float[width * height];
			float* pSrc = (float*)img.data;

			//数据复制
			memcpy(pDst, pSrc, width * height * sizeof(float));

			GenImage1(&hv_Img, "real", width, height, (Hlong)pDst);
			//GetImagePointer1(hv_Img, &hv_Pointer);
			//GetImagePointer1(hv_ImageCrop, &hv_Pointer, NULL, NULL, NULL);
			//uchar* pSrc = reinterpret_cast<uchar*>(hv_Pointer[0].L());
		}
		else if (type == CV_8UC1)
		{
			uchar* pDst = new uchar[width * height];
			uchar* pSrc = (uchar*)img.data;

			//数据复制
			memcpy(pDst, pSrc, width * height * sizeof(uchar));

			GenImage1(&hv_Img, "byte", width, height, (Hlong)pDst);
		}
		else
		{
			//图像类型有误
			return 2;
		}
	}
	catch (exception* e)
	{
		return 1;
	}
	return 0;
}

std::vector<std::string> split(const  std::string& s, const std::string& delim)
{
	std::vector<std::string> elems;
	size_t pos = 0;
	size_t len = s.length();
	size_t delim_len = delim.length();
	if (delim_len == 0) return elems;
	while (pos < len)
	{
		int find_pos = s.find(delim, pos);
		if (find_pos < 0)
		{
			elems.push_back(s.substr(pos, len - pos));
			break;
		}
		elems.push_back(s.substr(pos, find_pos - pos));
		pos = find_pos + delim_len;
	}
	return elems;
}

float f16_to_f32(int16_t* pn16) {
	half __x;
	memcpy(&__x, pn16, sizeof(__x));
	unsigned short n = *((unsigned short*)&__x);
	unsigned int x = (unsigned int)n;
	x = x & 0xffff;
	unsigned int sign = x & 0x8000;                   //符号位
	unsigned int exponent_f16 = (x & 0x7c00) >> 10;   //half指数位
	unsigned int mantissa_f16 = x & 0x03ff;           //half小数位
	unsigned int y = sign << 16;
	unsigned int exponent_f32;                        //float指数位
	unsigned int mantissa_f32;                        //float小数位
	unsigned int first_1_pos = 0;                     //（half小数位）最高位1的位置
	unsigned int mask;
	unsigned int hx;

	hx = x & 0x7fff;

	if (hx == 0) {
		return *((float*)&y);
	}
	if (hx == 0x7c00) {
		y |= 0x7f800000;
		return *((float*)&y);
	}
	if (hx > 0x7c00) {
		y = 0x7fc00000;
		return *((float*)&y);
	}

	exponent_f32 = 0x70 + exponent_f16;
	mantissa_f32 = mantissa_f16 << 13;

	for (first_1_pos = 0; first_1_pos < 10; first_1_pos++) {
		if ((mantissa_f16 >> (first_1_pos + 1)) == 0) {
			break;
		}
	}

	if (exponent_f16 == 0) {
		mask = (1 << 23) - 1;
		exponent_f32 = exponent_f32 - (10 - first_1_pos) + 1;
		mantissa_f32 = mantissa_f32 << (10 - first_1_pos);
		mantissa_f32 = mantissa_f32 & mask;
	}

	y = y | (exponent_f32 << 23) | mantissa_f32;

	return *((float*)&y);
}