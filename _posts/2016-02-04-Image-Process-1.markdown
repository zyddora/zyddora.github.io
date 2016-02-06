---
layout:     post
title:      "图像处理初探整理"
subtitle:   ""
date:       2016-02-04 19:57:00
author:     "Orchid"
header-img: "img/post-bg-img.jpg"
tags:
    - OpenCV
    - 图像处理
---
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

> 近日做了些与OpenCV、图像处理相关的工程，两个周的时间收获了挺多，故将涉及到的内容进行整理，方便以后查阅，共同学习。
本文主要介绍OpenCV使用、图像格式相关的内容。

### Catalog

1. [OpenCV2.4.10 + Win10 VS2015的安装配置](#opencv2410--win10-vs2015)
2. [OpenCV基本用法](#opencv)
3. [RGB/YUV色彩空间](#rgbyuv)

## OpenCV2.4.10 + Win10 VS2015的安装配置

### **工具**
- OpenCV 下载地址：http://opencv.org/downloads.html
- Visual Studio 2015

### **环境变量配置**
安装OpenCV并解压缩；

配置环境变量：

- 系统变量PATH：e.g. `E:\opencv\build\x86\vc12\bin`
- 用户变量：
	* 添加opencv变量：e.g. `E:\opencv\build`
	* 补全PATH变量：e.g. `E:\opencv\build\x86\vc12\bin`
	* 注：不管操作系统是32位或64位，建议上述目录均选择32位，因为一般都是32位的编译方式，若选择x64路径则可能会出错。

### **在VS中新建项目**
选择Visual C++中Win32 Console Application（Win32 控制台应用程序）进行创建；

进入Win32应用程序向导：

* 应用程序类型：选择控制台应用程序
* 附加选项：空项目、预编译头

### **工程目录的配置（Debug）**
在窗口右侧上栏找到`Property Manager（属性管理器）`，双击`Debug | Win32`：

* 按顺序找到 Common Properties - VC++ Directories - Include Directories，添加`E:\opencv\build\include`，`E:\opencv\build\include\opencv`，`E:\opencv\build\include\opencv2`。
* 按顺序找到 Common Properties - VC++ Directories - Library Directories，添加`E:\opencv\build\x86\vc12\lib`。
* 按顺序找到 Common Properties - Linker - Input - Addtional Dependencies，添加`E:\opencv\build\x86\vc12\lib`中的所有后缀带`d`的文件名。示例：

```
opencv_calib3d2410d.lib
opencv_contrib2410d.lib
opencv_core2410d.lib
......
```

### **工程目录的配置（Release）**
 - 类似地，双击`Release | Win32`，按照顺序找到 Common Properties - Linker - Input - Addtional Dependencies，添加`E:\opencv\build\x86\vc12\lib`中的所有后缀 **不带`d`** 的文件名。

### **测试代码**

```cpp
#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;

int main()
{
  IplImage * test;
  test = cvLoadImage("D:\\Sample_8.bmp");//路径，注意加双斜杠转义
  cvNamedWindow("test_demo", 1);
  cvShowImage("test_demo", test);
  cvWaitKey(0);
  cvDestroyWindow("test_demo");
  cvReleaseImage(&test);
  system("pause");
  return 0;
}
```
---

## OpenCV基本用法

### **基本操作**

OpenCV通过结构体`IplImage`存储图片的信息，通过指向`IplImage`的指针对图片进行操作。

```cpp
typedef struct _IplImage
{
  int  nSize;             /* sizeof(IplImage) */
  int  ID;                /* version (=0)*/
  int  nChannels;         /* Most of OpenCV functions support 1,2,3 or 4 channels */
  int  alphaChannel;      /* Ignored by OpenCV */
  int  depth;             /* Pixel depth in bits: IPL_DEPTH_8U, IPL_DEPTH_8S, IPL_DEPTH_16S,
                             IPL_DEPTH_32S, IPL_DEPTH_32F and IPL_DEPTH_64F are supported.  */
  char colorModel[4];     /* Ignored by OpenCV */
  char channelSeq[4];     /* ditto */
  int  dataOrder;         /* 0 - interleaved color channels, 1 - separate color channels.
                             cvCreateImage can only create interleaved images */
  int  origin;            /* 0 - top-left origin,
                             1 - bottom-left origin (Windows bitmaps style).  */
  int  align;             /* Alignment of image rows (4 or 8).
                             OpenCV ignores it and uses widthStep instead.    */
  int  width;             /* Image width in pixels.                           */
  int  height;            /* Image height in pixels.                          */
  struct _IplROI *roi;    /* Image ROI. If NULL, the whole image is selected. */
  struct _IplImage *maskROI;      /* Must be NULL. */
  void  *imageId;                 /* "           " */
  struct _IplTileInfo *tileInfo;  /* "           " */
  int  imageSize;         /* Image data size in bytes
                             (==image->height*image->widthStep
                             in case of interleaved data)*/
  char *imageData;        /* Pointer to aligned image data.         */
  int  widthStep;         /* Size of aligned image row in bytes.    */
  int  BorderMode[4];     /* Ignored by OpenCV.                     */
  int  BorderConst[4];    /* Ditto.                                 */
  char *imageDataOrigin;  /* Pointer to very origin of image data
                             (not necessarily aligned) -
                             needed for correct deallocation */
}IplImage;
```

其中，比较重要的成员为：

- `int nChannels`：图像通道数，如RGB为3，RGBA为4，YUV为1。
- `int depth`：图像深度，即各像素的数值类型，如IPL_DEPTH_8U为8bits unsigned char，其余支持类型见上述代码说明。
- `int width`：图像宽度。
- `int height`：图像高度。
- `int imageSize`：图像占用字节数，即`height * widthStep`。
- `char *imageData`：指向图像数据的`char`型指针。
- `int widthStep`：图像宽度步长。需要特别注意的值，因为图像每行需要内存对齐，所以`width`和`widthStep`是不同的。比如：`width = 311, widthStep = 312`，我自己测试在x86上widthStep必须是4的倍数。此概念对于操作图像具体像素值是非常重要的。

**读取图片**

```cpp
function: CVAPI(IplImage*) cvLoadImage( const char* filename, int iscolor CV_DEFAULT(CV_LOAD_IMAGE_COLOR));
example: IplImage *test_ori = cvLoadImage(argv[1], 1);
```

**创建图片**

```cpp
function: CVAPI(IplImage*)  cvCreateImage( CvSize size, int depth, int channels );
example: IplImage *rec_ori = cvCreateImage(cvSize(cvwidth, cvheight), IPL_DEPTH_8U, 3);
```

**显示图片**

```cpp
function: CVAPI(void) cvShowImage( const char* name, const CvArr* image );
example: cvShowImage("recover", rec_ori);
```

**转换图片色彩空间**

```cpp
function: CVAPI(void)  cvCvtColor( const CvArr* src, CvArr* dst, int code );
example: cvCvtColor(test_ori, cvt_yuv422, CV_BGR2YUV);
```

注：cvCvtColor函数虽然提供了多种自带转换，但不提供BGR-->YUV422I(YUY2、YUNV、YUYV、V422)的转换，反而提供YUV422I-->BGR的转换。此处，YUV422I是Android常用的图片格式，但如若进行图像处理，则一般需要转换成RGB。

**分通道显示图像**

```cpp
function: void cvSplit(const CvArr* src,CvArr *dst0,CvArr *dst1, CvArr *dst2, CvArr *dst3);
example: 
for (int i = 0; i < test_ori->nChannels; ++i) {
  imgChannel_[i] = cvCreateImage(cvGetSize(test_ori), test_ori->depth, 1);
}

cvSplit(test_ori, imgChannel_[0], imgChannel_[1], imgChannel_[2], 0);

cvShowImage("B_", imgChannel_[0]);
cvShowImage("G_", imgChannel_[1]);
cvShowImage("R_", imgChannel_[2]);
```

注：通常情况下，通道小于4，则可以填0或NULL以补全。显示每个通道时，也是灰色图像，如果要以蓝、绿、红色展示，有其他方法，但我个人感觉并不需要，因为颜色只是一种直观的判断方法，但是具体还是其中的数值。

### 其他

**图像数值显示**：一般的RGB、YUV或者灰度图像基本都是以8bits unsigned char的形式存储，范围在0~255之间。在做图像转换、恢复时，经常需要检查具体图像数据，因此建议以十六进制显示为佳。显示方式如下：

```cpp
printf("0x%.2x", x1); // x1为某数值
```
---

## RGB/YUV色彩空间

### 一般的图像传输流程（以YUV传输为例）

**发送端**： 彩色图像 --> 分色后放大校正得到RGB图像 --> 矩阵变换 --> 得到YUV --> 将三信号分别编码发送

**接收端**： 解码得到YUV --> 转换YUV到RGB ---> 进一步处理

### RGB与YUV的相互转换

RGB的原理是三者的组合可以构成任何颜色，用三个0~255的数构建一个像素的信息。然而，YUV更加符合人类视觉的习惯。

> 人类大脑最先感知的是亮度。

依据这个理论，结合人眼对于不同颜色的灵敏度相异，采用YUV的模型也能够更好反映色彩。三个值有不同的含义：

- Y代表亮度，是个正数，在0~255之间，0为黑，255为白；
- U（Cr）有正有负，正数时代表**<font color="#ff0000">红色</font>**，负数时代表<font color="#00ff00">绿色</font>；
- V（Cb）有正有负，正数时代表<font color="#0000ff">蓝色</font>，负数时代表<font color="#eeee00">黄色</font>。

二者的转换是线性的，Y代表该像素的亮度、U、V

$$
\begin{bmatrix}
Y\\ 
U\\ 
V
\end{bmatrix}=\begin{bmatrix}
0.299 & 0.587 & 0.114\\ 
-0.147 & -0.289 & 0.436\\ 
0.615 & -0.515 & -0.100
\end{bmatrix}\begin{bmatrix}
R\\ 
G\\ 
B
\end{bmatrix}
$$



![img](/img/in-post/post-DLS/RBM_structure.png)

---