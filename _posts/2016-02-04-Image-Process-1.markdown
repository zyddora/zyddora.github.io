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

1.  [OpenCV2.4.10 + Win10 VS2015的安装配置](#opencv2410--win10-vs2015)
2.  [OpenCV基本用法](#opencv)
3.  [RGB/YUV色彩空间](#rgbyuv)

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



---

## RGB/YUV色彩空间

![img](/img/in-post/post-DLS/RBM_structure.png)

---