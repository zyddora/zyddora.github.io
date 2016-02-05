---
layout:     post
title:      "图像处理初探整理"
subtitle:   ""
date:       2016-02-02 19:57:00
author:     "Orchid"
header-img: "img/post-bg-img.jpg"
tags:
    - OpenCV
    - 图像处理
---
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

最近做了些与图像处理相关的内容，

### Catalog

1.  [OpenCV2.4.10 + Win10 VS2015的安装配置](#opencv2410-win10-vs2015)
2.  [OpenCV基本用法](#opencv)
3.  [RGB/YUV色彩空间](#rgbyuv)

### OpenCV2.4.10 + Win10 VS2015的安装配置

1. 工具
- OpenCV 下载地址：http://opencv.org/downloads.html
- Visual Studio 2015

2. 步骤
- 安装OpenCV并解压缩
- 配置环境变量
	+ 系统变量PATH：e.g. `E:\opencv\build\x86\vc12\bin`
	+ 用户变量
		* 添加opencv变量：e.g. `E:\opencv\build`
		* 补全PATH变量：e.g. `E:\opencv\build\x86\vc12\bin`
	+ 注：不管操作系统是32位或64位，建议上述目录均选择32位，因为一般都是32位的编译方式，若选择x64路径则可能会出错。
- 在VS中新建项目
	+ 选择Visual C++中Win32 Console Application（Win32 控制台应用程序）进行创建
	+ 进入Win32应用程序向导
		* 应用程序类型：选择控制台应用程序
		* 附加选项：空项目、预编译头
- 工程目录的配置（Debug）
	+ 在窗口右侧上栏找到Property Manager（属性管理器），双击`Debug | Win32`
		* 按照顺序找到 Common Properties - VC++ Directories - Include Directories，添加`E:\opencv\build\include`，`E:\opencv\build\include\opencv`，`E:\opencv\build\include\opencv2`。
		* 按照顺序找到 Common Properties - VC++ Directories - Library Directories，添加`E:\opencv\build\x86\vc12\lib`。
		* 按照顺序找到 Common Properties - Linker - Input - Addtional Dependencies，添加`E:\opencv\build\x86\vc12\lib`中的所有后缀带`d`的文件名。示例：

		`
		opencv_calib3d2410d.lib
		opencv_contrib2410d.lib
		opencv_core2410d.lib
		......
		`


---

### OpenCV基本用法


---

### RGB/YUV色彩空间

![img](/img/in-post/post-DLS/RBM_structure.png)

---