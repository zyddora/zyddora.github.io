---
layout:     post
title:      "图像处理初探（二）——以Android中YUV422I旋转算法为例"
subtitle:   ""
date:       2016-03-16 18:43:00
author:     "Orchid"
header-img: "img/post-bg-img.jpg"
tags:
    - 图像处理
    - OpenCV
    - C/C++
---
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

> 在上一篇博文里介绍了[图像处理初探（一）——图像转换基础及OpenCV应用](http://zyddora.github.io/2016/02/05/image-process-1/)，本篇在此基础上重点谈图像旋转算法的原理与实现。本文以Android中常见的YUV422I（YUY2）格式为例。

### Catalog

1. [通用YUV422I旋转90/270算法](#yuv422i90270)
2. [通用YUV422I旋转0/180算法](#yuv422i0180)
3. [完善算法以支持不规则尺寸图像的旋转](#section-2)
4. [更加省时的图像旋转](#section-4)

## 通用YUV422I旋转90/270算法

### **解决思路**

**实现图像旋转算法的关键在于理解清楚：旋转前后图像格式排列规则之间的联系，这很重要。**初步考虑是，旋转90°或270°涉及相邻行列的像素点（最少2*2个像素点），将Y元素转置而UV元素进行适当地转换，UV的转换是重点。强烈建议画出前后的图像排列格式，再找转换关系。

### **YUV422I旋转90°示意**

![img](/img/in-post/rot90.jpg)

观察旋转前后可得以下几点信息：

- 旋转前的奇数行（以1为初始）中的UV信息都无用；
- 旋转后的偶数行（以1位初始）中的UV信息都是原图中没有的，需要推算；
  * 比如，图中的U10、V10需从Y10、U9、V9转换得到，通常思路为：(Y10, U9, V9) -> (R9, G9, B9) -> (U10, V10)
- 所有的Y信息都进行了转置且都有用（亮度信息必然有用）；
- `4*2`个YUV信息为旋转算法处理的最小单位；
- 旋转前图片格式为`(width * 2) * height`，旋转后为`(heigh * 2) * width`；

旋转90°的C程序框架：

```
const unsigned char *src_y = src; /* dst_1_1指向旋转前1-1处的指针 */
unsigned char *dst_y = dst_1_1; /* dst_1_1指向旋转后1-1处的指针 */

for (int i = 0; i < height / 2; ++i) { /* 假设一次外循环处理2行YUYV信息 */
  src = src_y;
  dst = dst_y;
  for (int j = 0; j < doublewidth / 4) { /* 假设一次处理每行相邻4个YUYV信息 */
    ... // 移动Y信息
    ... // 计算新的UV信息

    src += 4; /* 更新内循环的指针 */
    dst += doubleheight * 2;
  }
  src_y += doublewidth * 2; /* 更新外循环的指针 */
  dst_y -= 4;
}
```

### **YUV422I旋转270°示意**

![img](/img/in-post/rot270.jpg)

与旋转90°情况的不同在于：

- 旋转前的偶数行（以1为初始）中的UV信息都无用；
- 旋转后的奇数行（以1位初始）中的UV信息都是原图中没有的，需要推算；
  * 比如，图中的U24、V24需从Y24、U23、V23转换得到，通常思路为：(Y24, U23, V23) -> (R24, G24, B24) -> (U24, V24)

旋转270°的C程序框架与90°类似，此处不赘述。

---

## 通用YUV422I旋转0/180算法

### **解决思路**

旋转0°、180°较90°、270°更简单，仅涉及行上的像素点的转换（最少2*1个像素点）。

### **YUV422I旋转180°示意**

![img](/img/in-post/rot180.jpg)

观察可发现：

- 旋转前的所有YUV信息都有用；
- 旋转后的每行Y信息逆置，UV信息则需要推算，推算思路与上述类似；
- 旋转前后的图像规格不变，仍为`(width * 2) * height`；

旋转0°的本质：

- 图像数据的拷贝；

旋转180°的C程序框架：

```cpp
const unsigned char *src_y = src; /* dst_1_1指向旋转前1-1处的指针 */
unsigned char *dst_y = dst_1_1; /* dst_1_1指向旋转后1-1处的指针 */

for (int i = 0; i < height; ++i) {
  src = src_y;
  dst = dst_y;
  for (int j = 0; j < doublewidth / 4) { /* 假设一次处理每行相邻4个YUYV信息 */
    // 移动Y信息
    // 计算新的UV信息

    src += 4; /* 更新内循环的指针 */
    dst -= 4;
  }
  src_y += doublewidth; /* 更新外循环的指针 */
  dst_y -= doublewidth;
}
```

---

## 完善算法以支持不规则尺寸图像的旋转

上述算法仅是一个基础版本，对于不规则的行列数的图像是不支持的，为了更全面，我们需要算法程序具有以下功能：

1. 

### RGB与YUV的相互转换

RGB的原理是三者的组合可以构成任何颜色，用三个0~255的数构建一个像素的信息。然而，YUV更加符合人类视觉的习惯。

> 人类大脑最先感知的是亮度。

依据这个理论，结合人眼对于不同颜色的灵敏度相异，采用YUV的模型也能够更好反映色彩。三个值有不同的含义：

注：上述转换标准是SDTV with BT.601（ITU-R Recommendation BT.601）所规范的，是较为被普遍使用的一种，当然还有其他规范。依旧是线性变换，但区别在于具体参数不同。

### 数值近似

在众多的SIMD中，受限于计算性能要求，并不直接采用上述浮点数计算，而是用整数替代，2次幂的乘法、除法则用左移\\(\ll\\)、右移\\(\gg\\)来实现。

**RGB --> YUV**

$$
\begin{bmatrix}
{Y}'\\ U\\ V
\end{bmatrix}=
\begin{bmatrix}
66 & 129 & 25\\ -38 & -74 & 112\\ 112 & -94 & -18
\end{bmatrix}
\begin{bmatrix}
R\\ G\\B
\end{bmatrix}
$$

$$
\begin{matrix}
{Yu}'= \left ( \left ( {Y}'+128 \right )\gg 8 \right ) + 16\\ 
Uu = \left ( \left ( U + 128 \right )\gg 8 \right ) + 128\\ 
Vu = \left ( \left ( V + 128 \right )\gg 8 \right ) + 128
\end{matrix}
$$

**YUV --> RGB**

$$
\begin{matrix}
C = Y - 16\\ 
D = U - 128\\ 
E = V - 128
\end{matrix}
$$

$$
\begin{matrix}
R = clamp\left ( \left ( 298 \times C \qquad \qquad \quad + 409 \times E + 128  \right ) \gg 8 \right )\\ 
G = clamp\left ( \left ( 298 \times C - 100 \times D - 208 \times E + 128 \right ) \gg 8 \right )\\ 
B = clamp\left ( \left ( 298 \times C + 516 \times D \qquad \qquad \quad + 128 \right ) \gg 8 \right )
\end{matrix}
$$

注：\\(clamp\left (  \right ) \\)指将括号内数值整形至0~255之间。

**需要注意的几点**

- RGB --> YUV 转换无需整形至0~255，而 YUV --> RGB 需要；
- RGB --> YUV 转换时，U、V最后加上128目的是使其处于0~255之间，便于计算机采用8比特uchar计数，否则会有正有负。Y加16也是类似的考虑。
- 可观察到右移8位前，都会加128，这是考虑了四舍五入。因为右移8位即除以\\( 2^{8} = 256 \\)，而加上\\( 2^{7} = 128 \\)，故进行了四舍五入。这是一种惯用的ground方法，应该从二进制的视角来看。


## 更加省时的图像旋转

### YUV

YUV基本概念在上节已述，但实际上人眼对于3个8比特uchar表示1个像素的图片的感知，与略微降低采样频率后的图像相差无几。因此为了精简数据，有不同程度的采样比。若做图像色彩空间相互转换，实质肯定是有损的，但是依旧由于人眼对于这种细微差别的不敏感，采样后的图像经过转换回RGB，还是很不错的。

- **YUV444** 指平均4个像素采样4个Y、4个U、4个V；
- **YUV422** 指平均4个像素采样4个Y、2个U、2个V；
- **YUV420** 指平均4个像素采样4个Y、1个U、1个V；

下面以YUV422I为例，说明采样排列格式。这是一种在Android上经常遇到的图像格式。

YUV422I是YUV422采样中的一种格式，又被称为YUY2、YUNV、YUYV、V422，具体命名及格式请见[YUV pixel formats](http://www.fourcc.org/yuv.php)。不同的YUV422格式采样方式都是4:2:2，但是排列格式相互不同。比如YUV422I在编码时，在图像的每一行的每两个像素的维度上进行，这两个像素被称为**MacroPixel**。采样时保留这两个像素的亮度信号，记为Y1、Y2，但只保留第一个像素的U、V当做这个MacroPixel的U、V，记为U1、V1。排列时以`Y1 U1 Y2 V1`为准。请参见下图。

![img](/img/in-post/yuv.gif)

---