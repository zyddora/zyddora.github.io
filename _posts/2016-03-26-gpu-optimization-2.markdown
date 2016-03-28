---
layout:     post
title:      "GPU程序优化（二）——矩阵转置程序优化实例"
subtitle:   ""
date:       2016-03-26 13:46:00
author:     "Orchid"
header-img: "img/post-bg-cuda.jpg"
tags:
    - GPU
    - 并行程序优化
---

> 本文是**GPU并行程序优化**系列的第二部分，以矩阵转置为例，探究其在GPU上的程序优化过程。优化的基本原则请参加“GPU程序优化（一）——基本概念”一文。建议有一定的CUDA编程基础，再来阅读文本。

矩阵转置的例子，虽然简单，但涵盖了很多内容，更是对一些优化原则的实际运用，加深对其印象。

### Catalog

1. [矩阵转置程序目的](#section)
2. [C代码实现——串行代码](#c)
3. [第1个cuda版本实现——简单GPU代码](#cudagpu)
4. [第2个cuda版本实现——对行并行化](#cuda)
5. [第3个cuda版本实现——对各元素并行化](#cuda-1)


## 矩阵转置程序目的

输入：NxN维的原矩阵
输出：NxN维的转置后矩阵
目的：将原矩阵的(i, j)元素对应转置后的(j, i)输出。

注：为简便起见，将矩阵维数设为NxN。i->列，j->行，将矩阵按照行优先的顺序读取。

---

## C代码实现——串行代码

普通C代码很容易得到：

```cpp
#include <stdio.h>
#include <stdlib.h>

const int N = 1024; // matrix size is NxN

void transpose_CPU(float in[], float out[])
{
  for (int j = 0; j < N; j++) 
    for (int i = 0; i < N; i++) 
      out[j + i * N] = in[j + i * N];
}

int main(int argc, char **argv)
{
  int numbytes = N * N * sizeof(float);
  float *in = (float *)malloc(numbytes);
  float *out = (float *)malloc(numbytes);

  fill_matrix(in); // extra fill in matrix function, not listed here
  transpose_CPU(in, out);

  return 0;
}
```

这是CPU上运行的简单代码。此处假设为1024x1024维的矩阵，选择较大的数据有助于突显不同代码的性能差别。

---

## 第一个cuda版本实现——简单GPU代码

```cpp
#include <stdio.h>
#include <stdlib.h>
#include "gnutimer.h"

__global__ void transpose_serial(float in[], float out[])
{
  for (int j = 0; j < N; j++) 
    for (int i = 0; i < N; i++) 
      out[j + i*N] = in[j + i*N];
}

int main(int argc, char **argv)
{
  float *in = (float *)malloc(numbytes); // on Host
  float *out = (float *)malloc(numbytes);
  float *d_in, *d_out; // on Device

  cudaMalloc(&d_in, numbytes);
  cudaMalloc(&d_out, numbytes);
  cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);

  GpuTimer timer;
  timer.Start();
  transpose_serial<<<1, 1>>>(d_in, d_out); // launch kernel
  timer.Stop();

  cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);

  printf("transpose_serial: %g ms.\n", timer.Elapsed());
}
```

本代码仅是将CPU上C语言代码简单拷贝到了GPU上执行，较CPU代码增加了CPU与GPU传递数据的语句。

而且可以注意到，**调用kernel时仅用了一个单线程**，并未采用GPU上的多线程并行化处理。这里加入测试运行时间的代码，通过实测可以发现对于N较小的矩阵运行可以达到预期，但若N较大，则花费时间将大大增加。这说明如果在GPU上仅用类似于上述代码，则有些“大材小用”，没有发挥GPU并行计算的作用。

---

## 第2个cuda版本实现——对行并行化

接下来需要思考，如何对上述代码进行并行化？按照APOD的思路，恭喜！我们进行到了P (Parallelize)的阶段。

首先，不妨尝试对矩阵的每一行做并行化处理。

因此，对于i->列，j->行来说，i将会由线程ID代替，内部循环将在多个不同线程上运行。下面给出对每行设置多线程的代码。

```cpp
__gloabl__ void transpose_parallel_per_row(float in[], float out[])
{
  int i = threadIdx.x;

  for (int j = 0; j < N; j++)
    out[j + i*N] = in[j + j*N];
}
```

用`transpose_parallel_per_row<<<1,N>>>(d_in, d_out)`调用该新kernel函数，看看时间运行效果，可以发现有了一定的提升。这说明我们优化的方向是正确的。^-^  **为方便起见，在下文中给出每个cuda代码的具体运行时间对比，此处暂不列出。**

---

## 第3个cuda版本实现——对各元素并行化

既然对行并行化得到一定的性能提升，不妨再继续增大并行化程度——对每个元素启用并行化。

若一个线程块最多启动1024个线程，建议用多个线程块，每个线程块的维度是K*K个线程。那么矩阵将被分成片状进行转置处理，每片对应到一个线程区块中。

```cpp
const int K = 16; // tile size is K*K

__global__ void transpose_parallel_per_element(float in[], float out[])
{
  int i = blockIdx.x * K + threadIdx.x; // column
  int j = blockIdx.y * K + threadIdx.y; // row

  out[j + i*N] = in[i + j*N];
}
```

调用kernel函数的语句：

```cpp
dim3 blocks(N/K, N/K); // blocks per grid
dim3 threads(K, K); // threads per blocks

transpose_parallel_per_element<<<blocks, threads>>>(d_in, d_out);
```

这里假设N是K的整数倍。注意到与上例不同的是，此次要计算i和j的具体值，因为它们与区块索引`blockIdx.x/y`、线程索引`threadIdx.x/y`相关。经过测试，用时又一次下降，说明增加并行性确实有助于代码运行的效率提升。

但一味增加并行度是否一定有正面有效果呢？

事实证明，将并行性最大化并非总能带来最好的性能。相反地，每个线程做更多的工作反而更好些，这会引出现新的优化技术——粒度粗化 (granularity coarsening)。后面再具体探讨。

---