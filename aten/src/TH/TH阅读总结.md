# TH阅读总结

* `TH/generic/` 文件夹下写的是模板，然后通过宏替换的方式达到 C++ 模板编程的效果
* `TH/` 中的文件大部分是用来 通过模板生成特定类型代码的文件。

## 工具宏总结

**文件 TH/THTensor.h 中**

* `#define THTensor          TH_CONCAT_3(TH,Real,Tensor) ` 生成Token `THRealTensor`， class 的 Token(名字)。

* `#define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)` 生成 Token `THRealTensor_NAME`， 函数的 Token(名字)。


* `generic/THTensor.h` : 一些 shape，resize，创建操作。
* `THGeneral.h` : 一些 `Check` 操作。

**关于C的宏**
```c
#define NUM you
#define VECTOR NUM##1
// 这里需要注意的是，由于 ##，VECTOR 中的 NUM 是不会被替换成 you 的
// VECTOR 会替换成 NUM1
```

```c
#define NUM you
#define MID(A) A##1
#define VECTOR MID(NUM)

// 这里 VECTOR 也会被替换成 NUM1
```

```c
#define MID_(A) A##1
#define MID(A) MID_(A)
#define VECTOR MID(NUM)

#define NUM you
// 这时候 VECTOR 就能够被替换成 you1 了
```
所以，`THTensor` 宏是下面这样的
```c
#define TH_CONCAT_3_EXPAND(x,y,z) x ## y ## z  //在 THGeneral.h 中
#define TH_CONCAT_3(x,y,z) TH_CONCAT_3_EXPAND(x,y,z)   //在 THGeneral.h 中
#define THTensor TH_CONCAT_3(TH,Real,Tensor) //在 THTensor.h 中

// 之后通过设置 Real 个宏的值就能得到不同的 THTensor 真实代表的 Token了。
// THTensor_(NAME) 也是差不多一个德行
```

## 几个重要的结构体

```c
typedef struct THTensor
{
    int64_t *size; // 表示 shape
    int64_t *stride; // stride， 假设 size 为 [3, 2, 1, 4], 那么 stride 为 [8, 4, 4, 1]
                     // 如果 数据不是连续的， stride 就是另一种情况了。详见预备知识1
    int nDimension; // 表示有几维

    // Note: storage->size may be greater than the recorded size
    // of a tensor
    THStorage *storage;
    ptrdiff_t storageOffset; // Tensor_data的起始地址 TENSOR_data = TENSOR->storage->data+TENSOR->storageOffset
    int refcount;

    char flag;

} THTensor;
```

**stride**
由于数据实际上是在一维空间中存放着的，stride[d] 表示的意思是，如果 `d`-维 的 值增加 1,对应 在一维空间的步长是多少。
假设 a 是一个 2 维矩阵，size 为 `[2,3]`, 存放的值是 `[[1,2,3], [5,6,7]]`, 这些值在内存中实际上是 `[1,2,3,5,6,7]` 存储的。
a 的 stride 是 `[3, 1]`, 假设 a[0,1] 在内存中的位置 `pos1`, 根据 stride，a[0,2] 在内存中的位置是 `pos1+1`，a[1,1] 在内存中的位置是 `pos1+3`.

## TensorApply
如何使用

```c
void THTensor_(add)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
    TH_TENSOR_APPLY2_CONTIG(real, r_, real, t, THVector_(adds)(r__data, t_data, value, r__len););
  } else {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data + value;);
  }
}

// TH_TENSOR_APPLY2(值1类型, 值1, 值2类型, 值2, 对单个值定义的运算)， t_data 是 t中的某一个值
```