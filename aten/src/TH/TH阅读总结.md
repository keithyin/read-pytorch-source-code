# TH阅读总结

* `generic/` 文件夹下写的是模板，然后通过宏替换的方式达到 C++ 模板编程的效果

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

