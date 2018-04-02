# ATEN 阅读笔记

**THNN/init.c** 里面有很多 `Check` 宏定义。

## 预备知识 ##
阅读ATEN库时首先需要对低层Tensor的一些基本知识进行熟悉，比如最最重要的strided indexing scheme，详见预备知识1，了解了这个策略，对后面理解一些Tensor的数据结构、操作和加速就方便多了

## TH

* `#define THTensor          TH_CONCAT_3(TH,Real,Tensor) ` 生成Token `THRealTensor`， class 的 Token(名字)。
* `#define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)` 生成 Token `THRealTensor_NAME`， 函数的 Token(名字)。

* `generic/THTensor.h` : 一些 shape，resize，创建操作。
* `THGeneral.h` : 一些 `Check` 操作。



**TH中的几个重要Struct**

```c
typedef struct THTensor
{
    int64_t *size; // 表示 shape
    int64_t *stride; // stride， 假设 size 为 [3, 2, 1, 4], 那么 stride 为 [ 8, 4, 4, 1], outmost 到 innermost 的步长。
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

```c
// 指向分配的空间。
typedef struct THStorage
{
    real *data;
    ptrdiff_t size; // 空间大小。
    // 引用计数
    int refcount;
    char flag;  // TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
    // 用来 给 此 Storage 分配空间的类
    THAllocator *allocator;
    void *allocatorContext;
    struct THStorage *view;
} THStorage;
```

**real表示任何实数类型，Float，Int， Byte， Double**

```c
typedef struct THAllocator {
  void* (*malloc)(void*, ptrdiff_t);
  void* (*realloc)(void*, void*, ptrdiff_t);
  void (*free)(void*, void*);
} THAllocator;
```


## THNN

* `#define THNN_(NAME) TH_CONCAT_3(THNN_, Real, NAME) ` 生成 Token `THNN_RealNAME`
* `THNN_CHECK_SHAPE(I1, I2)` 检查 `I1,I2` 是不是形状相同。不相同会 报错。 
* `THNN.h` 中包含 所有 `NN` 方法的声明。


**几个疑问**

* accreal 是什么鬼 ： 在 TH/THGenerate**Type.h 中可以找到答案，就是某种 real 类型。
* THNNState* 是个空指针, 这玩意是用来干嘛的。

## THC

* #define THCTensor          TH_CONCAT_3(TH,CReal,Tensor)  生成 Token `THCRealTensor` , class 的 Token(名字)。
* #define THCTensor_(NAME)   TH_CONCAT_4(TH,CReal,Tensor_,NAME)， 生成 Token `THCRealTensor_NAME` , 函数的 Token(名字).

**几个重要类型**
```c
typedef struct THCTensor
{
    int64_t *size;
    int64_t *stride;
    int nDimension;

    THCStorage *storage;
    ptrdiff_t storageOffset;
    int refcount;

    char flag;

} THCTensor;
```

```c
typedef struct THCStorage
{
    real *data;
    ptrdiff_t size;
    int refcount;
    char flag;
    THCDeviceAllocator *allocator;
    void *allocatorContext;
    struct THCStorage *view;
    int device;
} THCStorage;
```

```c
struct THCState {
  struct THCRNGState* rngState;
  struct cudaDeviceProp* deviceProperties;
  /* Set of all allocated resources. resourcePerDevice[dev]->streams[0] is NULL,
     which specifies the per-device default stream. blasHandles and
     sparseHandles do not have a default and must be explicitly initialized.
     We always initialize 1 blasHandle and 1 sparseHandle but we can use more.
  */
  THCCudaResourcesPerDevice* resourcesPerDevice;
  /* Captured number of devices upon startup; convenience for bounds checking */
  int numDevices;
  /* Number of Torch defined resources available, indices 1 ... numStreams */
  int numUserStreams;
  int numUserBlasHandles;
  int numUserSparseHandles;

  /* Allocator using cudaMallocHost. */
  THAllocator* cudaHostAllocator;
  THAllocator* cudaUVAAllocator;
  THCDeviceAllocator* cudaDeviceAllocator;

  /* Index of the current selected BLAS handle. The actual BLAS handle used
     depends on the current device. */
  THCThreadLocal/*<int>*/ currentPerDeviceBlasHandle;
  /* Index of the current selected sparse handle. The actual sparse handle used
     depends on the current device. */
  THCThreadLocal/*<int>*/ currentPerDeviceSparseHandle;
  /* Array of thread locals containing the current stream for each device */
  THCThreadLocal* currentStreams;

  /* Table of enabled peer-to-peer access between directed pairs of GPUs.
     If i accessing allocs on j is enabled, p2pAccess[i][j] is 1; 0 otherwise. */
  int** p2pAccessEnabled;

  /* Is direct cross-kernel p2p access allowed? Normally, only cross-GPU
     copies are allowed via p2p if p2p access is enabled at all for
     the pair of GPUs in question, but if this flag is true, then
     all cross-GPU access checks are disabled, allowing kernels to
     directly access memory on another GPUs.
     Note that p2p access must exist and be enabled for the pair of
     GPUs in question. */
  int p2pKernelAccessEnabled;

  void (*cutorchGCFunction)(void *data);
  void *cutorchGCData;
  ptrdiff_t heapSoftmax;
  ptrdiff_t heapDelta;
};

```


## THCUNN




## ATen关键类型总结

* Tensor
* TensorBase Tensor的一个基类，主要用来 处理 reference counting

```c++
struct TensorBase {
  TensorBase(): TensorBase(UndefinedTensor::singleton(), false) {}
  TensorBase(TensorImpl * self, bool retain)
  : pImpl(self) {
    if (pImpl == nullptr) {
      throw std::runtime_error("TensorBase with nullptr not supported");
    }
    if(retain && pImpl != UndefinedTensor::singleton())
      pImpl->retain();
  }

  friend struct Type;

  //TODO(zach): sort out friend structes
public:
  // 一个 TensorImpl* 属性！！！！！！！！！！！
  TensorImpl * pImpl;
};
```

* TensorImpl : 实现特定类型需要继承的类 `CPUFloatTensor` 等等都得继承这个类。

```c++
struct TensorImpl {
  explicit TensorImpl(Type * type)
  :  refcount(1), is_scalar(false), type_(type) {}

  Type & type() const {
    return *type_;
  }
private:
  // TensorImpl 一个 refcount， 一个 is_scalar, 一个 type_
  std::atomic<int> refcount;
  bool is_scalar;
  Type * type_;
};
```

* Type : 

```c++
struct AT_API Type {
  explicit Type(Context * context)
  : context(context) {}
protected:
  // 只有一个 Context 属性
  Context* context;
```

* Context ： 进程期间就只有一个 Context 对象（单例模式）

```c++
class AT_API Context {
public:
  Context();
  // 获取Type 的工具方法
  // 根据 Backend 和 标量类型 获得 Type， Type 不就是 标量类型 加 Backend 吗
  Type & getType(Backend p, ScalarType s) {
    initCUDAIfNeeded(p);
    auto & type = type_registry[static_cast<int>(p)][static_cast<int>(s)];

    if(!type) {
      // there is only a single Undefined Type.
      if (p == Backend::Undefined || s == ScalarType::Undefined) {
        auto & undef = type_registry[static_cast<int>(Backend::Undefined)][static_cast<int>(ScalarType::Undefined)];
        if (undef) return *undef;
      }
      runtime_error("%s%sType is not enabled.",toString(p),toString(s));
    }
    return *type;
  }
  Generator & defaultGenerator(Backend p) {
    initCUDAIfNeeded(p);
    auto & generator = generator_registry[static_cast<int>(p)];
    if(!generator)
      runtime_error("%s backend type not enabled.",toString(p));
    return *generator;
  }
  bool hasCUDA() const;
  // defined in header so that getType has ability to inline
  // call_once check. getType is called fairly frequently
  THCState* lazyInitCUDA() {
    std::call_once(thc_init,[&] {
      doInitCUDA();
    });
    return thc_state;
  }
  ~Context();

  // 注册的 generator ？？？？
  // 一维数组： 生成器的数量
  std::unique_ptr<Generator>
    generator_registry[static_cast<int>(Backend::NumOptions)];

  // 系统的所有 Type 保存在这里。
  // type_registry[numBackend][numOptions]; 是个二维数组
  std::unique_ptr<Type> type_registry
    [static_cast<int>(Backend::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)];
  
  THCState * thc_state;
private:
  void initCUDAIfNeeded(Backend p) {
    if(p == Backend::CUDA)
      lazyInitCUDA();
  }
  void doInitCUDA();
  std::once_flag thc_init;
};

// 里面有一个静态 对象 Context。
AT_API Context & globalContext();

static inline void init() {
  globalContext();
}

// 通过 Backend 和 ScalarType 获取 Type。。。
static inline Type& getType(Backend p, ScalarType s) {
  return globalContext().getType(p,s);
}

// 用来获取类型，类型用来创建 Tensor， 666
static inline Type& CPU(ScalarType s) {
  return getType(Backend::CPU, s);
}

static inline Type& CUDA(ScalarType s) {
  return getType(Backend::CUDA, s);
}

static inline bool hasCUDA() {
  return globalContext().hasCUDA();
}
```

一般的创建 Tensor 过程：

```c++
Tensor d = CPU(kFloat).ones({3, 4}); 
// CPU(kFloat) 获取 Type 对象，然后由 Type 对象创建
// kFloat 是常量表达式。是个 ScalarType 对象。 在 ScalarType.
```


## ATEN 编译后

按照官网所述编译过程执行一遍，在 `ATen/build/src/ATen/Aten` 中会生成以下文件:
```
CUDAFloatStorage.cpp 
CUDAFloatStorage.h
CUDAFloatTensor.cpp
CUDAFloatTensor.h
CUDAFloatType.cpp
CUDAFloatType.h
```

**Tensor**

```c++
struct CPUFloatTensor final : public TensorImpl {
public:
  explicit CPUFloatTensor(Context* context);
  CPUFloatTensor(Context* context, THFloatTensor * tensor);
  virtual ~CPUFloatTensor();

public:
  THFloatTensor * tensor;
  Context* context;
  friend struct CPUFloatType;
};
```

**Storage**

```c++
struct CPUFloatStorage : public Storage {
public:
  explicit CPUFloatStorage(Context* context);
  CPUFloatStorage(Context* context, THFloatStorage *wrapped);
  CPUFloatStorage(Context* context, std::size_t size);
  CPUFloatStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CPUFloatStorage();
protected:
  friend struct CPUFloatType;
  THFloatStorage *storage;
  Context* context;
};
```

**Type**

```c++
// 这里面有 CPUFloatType 支持所有运算。
struct CPUFloatType final : public Type {
  explicit CPUFloatType(Context* context); 
```

```c++
struct CUDAFloatStorage : public Storage {
public:
  explicit CUDAFloatStorage(Context* context);
  CUDAFloatStorage(Context* context, THCudaStorage *wrapped);
  CUDAFloatStorage(Context* context, std::size_t size);
  CUDAFloatStorage(Context* context,
    void * data, std::size_t size, const std::function<void(void*)> & deleter);
  virtual ~CUDAFloatStorage();
protected:
  friend struct CUDAFloatType;
  THCudaStorage *storage;
  Context* context;
};


struct CUDAFloatTensor final : public TensorImpl {
public:
  explicit CUDAFloatTensor(Context* context);
  CUDAFloatTensor(Context* context, THCudaTensor * tensor);

//TODO(zach): sort of friend permissions later so this
// can be protected
public:
  THCudaTensor * tensor;
  Context* context;
  friend struct CUDAFloatType;
};

struct CUDAFloatType final : public Type {
  explicit CUDAFloatType(Context* context);
```
