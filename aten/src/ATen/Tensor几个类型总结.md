# A TENsor library 总结


# 几个类型总结

## Tenosr TensorBase TensorImpl Type Storage

* Type: 表示数据类型, CPUFloatType
* Storage ： 存数据的地方, CPUFloatStorage
* TensorImpl : 有一个 引用计数, （CPUFloatTensor，等等数据类型，都继承了 TensorImpl）
* TensorBase : 操作 TensorImpl 的引用计数
* Tensor : 上层建筑,  Tensor

> 每个数据类型都会有一个 Type，一个 Storage， 一个 Tensor， 例如： CPUFloatType, CPUFloatStorage, CPUFloatTensor， 等等。。。


Tensor 继承 TensorBase; 
TensorBase 包含 TensorImpl; 
TensorImpl 包含 （Type, 引用计数); （pytorch 中的所有类型都继承 TensorImpl）
THFLoatTensor 指向 Storage！！！！！，
Type 中包含 Context* 全局唯一： Context 中注册这 所有的 Type。Type 中包含了这个 Type 的Tensor能用的 方法。

```c++
//      doc/tensor.h
struct Tensor : public detail::TensorBase {
  Tensor() : TensorBase() {}
  Tensor(TensorImpl * self, bool retain) : TensorBase(self, retain) {}
  Tensor(const TensorBase & rhs) : TensorBase(rhs) {}
  Tensor(const Tensor & rhs) = default;
  Tensor(Tensor && rhs) noexcept = default;
```

```c++
// TensorBase.h, 专门用来处理 TensorImpl 中的 引用计数的（reference counting） 的。
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
  TensorBase(const TensorBase & rhs)
  : pImpl(rhs.pImpl) {
    if (pImpl != UndefinedTensor::singleton())
      pImpl->retain();
  }
  TensorBase(TensorBase && rhs) noexcept
  : pImpl(rhs.pImpl) {
    rhs.pImpl = UndefinedTensor::singleton();
  }
  ~TensorBase() {
    if (pImpl != UndefinedTensor::singleton())
      pImpl->release();
  }

public:
  // 一个 TensorImpl* 属性！！！！！！！！！！！
  TensorImpl * pImpl;
};

```

```c++
// TensorImpl.h ， 保存一个引用计数。如果引用计数为 0, 则销毁这个对象。
struct TensorImpl {
  explicit TensorImpl(Type * type)
  :  refcount(1), is_scalar(false), type_(type) {}

  Type & type() const {
    return *type_;
  }
  void retain() {
    ++refcount;
  }
  virtual void release() {
    if(--refcount == 0) {
      delete this;
    }
  }
  virtual ~TensorImpl() {}
  friend struct Type;

private:
  // TensorImpl 一个 refcount， 一个 is_scalar, 一个 type_
  std::atomic<int> refcount;
  bool is_scalar;
  Type * type_;
};
```

```c++
// pytorch 中的所有数据类型 (CPUFloat, CPUDouble, GPUFloat) 都继承 TensorImpl
// 这个文件 编译 Aten 后，在 build/src/Aten/Aten 中可以找到。
struct CPUFloatTensor final : public TensorImpl {
public:
  explicit CPUFloatTensor(Context* context);
  CPUFloatTensor(Context* context, THFloatTensor * tensor);
  virtual ~CPUFloatTensor();
//TODO(zach): sort of friend permissions later so this
// can be protected
public:
  THFloatTensor * tensor; // 
  Context* context; // 本机环境
  friend struct CPUFloatType;
};
```

```c++
// 在 TH/generic/THTensor.h, 表示 Tensor 的基本属性。预处理时这个会变成 THFloatTensor, THDoubleTensor 这种结构体。
typedef struct THTensor
{
    int64_t *size;
    int64_t *stride;
    int nDimension;

    // Note: storage->size may be greater than the recorded size
    // of a tensor
    THStorage *storage;
    ptrdiff_t storageOffset;
    int refcount;

    char flag;

} THTensor;

```

```c++

// 指向分配的空间。
typedef struct THStorage
{
    real *data;
    ptrdiff_t size;
    // 引用计数
    int refcount;
    char flag;  // TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
    // 用来 给 此 Storage 分配空间的类
    THAllocator *allocator;
    void *allocatorContext;
    struct THStorage *view;
} THStorage;
```



**Type可以**：

* 根据 *data 创建 storage 和 Tensor！！
* type 如何创建的 Tensor 嘞。Type 中有很多创建 Tensor 的方法，调用之会返回 Tensor 对象。



## Context
> 在 Aten/Context.h 文件中。

包含了什么属性：

* type_registry : 当前系统，所有的类型都注册在这里。（（这里面的值是由 Type 给注册进去的）
* generator：当前系统，所有 generator 都注册在这里
* THCState : 用来记录当前的 状态

这个类看样子是 用来表示当前 系统的 的环境的。是单例 的。

## Backend

用来表示后端的
```c++
enum class Backend {
  CPU,
  CUDA,
  SparseCPU,
  SparseCUDA,
  Undefined,
  NumOptions
};
```
## Generator
这个应该是随机数生成器。
