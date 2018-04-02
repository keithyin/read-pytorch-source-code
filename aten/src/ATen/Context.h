#pragma once

#include <memory>
#include <mutex>
#include "ATen/ATenGeneral.h"
#include "ATen/Generator.h"
#include "ATen/Type.h"
#include "ATen/Utils.h"

struct THCState;

namespace at {

/*
1. 检查系统是否有 CPU
2. 保存着 THCstate
3. 
*/


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

  // 注册的 type？？？
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

}
