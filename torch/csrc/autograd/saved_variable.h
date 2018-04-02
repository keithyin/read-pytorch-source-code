#pragma once

#include <mutex>
#include <memory>
#include <functional>
#include <ATen/ATen.h>

#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/variable_version.h"
#include "torch/csrc/Types.h"

namespace torch { namespace autograd {

struct Function;

extern const char* ERR_BACKWARD_TWICE;


// 就是 把 Variable 的属性全都搞过来！！！！！！！！！！！
// 逻辑是这样：
// 用 Conv 举例： ConvForward 的输入， 以 SavedVariable 保存到 ConvBackward 中！！！！！！！！ saved_for ConvBackward
// 这时， variable.grad_fn 是不等于 saved_for 的
// 所以 SavedVariable 的 grad_fn 设置成 variable 的 grad_fn
// 如果 variable.grad_fn 与 saved_for 这种是什么情况呢？ 就是 计算 sofmax 导数的时候！！！！！！！！！！！！！！！
// SavedVariable 是 函数 中保存的 Variable， 用来计算 反向传导时的梯度的！！！！！！！！！！！！！！！！
// 保存起来，用来计算梯度的！！！！！！！！！！！！！！！！， 和 InputBuffer 的作用不同（用来 累积梯度的）。
struct SavedVariable {
  SavedVariable()
    : data()
    , has_grad_fn(false)
    , version()
    , requires_grad(false)
    , is_volatile(false)
    , expected_version(-1) {}

  SavedVariable(const Variable& variable, Function* saved_for);


  at::Tensor data;
  // The gradient function associated with this node. If has_grad_fn
  // is false, then this is a leaf node. Note that the grad_fn is not saved if
  // it would create a circular reference. In that case, the grad_fn must be
  // passed in to the unpack function when reconstructing the Variable.
  bool has_grad_fn;
  // 变量的 梯度函数， 
  std::shared_ptr<Function> _grad_fn;
  std::weak_ptr<Function> grad_accumulator;
  SavedVersion version;
  bool requires_grad;
  bool is_volatile;
  int expected_version;
  int output_nr;
  Variable base;
  std::unique_ptr<jit::tracer::ValueTracingState> tracing_state;

  Variable unpack(std::shared_ptr<Function> saved_for=nullptr) const;
  at::Tensor unpack_data(std::shared_ptr<Function> saved_for=nullptr) const;
};

}} // namespace torch::autograd
