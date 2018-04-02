#pragma once

#include <memory>
#include <vector>

// A hook that's called on gradients

namespace torch { namespace autograd {

struct Variable;
using variable_list = std::vector<Variable>;

// prehook 在：执行 Fucntion 计算之前进行的操作
struct FunctionPreHook {
  virtual ~FunctionPreHook() {}
  virtual variable_list operator()(const variable_list& grads) = 0;
};

// posthook: 计算了 Function 之后，grad_input，正向输入的梯度， grad_output，正向输出的梯度
struct FunctionPostHook {
  virtual ~FunctionPostHook() {}
  virtual variable_list operator()(const variable_list& grad_input, const variable_list& grad_output) = 0;
};

}} // namespace torch::autograd
