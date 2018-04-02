#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/jit/tracer.h"

#include "torch/csrc/autograd/variable.h"

#include <sstream>

namespace torch { namespace autograd {

// 注意一下类型， inputs 是 variable， outputs 是 Tensor
// 这个函数 构建起了一个 反向传导 计算图
// 即使 输入的 Variable requires_grad 都为 False， 也会记录反向传导图。
variable_list wrap_outputs(const variable_list& inputs, tensor_list&& outputs,
                           function_constructor ctr) {

  // 使用 inputs variables 来计算 反向传导的 Function 的 flag (f.is_executable, f.is_volatile)和 next_functions
  // 这里需要搞清楚的一点是：inputs 是前向传导的 inputs，从它可获得的信息有：当前 函数的 反向传导函数 是否 可执行， next_functions 是什么？？？

  auto flags = Function::flags(inputs);
  variable_list result;

  // 开始创建 返回的 Variable 了。
  result.reserve(outputs.size());
  
  if (flags.is_volatile) {  // 如果 is_volatile=true, 那么输出的 Variable 的 is_volatile=true 
    for (auto& output : outputs) {
      if (output.defined()) { // 因为 可能返回 None 嘛，所以这里 check 一下
        result.emplace_back(make_variable(output, false, true)); // requires_grad=false, is_volatile=true
      } else {
        result.emplace_back(); 
      }
    }
  } else {  // 如果 volatile=false， 难道也不管 is_executable 了吗？ 
    // ctr 是一个 lambda 函数， 它返回一个 std::shared_ptr<GradFn>
    // 梯度 使用 Function::flags 计算出来的 flags 其实是给 Backward 用的。
    auto grad_fn = ctr(std::move(flags));  // 用 flags(is_executable, is_volatile) 创建出来一个 Function。
    for (auto& output : outputs) {
      if (output.defined()) {
        result.emplace_back(make_variable(output, grad_fn));
      } else {
        ++grad_fn->num_inputs;
        result.emplace_back();
      }
    }
  }
  return result;
}

void check_input_variables(const char* name, const variable_list& inputs, int args, int required_args) {
  if (required_args == -1) {
    required_args = args;
  }
  if (inputs.size() != (size_t)args) {
    std::stringstream ss;
    ss << name << ": expected " << args << " arguments (got " << inputs.size();
    ss << ")";
    throw std::runtime_error(ss.str());
  }
  for (int i = 0; i < required_args; ++i) {
    if (!inputs[i].defined()) {
      std::stringstream ss;
      ss << name << ": expected Variable at argument " << i << " (got None)";
      throw std::runtime_error(ss.str());
    }
  }
}

}}
