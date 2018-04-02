#include "basic_ops.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/utils/auto_gpu.h"

namespace torch { namespace autograd {

auto Error::apply(const variable_list& grad_outputs) -> variable_list {
  throw std::runtime_error(msg);
};

auto DelayedError::apply(const variable_list& inputs) -> variable_list {
  tensor_list outputs;
  outputs.reserve(inputs.size());
  for (auto& var : inputs) {
    // FIXME: share version counters
    outputs.emplace_back(var.defined() ? var.data() : Tensor());
  }
  return wrap_outputs(inputs, std::move(outputs), [&](FunctionFlags f) {
    return std::make_shared<Error>(msg, std::move(f));
  });
};

// Add function, 可以看到 ，apply 的时候，是 variable_list 作为输入， variable_list 作为输出。
auto Add::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("Add", inputs, 2);
  auto& input1 = inputs[0].data();
  auto& input2 = inputs[1].data();
  AutoGPU guard(input1);

  at::Tensor output;
  if (input1.type().isSparse()) {
    output = input2 + input1;
  } else {
    output = input1 + input2;
  }
  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<AddBackward_Deprecated>(std::move(f));
  });
};

auto AddBackward_Deprecated::apply(const variable_list& grad_outputs) -> variable_list {
  check_input_variables("AddBackward_Deprecated", grad_outputs, 1);
  return {grad_outputs[0], grad_outputs[0]};
};

auto Mul::apply(const variable_list& inputs) -> variable_list {
  // 这里在执行前向计算
  check_input_variables("Mul", inputs, 2);
  AutoGPU guard(inputs[0]);
  auto& input1 = inputs[0].data();
  auto& input2 = inputs[1].data();

  auto output = input1 * input2;
  // 这里在记录反向传导图！！！！
  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<MulBackward>(std::move(f));
  });
};

auto MulBackward::apply(const variable_list& grad_outputs) -> variable_list {
  check_input_variables("MulBackward", grad_outputs, 1);
  throw std::runtime_error("MulBackward::apply not implemented");
};

}} // namespace torch::autograd


// 前向过程可以总结为：
// 1. 创建一个 Function 对象
// 2. 执行 对象函数调用，variable_list 作为实参
// 3. 将 variable_list 中的 tensor 取出来，对其进行运算得到 Tensor outputs
// 4. 将 outputs 包装成 Variable， 并创建 反向计算 的 Function。
