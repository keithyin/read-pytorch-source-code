#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/utils/auto_gpu.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <typeinfo>
#include <sstream>
#include <TH/TH.h>

#ifdef WITH_CUDA
#include <cuda.h>
#include <THC/THC.h>
#endif

namespace torch { namespace autograd {

// NB: -1 indicates the CPU worker!
// -2 表示 NO_DEVIE
static constexpr int NO_DEVICE = -2;

// thread_local ： 线程生命周期，线程创建时存在，线程销毁时 销毁
// 这个变量会在每一个 使用它的 线程中创建一个副本，随着线程的 执行完毕 而销毁。
static thread_local int worker_device = NO_DEVICE;

// XXX: Changes to the way multithreading works in execute should be done with
// great care. Right now the implementation guarantees that a single function's
// apply will never be entered concurrently (even if multiple graphs are
// executed at the same time). Adding multiple threads per-device or removing
// engine thread affinity to the device can break this invariant, and we depend
// on it in a few places (e.g. AccumulateGrad function).

struct FunctionTask {
  // 每个 FunctionTask 中都维护着一个 base GraphTask
  GraphTask* base;
  std::shared_ptr<Function> fn; //这个是个反向求导函数，当前的
  // This buffer serves as an implicit "addition" node for all of the
  // gradients flowing here.  Once all the dependencies are finished, we
  // use the contents of this buffer to run the function.
  // buffer 是用来累积梯度的, 用来累积 fn 的输入梯度
  InputBuffer inputs; // 反向求导函数的输入，当前的

  FunctionTask(GraphTask* base, std::shared_ptr<Function> fn, InputBuffer inputs)
    : base(base)
    , fn(fn)
    , inputs(std::move(inputs)) {}
};

// 存放着 可以进行计算的 FunctionTask
struct ReadyQueue {
  // queue 是 FunctionTask 的 一个 双端队列
  std::deque<FunctionTask> queue;

  // std::condition_variable 条件变量，同步的时候会用到
  // 用 unique_lock (over mutex) 来进行操作
  // not_empty 是用来干什么的？
  std::condition_variable not_empty;
  std::mutex mutex;

  void push_front(FunctionTask item);
  FunctionTask pop_back();
};

struct GraphTask {
  std::exception_ptr exception;
  // Indicates if an error occurred while executing any task.  When this is
  // true, it signals all threads to stop executing.
  std::atomic_bool has_error;

  // 剩余 tasks， 在 ReadyQueue 的 push方法 中加一， 在 evaluate_function 中减一操作
  std::atomic<uint64_t> outstanding_tasks;
  bool keep_graph;
  bool has_any_work;
  // 用来 给 notify_all 加锁的
  std::mutex mutex;
  // Notified when a task finishes executing.  Check outstanding_tasks to see
  // if all tasks are done.
  std::condition_variable not_done;
  const Engine::pre_callback_map& pre_callbacks;
  const Engine::post_callback_map& post_callbacks;

  //用来存放没有 准备好的 FunctionTask
  std::unordered_map<Function*, InputBuffer> not_ready;
  std::unordered_map<Function*, int> dependencies;
  
  // 这个来 表示 GraphTask 是在哪个 device 上创建的
  int owner;

  GraphTask(bool keep_graph, const Engine::pre_callback_map& pre_callbacks, const Engine::post_callback_map& post_callbacks)
    : exception()
    , has_error(false)
    , outstanding_tasks(0)
    , keep_graph(keep_graph)
    , has_any_work(false)
    , mutex()
    , not_done()
    , pre_callbacks(pre_callbacks)
    , post_callbacks(post_callbacks)
    , not_ready()
    , dependencies()
    , owner(NO_DEVICE) {}
};

auto ReadyQueue::push_front(FunctionTask item) -> void {
  {
    // ReadyQueue 中 push 操作是线程安全的！！！， 即：queue 的push 操作是线程安全的
    std::lock_guard<std::mutex> lock(mutex);
    // -> 的优先级 大于 ++
    // GraphTask 记录了 outstanding_tasks，
    ++item.base->outstanding_tasks;
    queue.push_front(std::move(item));
  }

  // push 一次， 通知一个线程 来 pop 它
  not_empty.notify_one();
}

auto ReadyQueue::pop_back() -> FunctionTask {
  std::unique_lock<std::mutex> lock(mutex);
  // 只有 queue为 空，才会阻塞当前线程。 这部分代码也保证了 queue 的pop 线程安全！！！！
  not_empty.wait(lock, [this]{ return !queue.empty(); });
  auto task = std::move(queue.back()); queue.pop_back();
  return task;
}

Engine::Engine() : ready_queues() {
}

// This Engine's ReadyQueues and their corresponding threads are leaked here
Engine::~Engine() = default;

auto Engine::thread_init(int device) -> void {
  // 线程初始化是 用 nullptr 开 NUM_DEVICE 个 线程，准备操作 GraphTask
  // 设置一下线程个数？ 这个还不清楚到底到底干了些啥
  THInferNumThreads();

  //设置 GPU 的使用, device -1 代表 CPU， device 0 代表 GPU
  AutoGPU guard(device);

  // thread_init 的时候，给了 worker_device 值， 每个 thread 一个
  // worker_device: thread_local  类型，线程生存周期
  worker_device = device;
  
  // 开启线程， GraphTask 是一个 nullprt
  thread_main(nullptr);
}

// NOTE: graph_tasks do not necessarily form a stack. Imagine this
// case:
//
//    +----> Eval1
//  Root
//    +----> Eval2
//
// Once Root is executed, both Eval1 and Eval2 are added to the ready queue.
// Next, Eval1 is run and this causes the worker to enter thread_main again.
// Then, it pops the next task from the queue, but at this point it is Eval2.
// It enters thread_main once again, but now with graph_task of Eval2, which is
// completely unrelated to that of Eval1 (it's not a recursive call).
// It's all ok and is handled right now, but it should be accounted for
// in case this code is to be changed.

// 有几个 device， execute 中就开启了几个 此线程。
// 看了 Enginie::execute 代码，发现 NO_DEVICE 不会进入此方法
auto Engine::thread_main(GraphTask *graph_task) -> void {
  // worker_device 是 thread_local 类型的 变量
  // worker_device+1 是什么鬼，如果 CPU 的话， +1 就是 0, 
  // worker_device + 1 代表的是 worker_ 对应的 ReadyQueue
  // 从这里可以看出 ready_queues 中存的是 设备的 ready_queue
  auto queue = ready_queues[worker_device + 1];
  // 这里开始从 ReadyQueue中取出 FuncitonTask 开始操作
  // 1.DEVICE 线程在这做了些什么
  // 
  while (!graph_task || graph_task->outstanding_tasks > 0) {
    // 由于 DEVICE 线程是由 thread_start 开启的， 所以 DEVICE 线程中的 graph_task 为 nullptr
    // 所以 !graph_task 一直为 True， 这个循环一直在跑着。如果 pop 不出来了，就阻塞一下
    // DEVICE 线程在 生命周期是 进程的生命周期，并不是 一次 backward 的 生命周期
    // 如果 worker_device ！= NO_DEVICE， 则会执行以下操作
    FunctionTask task = queue->pop_back();
    if (task.fn && !task.base->has_error.load()) {
      try {
        evaluate_function(task);  // 计算 task，然后再把该放入 ReadyQueue 中的 的task 放进去
      } catch (std::exception& e) {
        thread_on_exception(task, e);
      }
    }

    // 取出 FunctionTask 的 base owner， 是 FunctionTask 所属 GraphTask 的 owner！
    auto base_owner = task.base->owner;

    // 简单情况下， base_owner 就是 NO_DEVICE
    // 这种情况下很简单，将 graph_task.outstanding_task -1 ,如果 为 0, 就 notify_all 一下， 告诉其它阻塞的伙计们，老子干完活了。
    // Task from a non-worker thread. Easy case.
    if (base_owner == NO_DEVICE) {
      if (--task.base->outstanding_tasks == 0) {

        // 给 notify_all 加个锁
        std::lock_guard<std::mutex> lock(task.base->mutex);
        // 给阻塞在 execute 中的伙计知会一声， 知会完之后， execute 就能够退出执行了。
        task.base->not_done.notify_all();
      }
    } else {
      // If it's a task initiated from this thread, decrease the counter, but
      // don't do anything - loop condition will do all checks for us next.
      if (base_owner == worker_device) {
        // 做完了一个 task， 所以 outstanding_tasks 减了一
        --task.base->outstanding_tasks;
      // Otherwise send a dummy function task to the owning thread just to
      // ensure that it's not sleeping. If it has work, it might see that
      // graph_task->outstanding_tasks == 0 before it gets to the task, but
      // it's a no-op anyway.
      } else if (base_owner != worker_device) {
        if (--task.base->outstanding_tasks == 0) {
          // Synchronize outstanding_tasks with queue mutex
          std::atomic_thread_fence(std::memory_order_release);
          ready_queue(base_owner).push_front(FunctionTask(task.base, nullptr, InputBuffer(0)));
        }
      }
    }
  }
}

auto Engine::thread_on_exception(FunctionTask& task, std::exception& e) -> void {
  std::lock_guard<std::mutex> lock(task.base->mutex);
  if (!task.base->has_error.load()) {
    task.base->exception = std::current_exception();
    task.base->has_error = true;
  }
}

// 从这可以看出， pre_hooks 和 post_hooks 都是注册在 Function 上的。

static variable_list call_pre_hooks(Function& fn, variable_list inputs) {
  for (auto& hook : fn.pre_hooks) {
    inputs = (*hook)(inputs);
  }
  return inputs;
}

static variable_list call_post_hooks(Function& fn, variable_list outputs, variable_list inputs) {
  for (auto& hook : fn.post_hooks) {
    outputs = (*hook)(outputs, inputs);
  }
  return outputs;
}

static variable_list call_function(FunctionTask& task) {
  auto& fn = *task.fn;
  auto inputs = call_pre_hooks(fn, InputBuffer::variables(std::move(task.inputs)));

  auto& pre_callbacks = task.base->pre_callbacks;
  for (auto it_p = pre_callbacks.equal_range(&fn); it_p.first != it_p.second; ++it_p.first) {
    auto& callback = it_p.first->second;
    if (!callback(&fn, inputs)) return variable_list(fn.next_functions.size());
  }

  auto outputs = fn(inputs);

  auto& post_callbacks = task.base->post_callbacks;
  for (auto it_p = post_callbacks.equal_range(&fn); it_p.first != it_p.second; ++it_p.first) {
    auto& callback = it_p.first->second;
    if (!callback(&fn, inputs, outputs)) return variable_list(fn.next_functions.size());
  }

  return call_post_hooks(fn, std::move(outputs), std::move(inputs));
}

// 计算 FuntionTask， 反向求导计算
auto Engine::evaluate_function(FunctionTask& task) -> void {
  // outputs， 是梯度的 输出
  auto outputs = call_function(task);

  auto& fn = *task.fn;
  if (!task.base->keep_graph) {
    // 如果不需要 keep graph 的话， 就直接 释放 变量就好了。
    fn.releaseVariables();
  }
  
  // 似乎这是反向求导的时候用的。。。。。。。奥奥， 明白了， pytorch 中的图是一个 反向求导图！！！
  // 判断当前 grad_fn 的返回值是否 和 next_functions 的个数一致， 如果不一致，则报错！！！
  if (outputs.size() != fn.next_functions.size()) {
    std::stringstream ss;
    ss << "Function '" << fn.name() << "' returned an invalid number of outputs - expected ";
    ss << fn.next_functions.size() << ", but got " << outputs.size();
    throw std::runtime_error(ss.str());
  }
  
  // 下面开始 对 next_function 做一些检查，看看他们是否已经 准备好计算了！！！
  int num_outputs = outputs.size();
  for (int i = 0; i < num_outputs; ++i) {
    auto& output = outputs[i];
    auto& next_fn = fn.next_functions[i].first;
    int input_nr = fn.next_functions[i].second;

    if (!next_fn) {
      continue;
    }

    // Stochastic functions are placed in the ready queue by
    // compute_dependencies, so we have to skip them here.
    if (next_fn->is_stochastic || !next_fn->is_executable) {
      continue;
    }

    // 这里用的是 GraphTask 的mutex， 整个图（BP图/反向传导图）都被锁住咯
    // 感觉 evaluation_function 可以并行啊， 这个函数可以并行操作啊！
    
    std::lock_guard<std::mutex> lock(task.base->mutex);
    /*
     给 GraphTask 加锁后干了些啥
     1. 将 next_functions 中的函数 放到 ReadyQueue 中还是 not_ready 中！！
    */

    // Check if the next function is ready to be computed
    // 当前这个 Function 被计算过后， 这个 function 的 next_functions 的 denpendencies 都会 -1
    bool is_ready = false;
    auto& dependencies = task.base->dependencies;

    // next_fn： 下一个要计算的反向求导函数。
    auto it = dependencies.find(next_fn.get());
    if (it == dependencies.end()) {
      auto name = next_fn->name();
      throw std::runtime_error(std::string("dependency not found for ") + name);
    } else if (--it->second == 0) {
      // 确定 next_function 是否准备好了
      dependencies.erase(it);
      is_ready = true;
    }
    
    // not_ready 放的是 没有准备好的 FunctionTask
    auto& not_ready = task.base->not_ready;
    // 看看这个 next_fn 是否已经在 base->not_ready 记录过
    auto not_ready_it = not_ready.find(next_fn.get());

    if (not_ready_it == not_ready.end()) {
      // 如果没有记录过，
      // No buffers have been allocated for the function
      InputBuffer input_buffer(next_fn->num_inputs);
      // 梯度累加, inputs_buffer 是 next_fn 的 输入 缓存区， 缓存好了就可以拿着输入去操作了
      input_buffer.add(input_nr, std::move(output));

      // 如果 ready 的话， 意味着 所有的 gradient 都已经累积完毕，可以进行下一步操作了。
      if (is_ready) {
        auto& queue = ready_queue(input_buffer.device());
        // push 到前面去 
        queue.push_front(FunctionTask(task.base, next_fn, std::move(input_buffer)));
      } else {
        not_ready.emplace(next_fn.get(), std::move(input_buffer));
      }
    } else {
      // The function already has a buffer
      auto &input_buffer = not_ready_it->second;
      input_buffer.add(input_nr, std::move(output));
      if (is_ready) {
        auto& queue = ready_queue(input_buffer.device());
        queue.push_front(FunctionTask(task.base, next_fn, std::move(input_buffer)));
        // 从 not_ready map中删除
        not_ready.erase(not_ready_it);
      }
    }
  }
}

/** Finds all stochastic functions and appends them to the queue */
auto Engine::find_stochastic_functions(function_queue& queue, Function* graph_root, GraphTask& task) -> void {
  std::unordered_set<Function*> seen {graph_root};
  function_queue search_queue {graph_root};
  while (search_queue.size() > 0) {
    auto fn = search_queue.back(); search_queue.pop_back();
    for (auto& next_fn_pair : fn->next_functions) {
      auto& next_fn = next_fn_pair.first;
      Function* next_ptr = next_fn.get();
      if (!next_ptr) continue;
      if (next_ptr->is_stochastic && next_ptr->is_executable && seen.count(next_ptr) == 0) {
        ready_queue(-1).push_front(FunctionTask(&task, next_fn, InputBuffer(0)));
        queue.push_back(next_ptr);
        task.has_any_work = true;
      }
      if (seen.count(next_ptr) == 0) {
        seen.insert(next_ptr);
        search_queue.push_back(next_ptr);
      }
    }
  }
}

/** Computes the number of dependencies for each function which requires grad */
// function_queue: std::vector<Function* >
auto Engine::compute_dependencies(function_queue queue, GraphTask& task) -> void {
  // Just to make sure that they will never be added to the queue again
  std::unordered_set<Function*> seen(queue.begin(), queue.end());

  // Queue contains all nodes that will start propagating gradients.
  // We no longer have to expand functions that don't require grad.
  // 
  auto& dependencies = task.dependencies;

  //停止 条件为 queue 为 空
  while (queue.size() > 0) {
    //  deque.back() 返回最后一个元素的引用， deque.pop_back(); 删除最后一个元素，返回为 void
    auto fn = std::move(queue.back()); queue.pop_back();
    
    for (auto& next_fn_pair : fn->next_functions) {
      Function* next_ptr = next_fn_pair.first.get();
      if (!next_ptr) continue;
      if (!next_ptr->is_executable) continue;
      if (next_ptr->is_stochastic) continue; // Stochastic nodes were in the queue already
      // 就算函数的依赖数量， 假设下一个函数需要当前函数传过去的梯度，那么下一个函数就依赖与当前 函数。
      dependencies[next_ptr] += 1;

      // 如果没有见过 next_ptr, 就把 next_ptr 加到 queue 中。 全部遍历一遍， 所有反向传导函数的 dependencies 个数都会保存在 base.dependencies 中！！！！
      if (seen.count(next_ptr) == 0) {
        seen.insert(next_ptr);
        queue.push_back(next_ptr);
      }
    }
  }
}

struct ClearCallbacks {
  ClearCallbacks(std::vector<std::function<void()>>& callbacks,
                 std::mutex &callbacks_lock)
    : callbacks(callbacks)
    , callbacks_lock(callbacks_lock) { clear(); }
  ~ClearCallbacks() { clear(); }

  void clear() {
    std::lock_guard<std::mutex> lock(callbacks_lock);
    callbacks.clear();
  }

  std::vector<std::function<void()>>& callbacks;
  std::mutex& callbacks_lock;
};

// function_list: std::vector<edge_type>, 
// edge_type: std::pair<std::shared_ptr<Function>, int>, Function 代表 grad_fn， int 的第几个输入（即：fn 的第几个输出）
// 为什么 function_list, 因为可能会有多个 root。
auto Engine::execute(const function_list& input_roots,
                     const variable_list& inputs,
                     bool keep_graph,
                     const pre_callback_map& pre_callbacks,
                     const post_callback_map& post_callbacks) -> void {
  
  // 只需要执行一次的函数， 而且线程安全！！！！！，
  //  执行 start_threads，开启了 NUM_DEVICE 个线程。
  // NUM_DEVICE 个线程是留着计算用的！！！！！！！！！！！

  std::call_once(start_threads_flag, &Engine::start_threads, this);
  // Callbacks are only valid for the duration of this run and should always be cleared
  // 这些 Callbacks 指的是什么？？？？？？？？？？？？？？？？？？
  ClearCallbacks _cb_guard(final_callbacks, post_callbacks_lock);
  
  // owner 默认为 NO_DEVICE, GraphTask 有个 owner
  GraphTask graph_task(keep_graph, pre_callbacks, post_callbacks);

  // lock 为 unique lock, 这儿加了个锁！！！！！！！！！！
  std::unique_lock<std::mutex> lock(graph_task.mutex);
  /*
  加锁的部分做的工作有：
  1. 创建了 graph_root 并将其放入 CPU 的 ReadyQueue 中
  2. 计算 反向传导图中 每个 Function 的依赖数量, 并放入 GraphTask 中
  3. 给 GraphTask 制定 owner = worker_device
  4. 然后释放锁， 执行 thread_main 
  */
  
  //这边的意思是，硬生生的搞出一个 graph 的 root 出来（Function）， 将 inputs 作为这个 roots 的 输出，将 inputs_roots 作为 这个 root 的 next_functions!!!!!!!
  auto graph_root = std::make_shared<GraphRoot>(input_roots, inputs);

  // function_queue: std::vector<Function*>
  function_queue roots;
  for (auto entry : input_roots) {
    if (entry.first->is_executable) {
      graph_task.has_any_work = true;
      roots.push_back(graph_root.get());
      // read_queue 返回一个 ReadyQueue， -1 为 device_id（代表CPU）, InputBuffer 为0 说明 root ,因为是 root， 没有输入，只有输出
      // 把这个 造好的 graph root 的 FunctionTask 放到 CPU 的 ReadyQueue 中
      ready_queue(-1).push_front(FunctionTask(&graph_task, graph_root, InputBuffer(0)));
      break;
    }
  }

  // Search the graph and find all stochastic functions. Append them to the queue.
  find_stochastic_functions(roots, graph_root.get(), graph_task);

  if (!graph_task.has_any_work) {
    throw std::runtime_error(
      "there are no graph nodes that require computing gradients");
  }

  // Now compute the dependencies for all executable functions
  // 计算 反向计算图 中 所有 Function 的依赖， 放到 GraphTask 中的 std::unordered_map<Function*, int> dependencies 中
  compute_dependencies(std::move(roots), graph_task);

  // Not a worker
  // NO_DEVICE 在 pytorch 中做什么工作呢？ 坐等 backward 完成，啥事都不干，然后 check 一下 backward 执行的怎么样
  if (worker_device == NO_DEVICE) {
    // Wait for all tasks to complete
    // 如果是 NO_DEVICE 线程， 静静的等待线程结束就好了
    // thread_main 中有 notify_all 这个操作
    // 第二个参数，返回为 true ， 则不阻塞 当前线程， 如果 为 false，即使 notify, 也不会释放
    graph_task.not_done.wait(lock, [&graph_task]{
      return graph_task.outstanding_tasks.load() == 0;
    });
  } else {
    graph_task.owner = worker_device;
    lock.unlock();
    thread_main(&graph_task);
  }

  // Check for an exception while running backwards
  if (graph_task.has_error.load()) {
    std::rethrow_exception(graph_task.exception);
  }

  if (!graph_task.not_ready.empty()) {
    throw std::runtime_error("could not compute gradients for some functions");
  }

  // Unlocking is necessary, because the callback can register
  // more callbacks (or they can be registered from other threads
  // while it's waiting.
  std::unique_lock<std::mutex> cb_lock(post_callbacks_lock);
  for (std::size_t i = 0; i < final_callbacks.size(); ++i) {
    cb_lock.unlock();
    final_callbacks[i]();
    cb_lock.lock();
  }
}

void Engine::queue_callback(std::function<void()> callback) {
  std::lock_guard<std::mutex> lock(post_callbacks_lock);
  final_callbacks.emplace_back(std::move(callback));
}

auto Engine::ready_queue(int device) -> ReadyQueue& {
  return *ready_queues.at(device + 1);
}

// 开启 NUM_DEVICES 个线程，放到后台处理 BP 提交的任务
// 这个函数在 Engine::execute 中被执行， 而且保证只倍执行一次
auto Engine::start_threads() -> void {
  int num_devices = 0;
#ifdef WITH_CUDA
  // check for case of compiled with CUDA but no available devices
  // 这里操作了一波 num_devices 如果有 cuda 的话， 就加了一
  if (cudaGetDeviceCount(&num_devices) != cudaSuccess) {
    cudaGetLastError();
    num_devices = 0;
  }
#endif
  // One for CPU, plus one for every GPU device
  // 
  int num_threads = num_devices + 1;
  ready_queues = std::vector<std::shared_ptr<ReadyQueue>>(num_threads);
  for (auto& queue : ready_queues)
    queue.reset(new ReadyQueue());
  for (int i = 0; i < num_threads; ++i) {
    // this, 显示将对象指针传进去！！！！！
    std::thread t(&Engine::thread_init, this, i - 1);
    t.detach();
  }
}

}} // namespace torch::autograd
