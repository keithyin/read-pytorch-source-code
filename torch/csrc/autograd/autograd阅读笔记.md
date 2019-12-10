# 理解 pytorch 的前向与反向过程

要想理解 pytorch 前向与反向过程，需要从以下几个方面考虑

## 前向：

1. Variable 在前向过程中充当什么角色
2. Function 在前向过程中充当什么角色



**pytorch 的前向过程，是动态创建反向传导图的过程 ！如何创建的反向传导图呢？**

前向过程, 调用 Function 的 apply 方法:
    1. 根据输入计算输出
    2. 调用wrap_outputs


构建反向传导图的核心是 `wrap_outputs`， `wrap_outputs` 做了什么：
    1. 根据 inputs 计算出 grad_fn 的 `is_volatile, is_executable, next_functions`
    2. 然后创建 grad_fn 
    3. 然后创建 forward_fn 的 输出 Variable, `grad_fn` 保存在这个对象里面。
    4. 由于 `inputs` 也保存了 `grad_fn`, 所以, 很容易就能构建 `grad_fn.next_functions`了

> 根据 `next_functions` 就能得到 反向传导图了。

## 反向：

1. pytorch 什么时候创建的 反向传导图，如何创建的
2. Variable 的梯度怎么处理的， 因为一个 Variable 可能有多个方向 传回来的梯度

看下反向传导的过程是啥样的：

**Engine.execute()**
1. std::call_once(start_threads_flag, &Engine::start_threads, this); 开启 NUM_DEVICES 个线程 放在后台 处理 GraphTask 中的任务 （只执行一次，和进程有相同的生命周期？）
    1. 根据设备的数量 (一个CPU, 几卡GPU), 创建 ReadyQueue, 一个设备负责一个ReadyQueue, 来执行其中FunctionTask. 设备的数量=线程的数量=ReadyQueue 的数量
    2. 执行 `Engine::thread_init`, 设置当前的 worker_device, 然后执行 `Engine::thread_main`
    3. `thread_main` 如何执行见 文章下面部分
2. 创建 GraphTask : {keep_graph, mutex, std::condition_variable not_done, std::unordered_map<Function*, InputBuffer> not_ready, std::unordered_map<Function*, int> dependencies, int owner}
3. 做一个  graph_root（是个 Function）， 然后将其包装成 FunctionTask 放到 CPU 的 ReadyQueue 中！
4. 计算 反向传导图中的 所有可计算函数的 依赖 个数 ，保存在 GraphTask 的 dependencies 中 (`Engine::compute_dependencies()`)
    1. 层次遍历DAG中的所有节点, 如果当前节点的下一个节点为 next_ptr, 那么 dependencies[next_ptr] += 1; 这样就可以计算出所有的节点的依赖个数.
5. 如果 NO_DEVICE， 就默默的等着计算结束就好了 (execute刚开始开的线程才是工作的线程, 执行execute的线程其实是NO_DEVICE)
6. 然后执行 Engin.thread_main(GraphTask* gt)： 第一步开启的 线程也是 执行下面的操作。
    1. 从 Engine 的 ready_queues 中拿出 当前 worker_device 所对应的 ReadyQueue
    2. 循环执行 Engine.evaluate_function(FunctionTask)： 从 相应的 ReadyQueue 中取出 FunctionTask（取的过程可能会阻塞）
        1. 进行 FunctionTask 的计算
        2. 遍历 当前 Function 的 所有 next_function, 将它们的 dependencies 减一， 看看他们是否已经 ready
        3. 如果 ready， 通过 InputBuffer 的 device 来确定 将其 放入到 那个 ReadyQueue 中！ ？？？ InputBuffer 的 device如何确定的？？？
        4. 如果没有准备好， 就放在 GraphTask 中的 not_ready map中。
        5. 如果 graph_task->outstanding_tasks <= 0 则退出循环
    3. 通过上面的循环， 可以执行完 GraphTask 中所有的 Function， 完成一次 BP
    4. NO_DEVICE 操作操作


**Engine.thread_main**
1. 从 Engine 的 ready_queues 中拿到属于当前 线程的 ReadyQueue
2. 判断 (!graph_task || graph_task->outstanding_tasks > 0), 对于worker_thread, `graph_task==nullptr`, 所以只需要考虑第二个条件就行了
3. 如果 判断条件满足, 则会从 ReadyQueue 中取出 FunctionTask, 执行这个FunctionTask (`Engine::evaluate_function`)
    1. 计算梯度
    2. 然后遍历所有的 `next_functions`, dependencies[next_func]--, 如果为0, 就直接 erase 了
    3. 然后再看 `graph_task.not_ready` 列表中有没有记录这个 `next_fn`. 如果没有记录, 那就构建 InputBuffer, 然后根据InputBuffer构建 FunctionTask, 然后根据具体情况确定是放入 `not_ready` 中 还是 `ready_queue` 中.
4. 

**关于outstanding_tasks**
* outstanding_tasks 的值总是会先加再减. 多线程情况下不会出现问题

**梯度计算时， GraphTask 的角色是**

* 记录 BP Graph 中所有 Function 的 dependencies 
* 记录着 还没有 准备好的 Function。(准备好的 Function 被包装成 FunctionTask，放到 ReadyQueue 中。)
* 管理着一些 GraphTask 的属性。 


**FunctionTask 的角色是**

* 在ReadyQueue中一个可以被执行的求导函数, 其中的输入已经准备好了


**反向的时候调用 call_function(), 这个函数干了些啥？**

1. 调用注册在 fn 上的 pre_hooks, 得到一个 新的 inputs
2. 然后执行  fn(inputs) 得到一个结果
3. 然后再调用注册在 fn 上的 post_hooks，然后返回 作为 outputs


**InputBuffer 的 device 是怎么得到的**
* 根据梯度累积的值的设备确定的.
    

**ReadyQueue**

*  
* push_front(...)
    * 会加锁，这个可以保证线程安全。
    * 同时 GraphTask 的 outstanding_tasks 会加一
    * not_empty 通知一次 
* pop_back() -> FunctionTask
    * 如果 queue 为空，则阻塞
    * 否则，返回一个 FunctionTask


## NO_DEVICE 在搞些什么

```c++
// execute 中, 如果是 NO_DEVICE 就不会执行 任何 FunctionTask ， 等着结束就好了
if (worker_device == NO_DEVICE) {
    // Wait for all tasks to complete
    // thread_main 中有 notify_all 这个操作
    // 第二个参数，返回为 true ， 则不阻塞 当前线程， 如果 为 false，即使 notify, 也不会释放
    graph_task.not_done.wait(lock, [&graph_task]{ return graph_task.outstanding_tasks.load() == 0});
}
```

```c++
// thead_main 中
// 如果 GraphTask 的 owner 是 NO_DEVICE
if (base_owner == NO_DEVICE) {
    if (--task.base->outstanding_tasks == 0) {
        // 给 notify_all 加个锁
        std::lock_guard<std::mutex> lock(task.base->mutex);
        task.base->not_done.notify_all();
}
```

## 疑问

* engine 是单线程的？？？ 
