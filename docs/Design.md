# Rusted vLLM Design

## Request Lifecycle 

```
Queued -> Active -> (Finished / Aborted / Failed)
```

### Request

- `prompt`
- `request_id`
- `state`

## InferenceEngine

模拟类似 `llama.cpp` 这样的模型引擎，LLM Inference本身作为黑盒

```Rust
trait Engine {
  aysnc execute_step(&self, &mut batch: RequestBatch)
}
```

`FauxInferenceEngine` 作为 Inference Engine和真实推理框架（Continuos batching，schedular ...) 的桥接

## Scheduler

```Rust
loop {
  tokio::select! {
    // 新 Request 入 Queue
    // 处理 Engine 返回的结果
  }
  // Schedule，组织batch派发
}
```

> Scheduler 拥有 State 的管理权 并进行状态迁移，Engine只负责产出token，做prefill和decode执行

### Scheduler

- `rx:Receiver`, `tx:Sender`,  通信通道
- `build_batch`: 打包batch
- 发送 `EngineTask`,  接收 `EngineResult` , 更新 `SequenceState` 和 `RequestState` 

## Protocol

主要定义 `Scheduler` 和 `Engine` 之间会发生交互的数据类型

- `RequestBatch`: `Scheduler` 决定怎么组织 batch 发送到 `Engine` 
- `EngineCommand`: `Scheduler` 真实发送给 `Engine` 的指令
  - `ExecuteStep(EngineTask(Prefill, Decode))`
  - `Shutdown`
- `EngineResult`: `Engine` 回传给 `Scheduler` 的执行结果
  - `StepOutput`:  `Vec<SequenceOutput>`
  - `EngineError`
- `Request`: `Scheduler` 接收的请求，包含 ID， prompt, 调用参数, 状态 `RequestState`

## Metrics

```Rust
struct RequestMetrics {
	queue_time,
	ttft,
	tpot,
	total_latency,
}
```

