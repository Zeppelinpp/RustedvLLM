# Rusted vLLM Design

## Request Lifecycle

```
Queued -> Active -> (Finished / Aborted / Failed)
```

### Request

```Rust
struct Request {
    request_id: RequestId,
    prompt: String,
    state: RequestState,  // Queued | Active | Finished | Aborted | Failed
}
```

## Protocol (Scheduler <-> Engine)

```Rust
// Scheduler 组织 batch 发送到 Engine
struct RequestBatch {
    batch_id: String,
    created_at: Instant,
    tasks: Vec<EngineTask>,
}

// EngineTask 按阶段拆分
enum EngineTask {
    Prefill  { request_id, input_tokens: Vec<TokenId> },
    Decode   { request_id, input_tokens: Vec<TokenId> },
}

// Scheduler -> Engine
enum EngineCommand {
    ExecuteStep(EngineTask),
    Shutdown,
}

// Engine -> Scheduler
enum EngineResult {
    StepOutput { outputs: Vec<SequenceOutput> },
    EngineError(String),
}

struct SequenceOutput {
    seq_id: RequestId,
    token: TokenId,
    kv: KVCache,
    state: SequenceState,  // Running | Finished(FinishReason) | Error
}
```

## InferenceEngine

Engine 本身作为黑盒，对外只暴露 `execute_step`：

```Rust
#[async_trait]
trait Engine: Send + Sync {
    async fn execute_step(&self, batch: &RequestBatch) -> EngineResult;
}
```

### MockEngine 实现

- 内部持有 `MockModel` + `MockTokenizer`
- `execute_step` 按 `Prefill` / `Decode` 自动分区、分别执行
- `prefill`: 调用 `model.forward(input_tokens, None)` → 产出首 token + KVCache
- `decode`:  调用 `model.forward(last_token, Some(kv))` → 产出 next token
- 每个 task 独立调用 model（TODO: 合并为一次 batch forward）
- 根据 eos_token_id 检测 `SequenceState`

### Model Layer

```Rust
#[async_trait]
trait Model {
    async fn forward(&self, input_ids: Vec<Token>, kv: Option<KVCache>) -> ModelOutput;
    async fn generate(&self, input_ids: Vec<Token>, max_tokens: usize) -> Vec<Token>;
}

enum ModelOutput {
    PrefillResult(PrefillResult { first_token, kv }),
    DecodeResult (DecodeResult  { new_token, kv }),
}
```

## Scheduler

```Rust
pub struct Scheduler {
    request_rx: Receiver<Request>,          // 外部请求入口
    engine_cmd_tx: Sender<EngineCommand>,   // -> Engine
    engine_result_rx: Receiver<EngineResult>, // <- Engine

    queued:   Vec<Request>,   // 待调度
    active:   Vec<Request>,   // 已 Prefill，进入 Decode 阶段
    finished: Vec<Request>,   // 终态
}
```

### 主循环

```Rust
async fn run(mut self) {
    loop {
        tokio::select! {
            Some(req)    = self.request_rx.recv()      => self.queued.push(req),
            Some(result) = self.engine_result_rx.recv() => self.handle_engine_result(result).await,
        }
        self.schedule().await;  // 检查状态 -> build_batch -> 派发 EngineTask -> 更新 State
    }
}
```

> Scheduler 拥有 State 的管理权并进行状态迁移；Engine 只负责产出 token，做 Prefill / Decode 执行。

## Metrics

```Rust
struct RequestMetrics {
    queue_time,
    ttft,
    tpot,
    total_latency,
}
```

