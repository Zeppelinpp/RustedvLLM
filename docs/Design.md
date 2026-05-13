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
    state: RequestState,      // Queued | Active | Finished | Aborted | Failed
    sampling_params: SamplingParams,
}
```

## Protocol (Scheduler <-> Engine 通信)

Protocol crate 只放**最小公共类型**，不放业务逻辑对象。

```Rust
// 基础类型
type RequestId = u64;
type TokenId  = u32;

struct Token { token_id: TokenId }

struct Vocab {
    vocab_size: Option<u32>,
    vocab: HashMap<u32, String>,
    eos_token_id: TokenId,
}

#[async_trait]
trait Tokenizer: Send + Sync { // Tokenizer被Scheduler使用，随 tokio::spawn 调用,Future需要实现 Send + 'static
    async fn tokenize(&self, prompt: &str) -> Vec<Token>;
    async fn decode(&self, input_ids: &Vec<Token>) -> String;
}
```

### Crate 依赖关系

```
        protocol  (中间数据交互类型定义)
       /        \
  scheduler    engine   (engine 不反向依赖 scheduler)
```

```Rust
// Scheduler 组织 batch 发送到 Engine
struct RequestBatch {
    batch_id: String,
    created_at: Instant,
    tasks: Vec<EngineTask>,
}

// EngineTask 按阶段拆分
enum EngineTask {
    Prefill { request_id, input_tokens: Vec<TokenId> },
    Decode  { request_id, input_tokens: Vec<TokenId>, kv: KVCache },
}

// Scheduler -> Engine
enum EngineCommand {
    ExecuteStep(RequestBatch),  // 一次调度一个 batch
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
    error: Option<String>,  // Engine 只报告执行错误，不做状态判断
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

- 内部持有 `MockModel`
- `execute_step` 按 `Prefill` / `Decode` 自动分区、分别执行
- `prefill`: 调用 `model.forward(input_tokens, None)` → 产出首 token + KVCache
- `decode`:  调用 `model.forward(last_token, Some(kv))` → 产出 next token
- 每个 task 独立调用 model（TODO: 合并为一次 batch forward）
- **Engine 只输出 token + kv + error，不做 SequenceState 判断**

### Model Trait

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

Scheduler 维护 `Sequence` 状态机，拥有请求从接入到完成的完整生命周期。

```Rust
pub struct Sequence {
    request_id: RequestId,
    prompt: String,
    sampling_params: SamplingParams,

    state: SequenceState,        // WaitingPrefill | WaitingDecode | RunningPrefill | RunningDecode | Finished | Error
    input_tokens: Vec<TokenId>,  // tokenize 后的输入
    output_tokens: Vec<TokenId>, // 已生成的 token
    kv_cache: Option<KVCache>,   // Engine 返回的 KV
}

impl Sequence {
    async fn from_request(req: Request, tokenizer: &Box<dyn Tokenizer + Send>) -> Self {
        let tokens = tokenizer.tokenize(&req.prompt).await;
        let input_tokens = tokens.into_iter().map(|t| t.token_id).collect();

        Sequence {
            request_id: req.request_id,
            prompt: req.prompt,
            sampling_params: req.sampling_params,
            state: SequenceState::WaitingPrefill,
            input_tokens,
            output_tokens: Vec::new(),
            kv_cache: None,
        }
    }
}

pub struct Scheduler {
    pub request_rx: Receiver<Request>,
    pub engine_cmd_tx: Sender<EngineCommand>,
    pub engine_result_rx: Receiver<EngineResult>,

    pub sequences: HashMap<RequestId, Sequence>,  // 统一维护所有序列状态
    pub tokenizer: Box<dyn Tokenizer + Send>,     // 随 Scheduler 拥有，满足 'static
    shutdown_rx: Receiver<()>,                    // Gracefully shutdown
}
```

### 主循环

```Rust
pub async fn run(mut self) -> Self {
    loop {
        tokio::select! {
            biased;

            Some(()) = self.shutdown_rx.recv() => {
                break;
            }

            Some(req) = self.request_rx.recv() => {
                let seq = Sequence::from_request(req, &self.tokenizer).await;
                self.sequences.insert(seq.request_id, seq);
            }

            Some(result) = self.engine_result_rx.recv() => {
                self.handle_engine_result(result).await;
            }
        }

        self.schedule().await;  // build batch -> 派发 -> 更新 State
    }
    self
}
```

> Scheduler 拥有 State 的管理权并进行状态迁移；Engine 只负责无状态计算，产出 raw token + kv + error。

## Metrics (TODO)

以下 metrics 尚未实现，仅作为设计预留：

```Rust
struct RequestMetrics {
    queue_time: Duration,
    ttft: Duration,           // Time To First Token
    tpot: Duration,           // Time Per Output Token
    total_latency: Duration,
}
```

