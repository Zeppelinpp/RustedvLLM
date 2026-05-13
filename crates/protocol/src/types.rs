use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Instant;
pub type RequestId = u64;
pub type TokenId = u32;

#[derive(Debug, Clone, Default)]
pub struct Token {
    pub token_id: TokenId,
}

#[derive(Debug, Clone)]
pub enum RequestState {
    Queued,
    Active,
    Finished,
    Aborted,
    Failed,
}

#[derive(Default, Clone)]
pub struct Vocab {
    pub vocab_size: Option<u32>,
    pub vocab: HashMap<u32, String>,
    pub eos_token_id: TokenId,
}

#[derive(Debug, Clone)]
pub struct Request {
    pub request_id: RequestId,
    pub prompt: String,
    pub state: RequestState,
    pub sampling_params: SamplingParams,
}

#[derive(Debug)]
pub struct RequestBatch {
    pub batch_id: String,
    pub created_at: Instant,
    pub tasks: Vec<EngineTask>,
}

#[derive(Debug, Clone, Default)]
pub struct SamplingParams {
    pub max_tokens: usize,
}

#[derive(Default, Debug, Clone)]
pub struct KVCache {
    pub addr: usize,
}

#[derive(Debug, Clone)]
pub enum FinishReason {
    Finished,
    Error,
}

#[async_trait]
pub trait Tokenizer: Send + Sync {
    async fn tokenize(&self, prompt: &str) -> Vec<Token>;
    async fn decode(&self, input_ids: &Vec<Token>) -> String;

    fn eos_token_id(&self) -> TokenId;
}

#[derive(Debug, Clone, Default)]
pub enum SequenceState {
    // Scheduler
    #[default]
    WaitingPrefill,
    WaitingDecode,

    // Engine
    RunningPrefill,
    RunningDecode,

    // Lifecycle end
    Finished(FinishReason),
    Error,
}

#[derive(Debug)]
pub enum EngineTask {
    Prefill {
        request_id: RequestId,
        input_tokens: Vec<TokenId>,
    },
    Decode {
        request_id: RequestId,
        input_tokens: Vec<TokenId>,
        kv: KVCache,
    },
}

#[derive(Debug)]
pub struct SequenceOutput {
    pub seq_id: RequestId,
    pub token: TokenId,
    pub kv: KVCache,
    pub error: Option<String>,
}

#[derive(Debug)]
pub enum EngineCommand {
    ExecuteStep(RequestBatch),
    Shutdown,
}

#[derive(Debug)]
pub enum EngineResult {
    StepOutput { outputs: Vec<SequenceOutput> },
    EngineError(String),
}
