use std::time::Instant;

pub type RequestId = u64;
pub type TokenId = u32;

#[derive(Debug, Clone)]
pub enum RequestState {
    Queued,
    Acitve,
    Finished,
    Aborted,
    Failed,
}

#[derive(Debug, Clone)]
pub struct Request {
    pub request_id: RequestId,
    pub prompt: String,
    pub state: RequestState,
}

#[derive(Debug)]
pub struct RequestBatch {
    pub batch_id: String,
    pub created_at: Instant,
    pub tasks: Vec<EngineTask>,
}

#[derive(Default)]
pub struct KVCache {
    pub addr: usize,
}

pub enum FinishReason {
    Finished,
    Error,
}

pub enum SequenceState {
    Running,
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
    },
}

pub struct SequenceOutput {
    pub seq_id: RequestId,
    pub token: TokenId,
    pub kv: KVCache,
    pub state: SequenceState,
}

pub enum EngineCommand {
    ExecuteStep(EngineTask),
    Shutdown,
}

pub enum EngineResult {
    StepOutput { outputs: Vec<SequenceOutput> },
    EngineError(String),
}
