use std::ops::Deref;

// Received -> Tokenize -> Prefill -> Output -> Finish or Decode
use crate::model::{
    self, DecodeResult, MockModel, MockTokenizer, Model, ModelOutput, PrefillResult, Token,
    Tokenizer,
};
use async_trait::async_trait;
use protocol::types::{
    EngineResult, EngineTask, FinishReason, KVCache, Request, RequestBatch, RequestId,
    RequestState, SequenceOutput, SequenceState, TokenId,
};

#[async_trait]
pub trait Engine: Send + Sync {
    // execute_step: Engine decide to prefill/decode base on the request state
    // return stepoutput: token
    async fn execute_step(&self, batch: &RequestBatch) -> EngineResult;
}

pub struct MockEngine {
    pub id: String,
    pub model: MockModel,
    pub tokenizer: MockTokenizer,
}

impl MockEngine {
    fn new(id: &str, model: &MockModel, tokenizer: &MockTokenizer) -> Self {
        MockEngine {
            id: id.to_string(),
            model: model.clone(),
            tokenizer: tokenizer.clone(),
        }
    }
    fn get_state(&self, token: TokenId) -> SequenceState {
        if token == self.model.tokenizer.vocab.eos_token_id {
            SequenceState::Finished(FinishReason::Finished)
        } else {
            SequenceState::Running
        }
    }

    async fn prefill(&self, tasks: &[&EngineTask]) -> EngineResult {
        let mut outputs = Vec::with_capacity(tasks.len());

        // TODO: 真正引擎应把所有 prefill tasks 合并成一次 batch forward，
        // 而不是逐个调用 model.forward。
        for task in tasks {
            let EngineTask::Prefill {
                request_id,
                input_tokens,
            } = task
            else {
                continue;
            };

            let model_output = self
                .model
                .forward(
                    input_tokens
                        .iter()
                        .map(|t| Token { token_id: *t })
                        .collect::<Vec<Token>>(),
                    None,
                )
                .await;

            if let ModelOutput::PrefillResult(PrefillResult { first_token, kv }) = model_output {
                // TODO: 集成 get_state 做 EOS 检测，更新 SequenceState
                let sequence_output = SequenceOutput {
                    seq_id: *request_id,
                    token: first_token.token_id,
                    kv: kv,
                    state: self.get_state(first_token.token_id),
                };
                outputs.push(sequence_output);
            } else {
                // TODO: 部分失败时，应把错误记录到对应 sequence，而不是让整个 batch 失败
                outputs.push(SequenceOutput {
                    seq_id: *request_id,
                    token: Token::default().token_id,
                    kv: KVCache::default(),
                    state: SequenceState::Error,
                })
            }
        }

        EngineResult::StepOutput { outputs }
    }

    async fn decode(&self, tasks: &[&EngineTask]) -> EngineResult {
        let mut outputs = Vec::with_capacity(tasks.len());

        // TODO: 真正引擎应把所有 decode tasks 合并成一次 batch forward。
        for task in tasks {
            let EngineTask::Decode {
                request_id,
                input_tokens,
            } = task
            else {
                continue;
            };

            // TODO: KV Cache 应该从 prefill 或上一步 decode 传递过来，
            // 而不是每次都新建 default。
            // 暂时保留 Mock
            let mock_kv = KVCache::default();
            let last_token = vec![
                input_tokens
                    .iter()
                    .map(|t| Token { token_id: *t })
                    .collect::<Vec<Token>>()
                    .last()
                    .unwrap_or(&Token::default())
                    .clone(),
            ];
            let model_output = self.model.forward(last_token, Some(mock_kv)).await;

            if let ModelOutput::DecodeResult(DecodeResult { new_token, kv }) = model_output {
                // TODO: 集成 get_state 做 EOS 检测，更新 SequenceState
                let sequence_output = SequenceOutput {
                    seq_id: *request_id,
                    token: new_token.token_id,
                    kv: kv,
                    state: self.get_state(new_token.token_id),
                };
                outputs.push(sequence_output);
            } else {
                outputs.push(SequenceOutput {
                    seq_id: *request_id,
                    token: Token::default().token_id,
                    kv: KVCache::default(),
                    state: SequenceState::Error,
                })
            }
        }

        EngineResult::StepOutput { outputs }
    }
}

#[async_trait]
impl Engine for MockEngine {
    async fn execute_step(&self, batch: &RequestBatch) -> EngineResult {
        let (prefill_tasks, decode_tasks): (Vec<&EngineTask>, Vec<&EngineTask>) = batch
            .tasks
            .iter()
            .partition(|t| matches!(t, EngineTask::Prefill { .. }));

        let mut outputs = Vec::with_capacity(batch.tasks.len());

        match self.prefill(&prefill_tasks).await {
            EngineResult::StepOutput { outputs: mut o } => outputs.append(&mut o),
            EngineResult::EngineError(e) => return EngineResult::EngineError(e),
        }

        match self.decode(&decode_tasks).await {
            EngineResult::StepOutput { outputs: mut o } => outputs.append(&mut o),
            EngineResult::EngineError(e) => return EngineResult::EngineError(e),
        }

        EngineResult::StepOutput { outputs }
    }
}
