use crate::model::{DecodeResult, MockModel, Model, ModelOutput, PrefillResult};
use async_trait::async_trait;
use protocol::types::{
    EngineResult, EngineTask, KVCache, RequestBatch, SequenceOutput, Token, TokenId, Vocab,
};

#[async_trait]
pub trait Engine: Send + Sync {
    async fn execute_step(&self, batch: &RequestBatch) -> EngineResult;
}

#[derive(Clone)]
pub struct MockTokenizer {
    pub vocab: Vocab,
}

fn parse_vocab(_vocab_path: &str) -> Vocab {
    Vocab::default()
}

impl MockTokenizer {
    pub fn new(vocab_path: &str) -> Self {
        MockTokenizer {
            vocab: parse_vocab(vocab_path),
        }
    }
}

#[async_trait]
impl protocol::types::Tokenizer for MockTokenizer {
    async fn decode(&self, _input_ids: &Vec<Token>) -> String {
        todo!()
    }
    async fn tokenize(&self, _prompt: &str) -> Vec<Token> {
        todo!()
    }
    fn eos_token_id(&self) -> TokenId {
        self.vocab.eos_token_id
    }
}

pub struct MockEngine {
    pub id: String,
    pub model: MockModel,
}

impl MockEngine {
    pub fn new(id: &str, model: &MockModel) -> Self {
        MockEngine {
            id: id.to_string(),
            model: model.clone(),
        }
    }

    async fn prefill(&self, tasks: &[&EngineTask]) -> EngineResult {
        let mut outputs = Vec::with_capacity(tasks.len());

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
                        .collect(),
                    None,
                )
                .await;

            if let ModelOutput::PrefillResult(PrefillResult { first_token, kv }) = model_output {
                outputs.push(SequenceOutput {
                    seq_id: *request_id,
                    token: first_token.token_id,
                    kv,
                    error: None,
                });
            } else {
                outputs.push(SequenceOutput {
                    seq_id: *request_id,
                    token: Token::default().token_id,
                    kv: KVCache::default(),
                    error: Some("prefill failed".to_string()),
                });
            }
        }

        EngineResult::StepOutput { outputs }
    }

    async fn decode(&self, tasks: &[&EngineTask]) -> EngineResult {
        let mut outputs = Vec::with_capacity(tasks.len());

        for task in tasks {
            let EngineTask::Decode {
                request_id,
                input_tokens,
                kv,
            } = task
            else {
                continue;
            };

            let last_token = vec![
                input_tokens
                    .iter()
                    .map(|t| Token { token_id: *t })
                    .collect::<Vec<Token>>()
                    .last()
                    .unwrap_or(&Token::default())
                    .clone(),
            ];
            let model_output = self.model.forward(last_token, Some(kv.clone())).await;

            if let ModelOutput::DecodeResult(DecodeResult { new_token, kv }) = model_output {
                outputs.push(SequenceOutput {
                    seq_id: *request_id,
                    token: new_token.token_id,
                    kv,
                    error: None,
                });
            } else {
                outputs.push(SequenceOutput {
                    seq_id: *request_id,
                    token: Token::default().token_id,
                    kv: KVCache::default(),
                    error: Some("decode failed".to_string()),
                });
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
