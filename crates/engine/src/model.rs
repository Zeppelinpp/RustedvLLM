use async_trait::async_trait;
use protocol::types::{KVCache, Token};

pub struct PrefillResult {
    pub first_token: Token,
    pub kv: KVCache,
}

pub struct DecodeResult {
    pub new_token: Token,
    pub kv: KVCache,
}

pub enum ModelOutput {
    PrefillResult(PrefillResult),
    DecodeResult(DecodeResult),
}

#[async_trait]
pub trait Model {
    async fn forward(&self, input_ids: Vec<Token>, kv: Option<KVCache>) -> ModelOutput;
    async fn generate(&self, input_ids: Vec<Token>, max_tokens: usize) -> Vec<Token>;
}

#[derive(Clone)]
pub struct MockModel {
    pub model: String,
}

#[async_trait]
impl Model for MockModel {
    async fn forward(&self, _input_ids: Vec<Token>, kv: Option<KVCache>) -> ModelOutput {
        let token = Token { token_id: 33 };

        if let Some(kv_cache) = kv {
            return ModelOutput::DecodeResult(DecodeResult {
                new_token: token,
                kv: kv_cache,
            });
        }
        ModelOutput::PrefillResult(PrefillResult {
            first_token: token,
            kv: KVCache::default(),
        })
    }

    async fn generate(&self, _input_ids: Vec<Token>, _max_tokens: usize) -> Vec<Token> {
        todo!()
    }
}
