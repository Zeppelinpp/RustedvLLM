use async_trait::async_trait;
use protocol::types::{KVCache, TokenId};
use std::{collections::HashMap, default};

#[derive(Debug, Clone, Default)]
pub struct Token {
    pub token_id: TokenId,
}

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

#[derive(Default)]
pub struct Vocab {
    pub vocab_size: Option<u32>,
    pub vocab: HashMap<u32, String>,
    pub eos_token_id: TokenId,
}

#[async_trait]
pub trait Tokenizer {
    async fn tokenize(&self, prompt: &str) -> Vec<Token>;
    async fn decode(&self, input_ids: &Vec<Token>) -> String;
}

#[async_trait]
pub trait Model {
    async fn forward(&self, input_ids: Vec<Token>, kv: Option<KVCache>) -> ModelOutput;
    async fn generate(&self, input_ids: Vec<Token>, max_tokens: usize) -> Vec<Token>;
}

#[derive(Clone)]
pub struct MockTokenizer {
    pub vocab: Vocab,
}

fn parse_vocab(vocab_path: &str) -> Vocab {
    Vocab::default()
}

impl MockTokenizer {
    fn new(vocab_path: &str) -> Self {
        MockTokenizer {
            vocab: parse_vocab(vocab_path),
        }
    }
}

#[derive(Clone)]
pub struct MockModel {
    pub model: String,
    pub tokenizer: MockTokenizer,
}

#[async_trait]
impl Model for MockModel {
    async fn forward(&self, input_ids: Vec<Token>, kv: Option<KVCache>) -> ModelOutput {
        // Mock RANDOM token id
        let token = Token { token_id: 33 };

        if let Some(kv_cache) = kv {
            // Decode
            return ModelOutput::DecodeResult(DecodeResult {
                new_token: token,
                kv: kv_cache,
            });
        }
        // Prefill
        ModelOutput::PrefillResult(PrefillResult {
            first_token: token,
            kv: KVCache::default(),
        })
    }

    async fn generate(&self, input_ids: Vec<Token>, max_tokens: usize) -> Vec<Token> {
        // Call forward multiple times until eos token
        todo!()
    }
}

#[async_trait]
impl Tokenizer for MockTokenizer {
    async fn decode(&self, input_ids: &Vec<Token>) -> String {
        todo!()
    }
    async fn tokenize(&self, prompt: &str) -> Vec<Token> {
        let tokenized_prompt: Vec<Token> = Vec::new();
        // TODO:Mock
        // Tokenization
        todo!() // tokenize
    }
}
