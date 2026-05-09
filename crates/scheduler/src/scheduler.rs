use protocol::types::{
    EngineCommand, EngineResult, EngineTask, FinishReason, KVCache, Request, RequestBatch,
    RequestId, SamplingParams, SequenceState, TokenId, Tokenizer,
};
use std::collections::HashMap;
use tokio::sync::mpsc::{Receiver, Sender};

#[derive(Debug)]
pub struct Sequence {
    pub request_id: RequestId,
    pub prompt: String,
    pub sampling_params: SamplingParams,

    pub state: SequenceState,

    pub input_tokens: Vec<TokenId>,
    pub output_tokens: Vec<TokenId>,

    pub kv_cache: Option<KVCache>,
}

impl Sequence {
    async fn from_request(req: Request, tokenizer: &dyn Tokenizer) -> Self {
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

    fn to_engine_task(&self) -> EngineTask {
        match self.state {
            SequenceState::WaitingPrefill => EngineTask::Prefill {
                request_id: self.request_id,
                input_tokens: self.input_tokens.clone(),
            },
            SequenceState::WaitingDecode => EngineTask::Decode {
                request_id: self.request_id,
                input_tokens: vec![*self.output_tokens.last().unwrap_or(&0)],
                kv: self.kv_cache.clone().unwrap_or_default(),
            },
            _ => panic!("invalid state for engine task: {:?}", self.state),
        }
    }
}

pub struct Scheduler {
    request_rx: Receiver<Request>,

    engine_cmd_tx: Sender<EngineCommand>,
    engine_result_rx: Receiver<EngineResult>,

    sequences: HashMap<RequestId, Sequence>,
}

impl Scheduler {
    pub fn new(
        request_rx: Receiver<Request>,
        engine_cmd_tx: Sender<EngineCommand>,
        engine_result_rx: Receiver<EngineResult>,
    ) -> Self {
        Scheduler {
            request_rx,
            engine_cmd_tx,
            engine_result_rx,
            sequences: HashMap::new(),
        }
    }

    pub async fn run(mut self, tokenizer: &dyn Tokenizer) {
        loop {
            tokio::select! {
                Some(req) = self.request_rx.recv() => {
                    let seq = Sequence::from_request(req, tokenizer).await;
                    self.sequences.insert(seq.request_id, seq);
                }

                Some(result) = self.engine_result_rx.recv() => {
                    self.handle_engine_result(result).await;
                }
            }

            self.schedule().await;
        }
    }

    async fn handle_engine_result(&mut self, result: EngineResult) {
        let EngineResult::StepOutput { outputs } = result else {
            return;
        };

        for output in outputs {
            let Some(seq) = self.sequences.get_mut(&output.seq_id) else {
                continue;
            };

            if let Some(ref _err) = output.error {
                seq.state = SequenceState::Error;
                continue;
            }

            seq.output_tokens.push(output.token);
            seq.kv_cache = Some(output.kv);

            let eos_token_id = 0; // TODO: get from tokenizer
            let max_tokens = 100; // TODO: from sampling_params

            if output.token == eos_token_id || seq.output_tokens.len() >= max_tokens {
                seq.state = SequenceState::Finished(FinishReason::Finished);
            } else {
                seq.state = SequenceState::WaitingDecode;
            }
        }
    }

    async fn schedule(&mut self) {
        let mut tasks = Vec::new();

        for seq in self.sequences.values() {
            if matches!(
                seq.state,
                SequenceState::WaitingPrefill | SequenceState::WaitingDecode
            ) {
                tasks.push(seq.to_engine_task());
            }
        }

        if tasks.is_empty() {
            return;
        }

        for seq in self.sequences.values_mut() {
            if matches!(
                seq.state,
                SequenceState::WaitingPrefill | SequenceState::WaitingDecode
            ) {
                seq.state = match seq.state {
                    SequenceState::WaitingPrefill => SequenceState::RunningPrefill,
                    SequenceState::WaitingDecode => SequenceState::RunningDecode,
                    _ => seq.state.clone(),
                };
            }
        }

        let batch = RequestBatch {
            batch_id: format!("batch-{}", std::time::Instant::now().elapsed().as_millis()),
            created_at: std::time::Instant::now(),
            tasks,
        };

        let _ = self.engine_cmd_tx.send(EngineCommand::ExecuteStep(batch)).await;
    }
}
