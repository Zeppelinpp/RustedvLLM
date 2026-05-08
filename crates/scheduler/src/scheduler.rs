use async_trait::async_trait;
use protocol::types::{EngineCommand, EngineResult, Request};
use tokio::sync::mpsc::{Receiver, Sender};

pub struct Scheduler {
    // Scheduler: Send(EngineCommand) + Receive(EngineResult + Request)
    // Engine: Send(EngineResult) + Receive(EngineCommand)
    request_rx: Receiver<Request>,

    engine_cmd_tx: Sender<EngineCommand>,
    engine_result_rx: Receiver<EngineResult>,

    queued: Vec<Request>,   // Queued
    active: Vec<Request>,   // Decode, Prefill
    finished: Vec<Request>, // Finished, Aborted, Failed
}

impl Scheduler {
    pub async fn run(mut self) {
        loop {
            // tick
            tokio::select! {
                Some(req) = self.request_rx.recv() => {
                    self.queued.push(req);
                }

                Some(result) = self.engine_result_rx.recv() => {
                    self.handle_engine_result(result).await;
                }
            }
            self.schedule().await;
        }
    }

    async fn handle_engine_result(&mut self, result: EngineResult) {
        todo!()
    }
    async fn schedule(&mut self) {
        // TODO:check state & build batch
        todo!()
        // TODO:dispatch engine task
        // TODO:Update Sequence and Request State
    }
}
