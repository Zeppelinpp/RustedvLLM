use engine::model::MockModel;
use engine::{Engine, MockEngine, MockTokenizer};
use protocol::types::{EngineCommand, EngineResult, Request, RequestState, SamplingParams};
use scheduler::Scheduler;
use std::time::Duration;
use tokio::sync::mpsc;

#[tokio::test]
async fn test_single_request() {
    let mock_tokenizer = MockTokenizer::new("mock_vocab");
    let model = MockModel {
        model: "mock_model".to_string(),
    };
    let engine = MockEngine::new("mock_engine", &model);

    let (request_tx, request_rx) = mpsc::channel::<Request>(16);
    let (engine_cmd_tx, mut engine_cmd_rx) = mpsc::channel::<EngineCommand>(16);
    let (engine_result_tx, engine_result_rx) = mpsc::channel::<EngineResult>(16);
    let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>(1);

    // Engine loop: receive commands, execute, return results
    let engine_loop = engine.clone();
    tokio::spawn(async move {
        while let Some(cmd) = engine_cmd_rx.recv().await {
            match cmd {
                EngineCommand::ExecuteStep(batch) => {
                    let result = engine_loop.execute_step(&batch).await;
                    let _ = engine_result_tx.send(result).await;
                }
                EngineCommand::Shutdown => break,
            }
        }
    });

    let scheduler = Scheduler::new(
        request_rx,
        engine_cmd_tx,
        engine_result_rx,
        shutdown_rx,
        Box::new(mock_tokenizer),
    );
    let scheduler_handle = tokio::spawn(async move { scheduler.run().await });

    // 1. Send a request inside the test
    let req = Request {
        request_id: 0,
        prompt: "Hi".to_string(),
        state: RequestState::Queued,
        sampling_params: SamplingParams { max_tokens: 3 },
    };
    request_tx.send(req).await.unwrap();

    // 2. Wait for scheduler + engine to process
    tokio::time::sleep(Duration::from_millis(50)).await;

    // 3. Shutdown and collect the scheduler instance back
    shutdown_tx.send(()).await.unwrap();
    let scheduler = scheduler_handle.await.unwrap();

    // 4. Assert results
    let seq = scheduler.sequences.get(&0).expect("sequence should exist");
    assert_eq!(
        seq.output_tokens.len(),
        3,
        "should generate exactly max_tokens tokens"
    );
    assert!(
        matches!(seq.state, protocol::SequenceState::Finished(_)),
        "sequence should reach Finished state"
    );
}

#[tokio::test]
async fn test_batch_request() {
    // TODO: send multiple requests and verify they are batched together
}
