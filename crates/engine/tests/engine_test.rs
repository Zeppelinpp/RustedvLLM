use engine::{
    engine::{Engine, MockEngine},
    model::{MockModel, MockTokenizer, Vocab},
};
use protocol::types::{EngineResult, EngineTask, RequestBatch, SequenceState};

#[tokio::test]
async fn test_prefill_batch() {
    let model = "MockModel".to_string();
    let tokenizer = MockTokenizer {
        vocab: Vocab::default(),
    };
    let mock_model = MockModel {
        model: model.clone(),
        tokenizer: tokenizer.clone(),
    };

    let mock_engine = MockEngine {
        id: model.clone(),
        model: mock_model.clone(),
        tokenizer: tokenizer.clone(),
    };

    // Batch
    let batch = RequestBatch {
        batch_id: "test-1".to_string(),
        created_at: std::time::Instant::now(),
        tasks: vec![
            EngineTask::Prefill {
                request_id: 1,
                input_tokens: vec![1, 2, 3],
            },
            EngineTask::Prefill {
                request_id: 2,
                input_tokens: vec![3, 2, 4],
            },
            EngineTask::Prefill {
                request_id: 3,
                input_tokens: vec![2, 2, 5],
            },
        ],
    };

    let result = mock_engine.execute_step(&batch).await;

    assert!(matches!(result, EngineResult::StepOutput { .. }));
    let EngineResult::StepOutput { outputs } = result else {
        panic!("Expected StepOutput, got {:?}", result);
    };

    assert_eq!(outputs.len(), 3);
    assert!(
        outputs
            .iter()
            .all(|o| matches!(o.state, SequenceState::Running))
    );
    assert!(outputs.iter().all(|o| o.token == 33));
}
