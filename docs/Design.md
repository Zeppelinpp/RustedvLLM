# Rusted vLLM Design

## Request Lifecycle 

```
Queued -> PrefillWaiting -> Prefilling -> DecodeWaiting -> Decoding -> Finished
- Cancelled
- Timeout
- Errored
- Evicted
```

## InferenceEngine

模拟类似 `llama.cpp` 这样的模型引擎，LLM Inference本身作为黑盒

```Rust
trait InferenceEngine {
  async fn prefill(batch);
  async fn decode_step(
  	&mut self,
    batch: DecodeBatch,
  ) -> Vec<TokenOutput>;
}
```

`FauxInferenceEngine` 作为 Inference Engine和真实推理框架（Continuos batching，schedular ...) 的桥接



## Scheduler



## Protocol



## Metrics

```Rust
struct RequestMetrics {
	queue_time,
	ttft,
	tpot,
	total_latency,
}
```

