# RustedvLLM

A Rust-based LLM inference framework focused on continuous batching, scheduling, and inference runtime lifecycle management.

## Architecture

The system is split into three crates, communicating via async message channels:

```
┌─────────────────┐     EngineCommand      ┌─────────────────┐
│   Scheduler     │  ───────────────────►  │     Engine      │
│  (State Owner)  │ ◄────────────────────  │  (Execute Step) │
└─────────────────┘     EngineResult       └─────────────────┘
        │
        ▼ Request
   External Client
```

### Crates

| Crate | Responsibility |
|-------|---------------|
| `protocol` | Shared types: `Request`, `EngineTask`, `SequenceOutput`, `EngineCommand`, `EngineResult` |
| `scheduler` | Request lifecycle management, batch construction, state transitions |
| `engine` | `Engine` trait + `MockEngine` implementation; executes `Prefill` and `Decode` steps |

### Request Lifecycle

```
Queued -> Active -> (Finished / Aborted / Failed)
```

The scheduler owns all state transitions. The engine is a black box that only exposes:

```rust
async fn execute_step(&self, batch: &RequestBatch) -> EngineResult;
```

### Engine Execution

Each batch is partitioned into two phases:

1. **Prefill** — processes new prompts, produces the first token and KV cache
2. **Decode** — generates the next token using the cached KV state

The `MockEngine` simulates this flow with a `MockModel` and `MockTokenizer`.

## Project Status

This is a learning project for building production-grade AI infrastructure in Rust. Core scheduling logic and batched model forwarding are still in progress.

See [`docs/Design.md`](docs/Design.md) for detailed design docs.
