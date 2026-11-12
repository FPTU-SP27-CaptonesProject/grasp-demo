# GRASP Demo Architecture

## Purpose

`grasp-demo` is a local presentation application. It runs five question-answering
methods on the same question and streams their intermediate decisions to a
browser:

| Column | Implementation | Model that decides retrieval |
| --- | --- | --- |
| RAG 7B | `grasp_grpo.baselines.run_single_step` | None; retrieve once with the question |
| RAG 235B | `grasp_grpo.baselines.run_single_step` + Bedrock reader | None; retrieve once with the question |
| IRCoT 7B | `grasp_grpo.baselines.run_ircot` | The reader's generated thought |
| GRASP SFT | `grasp_grpo.infer.run_example` | Qwen2.5-3B with the SFT adapter |
| GRASP GRPO | `grasp_grpo.infer.run_example` | Qwen2.5-3B with the GRPO adapter |

The demo is not a training program and does not update any model weights.

## Repository Layout

```text
grasp-demo/
  models.yaml                 # Paths, endpoints, adapter choices and run limits
  .env                        # Optional local secrets, never committed
  static/
    index.html                # Five-column presentation interface
    app.js                    # API, SSE rendering, history and replay
    style.css                 # Light/dark operational UI styling
  src/grasp_demo/
    server.py                 # FastAPI API, run orchestration and SSE endpoints
    services.py               # Load shared reader, retriever, proxy and Bedrock client once
    runners.py                # Adapt each of the five columns to one event protocol
    store.py                  # Thread-safe events, JSONL persistence and replay
    config.py                 # YAML/.env loading and path resolution
    metrics.py                # Per-question F1, EM and context recall
  scripts/
    fetch_data.py             # Download frozen retrieval/evidence data and make corpus.json
    fetch_checkpoint.py       # Download a PEFT adapter from Hugging Face
    run_reader.ps1            # Start llama.cpp's local 7B reader on Windows
  data/                       # Downloaded corpus, evidence QA and corpus.json; ignored by Git
  checkpoints/                # Downloaded SFT/GRPO adapters; ignored by Git
  runs/                       # Recorded demo runs as JSONL; ignored by Git
```

## Relationship with `grasp-grpo-trainer`

`grasp-demo` intentionally does not duplicate GRASP or baseline algorithms. It
installs and imports `../grasp-grpo-trainer` as a Python package:

```powershell
pip install -e ..\grasp-grpo-trainer[rollout,quantization]
pip install -e .[dev]
```

The shared modules are used as follows:

| Trainer module | Demo usage |
| --- | --- |
| `grasp_grpo.retrieval` | `CorpusRouter`, `DenseRetriever` and `BoundRetriever`; E5 + FAISS + SQLite passages/triples |
| `grasp_grpo.reader` | `GraspReader`, reader prompt, OpenAI completion client and chat template |
| `grasp_grpo.baselines` | Single-step RAG and IRCoT baseline algorithms |
| `grasp_grpo.infer` | Greedy GRASP policy loop (`RD -> QF -> KI`) |
| `grasp_grpo.model` | Load the Qwen 3B base and PEFT LoRA adapters |
| `grasp_grpo.evaluate` | Identical F1 and exact-match normalization |

This ensures report evaluation and the visual demo follow the same retrieval,
prompt, baseline, GRASP action parsing and metric definitions.

## Checkpoint Loading

Paths are declared in `models.yaml`:

```yaml
proxy:
  base: Qwen/Qwen2.5-3B-Instruct
  adapters:
    sft: checkpoints/grasp-qwen2.5-3b-sft
    grpo: checkpoints/grpo-pilot-step100
```

Each adapter directory must contain at least:

```text
adapter_config.json
adapter_model.safetensors
```

The GRPO directory may also contain `grpo_config.json`, which the demo reads to
show checkpoint metadata, such as training step or recorded F1, in the GRASP
column header.

At server startup, `DemoServices` performs this sequence:

1. Call `grasp_grpo.model.load_policy(sft_path, trainable=False)`.
   This loads Qwen2.5-3B-Instruct once and attaches the SFT LoRA adapter as the
   default PEFT adapter.
2. Call `model.load_adapter(grpo_path, adapter_name="grpo", is_trainable=False)`.
   This attaches the GRPO LoRA to the *same base Qwen model*.
3. Store both names in `ProxyPolicy`:
   - `sft` maps to PEFT's `default` adapter.
   - `grpo` maps to PEFT adapter name `grpo`.
4. For a GRASP request, acquire the proxy lock and use
   `grasp_grpo.model.active_adapter(...)` to temporarily select either SFT or
   GRPO. The previous adapter is restored after generation.

Therefore the two GRASP columns share one copy of Qwen 3B in VRAM. They do not
load two base models and do not run SFT then GRPO sequentially for one action.
Each column selects one complete LoRA policy:

```text
GRASP SFT:  Qwen2.5-3B base + SFT LoRA
GRASP GRPO: Qwen2.5-3B base + GRPO LoRA
```

The GRPO adapter is the final policy emitted by GRPO training. It is not an
extra adapter layered on top of the SFT adapter at inference time. The SFT
checkpoint is used as a separate baseline policy in the comparison.

The proxy lock is deliberate: PEFT's active adapter is mutable global model
state, so GRASP SFT and GRASP GRPO generation must take turns. Retrieval and
reader requests can still overlap with other columns.

## Request Flow

```text
Browser
  POST /api/ask
    -> FastAPI creates one run ID
    -> starts up to five runner threads
    -> each runner emits events
  GET /api/stream/<run_id>
    -> Server-Sent Events (SSE)
    -> browser appends events to the matching column
```

For an evaluation question, the server loads its question, answers,
`gold_doc_ids` and corpus name from `data/grasp-gold-evidence/dev.jsonl`. For a
free question it creates a row with the selected corpus but no answers/evidence;
the UI then displays latency and retrieval count but no score.

The retriever is wrapped by `BoundRetriever` for every column/run. It pins the
mixed-dataset row to the correct corpus and serializes E5/FAISS/SQLite access.
All columns use the same frozen index, passage store, triple cache and `top_k`.

## GRASP Event Flow

`grasp_grpo.infer.run_example` has an optional `emit(kind, data)` callback.
When it is omitted, ordinary evaluation behavior and returned prediction records
are unchanged. In the demo it emits the actual trace:

```text
retrieve(step=0, query=question)
thought
rd       # Retrieval Decision
qf       # Query Formulation
retrieve(query=qf.query)
ki       # Knowledge Integration / chosen triple
answer
done     # demo layer adds elapsed time, retrieval count and metrics
```

Document events are copied before transmission and marked `gold: true` only if
their `doc_id` is in the evaluation row's `gold_doc_ids`. Gold evidence is
never inserted into a model prompt.

## Data Requirements

`data/corpus.json` is a routing map. Each key must exactly match the dataset
field in an evaluation row:

```json
{
  "2wiki_ircot": {
    "index_dir": ".../grasp-retrieval/2wiki_ircot",
    "triple_cache": ".../grasp-triples/2wiki.sqlite"
  },
  "hotpotqa_abstracts_2017": {
    "index_dir": ".../grasp-retrieval/hotpotqa_abstracts_2017",
    "triple_cache": ".../grasp-triples/hotpotqa.sqlite"
  },
  "wiki18": {
    "index_dir": ".../grasp-retrieval/wiki18",
    "triple_cache": ".../grasp-triples/wiki18.sqlite"
  }
}
```

Each `index_dir` must contain `index.faiss` and `passages.sqlite`. Triple cache
is required because GRASP's `KI` action ranks and chooses triples from it.

## Reliability Behavior

- The server records component readiness in `/api/health`.
- Missing Bedrock credentials disable only `RAG 235B`.
- Missing GRPO adapter disables only `GRASP GRPO`.
- An exception in one runner produces that column's `error` event; other
  columns keep running.
- Every run is written to `runs/<timestamp>-<run_id>.jsonl` and can be replayed
  without calling a model or an external API.
