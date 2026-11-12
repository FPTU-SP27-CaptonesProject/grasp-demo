# GRASP Demo

Local presentation application that runs GRASP beside RAG and IRCoT baselines
using the shared implementation in `../grasp-grpo-trainer`.

## Setup

```powershell
cd D:\FU\DoAn\grasp-demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ..\grasp-grpo-trainer[rollout,quantization]
pip install -e .[dev]
Copy-Item models.yaml models.local.yaml
```

Set `BEDROCK_API_KEY` in `.env` beside `models.local.yaml` or in the
environment when the 235B column is enabled. Fetch frozen artifacts, then
start the local reader and run:

```powershell
python scripts/fetch_data.py --evidence-repo <your-evidence-HF-dataset> --triple-root ..\grasp-grpo-trainer\data\grasp-triples
python scripts/fetch_checkpoint.py doancuavahk/grasp-qwen2.5-3b-sft grasp-qwen2.5-3b-sft
python scripts/run_reader.ps1 -Model C:\models\Qwen2.5-7B-Instruct-Q6_K.gguf
```

The retrieval and evidence repositories may be private; log in using `hf auth
login` before fetching them. The fetch script refuses to create `corpus.json`
without a triple cache, because the GRASP `KI` action needs it.

Then run:

```powershell
grasp-demo --config models.local.yaml
```

Open `http://127.0.0.1:7860`. The server remains useful when an optional
component is unavailable: health reports the disabled column rather than
preventing the interface from starting.
