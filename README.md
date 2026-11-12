# GRASP Demo

Five columns side by side on one question: RAG 7B, RAG 235B (Bedrock), IRCoT 7B, GRASP SFT, GRASP GRPO.
Every column runs the trainer's own code (`grasp_grpo.baselines`, `grasp_grpo.infer.run_example`).
Specs: `docs/superpowers/specs/`.

## Install
Needs `grasp-grpo-trainer` on branch `feat/grpo-training-protocol` (commit 9ff7150 or later).

```powershell
python -m venv .venv; .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # Blackwell (RTX 50xx) needs cu128+
pip install -e "../grasp-grpo-trainer[rollout]" -e ".[dev]"
```

## Data and checkpoints
```powershell
python scripts/fetch_data.py --trainer-dir ../grasp-grpo-trainer
python scripts/fetch_checkpoint.py doancuavahk/grasp-qwen2.5-3b-sft grasp-qwen2.5-3b-sft
python scripts/fetch_checkpoint.py doancuavahk/grasp-grpo-checkpoints grpo-s100 --subfolder s100
```
`.env` holds `BEDROCK_API_KEY` (copy it from `grasp/.env`; it is gitignored). Without it, only the 235B column is off.

## Run
- Local: start the reader with `scripts/run_reader.ps1 -Model <Qwen2.5-7B-Instruct-Q6_K.gguf>`, then run
  `python -m grasp_demo --config models.local.yaml` and open http://localhost:7860.
- Vast: with the eval stack running on the box, run
  `python scripts/fetch_data.py --skip-download --root /workspace/data --trainer-dir /workspace/grasp-grpo-trainer`,
  then `python -m grasp_demo --config models.vast.yaml`. From your PC, run
  `ssh -p <port> root@<host> -L 7860:localhost:7860` and open http://localhost:7860.

## Before the defense
1. `python scripts/preflight.py --config models.<local|vast>.yaml --report <test999 grasp-grpo-*.json>`. Use the
   local machine only if its verdict says `local_ok: true`.
2. `python -m grasp_demo.showcase --qa data/grasp-gold-evidence/dev.jsonl --ids data/test_ids.json --grpo <grpo report> --baseline <single.json> --baseline <ircot-k1.json> --output data/showcase.json`
3. In the page: Lịch sử → "Chạy trước câu tiêu biểu". The saved runs replay without models if anything fails on the day.
