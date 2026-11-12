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
python scripts/fetch_checkpoint.py doancuavahk/grasp-grpo-checkpoints grpo-s250 --subfolder s250
```
`.env` holds `BEDROCK_API_KEY` (copy it from `grasp/.env`; it is gitignored). Without it, only the 235B column is off.

## Run
- Local: start the reader with `scripts/run_reader.ps1 -Model <Qwen2.5-7B-Instruct-Q6_K.gguf>`, then run
  `python -m grasp_demo --config models.local.yaml` and open http://localhost:7860.
  The script defaults to `-Context 32768`: llama-server splits `-c` across the `--parallel` slots, so 8192 / 4 = 2048
  tokens per slot is too small for GRASP/IRCoT prompts. If VRAM is tight, use a Q5_K_M or Q4_K_M GGUF instead of Q6_K
  (keep the context).
- Vast: with the eval stack running on the box:
  1. Clone grasp-demo to `/workspace/grasp-demo` and install it into the trainer venv:
     `pip install -e /workspace/grasp-demo` (or `uv pip install -e /workspace/grasp-demo`).
  2. `export READER_API_KEY=... BEDROCK_API_KEY=...` in the shell, or put them in `/workspace/grasp-demo/.env`.
  3. Fetch the GRPO checkpoint on a fresh box:
     `python scripts/fetch_checkpoint.py doancuavahk/grasp-grpo-checkpoints s250 --subfolder s250 --root /workspace/ckpts`
  4. `python scripts/fetch_data.py --skip-download --root /workspace/data --trainer-dir /workspace/grasp-grpo-trainer`
  5. Check `models.vast.yaml`: `reader.model` must equal the box's `READER_MODEL`, and `encoder_device: "cuda:1"` needs
     a 2-GPU box (use `"cuda:0"` or `cpu` on a single GPU).
  6. `python -m grasp_demo --config models.vast.yaml`. From your PC, run
     `ssh -p <port> root@<host> -L 7860:localhost:7860` and open http://localhost:7860.
  7. After the pre-run, copy the saved runs back so replays work offline:
     `scp -P <port> root@<host>:/workspace/grasp-demo/runs/*.jsonl runs/`

## Before the defense
1. Run the preflight twice for the local config, once with `encoder_device: cpu` and once with `cuda`, then decide:
   `python scripts/preflight.py --config models.local.yaml --report <test999 grasp-grpo-*.json>`.
   Use the local machine only if its verdict says `local_ok: true`. The verdict applies to the local config only; on
   Vast, vLLM preallocates VRAM so `vram` is always False there. Without `--report`, `grpo_matches` is unknown and
   `local_ok` cannot be True.
2. Showcase (on Vast use `--output /workspace/data/showcase.json`); the SFT report is passed as a `--baseline` so the
   showcase questions contrast GRPO with SFT:
   `python -m grasp_demo.showcase --qa data/grasp-gold-evidence/dev.jsonl --ids data/test_ids.json --grpo <grpo report> --baseline <sft report> --baseline <single.json> --baseline <ircot-k1.json> --output data/showcase.json`
3. In the page: Lịch sử -> "Chạy trước câu tiêu biểu". The saved runs replay without models if anything fails on the day.

## Day-of tips
- Set `HF_HUB_OFFLINE=1` once the models are cached.
- Delete any sample runs from `runs/` before presenting.
- The demo's retrieval count for GRASP includes the initial question search, while the report counts QF searches only,
  so the demo shows one more.
