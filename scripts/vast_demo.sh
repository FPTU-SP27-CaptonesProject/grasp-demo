#!/usr/bin/env bash
# Build and run the GRASP demo on a fresh rented Vast box (2 GPUs, e.g. 2x RTX 4090), with the same stack as the
# report evals: vLLM FP8 reader on GPU 0, vLLM proxy (base 3B + LoRA) on GPU 1, E5 on GPU 1.
#
#   git clone -b feat/demo-backends https://github.com/FPTU-SP27-CaptonesProject/grasp-demo.git /workspace/grasp-demo
#   bash /workspace/grasp-demo/scripts/vast_demo.sh setup    # once, ~20-30 min: trainer + vLLM, data, adapters, keys
#   bash /workspace/grasp-demo/scripts/vast_demo.sh start    # reader, proxy and demo in tmux; waits until healthy
#   bash /workspace/grasp-demo/scripts/vast_demo.sh prerun   # runs the showcase questions, packs runs/ for your PC
#   bash /workspace/grasp-demo/scripts/vast_demo.sh stop
#
# Then on your PC: ssh -p <port> root@<host> -L 7860:localhost:7860  and open http://localhost:7860
# GRPO_CKPT (default s250) picks the folder of doancuavahk/grasp-grpo-checkpoints shown as "GRASP · GRPO".
set -euo pipefail

DEMO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINER_DIR="${TRAINER_DIR:-/workspace/grasp-grpo-trainer}"
TRAINER_BRANCH="${TRAINER_BRANCH:-feat/grpo-training-protocol}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
SFT_DIR="${SFT_DIR:-/workspace/checkpoints/grasp-qwen2.5-3b-sft}"
GRPO_CKPT="${GRPO_CKPT:-s250}"
GRPO_LABEL="${GRPO_LABEL:-GRPO $GRPO_CKPT · test F1 0.552}"
CKPT_ROOT="${CKPT_ROOT:-/workspace/ckpts}"
TRIPLES="${TRIPLES:-cache/triples-train-20261112.sqlite}"
VAST_ENV="$TRAINER_DIR/config/vast.env"
PY="$TRAINER_DIR/.venv/bin/python"
CONFIG="$DEMO_DIR/demo.vast.yaml"
export UV_CACHE_DIR=/workspace/.cache/uv HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}" TMPDIR=/workspace/tmp
mkdir -p "$TMPDIR"

say() { printf '\n== %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# Set KEY=VALUE in vast.env (replace or append). Values never go to the terminal.
setv() { grep -q "^$1=" "$VAST_ENV" && sed -i "s#^$1=.*#$1=$2#" "$VAST_ENV" || echo "$1=$2" >> "$VAST_ENV"; }

load_env() {
  [[ -f "$VAST_ENV" ]] || die "$VAST_ENV missing: run '$0 setup' first"
  set -a; source "$VAST_ENV"; [[ -f "$DEMO_DIR/.env" ]] && source "$DEMO_DIR/.env"; set +a
}

wait_http() {  # wait_http <name> <url> [bearer] : up to 15 min (vLLM downloads and compiles on first start)
  local name=$1 url=$2 key=${3:-}
  for _ in $(seq 1 180); do
    if curl -sf -m 5 ${key:+-H "Authorization: Bearer $key"} "$url" >/dev/null; then echo "$name ready"; return 0; fi
    sleep 5
  done
  die "$name not ready after 15 min ($url); see: tmux capture-pane -pt ${name} | tail -40"
}

cmd_setup() {
  local gpus; gpus=$(nvidia-smi -L | wc -l)
  (( gpus >= 2 )) || die "this script expects 2 GPUs (found $gpus); the single-GPU layout ran out of memory on 2026-11-12"

  say "trainer ($TRAINER_BRANCH) + vLLM venvs"
  [[ -d "$TRAINER_DIR/.git" ]] || git clone -q -b "$TRAINER_BRANCH" https://github.com/FPTU-SP27-CaptonesProject/grasp-grpo-trainer.git "$TRAINER_DIR"
  git -C "$TRAINER_DIR" pull -q
  [[ -x "$PY" && -x /workspace/grasp-reader-venv/bin/vllm ]] || bash "$TRAINER_DIR/scripts/setup_vast.sh"
  export PATH="$HOME/.local/bin:$PATH"
  uv pip install -q --python "$PY" -e "$DEMO_DIR"

  say "Hugging Face access (the repos are private)"
  # Check real read access, not just a stored token: a stale HF_TOKEN (e.g. set in the Vast account) passes
  # `whoami` checks poorly and then fails every download with 401 (2026-11-13).
  can_read() { "$PY" -c 'from huggingface_hub import HfApi; HfApi().repo_info("doancuavahk/grasp-retrieval", repo_type="dataset"); HfApi().repo_info("doancuavahk/grasp-grpo-checkpoints")' >/dev/null 2>&1; }
  if ! can_read; then
    [[ -n "${HF_TOKEN:-}" ]] && { echo "HF_TOKEN in the environment cannot read the repos: ignoring it"; unset HF_TOKEN; }
    can_read || "$TRAINER_DIR/.venv/bin/hf" auth login
    can_read || die "this Hugging Face account cannot read doancuavahk/grasp-retrieval and grasp-grpo-checkpoints"
  fi

  say "keys"
  [[ -f "$VAST_ENV" ]] || cp "$TRAINER_DIR/config/vast.env.example" "$VAST_ENV"
  chmod 600 "$VAST_ENV"
  if ! grep -q '^READER_API_KEY=[0-9a-f]\{32\}$' "$VAST_ENV"; then setv READER_API_KEY "$(openssl rand -hex 16)"; fi
  if [[ ! -s "$DEMO_DIR/.env" ]] || ! grep -q '^BEDROCK_API_KEY=.' "$DEMO_DIR/.env"; then
    read -rsp "BEDROCK_API_KEY (Enter to skip; only the RAG 235B column needs it): " key; echo
    umask 077; [[ -n "$key" ]] && printf 'BEDROCK_API_KEY=%s\n' "$key" > "$DEMO_DIR/.env" || : > "$DEMO_DIR/.env"
  fi
  # Two-GPU layout used for the test999 evals: reader alone on GPU 0, proxy + E5 on GPU 1.
  setv READER_GPU 0; setv READER_GPU_UTIL 0.85; setv POLICY_GPU 1; setv POLICY_GPU_UTIL 0.60; setv RETRIEVER_GPU 1
  setv SFT_CHECKPOINT "$SFT_DIR"; setv EXTRACT_TRIPLES 0

  say "data: corpora, $TRIPLES, dev.jsonl, test_ids.json"
  (cd "$DEMO_DIR" && "$PY" scripts/fetch_data.py --root "$DATA_DIR" --trainer-dir "$TRAINER_DIR" --triples "$TRIPLES")

  say "adapters: SFT and GRPO $GRPO_CKPT"
  (cd "$DEMO_DIR" && "$PY" scripts/fetch_checkpoint.py doancuavahk/grasp-qwen2.5-3b-sft "$(basename "$SFT_DIR")" --root "$(dirname "$SFT_DIR")")
  (cd "$DEMO_DIR" && "$PY" scripts/fetch_checkpoint.py doancuavahk/grasp-grpo-checkpoints "$GRPO_CKPT" --subfolder "$GRPO_CKPT" --root "$CKPT_ROOT")
  cp "$DEMO_DIR/showcase/$GRPO_CKPT.json" "$DATA_DIR/showcase.json" 2>/dev/null || echo "no showcase/$GRPO_CKPT.json: the showcase filter will be empty"

  write_config
  say "setup done. Next: bash $0 start"
}

write_config() {
  load_env
  "$PY" - "$DEMO_DIR/models.vast.yaml" "$CONFIG" "$CKPT_ROOT/$GRPO_CKPT" "$GRPO_LABEL" "$SFT_DIR" "$DATA_DIR" <<'EOF'
import os, sys, yaml
src, dst, grpo, label, sft, data = sys.argv[1:]
c = yaml.safe_load(open(src, encoding="utf-8"))
c["reader"].update(url=os.environ["READER_URL"], model=os.environ["READER_MODEL"], tokenizer=os.environ["READER_MODEL"])
c["proxy"]["url"] = os.environ["POLICY_URL"]
c["proxy"]["adapters"] = {"sft": {"path": sft, "label": "SFT"}, "grpo": {"path": grpo, "label": label}}
c["retrieval"].update(corpus_config=f"{data}/corpus.json", encoder_device=f"cuda:{os.environ.get('RETRIEVER_GPU', '1')}")
c["eval"] = {"qa": f"{data}/grasp-gold-evidence/dev.jsonl", "ids": f"{data}/test_ids.json", "showcase": f"{data}/showcase.json"}
yaml.safe_dump(c, open(dst, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
print(f"wrote {dst}: reader {c['reader']['model']} at {c['reader']['url']}, proxy at {c['proxy']['url']}, GRPO {grpo}")
EOF
}

cmd_start() {
  load_env
  [[ -f "$CONFIG" ]] || write_config
  say "reader (GPU $READER_GPU) and proxy (GPU $POLICY_GPU)"
  tmux has-session -t reader 2>/dev/null || tmux new -d -s reader "bash $TRAINER_DIR/scripts/run_reader_vast.sh 2>&1 | tee /workspace/reader.log"
  wait_http reader "${READER_URL%/}/models" "$READER_API_KEY"
  tmux has-session -t policy 2>/dev/null || tmux new -d -s policy "bash $TRAINER_DIR/scripts/run_policy_vast.sh 2>&1 | tee /workspace/policy.log"
  wait_http policy "${POLICY_URL%/}/models" "$READER_API_KEY"

  say "demo"
  tmux kill-session -t demo 2>/dev/null || true
  # HF_HOME passed explicitly: a tmux server started earlier would not inherit this shell's environment.
  tmux new -d -s demo "cd $DEMO_DIR && set -a && source $VAST_ENV && source $DEMO_DIR/.env && set +a && HF_HOME=$HF_HOME $PY -m grasp_demo --config $CONFIG --port 7860 2>&1 | tee /workspace/demo.log"
  wait_http demo http://127.0.0.1:7860/api/health
  curl -s http://127.0.0.1:7860/api/health | "$PY" -c '
import json, sys
h = json.load(sys.stdin)
for name, c in h["columns"].items():
    state = "ok" if c["enabled"] else "OFF: " + str(c["reason"])
    print("  %-11s %s" % (name, state))
print("  GRPO label:", h["adapters"].get("grpo", {}).get("label"), "| eval questions:", h["eval_questions"], "| showcase:", h["showcase_questions"])'
  say "open on your PC: ssh -p <port> root@<host> -L 7860:localhost:7860  ->  http://localhost:7860"
}

cmd_prerun() {
  load_env
  curl -sf -X POST http://127.0.0.1:7860/api/prerun >/dev/null || die "pre-run refused (already running, or no showcase questions); see /api/prerun"
  while :; do
    s=$(curl -s http://127.0.0.1:7860/api/prerun)
    echo "  $s"
    "$PY" -c 'import json,sys; sys.exit(0 if json.loads(sys.argv[1])["running"] else 1)' "$s" || break
    sleep 15
  done
  tar -czf /workspace/demo-runs.tgz -C "$DEMO_DIR" runs
  say "saved runs packed: /workspace/demo-runs.tgz. On your PC (keep them for offline replay):"
  echo "  scp -P <port> root@<host>:/workspace/demo-runs.tgz . ; tar -xzf demo-runs.tgz -C D:\\DoAn\\grasp-demo"
}

cmd_stop() {
  for s in demo policy reader; do tmux kill-session -t "$s" 2>/dev/null && echo "stopped $s" || true; done
}

case "${1:-}" in
  setup) cmd_setup ;;
  start) cmd_start ;;
  prerun) cmd_prerun ;;
  stop) cmd_stop ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
