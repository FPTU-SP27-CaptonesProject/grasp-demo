# Demo backends, data and preflight: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the GRASP demo run end to end, both locally (llama-server + transformers) and on a rented Vast box (vLLM). It loads the data and checkpoints that actually exist on HF, and comes with a preflight that decides which of the two machines to use.

**Architecture:** One FastAPI server (`grasp_demo.server`) runs five columns through the trainer's own loops (`grasp_grpo.baselines`, `grasp_grpo.infer.run_example(emit=)`). A proxy backend (`hf` or `vllm`, in `grasp_demo.proxy`) hides where the 3B policy runs. Data and checkpoint fetching, the showcase pick and the preflight are package modules with thin scripts on top, so they can be tested.

**Tech Stack:** Python 3.12, FastAPI, sse via StreamingResponse, PyYAML, huggingface_hub, `grasp_grpo` (trainer, branch `feat/grpo-training-protocol` ≥ 9ff7150), vanilla JS/CSS frontend.

**Spec:** `docs/superpowers/specs/2026-11-12-demo-backends-design.md` (addendum to `2026-11-11-grasp-demo-design.md`).

## Global Constraints

- Work on branch `feat/demo-backends` in `D:\DoAn\grasp-demo`. Commit per task. Never push to `main`. Push the branch at the end (Task 11).
- Python for every command: `D:/DoAn/grasp-grpo-trainer/.venv/Scripts/python.exe`, referred to below as `$PY`. It already has torch 2.11+cu128 and `grasp_grpo` (editable). Run all commands from `D:/DoAn/grasp-demo`.
- Never change GRASP or baseline logic in the demo. Everything goes through `grasp_grpo`.
- No triple extraction at run time (same as eval with `EXTRACT_TRIPLES=0`).
- `strong` (Bedrock): `timeout: 60`, `retries: 1`.
- `VLLMPolicy` timeout in the demo: 60 s.
- vLLM adapter names: `demo-sft`, `demo-grpo`. Load with `POST {url}/load_lora_adapter`, unload with `POST {url}/unload_lora_adapter`, Bearer `READER_API_KEY`.
- Triple cache default: `cache/triples-train-20261112.sqlite` in `doancuavahk/grasp-retrieval`. It can be overridden with `--triples`.
- Corpus layout: `<root>/grasp-retrieval/<corpus>/passages.sqlite` and `<root>/grasp-retrieval/<corpus>/e5-large-v2/index.faiss`. `index_dir` = corpus root.
- Gold QA: `doancuavahk/grasp-gold-evidence`, file `dev.jsonl`. Question set: `test_ids.json` (999 ids).
- Preflight thresholds:
  - every component ready except `strong`;
  - VRAM peak ≤ 15 GB;
  - each question ≤ 60 s for all columns;
  - each GRASP column ≤ 40 s;
  - GRPO answer matches the test999 report on ≥ 4 of 6 questions.
- UI text is in Vietnamese. Data from the server is inserted only with `textContent`, never `innerHTML`. No `alert()`.
- Never commit secrets, `.env`, `data/`, `checkpoints/` or `runs/*.jsonl` (all already in `.gitignore`).

## Plan-time decision (differs from spec §4)

`GET /api/doc/{corpus}/{doc_id}` is **dropped**. `passages.sqlite` has no index on `doc_id`, so a lookup in wiki18 (about 21M rows) would be a full scan. Every `retrieve` event already carries each document's full `text`, so the UI opens the full text from the event itself.

The gold list in the question panel shows the gold doc ids. Each one is marked "tìm thấy bởi: …" once any column retrieves it. Task 11 records this change in the spec.

## File map

| File | Responsibility |
|---|---|
| `src/grasp_demo/config.py` (modify) | YAML loading, path resolution, adapter entries `{path, label}`, `reader_api_key` |
| `src/grasp_demo/evalset.py` (create) | Load eval rows (QA filtered by ids) and id lists (showcase) |
| `src/grasp_demo/store.py` (modify) | Run events, one read position per stream connection, persist, history, replay |
| `src/grasp_demo/proxy.py` (create) | `HFProxy`, `VLLMProxy`: the same `names`, `generate_fn(adapter)` and `close()` |
| `src/grasp_demo/services.py` (modify) | Component loaders with real health probes, proxy backend choice, `enabled(column)` |
| `src/grasp_demo/runners.py` (modify) | Use `services.proxy.generate_fn(adapter)` |
| `src/grasp_demo/server.py` (modify) | `/api/eval?showcase=1`, `/api/prerun` (POST and GET), labels in health |
| `src/grasp_demo/setup_data.py` (create) | Fetch and validate data, build `corpus.json`, fetch adapters (with subfolder) |
| `scripts/fetch_data.py`, `scripts/fetch_checkpoint.py` (rewrite) | Thin CLIs over `setup_data` |
| `src/grasp_demo/showcase.py` (create) | Pick showcase ids from test999 reports |
| `src/grasp_demo/preflight.py`, `scripts/preflight.py` (create) | Run 6 questions, measure, judge local versus rented |
| `static/index.html`, `static/app.js`, `static/style.css` (rewrite) | Readable step cards, gold marks, pinned GRPO column, history and pre-run |
| `models.local.yaml`, `models.vast.yaml` (create), `models.yaml` (delete), `README.md` (modify) | The two deployment modes |

---

### Task 0: Test environment

**Files:** none (environment only)

- [ ] **Step 1: Install the demo's web dependencies into the trainer venv**

Run: `$PY -m pip install "fastapi>=0.115,<1" "uvicorn[standard]>=0.30,<1" "PyYAML>=6,<7" "httpx>=0.28" "pytest>=8" "huggingface_hub>=0.34,<1"`
Expected: `Successfully installed …` or `Requirement already satisfied`.

- [ ] **Step 2: Run the existing tests**

Run: `$PY -m pytest -q`
Expected: `5 passed`. If anything fails at import time, stop and report: the trainer branch is wrong (must be ≥ 9ff7150).

---

### Task 1: Config with adapter entries, and eval-set loading

**Files:**
- Modify: `src/grasp_demo/config.py`
- Create: `src/grasp_demo/evalset.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `load_config(path) -> dict`, where `config["proxy"]["adapters"][name] == {"path": <abs str>, "label": str}`. Paths for `retrieval.corpus_config`, `eval.qa`, `eval.ids`, `eval.showcase` and `runs_dir` are absolute strings.
  - `reader_api_key(values: dict) -> str`
  - `load_dotenv(root)` (unchanged)
  - `evalset.load_ids(path: str | None) -> set[str]`
  - `evalset.load_eval_rows(config) -> list[dict]` (rows as `grasp_grpo.data.read_examples` returns them, filtered by `eval.ids`; ids missing from the QA file are ignored)

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:
```python
import json

from grasp_demo.config import load_config, reader_api_key
from grasp_demo.evalset import load_eval_rows, load_ids


def write(tmp_path, text):
    path = tmp_path / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_adapters_accept_strings_and_entries(tmp_path):
    config = load_config(write(tmp_path, "proxy:\n  adapters:\n    sft: ck/sft\n    grpo: {path: ck/s100, label: GRPO s100}\n"))
    adapters = config["proxy"]["adapters"]
    assert adapters["sft"] == {"path": str(tmp_path / "ck" / "sft"), "label": "sft"}
    assert adapters["grpo"] == {"path": str(tmp_path / "ck" / "s100"), "label": "GRPO s100"}


def test_eval_paths_are_resolved(tmp_path):
    config = load_config(write(tmp_path, "eval: {qa: d/dev.jsonl, ids: d/test_ids.json, showcase: d/showcase.json}\nruns_dir: runs\n"))
    assert config["eval"]["showcase"] == str(tmp_path / "d" / "showcase.json")
    assert config["runs_dir"] == str(tmp_path / "runs")


def test_reader_api_key_prefers_named_env(monkeypatch):
    monkeypatch.setenv("DEMO_KEY", "secret")
    assert reader_api_key({"api_key_env": "DEMO_KEY", "api_key": "x"}) == "secret"
    monkeypatch.delenv("DEMO_KEY")
    assert reader_api_key({"api_key_env": "DEMO_KEY"}) == "EMPTY"
    assert reader_api_key({"api_key": "x"}) == "x"


def test_eval_rows_are_filtered_by_ids(tmp_path):
    qa = tmp_path / "dev.jsonl"
    qa.write_text("\n".join(json.dumps({"id": i, "question": f"Q{i}?", "gold_answers": ["a"], "dataset": "wiki18"}) for i in ("a", "b", "c")), encoding="utf-8")
    (tmp_path / "ids.json").write_text(json.dumps({"ids": ["a", "c", "zzz"]}), encoding="utf-8")
    config = load_config(write(tmp_path, "eval: {qa: dev.jsonl, ids: ids.json}\n"))
    assert [row["id"] for row in load_eval_rows(config)] == ["a", "c"]
    assert load_ids(None) == set()
    assert load_ids(str(tmp_path / "missing.json")) == set()
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_config.py -q`
Expected: FAIL, `ImportError: cannot import name 'reader_api_key'` (and `No module named 'grasp_demo.evalset'`).

- [ ] **Step 3: Implement**

Replace `src/grasp_demo/config.py` with:
```python
"""Configuration loading with paths resolved relative to the YAML file."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PATH_KEYS = (("retrieval", "corpus_config"), ("eval", "qa"), ("eval", "ids"), ("eval", "showcase"))


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"configuration does not exist: {source}")
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("models.yaml must contain a mapping")
    config["_root"] = source.parent
    for section, key in PATH_KEYS:
        if config.get(section, {}).get(key):
            config[section][key] = str(resolve(config, config[section][key]))
    adapters = config.get("proxy", {}).get("adapters", {}) or {}
    for name, value in list(adapters.items()):
        entry = {"path": value} if isinstance(value, str) else dict(value or {})
        if not entry.get("path"):
            raise ValueError(f"proxy.adapters.{name} needs a path")
        entry["path"] = str(resolve(config, entry["path"]))
        entry["label"] = str(entry.get("label") or Path(entry["path"]).name)
        adapters[name] = entry
    if config.get("runs_dir"):
        config["runs_dir"] = str(resolve(config, config["runs_dir"]))
    return config


def resolve(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(config["_root"]) / path


def reader_api_key(values: dict[str, Any]) -> str:
    """Key from the environment variable named by `api_key_env`, else `api_key`, else EMPTY."""
    name = values.get("api_key_env")
    if name:
        return os.environ.get(name) or "EMPTY"
    return str(values.get("api_key") or "EMPTY")


def load_dotenv(root: str | Path) -> None:
    """Small dependency-free .env loader; environment variables win over disk."""
    path = Path(root) / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")
```

Create `src/grasp_demo/evalset.py`:
```python
"""Eval questions shown in the demo: the QA file filtered by an id list (test_ids.json)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_ids(path: str | None) -> set[str]:
    if not path or not Path(path).is_file():
        return set()
    return {str(value) for value in json.loads(Path(path).read_text(encoding="utf-8")).get("ids", [])}


def load_eval_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    from grasp_grpo.data import read_examples

    values = config.get("eval", {})
    path = values.get("qa")
    if not path or not Path(path).is_file():
        return []
    rows = read_examples(path)
    ids = load_ids(values.get("ids"))
    return [row for row in rows if row["id"] in ids] if ids else rows
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_config.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/grasp_demo/config.py src/grasp_demo/evalset.py tests/test_config.py
git commit -m "Config: adapter entries with labels, reader key from env, eval-set loader"
```

---

### Task 2: One read position per stream connection

**Files:**
- Modify: `src/grasp_demo/store.py`
- Test: `tests/test_store.py` (add a test)

**Interfaces:**
- Produces:
  - `DemoRun.publish(event)`, `DemoRun.close()`, `DemoRun.wait_from(index: int, timeout: float = 15.0) -> tuple[list[dict], bool]`
  - `event_stream(run) -> AsyncIterator[str]`
  - `RunStore.create(request) -> DemoRun`, `.get`, `.persist`, `.history`, `.replay` (signatures unchanged)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_store.py`:
```python
def test_two_connections_each_receive_every_event(tmp_path):
    store = RunStore(tmp_path)
    run = store.create({})
    run.publish({"type": "thought", "data": {}, "t": 0.0})

    async def collect():
        return [chunk async for chunk in event_stream(run)]

    async def both():
        first = asyncio.create_task(collect())
        second = asyncio.create_task(collect())
        await asyncio.sleep(0.05)
        run.publish({"type": "answer", "data": {}, "t": 0.1})
        run.close()
        return await first, await second

    first, second = asyncio.run(both())
    for chunks in (first, second):
        assert sum('"thought"' in chunk for chunk in chunks) == 1
        assert sum('"answer"' in chunk for chunk in chunks) == 1
        assert chunks[-1].startswith("event: close")
```

- [ ] **Step 2: Run it to see it fail**

Run: `$PY -m pytest tests/test_store.py -q`
Expected: the new test fails or hangs past pytest's default (both consumers share one queue). If it hangs, stop it with Ctrl+C. That counts as the expected failure.

- [ ] **Step 3: Implement**

In `src/grasp_demo/store.py`, replace the imports, the `DemoRun` class and `event_stream` with:
```python
import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator


@dataclass
class DemoRun:
    run_id: str
    request: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False
    changed: threading.Condition = field(default_factory=threading.Condition)

    def publish(self, event: dict[str, Any]) -> None:
        with self.changed:
            self.events.append(event)
            self.changed.notify_all()

    def close(self) -> None:
        with self.changed:
            self.closed = True
            self.changed.notify_all()

    def wait_from(self, index: int, timeout: float = 15.0) -> tuple[list[dict[str, Any]], bool]:
        """Events from `index` on (waiting up to `timeout` for one), and whether the run has closed."""
        with self.changed:
            if index >= len(self.events) and not self.closed:
                self.changed.wait(timeout)
            return list(self.events[index:]), self.closed
```
```python
async def event_stream(run: DemoRun) -> AsyncIterator[str]:
    """Every event of the run for this connection alone (its own read position), then a close event."""
    sent = 0
    while True:
        batch, closed = await asyncio.to_thread(run.wait_from, sent)
        for event in batch:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        sent += len(batch)
        if closed and not batch:
            yield "event: close\ndata: {}\n\n"
            return
        if not batch:
            yield ": keepalive\n\n"
```
In `RunStore.persist`, read the events under the condition:
```python
    def persist(self, run: DemoRun) -> Path:
        created = time.strftime("%Y%m%d-%H%M%S", time.localtime(run.created_at))
        path = self.runs_dir / f"{created}-{run.run_id}.jsonl"
        with run.changed:
            events = list(run.events)
        rows = [{"type": "request", "data": run.request, "run_id": run.run_id, "created_at": run.created_at}, *events]
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path
```
Delete the `import queue` line. `RunStore.__init__`, `create`, `get`, `history` and `replay` stay as they are.

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_store.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/grasp_demo/store.py tests/test_store.py
git commit -m "Stream: one read position per connection, so two tabs both see every event"
```

---

### Task 3: Proxy backends (hf, vllm)

**Files:**
- Create: `src/grasp_demo/proxy.py`
- Test: `tests/test_proxy.py`

**Interfaces:**
- Consumes: adapter entries `{"path", "label"}` (Task 1).
- Produces:
  - `HFProxy(adapters: dict[str, dict], *, load=None, generate=None)`
  - `VLLMProxy(url: str, api_key: str, adapters: dict[str, dict], *, admin=None, policy_factory=None)`
  - Both have `.names: dict[str, str]` (demo adapter name to served or PEFT name), `.generate_fn(adapter: str) -> Callable[[str, str, int], list[Action]]` and `.close() -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_proxy.py`:
```python
from grasp_grpo.model import Action

from grasp_demo.proxy import HFProxy, VLLMProxy

ADAPTERS = {"sft": {"path": "/ck/sft", "label": "SFT"}, "grpo": {"path": "/ck/s100", "label": "GRPO"}}


class FakeModel:
    def __init__(self):
        self.active_adapter, self.loaded, self.seen = "default", [], []

    def load_adapter(self, path, adapter_name, is_trainable):
        self.loaded.append((path, adapter_name, is_trainable))

    def set_adapter(self, name):
        self.active_adapter = name

    def parameters(self):
        return []


def test_hf_proxy_loads_the_first_adapter_as_base_and_switches():
    model = FakeModel()

    def generate(m, tokenizer, prompt, **kwargs):
        m.seen.append((m.active_adapter, kwargs["temperature"], kwargs["samples"]))
        return [Action("Yes", [], [])]

    proxy = HFProxy(ADAPTERS, load=lambda path, trainable: (model, "tok", "base"), generate=generate)
    assert proxy.names == {"sft": "default", "grpo": "grpo"}
    assert model.loaded == [("/ck/s100", "grpo", False)]
    proxy.generate_fn("grpo")("rd", "prompt", 1)
    proxy.generate_fn("sft")("rd", "prompt", 1)
    assert model.seen == [("grpo", 0.0, 1), ("default", 0.0, 1)]
    assert model.active_adapter == "default"


class FakeAdmin:
    def __init__(self, fail_first_load=False):
        self.calls, self.fail_first_load = [], fail_first_load

    def post(self, endpoint, payload):
        self.calls.append((endpoint, payload["lora_name"]))
        if self.fail_first_load and endpoint == "load_lora_adapter" and len(self.calls) == 1:
            raise RuntimeError("already loaded")


class FakePolicy:
    def __init__(self, served):
        self.served = served

    def generate(self, prompt, *, samples, max_new_tokens, temperature, top_p):
        return [Action(f"{self.served}:{temperature}", [], [])]


def test_vllm_proxy_loads_named_adapters_and_unloads_on_close():
    admin = FakeAdmin()
    proxy = VLLMProxy("http://x/v1", "key", ADAPTERS, admin=admin, policy_factory=lambda served, path: FakePolicy(served))
    assert proxy.names == {"sft": "demo-sft", "grpo": "demo-grpo"}
    assert proxy.generate_fn("grpo")("qf", "p", 1)[0].text == "demo-grpo:0.0"
    proxy.close()
    assert admin.calls == [("load_lora_adapter", "demo-sft"), ("load_lora_adapter", "demo-grpo"),
                           ("unload_lora_adapter", "demo-sft"), ("unload_lora_adapter", "demo-grpo")]


def test_vllm_proxy_reloads_a_name_left_by_an_old_server():
    admin = FakeAdmin(fail_first_load=True)
    VLLMProxy("http://x/v1", "key", {"sft": ADAPTERS["sft"]}, admin=admin, policy_factory=lambda served, path: FakePolicy(served))
    assert admin.calls == [("load_lora_adapter", "demo-sft"), ("unload_lora_adapter", "demo-sft"), ("load_lora_adapter", "demo-sft")]
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_proxy.py -q`
Expected: FAIL, `No module named 'grasp_demo.proxy'`.

- [ ] **Step 3: Implement**

`src/grasp_demo/proxy.py`:
```python
"""Where the 3B proxy runs: transformers in this process (hf) or a vLLM policy server (vllm).

Both expose `names`, `generate_fn(adapter)` (greedy, the trainer's role generate function) and `close()`.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

PREFIX = "demo-"


class HFProxy:
    """One base model with every adapter: the first adapter is PEFT's `default`, the others keep their names.
    One lock: one column generates at a time."""

    def __init__(self, adapters: dict[str, dict[str, Any]], *, load: Callable | None = None, generate: Callable | None = None):
        if load is None:
            from grasp_grpo.model import load_policy as load
        if generate is None:
            from grasp_grpo.model import generate
        self._generate = generate
        items = list(adapters.items())
        first, entry = items[0]
        self.model, self.tokenizer, _ = load(entry["path"], trainable=False)
        self.names = {first: "default"}
        for name, entry in items[1:]:
            self.model.load_adapter(entry["path"], adapter_name=name, is_trainable=False)
            self.names[name] = name
        self.lock = threading.Lock()

    def generate_fn(self, adapter: str):
        from grasp_grpo.batcher import role_generate_fn
        from grasp_grpo.model import active_adapter

        peft_name = self.names[adapter]

        def raw(prompt: str, **kwargs):
            with self.lock, active_adapter(self.model, peft_name):
                return self._generate(self.model, self.tokenizer, prompt, **kwargs)

        return role_generate_fn(raw, temperature=0.0, top_p=1.0)

    def close(self) -> None:
        return None


class VLLMProxy:
    """Adapters loaded on a running vLLM policy server as demo-<name>; unloaded on close.
    The adapter paths must exist on the server's machine (the demo runs there in vast mode)."""

    def __init__(self, url: str, api_key: str, adapters: dict[str, dict[str, Any]], *, admin=None, policy_factory: Callable | None = None):
        if admin is None:
            from grasp_grpo.policy_server import _Admin

            admin = _Admin(url, api_key)
        if policy_factory is None:
            def policy_factory(served: str, path: str):
                from grasp_grpo.model import load_tokenizer
                from grasp_grpo.policy_server import VLLMPolicy

                return VLLMPolicy(url, api_key, load_tokenizer(path), model=served, timeout=60.0)
        self.admin, self.names, self.policies = admin, {}, {}
        for name, entry in adapters.items():
            served = f"{PREFIX}{name}"
            self._load(served, entry["path"])
            self.names[name] = served
            self.policies[name] = policy_factory(served, entry["path"])

    def _load(self, served: str, path: str) -> None:
        payload = {"lora_name": served, "lora_path": path}
        try:
            self.admin.post("load_lora_adapter", payload)
        except RuntimeError:  # a previous demo server left the name loaded
            self.admin.post("unload_lora_adapter", {"lora_name": served})
            self.admin.post("load_lora_adapter", payload)

    def generate_fn(self, adapter: str):
        from grasp_grpo.batcher import role_generate_fn

        return role_generate_fn(self.policies[adapter].generate, temperature=0.0, top_p=1.0)

    def close(self) -> None:
        for served in self.names.values():
            try:
                self.admin.post("unload_lora_adapter", {"lora_name": served})
            except Exception:  # noqa: BLE001 - shutting down; the server may already be gone
                pass
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_proxy.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/grasp_demo/proxy.py tests/test_proxy.py
git commit -m "Proxy backends: hf (one base, LoRA switch, lock) and vllm (demo-<name> adapters)"
```

---

### Task 4: Services with real health probes and backend choice

**Files:**
- Modify: `src/grasp_demo/services.py` (full rewrite)
- Test: `tests/test_services.py`

**Interfaces:**
- Consumes: `HFProxy`, `VLLMProxy` (Task 3); `reader_api_key` (Task 1).
- Produces:
  - `DemoServices(config, *, loaders: dict[str, Callable[[DemoServices], None]] | None = None)`
  - `DemoServices.from_config(config, *, loaders=None)`
  - Attributes `.reader`, `.strong_reader`, `.retriever`, `.retriever_lock`, `.proxy`, `.health: dict[str, {"ready": bool, "reason"?: str}]`, `.adapter_metadata: dict[str, {"label", "path"}]`, `.adapter_errors: dict[str, str]`
  - `.enabled(column) -> tuple[bool, str | None]`, `.close()`
  - Module functions `load_reader`, `load_retriever`, `load_proxy`, `load_strong`, each taking the services object.

- [ ] **Step 1: Write the failing tests**

`tests/test_services.py`:
```python
import pytest

from grasp_demo import services as services_module
from grasp_demo.services import DemoServices, load_proxy

ADAPTERS = {"sft": {"path": "", "label": "SFT"}, "grpo": {"path": "", "label": "GRPO s100"}}


def ok(name):
    def loader(services):
        setattr(services, name, object())
    return loader


def fail(message):
    def loader(services):
        raise RuntimeError(message)
    return loader


class Proxy:
    def __init__(self, names):
        self.names = names

    def close(self):
        pass


def config():
    return {"proxy": {"adapters": {k: dict(v) for k, v in ADAPTERS.items()}}}


def test_reader_down_disables_every_reader_column():
    services = DemoServices.from_config(config(), loaders={"reader": fail("reader down"), "retriever": ok("retriever"),
                                                           "proxy": lambda s: setattr(s, "proxy", Proxy({"sft": "default", "grpo": "grpo"})), "strong": ok("strong_reader")})
    assert services.enabled("rag_235b") == (True, None)
    for column in ("rag_7b", "ircot_7b", "grasp_sft", "grasp_grpo"):
        enabled, reason = services.enabled(column)
        assert not enabled and "reader down" in reason
    assert services.adapter_metadata["grpo"]["label"] == "GRPO s100"


def test_missing_bedrock_key_disables_only_the_235b_column():
    services = DemoServices.from_config(config(), loaders={"reader": ok("reader"), "retriever": ok("retriever"),
                                                           "proxy": lambda s: setattr(s, "proxy", Proxy({"sft": "default", "grpo": "grpo"})), "strong": fail("BEDROCK_API_KEY is not set")})
    assert services.enabled("rag_235b")[0] is False
    assert all(services.enabled(c)[0] for c in ("rag_7b", "ircot_7b", "grasp_sft", "grasp_grpo"))


def test_missing_grpo_adapter_disables_only_its_column(tmp_path, monkeypatch):
    sft = tmp_path / "sft"
    sft.mkdir()
    (sft / "adapter_config.json").write_text("{}", encoding="utf-8")
    values = config()
    values["proxy"]["adapters"]["sft"]["path"] = str(sft)
    values["proxy"]["adapters"]["grpo"]["path"] = str(tmp_path / "nope")
    monkeypatch.setattr(services_module, "HFProxy", lambda adapters: Proxy({name: name for name in adapters}))
    services = DemoServices.from_config(values, loaders={"reader": ok("reader"), "retriever": ok("retriever"), "proxy": load_proxy, "strong": ok("strong_reader")})
    assert services.enabled("grasp_sft") == (True, None)
    enabled, reason = services.enabled("grasp_grpo")
    assert not enabled and "nope" in reason


def test_vllm_backend_is_chosen_from_config(tmp_path, monkeypatch):
    sft = tmp_path / "sft"
    sft.mkdir()
    (sft / "adapter_config.json").write_text("{}", encoding="utf-8")
    seen = {}
    monkeypatch.setenv("READER_API_KEY", "k")
    monkeypatch.setattr(services_module, "VLLMProxy", lambda url, key, adapters: seen.update(url=url, key=key, names=list(adapters)) or Proxy({"sft": "demo-sft"}))
    values = {"proxy": {"backend": "vllm", "url": "http://localhost:8001/v1", "adapters": {"sft": {"path": str(sft), "label": "SFT"}}}}
    services = DemoServices(values)
    load_proxy(services)
    assert seen == {"url": "http://localhost:8001/v1", "key": "k", "names": ["sft"]}


def test_unknown_backend_is_an_error(tmp_path):
    sft = tmp_path / "sft"
    sft.mkdir()
    (sft / "adapter_config.json").write_text("{}", encoding="utf-8")
    services = DemoServices({"proxy": {"backend": "tgi", "adapters": {"sft": {"path": str(sft), "label": "SFT"}}}})
    with pytest.raises(ValueError, match="tgi"):
        load_proxy(services)
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_services.py -q`
Expected: FAIL, `ImportError: cannot import name 'load_proxy'`.

- [ ] **Step 3: Implement**

Replace `src/grasp_demo/services.py` with:
```python
"""Resources shared by every column, loaded once. A component that fails its probe disables only the
columns that need it (with the reason); the server keeps running."""
from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import reader_api_key
from .proxy import HFProxy, VLLMProxy

NEEDS = {
    "rag_7b": ("reader", "retriever"),
    "rag_235b": ("strong", "retriever"),
    "ircot_7b": ("reader", "retriever"),
    "grasp_sft": ("reader", "retriever", "proxy"),
    "grasp_grpo": ("reader", "retriever", "proxy"),
}


def load_reader(services: DemoServices) -> None:
    from grasp_grpo.reader import GraspReader, chat_renderer, openai_completion

    values = services.config["reader"]
    complete = openai_completion(values["url"], reader_api_key(values), values["model"], temperature=0.0)
    complete("Say OK.", stop=None, max_tokens=4)  # probe: raises when the reader is down
    services.reader = GraspReader(complete, chat_renderer(values.get("tokenizer") or values["model"]))


def load_retriever(services: DemoServices) -> None:
    from grasp_grpo.retrieval import CorpusRouter

    values = services.config["retrieval"]
    services.retriever = CorpusRouter(values["corpus_config"], values.get("encoder", "intfloat/e5-large-v2"),
                                      top_k=int(values.get("top_k", 10)), max_candidate_triples=int(values.get("max_candidate_triples", 20)),
                                      encoder_device=values.get("encoder_device"))


def load_proxy(services: DemoServices) -> None:
    values = services.config["proxy"]
    usable = {}
    for name, entry in values.get("adapters", {}).items():
        if Path(entry["path"], "adapter_config.json").is_file():
            usable[name] = entry
        else:
            services.adapter_errors[name] = f"missing adapter_config.json in {entry['path']}"
    if not usable:
        raise FileNotFoundError("; ".join(services.adapter_errors.values()) or "no proxy adapters configured")
    backend = values.get("backend", "hf")
    if backend == "hf":
        services.proxy = HFProxy(usable)
    elif backend == "vllm":
        key = os.environ.get(values.get("api_key_env", "READER_API_KEY")) or "EMPTY"
        services.proxy = VLLMProxy(values["url"], key, usable)
    else:
        raise ValueError(f"unknown proxy backend {backend!r} (hf or vllm)")


def load_strong(services: DemoServices) -> None:
    from openai import OpenAI

    from grasp_grpo.baselines import BEDROCK_URL, ChatReader

    key = os.environ.get("BEDROCK_API_KEY")
    if not key:
        raise RuntimeError("BEDROCK_API_KEY is not set")
    values = services.config["strong"]
    client = OpenAI(base_url=values.get("url", BEDROCK_URL), api_key=key, timeout=float(values.get("timeout", 60)), max_retries=int(values.get("retries", 1)))
    client.models.list()  # probe: GET /models checks the key and the endpoint
    services.strong_reader = ChatReader(client, values["model"])


LOADERS: dict[str, Callable[[Any], None]] = {"reader": load_reader, "retriever": load_retriever, "proxy": load_proxy, "strong": load_strong}


class DemoServices:
    def __init__(self, config: dict[str, Any], *, loaders: dict[str, Callable[[Any], None]] | None = None):
        self.config, self.loaders = config, LOADERS if loaders is None else loaders  # {} = load nothing (UI checks)
        self.reader = self.strong_reader = self.retriever = self.proxy = None
        self.retriever_lock = threading.Lock()
        self.health: dict[str, dict[str, Any]] = {}
        self.adapter_errors: dict[str, str] = {}
        adapters = config.get("proxy", {}).get("adapters", {}) or {}
        self.adapter_metadata = {name: {"label": entry["label"], "path": entry["path"]} for name, entry in adapters.items()}

    @classmethod
    def from_config(cls, config: dict[str, Any], *, loaders: dict[str, Callable[[Any], None]] | None = None) -> DemoServices:
        services = cls(config, loaders=loaders)
        services.load()
        return services

    def load(self) -> None:
        for name, loader in self.loaders.items():
            try:
                loader(self)
                self.health[name] = {"ready": True}
            except Exception as error:  # a presentation server must survive one unavailable component
                self.health[name] = {"ready": False, "reason": f"{type(error).__name__}: {error}"}

    def enabled(self, column: str) -> tuple[bool, str | None]:
        missing = [item for item in NEEDS[column] if not self.health.get(item, {}).get("ready")]
        if missing:
            return False, "; ".join(self.health.get(item, {}).get("reason", f"{item} not loaded") for item in missing)
        if column.startswith("grasp_"):
            adapter = column.removeprefix("grasp_")
            if adapter not in self.proxy.names:
                return False, self.adapter_errors.get(adapter, f"adapter {adapter} is not loaded")
        return True, None

    def close(self) -> None:
        if self.proxy is not None:
            self.proxy.close()
        if self.retriever is not None:
            self.retriever.close()
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_services.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/grasp_demo/services.py tests/test_services.py
git commit -m "Services: real reader/Bedrock probes, hf|vllm proxy backend, per-adapter column state"
```

---

### Task 5: Runners use the proxy interface

**Files:**
- Modify: `src/grasp_demo/runners.py:181-182`
- Test: `tests/test_runners.py` (add a test)

**Interfaces:**
- Consumes: `services.proxy.generate_fn(adapter)` (Task 3), `run_example(..., emit=)` (trainer 9ff7150).
- Produces: `run_column(services, column, example, emit) -> dict` (unchanged signature). It emits the trainer's events, then `done` with `{seconds, retrievals, f1, em, recall}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_runners.py`:
```python
import threading

from grasp_grpo.evaluate import f1
from grasp_grpo.model import Action
from grasp_grpo.reader import ReaderStep

EXAMPLE = {"id": "q", "question": "Q?", "gold_answers": ["A"], "gold_doc_ids": ["d2"], "metadata": {"dataset": "wiki18"}}


class Retriever:
    def search(self, query):
        return [{"doc_id": "d2" if query == "gold query" else "d1", "title": "T", "text": "x", "triples": [{"h": 1}]}]

    def candidate_triples(self, query, passages, question=None):
        return [{"doc_id": p["doc_id"], "head": "H", "relation": "r", "tail": "T"} for p in passages]


class Reader:
    def __init__(self):
        self.calls = 0

    def step(self, question, context, thoughts, transcript):
        self.calls += 1
        return ReaderStep("<think> need </think>\n", "need", None) if self.calls == 1 else ReaderStep("<answer> A </answer>", None, "A")

    def final(self, *args):
        return "A"


class Proxy:
    names = {"sft": "default", "grpo": "grpo"}

    def generate_fn(self, adapter):
        texts = {"rd": "Yes (retrieval is required)", "qf": "Reasoning Chain: None\nKnowledge Gap: need\nQuery: gold query", "ki": "<H; r; T>"}
        return lambda role, prompt, samples: [Action(texts[role], [1], [0.0])]


class Services:
    config = {"ircot": {}, "grasp": {}}

    def __init__(self):
        self.retriever, self.retriever_lock, self.reader, self.proxy = Retriever(), threading.Lock(), Reader(), Proxy()

    def enabled(self, column):
        return True, None


def test_grasp_column_streams_steps_and_scores():
    events = []
    result = run_column(Services(), "grasp_grpo", EXAMPLE, lambda kind, data: events.append((kind, data)))
    kinds = [kind for kind, _ in events]
    assert kinds == ["retrieve", "thought", "rd", "qf", "retrieve", "ki", "answer", "done"]
    first_docs = events[0][1]["docs"]
    assert "triples" not in first_docs[0] and first_docs[0]["gold"] is False
    assert events[4][1]["docs"][0]["gold"] is True
    done = events[-1][1]
    assert done["f1"] == f1(result["prediction"], ["A"]) == 1.0 and done["recall"] == 1.0 and done["retrievals"] == 2
```

- [ ] **Step 2: Run it to see it fail**

Run: `$PY -m pytest tests/test_runners.py -q`
Expected: FAIL, `AttributeError: 'Proxy' object has no attribute 'generate'`.

- [ ] **Step 3: Implement**

In `src/grasp_demo/runners.py`, replace the GRASP branch:
```python
    else:
        adapter = column.removeprefix("grasp_")
        result = run_example(services.proxy.generate_fn(adapter), services.reader, bound, example, int(services.config.get("grasp", {}).get("max_steps", 5)), emit=traced)
```

- [ ] **Step 4: Run all tests**

Run: `$PY -m pytest -q`
Expected: every test passes.

- [ ] **Step 5: Commit**

```bash
git add src/grasp_demo/runners.py tests/test_runners.py
git commit -m "Runners: GRASP columns take generate_fn from the proxy backend"
```

---

### Task 6: Server: showcase filter, pre-run, labels

**Files:**
- Modify: `src/grasp_demo/server.py`
- Test: `tests/test_server.py` (add tests; update `make_config`)

**Interfaces:**
- Consumes: `load_eval_rows`, `load_ids` (Task 1), `DemoServices.adapter_metadata` (Task 4).
- Produces:
  - `GET /api/eval?dataset=&query=&showcase=0|1` returns `[{id, dataset, question, gold_answers, gold_doc_ids, showcase: bool}]`.
  - `POST /api/prerun` returns `{running, done, total, error}` (409 if one is running, 404 if no showcase).
  - `GET /api/prerun` returns the same status.
  - `GET /api/health` returns `{uptime_seconds, components, columns, adapters, eval_questions, showcase_questions}`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_server.py`:
- add `import threading` at the top;
- give the fake `Services` class an `adapter_metadata = {"sft": {"label": "SFT", "path": "x"}}` class attribute;
- replace `make_config` with:
```python
def make_config(tmp_path):
    qa = tmp_path / "qa.jsonl"
    rows = [{"id": i, "question": f"Question {i}?", "gold_answers": ["answer"], "gold_doc_ids": ["doc"], "dataset": "wiki18"} for i in ("q1", "q2")]
    qa.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (tmp_path / "ids.json").write_text(json.dumps({"ids": ["q1", "q2"]}), encoding="utf-8")
    (tmp_path / "showcase.json").write_text(json.dumps({"ids": ["q2"]}), encoding="utf-8")
    config = tmp_path / "models.yaml"
    config.write_text("eval:\n  qa: qa.jsonl\n  ids: ids.json\n  showcase: showcase.json\nruns_dir: runs\n", encoding="utf-8")
    return config
```
Then append:
```python
def test_eval_lists_showcase_questions(tmp_path):
    app = create_app(make_config(tmp_path), services=Services())
    with TestClient(app) as client:
        assert [row["id"] for row in client.get("/api/eval").json()] == ["q1", "q2"]
        assert [row["id"] for row in client.get("/api/eval?showcase=1").json()] == ["q2"]
        assert client.get("/api/health").json()["showcase_questions"] == 1


def test_prerun_refuses_a_second_concurrent_prerun(tmp_path, monkeypatch):
    release = threading.Event()

    def slow_run(services, column, example, emit):
        release.wait(5)
        emit("done", {"seconds": 0.1, "retrievals": 0, "f1": 1.0, "em": 1.0, "recall": 1.0})

    monkeypatch.setattr("grasp_demo.server.run_column", slow_run)
    app = create_app(make_config(tmp_path), services=Services())
    with TestClient(app) as client:
        first = client.post("/api/prerun")
        assert first.status_code == 200 and first.json()["total"] == 1
        assert client.post("/api/prerun").status_code == 409
        release.set()
        for _ in range(50):
            if not client.get("/api/prerun").json()["running"]:
                break
            threading.Event().wait(0.1)
        status = client.get("/api/prerun").json()
        assert status == {"running": False, "done": 1, "total": 1, "error": None}
        assert len(client.get("/api/history").json()) == 1
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_server.py -q`
Expected: the two new tests FAIL (404 on `/api/prerun`, the showcase filter is ignored). The old tests still pass.

- [ ] **Step 3: Implement**

Replace `src/grasp_demo/server.py` with:
```python
"""FastAPI server for the GRASP presentation demo."""
from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import load_config, load_dotenv
from .evalset import load_eval_rows, load_ids
from .runners import COLUMNS, run_column
from .services import DemoServices
from .store import DemoRun, RunStore, event_stream


class AskRequest(BaseModel):
    eval_id: str | None = None
    question: str | None = None
    corpus: str = "wiki18"
    columns: list[str] = Field(default_factory=lambda: list(COLUMNS))


def create_app(config_path: str | Path, *, services: DemoServices | None = None) -> FastAPI:
    config = load_config(config_path)
    load_dotenv(config["_root"])
    store = RunStore(config.get("runs_dir", Path(config["_root"]) / "runs"))
    eval_rows = load_eval_rows(config)
    showcase_ids = load_ids(config.get("eval", {}).get("showcase")) & {row["id"] for row in eval_rows}
    services = services or DemoServices.from_config(config)
    started = time.monotonic()
    prerun: dict[str, Any] = {"running": False, "done": 0, "total": 0, "error": None}
    prerun_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        services.close()

    app = FastAPI(title="GRASP Demo", lifespan=lifespan)
    static = Path(__file__).resolve().parents[2] / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    def execute(run: DemoRun, example: dict[str, Any], columns: list[str]) -> None:
        begun = time.monotonic()

        def publish(column: str, kind: str, data: dict[str, Any]) -> None:
            run.publish({"run_id": run.run_id, "column": column, "type": kind, "data": data, "t": time.monotonic() - begun})

        def one(column: str) -> None:
            try:
                run_column(services, column, example, lambda kind, data: publish(column, kind, data))
            except Exception as error:  # a failing column must not block the others
                publish(column, "error", {"message": f"{type(error).__name__}: {error}"})

        try:
            with ThreadPoolExecutor(max_workers=len(columns)) as pool:
                for future in as_completed([pool.submit(one, column) for column in columns]):
                    future.result()
        finally:
            store.persist(run)
            run.close()

    def new_run(example: dict[str, Any], columns: list[str], **extra: Any) -> DemoRun:
        return store.create({"example": example, "columns": columns, "config": str(config.get("_root")), **extra})

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/health")
    def health():
        columns = {}
        for column in COLUMNS:
            enabled, reason = services.enabled(column)
            columns[column] = {"enabled": enabled, "reason": reason}
        return {"uptime_seconds": time.monotonic() - started, "components": services.health, "columns": columns,
                "adapters": services.adapter_metadata, "eval_questions": len(eval_rows), "showcase_questions": len(showcase_ids)}

    @app.get("/api/eval")
    def eval_questions(dataset: str | None = None, query: str = "", showcase: int = 0):
        wanted = bool(showcase)
        needle = query.casefold().strip()
        rows = [row for row in eval_rows
                if (not dataset or row.get("metadata", {}).get("dataset") == dataset)
                and (not needle or needle in row["question"].casefold())
                and (not wanted or row["id"] in showcase_ids)]
        return [{"id": row["id"], "dataset": row.get("metadata", {}).get("dataset"), "question": row["question"],
                 "gold_answers": row["gold_answers"], "gold_doc_ids": row.get("gold_doc_ids", []), "showcase": row["id"] in showcase_ids} for row in rows]

    def resolve(request: AskRequest) -> dict[str, Any]:
        if request.eval_id:
            found = next((row for row in eval_rows if row["id"] == request.eval_id), None)
            if found is None:
                raise HTTPException(404, f"unknown eval_id: {request.eval_id}")
            return found
        if not request.question or not request.question.strip():
            raise HTTPException(422, "provide eval_id or a non-empty question")
        return {"id": f"free::{int(time.time() * 1000)}", "question": request.question.strip(), "gold_answers": [], "gold_doc_ids": [], "metadata": {"dataset": request.corpus}}

    @app.post("/api/ask")
    def ask(request: AskRequest):
        invalid = sorted(set(request.columns) - set(COLUMNS))
        if invalid:
            raise HTTPException(422, f"unknown columns: {', '.join(invalid)}")
        if not request.columns:
            raise HTTPException(422, "select at least one column")
        example = resolve(request)
        run = new_run(example, request.columns)
        threading.Thread(target=execute, args=(run, example, request.columns), name=f"grasp-demo-{run.run_id}", daemon=True).start()
        return {"run_id": run.run_id}

    @app.post("/api/prerun")
    def start_prerun():
        rows = [row for row in eval_rows if row["id"] in showcase_ids]
        if not rows:
            raise HTTPException(404, "no showcase questions (eval.showcase)")
        with prerun_lock:
            if prerun["running"]:
                raise HTTPException(409, "a pre-run is already in progress")
            prerun.update(running=True, done=0, total=len(rows), error=None)

        def work() -> None:
            try:
                for row in rows:
                    columns = [column for column in COLUMNS if services.enabled(column)[0]]
                    execute(new_run(row, columns, prerun=True), row, columns)
                    prerun["done"] += 1
            except Exception as error:  # reported through GET /api/prerun
                prerun["error"] = f"{type(error).__name__}: {error}"
            finally:
                prerun["running"] = False

        threading.Thread(target=work, name="grasp-demo-prerun", daemon=True).start()
        return dict(prerun)

    @app.get("/api/prerun")
    def prerun_status():
        return dict(prerun)

    @app.get("/api/stream/{run_id}")
    async def stream(run_id: str):
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        return StreamingResponse(event_stream(run), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/history")
    def history():
        return store.history()

    @app.get("/api/replay/{filename}")
    def replay(filename: str):
        try:
            return JSONResponse(store.replay(filename))
        except FileNotFoundError:
            raise HTTPException(404, "saved run not found") from None

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GRASP demonstration")
    parser.add_argument("--config", default="models.local.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.config), host=args.host, port=args.port)
```
Note: in `eval_questions`, `showcase` is the query parameter (`?showcase=1`). The id set is `showcase_ids`, so the two never clash. Also delete the old `_load_eval` function, now replaced by `load_eval_rows`.

Also add `src/grasp_demo/__main__.py`, so `$PY -m grasp_demo --config …` works:
```python
from .server import main

main()
```

- [ ] **Step 4: Run all tests**

Run: `$PY -m pytest -q`
Expected: every test passes (`test_server.py`: 4).

- [ ] **Step 5: Commit**

```bash
git add src/grasp_demo/server.py src/grasp_demo/__main__.py tests/test_server.py
git commit -m "Server: showcase filter, one-at-a-time pre-run of showcase questions, adapter labels in health"
```

---

### Task 7: Fetch data and checkpoints that match HF

**Files:**
- Create: `src/grasp_demo/setup_data.py`
- Rewrite: `scripts/fetch_data.py`, `scripts/fetch_checkpoint.py`
- Test: `tests/test_setup_data.py`

**Interfaces:**
- Produces:
  - `CORPORA`, `DEFAULT_TRIPLES = "cache/triples-train-20261112.sqlite"`
  - `corpus_config(retrieval_root: Path, triples: Path) -> dict[str, dict[str, str]]` (raises `FileNotFoundError` listing every bad path)
  - `check_eval_files(qa: Path, ids: Path) -> int` (raises `ValueError` naming missing ids)
  - `fetch_adapter(repo: str, name: str, root: Path, *, subfolder: str | None = None, revision: str | None = None, repo_type: str = "model", download=None) -> Path`
  - `main_data(argv=None)`, `main_checkpoint(argv=None)`

- [ ] **Step 1: Write the failing tests**

`tests/test_setup_data.py`:
```python
import json
import sqlite3

import pytest

from grasp_demo.setup_data import CORPORA, check_eval_files, corpus_config, fetch_adapter


def make_tree(root, *, skip=None, triples_table=True):
    for corpus in CORPORA:
        (root / corpus / "e5-large-v2").mkdir(parents=True)
        (root / corpus / "passages.sqlite").write_bytes(b"")
        if corpus != skip:
            (root / corpus / "e5-large-v2" / "index.faiss").write_bytes(b"")
    (root / "cache").mkdir()
    db = sqlite3.connect(root / "cache" / "t.sqlite")
    if triples_table:
        db.execute("CREATE TABLE triples (doc_id TEXT, model TEXT, triples TEXT)")
    db.commit()
    db.close()
    return root / "cache" / "t.sqlite"


def test_corpus_config_points_every_corpus_at_one_triple_cache(tmp_path):
    triples = make_tree(tmp_path)
    config = corpus_config(tmp_path, triples)
    assert set(config) == set(CORPORA)
    assert config["wiki18"] == {"index_dir": str((tmp_path / "wiki18").resolve()), "triple_cache": str(triples.resolve())}


def test_corpus_config_names_the_missing_index(tmp_path):
    triples = make_tree(tmp_path, skip="wiki18")
    with pytest.raises(FileNotFoundError, match=r"wiki18.e5-large-v2.index\.faiss"):
        corpus_config(tmp_path, triples)


def test_corpus_config_rejects_a_cache_without_triples_table(tmp_path):
    triples = make_tree(tmp_path, triples_table=False)
    with pytest.raises(FileNotFoundError, match="no triples table"):
        corpus_config(tmp_path, triples)


def test_check_eval_files_reports_missing_ids(tmp_path):
    qa = tmp_path / "dev.jsonl"
    qa.write_text(json.dumps({"id": "a", "question": "Q?", "gold_answers": ["x"]}) + "\n", encoding="utf-8")
    ids = tmp_path / "test_ids.json"
    ids.write_text(json.dumps({"ids": ["a"]}), encoding="utf-8")
    assert check_eval_files(qa, ids) == 1
    ids.write_text(json.dumps({"ids": ["a", "b"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="b"):
        check_eval_files(qa, ids)


def test_fetch_adapter_lifts_a_subfolder_to_the_top(tmp_path):
    def download(repo, *, repo_type, revision, local_dir, allow_patterns):
        assert allow_patterns == ["s100/*"]
        (local_dir / "s100").mkdir(parents=True)
        (local_dir / "s100" / "adapter_config.json").write_text("{}", encoding="utf-8")
        (local_dir / "s100" / "adapter_model.safetensors").write_bytes(b"w")

    target = fetch_adapter("org/ckpts", "grpo-s100", tmp_path, subfolder="s100", download=download)
    assert target == tmp_path / "grpo-s100"
    assert sorted(p.name for p in target.iterdir()) == ["adapter_config.json", "adapter_model.safetensors"]
    assert not (tmp_path / ".staging").exists()
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_setup_data.py -q`
Expected: FAIL, `No module named 'grasp_demo.setup_data'`.

- [ ] **Step 3: Implement**

`src/grasp_demo/setup_data.py`:
```python
"""Fetch and check the demo's data and adapters. The layout is the one on HF (and on the Vast boxes):

<root>/grasp-retrieval/<corpus>/passages.sqlite, <root>/grasp-retrieval/<corpus>/e5-large-v2/index.faiss,
<root>/grasp-retrieval/cache/<one shared triple cache>, <root>/grasp-gold-evidence/dev.jsonl, <root>/test_ids.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

CORPORA = ("2wiki_ircot", "hotpotqa_abstracts_2017", "wiki18")
DEFAULT_TRIPLES = "cache/triples-train-20261112.sqlite"
PROJECT = Path(__file__).resolve().parents[2]


def _has_triples_table(path: Path) -> bool:
    try:
        db = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
        try:
            return db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'triples'").fetchone() is not None
        finally:
            db.close()
    except sqlite3.Error:
        return False


def corpus_config(retrieval_root: Path, triples: Path) -> dict[str, dict[str, str]]:
    problems = []
    for corpus in CORPORA:
        for path in (retrieval_root / corpus / "passages.sqlite", retrieval_root / corpus / "e5-large-v2" / "index.faiss"):
            if not path.is_file():
                problems.append(str(path))
    if not triples.is_file():
        problems.append(str(triples))
    elif not _has_triples_table(triples):
        problems.append(f"{triples}: no triples table")
    if problems:
        raise FileNotFoundError("missing or invalid: " + ", ".join(problems))
    return {corpus: {"index_dir": str((retrieval_root / corpus).resolve()), "triple_cache": str(triples.resolve())} for corpus in CORPORA}


def check_eval_files(qa: Path, ids: Path) -> int:
    from grasp_grpo.data import read_examples

    wanted = [str(value) for value in json.loads(ids.read_text(encoding="utf-8"))["ids"]]
    have = {row["id"] for row in read_examples(qa)}
    missing = [value for value in wanted if value not in have]
    if missing:
        raise ValueError(f"{len(missing)} ids of {ids} are not in {qa}, e.g. {missing[:3]}")
    return len(wanted)


def fetch_adapter(repo: str, name: str, root: Path, *, subfolder: str | None = None, revision: str | None = None, repo_type: str = "model", download=None) -> Path:
    if download is None:
        from huggingface_hub import snapshot_download as download
    target = root / name
    if subfolder is None:
        download(repo, repo_type=repo_type, revision=revision, local_dir=target, allow_patterns=None)
    else:
        staging = root / ".staging" / name
        download(repo, repo_type=repo_type, revision=revision, local_dir=staging, allow_patterns=[f"{subfolder.strip('/')}/*"])
        source = staging / subfolder.strip("/")
        if not source.is_dir():
            raise FileNotFoundError(f"{repo} has no folder {subfolder}")
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            destination = target / item.name
            if destination.exists():
                shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
            shutil.move(str(item), destination)
        shutil.rmtree(root / ".staging", ignore_errors=True)
    if not (target / "adapter_config.json").is_file():
        raise FileNotFoundError(f"{target} is not a PEFT adapter (no adapter_config.json)")
    return target


def main_data(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch (or only check) the demo data and write <root>/corpus.json")
    parser.add_argument("--root", type=Path, default=PROJECT / "data")
    parser.add_argument("--retrieval-repo", default="doancuavahk/grasp-retrieval")
    parser.add_argument("--gold-repo", default="doancuavahk/grasp-gold-evidence")
    parser.add_argument("--triples", default=DEFAULT_TRIPLES, help="triple cache path inside the retrieval repo")
    parser.add_argument("--trainer-dir", type=Path, help="grasp-grpo-trainer checkout: test_ids.json is copied from its config/")
    parser.add_argument("--skip-download", action="store_true", help="only check the files and write corpus.json (Vast: --root /workspace/data)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not args.skip_download:
        from huggingface_hub import snapshot_download

        snapshot_download(args.retrieval_repo, repo_type="dataset", local_dir=root / "grasp-retrieval", allow_patterns=[f"{c}/**" for c in CORPORA] + [args.triples])
        snapshot_download(args.gold_repo, repo_type="dataset", local_dir=root / "grasp-gold-evidence", allow_patterns=["dev.jsonl"])
    ids = root / "test_ids.json"
    if args.trainer_dir:
        shutil.copyfile(args.trainer_dir / "config" / "test_ids.json", ids)
    if not ids.is_file():
        raise SystemExit(f"{ids} is missing: pass --trainer-dir <grasp-grpo-trainer checkout>")
    config = corpus_config(root / "grasp-retrieval", root / "grasp-retrieval" / args.triples)
    count = check_eval_files(root / "grasp-gold-evidence" / "dev.jsonl", ids)
    (root / "corpus.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"corpus_config": str(root / "corpus.json"), "test_questions": count, "triples": args.triples}))


def main_checkpoint(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download one PEFT adapter into <root>/<name>")
    parser.add_argument("repository")
    parser.add_argument("name")
    parser.add_argument("--subfolder", help="folder inside a multi-checkpoint repo, e.g. s100")
    parser.add_argument("--root", type=Path, default=PROJECT / "checkpoints")
    parser.add_argument("--revision")
    parser.add_argument("--repo-type", default="model")
    args = parser.parse_args(argv)
    print(fetch_adapter(args.repository, args.name, args.root, subfolder=args.subfolder, revision=args.revision, repo_type=args.repo_type))
```

Replace `scripts/fetch_data.py` with:
```python
"""python scripts/fetch_data.py [--trainer-dir ../grasp-grpo-trainer] [--skip-download --root /workspace/data]"""
from grasp_demo.setup_data import main_data

if __name__ == "__main__":
    main_data()
```
Replace `scripts/fetch_checkpoint.py` with:
```python
"""python scripts/fetch_checkpoint.py doancuavahk/grasp-grpo-checkpoints grpo-s100 --subfolder s100"""
from grasp_demo.setup_data import main_checkpoint

if __name__ == "__main__":
    main_checkpoint()
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_setup_data.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/grasp_demo/setup_data.py scripts/fetch_data.py scripts/fetch_checkpoint.py tests/test_setup_data.py
git commit -m "Data and adapters as on HF: shared triple cache, test_ids, subfolder checkpoints, checks"
```

---

### Task 8: Showcase pick from the test999 reports

**Files:**
- Create: `src/grasp_demo/showcase.py`
- Test: `tests/test_showcase.py`

**Interfaces:**
- Produces:
  - `pick_showcase(examples: list[dict], grpo: dict, baselines: list[dict], *, per_dataset: int = 3) -> list[str]`, where reports are policy_eval/baselines JSON with `per_question: [{id, f1, em}]`
  - CLI `python -m grasp_demo.showcase --qa … --ids … --grpo … --baseline … --baseline … --output data/showcase.json`

- [ ] **Step 1: Write the failing test**

`tests/test_showcase.py`:
```python
from grasp_demo.showcase import pick_showcase


def report(scores):
    return {"per_question": [{"id": k, "f1": v, "em": v} for k, v in scores.items()]}


def test_pick_prefers_grpo_wins_with_more_gold_docs_per_dataset():
    examples = [
        {"id": "a", "gold_doc_ids": ["1"], "metadata": {"dataset": "wiki18"}},
        {"id": "b", "gold_doc_ids": ["1", "2"], "metadata": {"dataset": "wiki18"}},
        {"id": "c", "gold_doc_ids": ["1", "2"], "metadata": {"dataset": "2wiki_ircot"}},
        {"id": "d", "gold_doc_ids": ["1", "2"], "metadata": {"dataset": "2wiki_ircot"}},
        {"id": "e", "gold_doc_ids": ["1"], "metadata": {"dataset": "wiki18"}},
    ]
    grpo = report({"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.5, "e": 1.0})
    single = report({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0})
    ircot = report({"a": 0.2, "b": 0.0, "c": 0.4, "d": 0.0, "e": 0.9})
    assert pick_showcase(examples, grpo, [single, ircot], per_dataset=1) == ["c", "b"]
    assert pick_showcase(examples, grpo, [single, ircot], per_dataset=3) == ["c", "b", "a"]
```

- [ ] **Step 2: Run it to see it fail**

Run: `$PY -m pytest tests/test_showcase.py -q`
Expected: FAIL, `No module named 'grasp_demo.showcase'`.

- [ ] **Step 3: Implement**

`src/grasp_demo/showcase.py`:
```python
"""Showcase questions: GRPO answers exactly (F1 = 1) where every baseline is below 0.5; per dataset, more
gold documents first (multi-hop), then id. Output {"ids": [...], "source": ...} for eval.showcase."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _scores(report: dict[str, Any]) -> dict[str, float]:
    return {str(row["id"]): float(row["f1"]) for row in report["per_question"]}


def pick_showcase(examples: list[dict[str, Any]], grpo: dict[str, Any], baselines: list[dict[str, Any]], *, per_dataset: int = 3) -> list[str]:
    ours, theirs = _scores(grpo), [_scores(report) for report in baselines]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        key = example["id"]
        if ours.get(key) == 1.0 and all(scores.get(key, 0.0) < 0.5 for scores in theirs):
            by_dataset[str(example.get("metadata", {}).get("dataset", ""))].append(example)
    chosen = []
    for dataset in sorted(by_dataset):
        rows = sorted(by_dataset[dataset], key=lambda row: (-len(row.get("gold_doc_ids", [])), row["id"]))
        chosen.extend(row["id"] for row in rows[:per_dataset])
    return chosen


def main(argv: list[str] | None = None) -> None:
    from grasp_grpo.data import read_examples

    parser = argparse.ArgumentParser(description="Pick showcase questions from test999 reports")
    parser.add_argument("--qa", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--grpo", required=True, help="report of the GRPO checkpoint the demo shows")
    parser.add_argument("--baseline", action="append", required=True, help="baseline report (repeat: single, ircot-k1)")
    parser.add_argument("--per-dataset", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    ids = set(json.loads(Path(args.ids).read_text(encoding="utf-8"))["ids"])
    examples = [row for row in read_examples(args.qa) if row["id"] in ids]
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))  # noqa: E731
    chosen = pick_showcase(examples, load(args.grpo), [load(path) for path in args.baseline], per_dataset=args.per_dataset)
    Path(args.output).write_text(json.dumps({"ids": chosen, "source": Path(args.grpo).name}, indent=2), encoding="utf-8")
    print(json.dumps({"showcase": len(chosen), "output": args.output}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test**

Run: `$PY -m pytest tests/test_showcase.py -q`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/grasp_demo/showcase.py tests/test_showcase.py
git commit -m "Showcase pick: GRPO exact where baselines miss, multi-hop first, per dataset"
```

---

### Task 9: Preflight

**Files:**
- Create: `src/grasp_demo/preflight.py`, `scripts/preflight.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: `DemoServices.from_config` (Task 4), `run_column` (Task 5), `load_eval_rows` (Task 1), `COLUMNS`.
- Produces:
  - `THRESHOLDS`
  - `pick_questions(rows, per_dataset=2) -> list[dict]`
  - `judge(result: dict) -> dict[str, bool | None]` (keys `health`, `vram`, `speed`, `grpo_matches`, `local_ok`)
  - `main(argv=None)` writes `runs/preflight-<mode>-<time>.json`

- [ ] **Step 1: Write the failing tests**

`tests/test_preflight.py`:
```python
from grasp_demo.preflight import judge, pick_questions


def question(seconds, grasp=10.0, match=True):
    columns = {"rag_7b": {"seconds": 3.0}, "grasp_sft": {"seconds": grasp}, "grasp_grpo": {"seconds": grasp}}
    return {"seconds": seconds, "columns": columns, "grpo_matches_report": match}


def result(**overrides):
    base = {"health": {"reader": {"ready": True}, "retriever": {"ready": True}, "proxy": {"ready": True}, "strong": {"ready": False}},
            "vram_peak_gb": 14.2, "questions": [question(30.0) for _ in range(6)]}
    base.update(overrides)
    return base


def test_judge_passes_when_every_threshold_holds_and_ignores_bedrock():
    verdict = judge(result())
    assert verdict == {"health": True, "vram": True, "speed": True, "grpo_matches": True, "local_ok": True}


def test_judge_fails_on_vram_speed_or_matches():
    assert judge(result(vram_peak_gb=15.5))["local_ok"] is False
    assert judge(result(questions=[question(61.0)] + [question(30.0)] * 5))["speed"] is False
    assert judge(result(questions=[question(30.0, grasp=41.0)] * 6))["speed"] is False
    assert judge(result(questions=[question(30.0, match=False)] * 3 + [question(30.0)] * 3))["grpo_matches"] is False


def test_errored_grasp_column_fails_speed_and_missing_report_is_unknown():
    broken = question(30.0)
    broken["columns"]["grasp_grpo"] = {"error": "boom"}
    assert judge(result(questions=[broken] * 6))["speed"] is False
    unknown = judge(result(questions=[question(30.0, match=None)] * 6))
    assert unknown["grpo_matches"] is None and unknown["local_ok"] is False


def test_pick_two_questions_per_dataset_in_file_order():
    rows = [{"id": f"{d}{i}", "metadata": {"dataset": d}} for d in ("a", "b") for i in range(3)]
    assert [row["id"] for row in pick_questions(rows)] == ["a0", "a1", "b0", "b1"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_preflight.py -q`
Expected: FAIL, `No module named 'grasp_demo.preflight'`.

- [ ] **Step 3: Implement**

`src/grasp_demo/preflight.py`:
```python
"""Run 6 test questions (2 per dataset) through every column, measure time and VRAM, compare GRPO's answers
with the test999 report, and judge whether this machine can host the demo (spec 2026-11-12 section 6)."""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

THRESHOLDS = {"vram_gb": 15.0, "question_seconds": 60.0, "grasp_seconds": 40.0, "grpo_matches": 4}
GRASP = ("grasp_sft", "grasp_grpo")


def pick_questions(rows: list[dict[str, Any]], per_dataset: int = 2) -> list[dict[str, Any]]:
    taken: dict[str, int] = defaultdict(int)
    chosen = []
    for row in rows:
        dataset = str(row.get("metadata", {}).get("dataset", ""))
        if taken[dataset] < per_dataset:
            chosen.append(row)
            taken[dataset] += 1
    return chosen


def judge(result: dict[str, Any]) -> dict[str, bool | None]:
    health = all(item.get("ready") for name, item in result["health"].items() if name != "strong")
    vram = result.get("vram_peak_gb") is not None and result["vram_peak_gb"] <= THRESHOLDS["vram_gb"]
    speed = True
    for question in result["questions"]:
        if question["seconds"] > THRESHOLDS["question_seconds"]:
            speed = False
        for column in GRASP:
            values = question["columns"].get(column)
            if values is not None and ("seconds" not in values or values["seconds"] > THRESHOLDS["grasp_seconds"]):
                speed = False
    compared = [q.get("grpo_matches_report") for q in result["questions"] if q.get("grpo_matches_report") is not None]
    matches = None if not compared else sum(compared) >= THRESHOLDS["grpo_matches"]
    verdict = {"health": health, "vram": vram, "speed": speed, "grpo_matches": matches}
    verdict["local_ok"] = all(value is True for value in verdict.values())
    return verdict


class VramSampler(threading.Thread):
    """Peak of the summed memory.used over every GPU, sampled each second (None without nvidia-smi)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_mib: float | None = None
        self.stopped = threading.Event()

    def run(self) -> None:
        while not self.stopped.is_set():
            try:
                output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                used = sum(float(line) for line in output.split() if line.strip())
                self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stopped.wait(1.0)


def main(argv: list[str] | None = None) -> None:
    from grasp_grpo.evaluate import exact

    from .config import load_config, load_dotenv
    from .evalset import load_eval_rows
    from .runners import COLUMNS, run_column
    from .services import DemoServices

    parser = argparse.ArgumentParser(description="Measure the demo on 6 test questions and judge this machine")
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", help="test999 report of the configured GRPO checkpoint (has predictions)")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    load_dotenv(config["_root"])
    sampler = VramSampler()
    sampler.start()
    services = DemoServices.from_config(config)
    for name, item in services.health.items():
        print(f"{name:10s} {'ok' if item['ready'] else 'FAIL ' + item.get('reason', '')}")
    reported = {}
    if args.report:
        reported = {p["id"]: p["prediction"] for p in json.loads(Path(args.report).read_text(encoding="utf-8")).get("predictions", [])}
    columns = [column for column in COLUMNS if services.enabled(column)[0]]
    questions = []
    for row in pick_questions(load_eval_rows(config)):
        def one(column: str, row=row) -> tuple[str, dict[str, Any]]:
            events: list[tuple[str, dict[str, Any]]] = []
            try:
                prediction = run_column(services, column, row, lambda kind, data: events.append((kind, data)))["prediction"]
                done = next(data for kind, data in reversed(events) if kind == "done")
                return column, {"seconds": done["seconds"], "f1": done.get("f1"), "prediction": prediction}
            except Exception as error:  # noqa: BLE001 - recorded; the judge counts it as a failure
                return column, {"error": f"{type(error).__name__}: {error}"}

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=max(1, len(columns))) as pool:
            per_column = dict(pool.map(one, columns))
        entry = {"id": row["id"], "dataset": row.get("metadata", {}).get("dataset"), "seconds": time.monotonic() - started, "columns": per_column}
        grpo = per_column.get("grasp_grpo", {})
        if row["id"] in reported and "prediction" in grpo:
            entry["grpo_matches_report"] = exact(grpo["prediction"], [reported[row["id"]]]) == 1.0
        questions.append(entry)
        cells = "  ".join(f"{c}={v['seconds']:.1f}s" if "seconds" in v else f"{c}=ERR" for c, v in per_column.items())
        print(f"{row['id']:>24s} {entry['seconds']:5.1f}s  {cells}")
    sampler.stopped.set()
    sampler.join(timeout=5)
    services.close()
    result = {"config": str(Path(args.config).resolve()), "health": services.health, "columns": columns,
              "vram_peak_gb": None if sampler.peak_mib is None else round(sampler.peak_mib / 1024, 2), "questions": questions, "thresholds": THRESHOLDS}
    result["verdict"] = judge(result)
    mode = Path(args.config).stem.replace("models.", "")
    output = Path(config.get("runs_dir", "runs")) / f"preflight-{mode}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"vram_peak_gb": result["vram_peak_gb"], **result["verdict"], "output": str(output)}))


if __name__ == "__main__":
    main()
```
`scripts/preflight.py`:
```python
"""python scripts/preflight.py --config models.local.yaml [--report <test999 grasp-grpo-*.json>]"""
from grasp_demo.preflight import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests**

Run: `$PY -m pytest -q`
Expected: every test passes (`test_preflight.py`: 4).

- [ ] **Step 5: Commit**

```bash
git add src/grasp_demo/preflight.py scripts/preflight.py tests/test_preflight.py
git commit -m "Preflight: 6 questions, time and VRAM, GRPO vs test999 report, local-or-rent verdict"
```

---

### Task 10: Frontend

**Files:**
- Rewrite: `static/index.html`, `static/app.js`, `static/style.css`

**Interfaces:**
- Consumes:
  - `/api/health` (`adapters.<name>.label`, `columns.<c>.enabled/reason`, `showcase_questions`);
  - `/api/eval?dataset=&showcase=1`;
  - `/api/ask`, `/api/stream/{id}` (events `{column, type, data, t}`);
  - `/api/history`, `/api/replay/{file}`, `POST`/`GET /api/prerun`.
- Event data shapes, from trainer 9ff7150 and the demo runner:

| type | data |
|---|---|
| `retrieve` | `{step, query, docs:[{doc_id,title,text,score,gold}]}` |
| `thought` | `{step, text}` |
| `rd` | `{step, retrieve, raw, valid}` |
| `qf` | `{step, query, chain:[{head,relation,tail}], raw, valid}` |
| `ki` | `{step, candidates:[{doc_id,head,relation,tail}], chosen, raw, valid, added_doc}` |
| `answer` | `{text, raw}` |
| `done` | `{seconds, retrievals, f1, em, recall}` |
| `error` | `{message}` |

- [ ] **Step 1: Write `static/index.html`**

```html
<!doctype html>
<html lang="vi">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>GRASP Demo</title>
    <link rel="stylesheet" href="/static/style.css">
  </head>
  <body>
    <header class="topbar">
      <div class="brand"><strong>GRASP</strong><span>So sánh với các baseline RAG</span></div>
      <div id="health" class="health">Đang kiểm tra dịch vụ…</div>
      <button id="theme" class="icon" title="Đổi chế độ sáng/tối" aria-label="Đổi chế độ sáng/tối">◐</button>
    </header>
    <div id="notice" class="notice" role="alert" hidden></div>
    <main>
      <section class="controls" aria-label="Chọn câu hỏi">
        <label>Dataset
          <select id="dataset"><option value="">Tất cả</option><option value="2wiki_ircot">2Wiki</option><option value="hotpotqa_abstracts_2017">HotpotQA</option><option value="wiki18">NQ / Wiki18</option></select>
        </label>
        <label class="check"><input type="checkbox" id="showcase"> Câu tiêu biểu</label>
        <label class="wide">Câu hỏi trong tập test <input id="eval-search" list="eval-list" placeholder="Gõ để tìm câu hỏi"></label>
        <datalist id="eval-list"></datalist>
        <button id="random" class="icon" title="Chọn ngẫu nhiên" aria-label="Chọn ngẫu nhiên">↻</button>
        <span class="or">hoặc</span>
        <label class="wide">Câu hỏi tự do <input id="free-question" placeholder="Gõ câu hỏi bất kỳ"></label>
        <label>Corpus
          <select id="free-corpus"><option value="wiki18">NQ / Wiki18</option><option value="2wiki_ircot">2Wiki</option><option value="hotpotqa_abstracts_2017">HotpotQA</option></select>
        </label>
        <fieldset id="columns"><legend>Phương pháp</legend>
          <label><input type="checkbox" value="rag_7b" checked> RAG 7B</label>
          <label><input type="checkbox" value="rag_235b" checked> RAG 235B</label>
          <label><input type="checkbox" value="ircot_7b" checked> IRCoT 7B</label>
          <label><input type="checkbox" value="grasp_sft" checked> GRASP SFT</label>
          <label><input type="checkbox" value="grasp_grpo" checked> GRASP GRPO</label>
        </fieldset>
        <button id="ask" class="ask">Hỏi</button>
        <button id="history" class="icon" title="Lịch sử và phát lại" aria-label="Lịch sử">◷</button>
      </section>
      <section id="question-info" class="question-info" hidden>
        <p id="question-text" class="question-text"></p>
        <p><span class="muted">Đáp án đúng:</span> <strong id="gold-answer"></strong></p>
        <div><span class="muted">Tài liệu gold:</span><ul id="gold-docs" class="gold-docs"></ul></div>
      </section>
      <p id="free-note" class="muted" hidden>Câu tự do không có đáp án nên không chấm điểm. Tài liệu chưa có trong cache triple sẽ không có triple cho KI.</p>
      <section id="board" class="board" aria-live="polite"></section>
    </main>
    <dialog id="history-dialog">
      <header><strong>Lịch sử</strong>
        <label>Tốc độ <select id="replay-speed"><option value="1">1×</option><option value="4">4×</option></select></label>
        <button class="icon" id="close-history" aria-label="Đóng">×</button>
      </header>
      <div class="prerun"><button id="prerun">Chạy trước câu tiêu biểu</button><span id="prerun-status" class="muted"></span></div>
      <div id="history-list"></div>
    </dialog>
    <dialog id="doc-dialog">
      <header><strong id="doc-title"></strong><button class="icon" id="close-doc" aria-label="Đóng">×</button></header>
      <p id="doc-meta" class="muted"></p>
      <div id="doc-text" class="doc-text"></div>
    </dialog>
    <template id="column-template">
      <article class="column"><header><div><h2></h2><small class="checkpoint"></small></div><div class="metrics">Đang chờ</div></header><div class="trace"></div><footer class="answer"></footer></article>
    </template>
    <script src="/static/app.js"></script>
  </body>
</html>
```

- [ ] **Step 2: Write `static/app.js`**

```js
const LABELS = { rag_7b: 'RAG · 7B', rag_235b: 'RAG · 235B', ircot_7b: 'IRCoT · 7B', grasp_sft: 'GRASP · SFT', grasp_grpo: 'GRASP · GRPO' };
const ORDER = ['rag_7b', 'rag_235b', 'ircot_7b', 'grasp_sft', 'grasp_grpo'];
const state = { questions: [], selected: null, adapters: {}, stream: null, replaying: false };
const $ = (selector) => document.querySelector(selector);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}
function notice(message) { const box = $('#notice'); box.textContent = message || ''; box.hidden = !message; }
async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}
const num = (value) => (value == null ? '–' : value.toFixed(2));
const triple = (t) => `<${t.head}; ${t.relation}; ${t.tail}>`;

function renderBoard(columns) {
  const board = $('#board');
  board.replaceChildren();
  for (const name of ORDER.filter((c) => columns.includes(c))) {
    const node = $('#column-template').content.firstElementChild.cloneNode(true);
    node.dataset.column = name;
    node.querySelector('h2').textContent = LABELS[name];
    if (name.startsWith('grasp_')) node.querySelector('.checkpoint').textContent = state.adapters[name.slice(6)]?.label || '';
    if (name === 'grasp_grpo') node.classList.add('pinned');
    board.append(node);
  }
  for (const item of document.querySelectorAll('#gold-docs li')) { item.classList.remove('found'); item.querySelector('.by').textContent = ''; }
}
const column = (name) => document.querySelector(`.column[data-column="${name}"]`);

function markGold(docId, columnName) {
  for (const item of document.querySelectorAll('#gold-docs li')) {
    if (item.dataset.doc !== String(docId)) continue;
    item.classList.add('found');
    const by = item.querySelector('.by');
    const names = new Set(by.textContent ? by.textContent.replace('tìm thấy bởi: ', '').split(', ') : []);
    names.add(LABELS[columnName]);
    by.textContent = `tìm thấy bởi: ${[...names].join(', ')}`;
  }
}

function openDoc(doc) {
  $('#doc-title').textContent = doc.title || doc.doc_id;
  $('#doc-meta').textContent = `${doc.doc_id}${doc.score != null ? ` · score ${doc.score.toFixed(3)}` : ''}${doc.gold ? ' · ★ tài liệu gold' : ''}`;
  $('#doc-text').textContent = doc.text || '';
  $('#doc-dialog').showModal();
}

function docList(docs, columnName) {
  const list = el('ol', 'docs');
  for (const doc of docs) {
    const item = el('li', doc.gold ? 'doc gold' : 'doc');
    const title = el('button', 'doc-title', `${doc.gold ? '★ ' : ''}${doc.title || doc.doc_id}`);
    title.onclick = () => openDoc(doc);
    item.append(title, el('span', 'snippet', (doc.text || '').slice(0, 200)));
    list.append(item);
    if (doc.gold) markGold(doc.doc_id, columnName);
  }
  return list;
}

function stepCard(type, data, columnName, t) {
  const card = el('details', `event ${type}`);
  card.open = type !== 'retrieve' || (data.step ?? 0) > 0;
  const summary = el('summary');
  const body = el('div', 'body');
  const step = data.step ? ` · bước ${data.step}` : '';
  if (type === 'retrieve') {
    summary.textContent = `Truy xuất${step} (${(data.docs || []).length} tài liệu)`;
    body.append(el('div', 'query', `Query: ${data.query}`), docList(data.docs || [], columnName));
  } else if (type === 'thought') {
    summary.textContent = `Suy nghĩ${step}`;
    body.append(el('p', null, data.text));
  } else if (type === 'rd') {
    summary.textContent = `RD${step}: ${data.retrieve ? 'cần truy xuất' : 'không truy xuất'}`;
  } else if (type === 'qf') {
    summary.textContent = `QF${step}`;
    body.append(el('div', 'query', `Query: ${data.query}`));
    if (data.chain && data.chain.length) body.append(el('div', 'muted', `Chuỗi: ${data.chain.map(triple).join(' → ')}`));
  } else if (type === 'ki') {
    summary.textContent = `KI${step}`;
    if (!data.candidates || !data.candidates.length) {
      body.append(el('div', 'muted', 'Không có triple ứng viên (tài liệu chưa có trong cache triple)'));
    } else {
      body.append(el('div', 'triple', triple(data.candidates[data.chosen ?? 0])));
      body.append(el('div', 'muted', data.added_doc ? `Thêm tài liệu ${data.added_doc} vào ngữ cảnh` : 'Tài liệu đã có trong ngữ cảnh'));
    }
  } else if (type === 'answer') {
    summary.textContent = 'Trả lời';
    body.append(el('p', 'answer-text', data.text || '(trống)'));
  } else if (type === 'error') {
    summary.textContent = 'Lỗi';
    body.append(el('p', null, data.message));
  } else {
    summary.textContent = type;
  }
  summary.append(el('span', 'time', `${t.toFixed(1)} s`));
  if (data.valid === false) summary.append(el('span', 'badge warn', '⚠ sai format'));
  if (data.raw) {
    const raw = el('details', 'raw');
    raw.append(el('summary', null, 'raw'), el('pre', null, data.raw));
    body.append(raw);
  }
  card.append(summary, body);
  return card;
}

function renderMetrics(host, d) {
  const metrics = host.querySelector('.metrics');
  metrics.replaceChildren();
  const add = (text, good) => metrics.append(el('span', good == null ? 'metric' : good ? 'metric good' : 'metric bad', text));
  if (d.f1 != null) {
    add(`F1 ${num(d.f1)}`, d.f1 >= 1 ? true : d.f1 <= 0 ? false : null);
    add(`EM ${num(d.em)}`, d.em >= 1);
  }
  if (d.recall != null) add(`Recall ${num(d.recall)}`, d.recall >= 1 ? true : d.recall <= 0 ? false : null);
  add(`${d.seconds.toFixed(1)} s`);
  add(`${d.retrievals} lần truy xuất`);
}

function applyEvent(event) {
  const host = column(event.column);
  if (!host) return;
  const data = event.data || {};
  if (event.type === 'done') { renderMetrics(host, data); return; }
  host.querySelector('.trace').append(stepCard(event.type, data, event.column, event.t || 0));
  if (event.type === 'answer') host.querySelector('.answer').textContent = data.text || '(trống)';
  if (event.type === 'error') {
    host.classList.add('failed');
    host.querySelector('.metrics').textContent = 'Lỗi';
  }
}

function showQuestion(question) {
  const scored = question && question.gold_answers && question.gold_answers.length;
  $('#question-info').hidden = !scored;
  $('#free-note').hidden = !!scored;
  if (!scored) return;
  $('#question-text').textContent = question.question;
  $('#gold-answer').textContent = question.gold_answers.join(' / ');
  $('#gold-docs').replaceChildren(...(question.gold_doc_ids || []).map((id) => {
    const item = el('li', null);
    item.dataset.doc = String(id);
    item.append(el('span', 'gold-id', `★ ${id}`), el('span', 'by muted', ''));
    return item;
  }));
}

function choose(question) {
  state.selected = question;
  $('#free-question').value = '';
  $('#eval-search').value = question.question;
  showQuestion(question);
}

async function loadQuestions() {
  const params = new URLSearchParams({ dataset: $('#dataset').value });
  if ($('#showcase').checked) params.set('showcase', '1');
  state.questions = await api(`/api/eval?${params}`);
  $('#eval-list').replaceChildren(...state.questions.map((q) => { const option = document.createElement('option'); option.value = q.question; return option; }));
}

async function ask() {
  notice('');
  const free = $('#free-question').value.trim();
  const columns = [...document.querySelectorAll('#columns input:checked')].map((x) => x.value);
  if (!columns.length) { notice('Chọn ít nhất một phương pháp.'); return; }
  const body = free ? { question: free, corpus: $('#free-corpus').value, columns } : state.selected ? { eval_id: state.selected.id, columns } : null;
  if (!body) { notice('Chọn một câu trong tập test hoặc gõ câu hỏi tự do.'); return; }
  if (free) showQuestion(null);
  if (state.stream) state.stream.close();
  renderBoard(columns);
  const { run_id } = await api('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const stream = new EventSource(`/api/stream/${run_id}`);
  state.stream = stream;
  stream.onmessage = (e) => applyEvent(JSON.parse(e.data));
  stream.addEventListener('close', () => stream.close());
  stream.onerror = () => { if (stream.readyState === EventSource.CLOSED) notice('Mất kết nối tới server.'); stream.close(); };
}

async function health() {
  const h = await api('/api/health');
  state.adapters = h.adapters || {};
  const off = Object.entries(h.columns).filter(([, v]) => !v.enabled);
  $('#health').textContent = off.length ? `Không dùng được: ${off.map(([k]) => LABELS[k]).join(', ')}` : 'Mọi cột sẵn sàng';
  $('#health').title = off.map(([k, v]) => `${LABELS[k]}: ${v.reason}`).join('\n');
  $('#health').classList.toggle('bad', off.length > 0);
  for (const [name, v] of Object.entries(h.columns)) {
    const box = document.querySelector(`#columns input[value="${name}"]`);
    if (box && !v.enabled) { box.checked = false; box.disabled = true; box.parentElement.title = v.reason || ''; }
  }
}

async function replay(row) {
  const events = await api(`/api/replay/${encodeURIComponent(row.file)}`);
  $('#history-dialog').close();
  if (state.stream) state.stream.close();
  showQuestion(row.request.example || null);
  renderBoard([...new Set(events.map((x) => x.column))]);
  const speed = Number($('#replay-speed').value);
  let previous = 0;
  for (const event of events) {
    await new Promise((resolve) => setTimeout(resolve, Math.max(0, ((event.t - previous) * 1000) / speed)));
    applyEvent(event);
    previous = event.t;
  }
}

async function showHistory() {
  const rows = await api('/api/history');
  $('#history-list').replaceChildren(...rows.map((row) => {
    const item = el('div', 'run');
    const when = row.created_at ? new Date(row.created_at * 1000).toLocaleString('vi-VN') : '';
    item.append(el('span', null, `${row.request.prerun ? '★ ' : ''}${row.request.example?.question || row.file}`), el('span', 'muted', when));
    const button = el('button', null, 'Phát lại');
    button.onclick = () => replay(row).catch((e) => notice(e.message));
    item.append(button);
    return item;
  }));
  $('#history-dialog').showModal();
}

async function prerun() {
  try { await api('/api/prerun', { method: 'POST' }); } catch (e) { notice(e.message); }
  const poll = async () => {
    const s = await api('/api/prerun');
    $('#prerun-status').textContent = s.error ? `Lỗi: ${s.error}` : s.running ? `Đang chạy ${s.done}/${s.total}…` : `Xong ${s.done}/${s.total}`;
    if (s.running) setTimeout(poll, 3000); else showHistory().catch((e) => notice(e.message));
  };
  poll();
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('grasp-theme', theme); } catch (e) { /* storage unavailable */ }
}

$('#dataset').onchange = () => loadQuestions().catch((e) => notice(e.message));
$('#showcase').onchange = () => loadQuestions().catch((e) => notice(e.message));
$('#eval-search').onchange = () => { const found = state.questions.find((q) => q.question === $('#eval-search').value); if (found) choose(found); };
$('#random').onclick = () => { if (state.questions.length) choose(state.questions[Math.floor(Math.random() * state.questions.length)]); };
$('#ask').onclick = () => ask().catch((e) => notice(e.message));
$('#history').onclick = () => showHistory().catch((e) => notice(e.message));
$('#close-history').onclick = () => $('#history-dialog').close();
$('#close-doc').onclick = () => $('#doc-dialog').close();
$('#prerun').onclick = () => prerun();
$('#theme').onclick = () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
try { const saved = localStorage.getItem('grasp-theme'); if (saved) document.documentElement.dataset.theme = saved; } catch (e) { /* storage unavailable */ }
loadQuestions().catch((e) => notice(e.message));
health().catch((e) => { $('#health').textContent = `Không kiểm tra được dịch vụ: ${e.message}`; });
```

- [ ] **Step 3: Write `static/style.css`**

```css
:root { color-scheme: light; --bg:#f4f7f6; --surface:#ffffff; --ink:#16211f; --muted:#5d6b67; --line:#c9d4d0; --accent:#006a5b; --warn:#9b4a00; --bad:#b42318; --good:#1a7f37; --gold:#8c6500; --gold-bg:#fff4d6; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --bg:#121716; --surface:#1b2220; --ink:#edf5f2; --muted:#a7b5b1; --line:#3a4944; --accent:#54d6bb; --warn:#ffb86e; --bad:#ff9088; --good:#7ee2a0; --gold:#ffd66b; --gold-bg:#3a3214; } }
:root[data-theme="dark"] { color-scheme: dark; --bg:#121716; --surface:#1b2220; --ink:#edf5f2; --muted:#a7b5b1; --line:#3a4944; --accent:#54d6bb; --warn:#ffb86e; --bad:#ff9088; --good:#7ee2a0; --gold:#ffd66b; --gold-bg:#3a3214; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 "Segoe UI",Inter,Arial,sans-serif; }
.muted { color:var(--muted); }
.topbar { min-height:56px; display:flex; align-items:center; gap:16px; padding:0 24px; border-bottom:1px solid var(--line); background:var(--surface); position:sticky; top:0; z-index:3; }
.brand strong { font-size:20px; color:var(--accent); margin-right:10px; } .brand span { color:var(--muted); }
.health { margin-left:auto; color:var(--muted); font-size:14px; } .health.bad { color:var(--bad); }
.notice { margin:12px 24px 0; padding:10px 14px; border:1px solid var(--bad); border-radius:6px; color:var(--bad); background:var(--surface); }
main { padding:12px 24px 28px; }
.controls { display:flex; flex-wrap:wrap; gap:12px; align-items:end; padding:12px 0; border-bottom:1px solid var(--line); }
label { display:grid; gap:4px; color:var(--muted); font-size:14px; } label.check { display:flex; align-items:center; gap:6px; padding-bottom:8px; }
input, select, button { color:inherit; font:inherit; }
input, select { height:38px; min-width:130px; background:var(--surface); border:1px solid var(--line); border-radius:4px; padding:0 10px; }
label.wide input { width:min(380px, 80vw); } label.check input, fieldset input { height:auto; min-width:0; }
button { height:38px; border:1px solid var(--accent); border-radius:4px; background:var(--surface); cursor:pointer; padding:0 14px; }
button:hover { background:var(--accent); color:var(--surface); } button.icon { width:38px; padding:0; font-size:18px; }
button.ask { background:var(--accent); color:var(--surface); font-weight:700; } .or { color:var(--muted); padding-bottom:8px; }
fieldset { border:0; margin:0; padding:0; display:flex; gap:10px; flex-wrap:wrap; } fieldset legend { color:var(--muted); font-size:14px; } fieldset label { display:flex; gap:4px; align-items:center; white-space:nowrap; }
.question-info { padding:12px 0; border-bottom:1px solid var(--line); } .question-info p { margin:4px 0; } .question-text { font-size:18px; font-weight:600; }
.gold-docs { display:flex; flex-wrap:wrap; gap:8px; list-style:none; padding:0; margin:6px 0 0; }
.gold-docs li { border:1px dashed var(--gold); border-radius:4px; padding:2px 8px; color:var(--gold); } .gold-docs li.found { border-style:solid; background:var(--gold-bg); }
.gold-docs .by { margin-left:6px; font-size:13px; }
.board { display:flex; gap:12px; overflow-x:auto; padding:16px 0; align-items:flex-start; }
.column { flex:1 0 320px; min-width:320px; min-height:360px; background:var(--surface); border:1px solid var(--line); border-radius:6px; display:flex; flex-direction:column; }
.column.pinned { position:sticky; right:0; z-index:1; box-shadow:-6px 0 12px rgba(0,0,0,.12); }
.column.failed { border-color:var(--bad); }
.column header { padding:12px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:8px; }
h2 { font-size:17px; margin:0; } .checkpoint { color:var(--muted); font-size:13px; }
.metrics { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:4px 8px; font-size:13px; color:var(--muted); text-align:right; }
.metric.good { color:var(--good); font-weight:700; } .metric.bad { color:var(--bad); font-weight:700; }
.trace { padding:8px; display:grid; gap:8px; max-height:64vh; overflow:auto; }
details.event { border-left:3px solid var(--accent); background:color-mix(in srgb, var(--surface) 90%, var(--accent)); padding:6px 10px; border-radius:0 4px 4px 0; }
details.event.error { border-color:var(--bad); } details.event.answer { border-color:var(--good); }
details.event > summary { cursor:pointer; font-weight:600; } .time { margin-left:8px; color:var(--muted); font-weight:400; font-size:13px; }
.badge.warn { margin-left:8px; color:var(--warn); font-size:13px; }
.body p { margin:6px 0; } .query { font-style:italic; margin:4px 0; } .triple { font-family:ui-monospace,Consolas,monospace; font-size:14px; margin:4px 0; }
.docs { margin:6px 0 0; padding-left:20px; display:grid; gap:6px; }
.doc .doc-title { height:auto; border:0; padding:0; background:none; color:var(--accent); text-align:left; font-weight:600; cursor:pointer; }
.doc .doc-title:hover { text-decoration:underline; background:none; color:var(--accent); }
.doc.gold { background:var(--gold-bg); border-radius:4px; padding:2px 4px; } .doc.gold .doc-title { color:var(--gold); }
.snippet { display:block; color:var(--muted); font-size:14px; }
details.raw summary { color:var(--muted); font-size:13px; cursor:pointer; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; margin:6px 0 2px; color:var(--muted); font:13px/1.4 ui-monospace,Consolas,monospace; }
.answer { margin-top:auto; padding:12px; border-top:1px solid var(--line); font-weight:700; font-size:17px; } .answer:empty { display:none; }
dialog { color:var(--ink); background:var(--surface); border:1px solid var(--line); border-radius:6px; width:min(760px, 94vw); max-height:86vh; }
dialog header { display:flex; justify-content:space-between; align-items:center; gap:12px; border-bottom:1px solid var(--line); padding-bottom:8px; }
.doc-text { white-space:pre-wrap; line-height:1.6; }
.prerun { display:flex; gap:12px; align-items:center; padding:10px 0; border-bottom:1px solid var(--line); }
.run { display:grid; grid-template-columns:1fr auto auto; gap:10px; align-items:center; padding:10px 0; border-bottom:1px solid var(--line); }
@media (max-width:700px) { main { padding:12px 16px; } .topbar { padding:0 16px; } .controls { align-items:stretch; } .column { flex-basis:85vw; min-width:85vw; } }
```

- [ ] **Step 4: Check by hand with a replay (no models needed)**

1. Create `runs/20261112-000000-sample.jsonl` (gitignored). The first line is `{"type":"request","run_id":"sample","created_at":1762900000,"data":{"example":{"id":"x","question":"Who directed Up?","gold_answers":["Pete Docter"],"gold_doc_ids":["d2"]},"columns":["rag_7b","grasp_grpo"]}}`. Then one event of every type for `grasp_grpo` and `retrieve`, `answer`, `done` for `rag_7b`, with `t` increasing from 0.1 to 3.0. Include one `qf` with `"valid": false` and one `retrieve` doc with `"gold": true, "doc_id": "d2"`.
2. Create `runs/ui.yaml` containing the single line `runs_dir: .`. Do not commit it; delete it after the check. Then run the server with no models loaded (every column shows as unavailable, and replay still works): `$PY -c "import uvicorn; from grasp_demo.server import create_app; from grasp_demo.services import DemoServices; uvicorn.run(create_app('runs/ui.yaml', services=DemoServices({}, loaders={})), port=7860)"`
3. Open `http://localhost:7860` in Chrome, then Lịch sử → Phát lại (1×). Check:
   - the gold doc shows ★ and is highlighted, and the gold chip turns solid;
   - "⚠ sai format" appears on the QF card;
   - clicking a doc opens the full text;
   - the GRPO column stays visible at the right when the board scrolls horizontally (narrow the window to 1000 px);
   - light and dark themes both work;
   - no console errors.
4. Take screenshots (light and dark, width 1366) for the PR.

- [ ] **Step 5: Commit**

```bash
git add static/index.html static/app.js static/style.css
git commit -m "UI: readable step cards, gold marks, full-text dialog, format warnings, pinned GRPO column, pre-run"
```

---

### Task 11: Deployment configs, README, spec note, push

**Files:**
- Create: `models.local.yaml`, `models.vast.yaml`
- Delete: `models.yaml`
- Modify: `README.md`, `docs/superpowers/specs/2026-11-12-demo-backends-design.md` (record the `/api/doc` decision)

- [ ] **Step 1: Write `models.local.yaml`**

```yaml
# Local RTX 5060 Ti: reader = llama-server (scripts/run_reader.ps1), proxy = transformers bf16 in this process.
reader:   {url: http://localhost:8080/v1, model: qwen2.5-7b-instruct, tokenizer: Qwen/Qwen2.5-7B-Instruct, api_key: EMPTY}
strong:   {url: https://bedrock-mantle.ap-southeast-2.api.aws/v1, model: qwen.qwen3-235b-a22b-2507, timeout: 60, retries: 1}
proxy:
  backend: hf
  adapters:
    sft:  {path: checkpoints/grasp-qwen2.5-3b-sft, label: "SFT"}
    grpo: {path: checkpoints/grpo-s100, label: "GRPO s100"}
retrieval: {corpus_config: data/corpus.json, top_k: 10, max_candidate_triples: 20, encoder: intfloat/e5-large-v2, encoder_device: cpu}
ircot:    {max_steps: 5, docs_per_step: 3}
grasp:    {max_steps: 5}
eval:     {qa: data/grasp-gold-evidence/dev.jsonl, ids: data/test_ids.json, showcase: data/showcase.json}
runs_dir: runs
```

- [ ] **Step 2: Write `models.vast.yaml`**

```yaml
# Vast box with the eval stack running (run_reader_vast.sh on :8000, run_policy_vast.sh on :8001). Run the demo
# on the box, then from your PC: ssh -p <port> root@<host> -L 7860:localhost:7860 and open http://localhost:7860
reader:   {url: http://localhost:8000/v1, model: Qwen/Qwen2.5-7B-Instruct, tokenizer: Qwen/Qwen2.5-7B-Instruct, api_key_env: READER_API_KEY}
strong:   {url: https://bedrock-mantle.ap-southeast-2.api.aws/v1, model: qwen.qwen3-235b-a22b-2507, timeout: 60, retries: 1}
proxy:
  backend: vllm
  url: http://localhost:8001/v1
  api_key_env: READER_API_KEY
  adapters:
    sft:  {path: /workspace/checkpoints/grasp-qwen2.5-3b-sft, label: "SFT"}
    grpo: {path: /workspace/ckpts/s100, label: "GRPO s100"}
retrieval: {corpus_config: /workspace/data/corpus.json, top_k: 10, max_candidate_triples: 20, encoder: intfloat/e5-large-v2, encoder_device: "cuda:1"}
ircot:    {max_steps: 5, docs_per_step: 3}
grasp:    {max_steps: 5}
eval:     {qa: /workspace/data/grasp-gold-evidence/dev.jsonl, ids: /workspace/data/test_ids.json, showcase: /workspace/data/showcase.json}
runs_dir: runs
```
Before the first vast run, check on the box:
- `grep -E "^(READER_MODEL|POLICY_URL|READER_URL)=" /workspace/grasp-grpo-trainer/config/vast.env` and make `reader.model`, `reader.url` and `proxy.url` match it;
- `ls /workspace/data/grasp-retrieval/cache/` and pass the triple file the evals used (from `CORPUS_CONFIG`) to `fetch_data.py --skip-download --triples cache/<that file>`.

- [ ] **Step 3: Delete `models.yaml` and rewrite `README.md`**

Run: `git rm models.yaml`

`README.md`:
````markdown
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
````

- [ ] **Step 4: Record the `/api/doc` decision in the spec**

In `docs/superpowers/specs/2026-11-12-demo-backends-design.md`, section 4, replace the bullet that starts with "`GET /api/doc/{corpus}/{doc_id}`" with:
```markdown
  - ~~`GET /api/doc/{corpus}/{doc_id}`~~: bỏ khi lập kế hoạch. `passages.sqlite` không có index trên `doc_id`,
    nên tra cứu trong wiki18 phải quét cả bảng. Sự kiện `retrieve` đã chứa toàn văn từng tài liệu, nên giao diện
    mở toàn văn từ chính sự kiện đó. Danh sách gold trong phần câu hỏi hiện doc id và đánh dấu "tìm thấy bởi …"
    khi có cột truy xuất được tài liệu đó.
```

- [ ] **Step 5: Run every test, commit, push the branch**

Run: `$PY -m pytest -q`
Expected: all tests pass.

```bash
git add -A models.local.yaml models.vast.yaml README.md docs
git commit -m "Deployment configs (local, vast), README, spec note on full-text documents"
git push https://github.com/FPTU-SP27-CaptonesProject/grasp-demo.git feat/demo-backends
```
If the push is rejected for lack of rights, stop and ask the user. Do not push anywhere else. Then give the user the PR link: `https://github.com/FPTU-SP27-CaptonesProject/grasp-demo/compare/main...feat/demo-backends`.

---

## After the plan (needs the machines, not code)

1. **When the test999 evals finish:**
   - read `reports/test_ids/table.md` and pick the GRPO checkpoint with the best F1;
   - set `path` and `label` in both configs;
   - build `showcase.json` on the eval box from its reports (they have `predictions`).
2. **On the eval box C.60319605, before it is destroyed:**
   - `git clone` the demo branch;
   - install the web dependencies into the trainer venv;
   - run `fetch_data.py --skip-download …`, then the vast preflight. Save the JSON to the trainer's HF log repo.
3. **On the local machine:**
   - download the data (about 21 GB), the GGUF and llama-server;
   - run the local preflight;
   - choose local or rented from the two verdicts.
