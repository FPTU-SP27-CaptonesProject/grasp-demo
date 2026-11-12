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
