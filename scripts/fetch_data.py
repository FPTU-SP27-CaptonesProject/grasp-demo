"""Fetch the frozen retrieval/evidence artifacts and create data/corpus.json.

The script deliberately does not build or alter FAISS indexes. It only downloads
the exact release selected by the user, then verifies that each corpus has the
index, passages store and triple cache that the shared GRASP retriever needs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CORPORA = ("2wiki_ircot", "hotpotqa_abstracts_2017", "wiki18")


def unique(path: Path, pattern: str) -> Path:
    found = sorted(path.glob(pattern))
    if len(found) != 1:
        raise RuntimeError(f"{path}: expected one {pattern}, found {found}")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-repo", default="doancuavahk/grasp-retrieval")
    parser.add_argument("--retrieval-revision")
    parser.add_argument("--evidence-repo", help="Optional HF dataset containing grasp-gold-evidence/")
    parser.add_argument("--evidence-revision")
    parser.add_argument("--triple-root", type=Path, help="Directory containing <corpus>/triples.sqlite when triples are not in retrieval release")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    root = args.root.resolve()
    retrieval_root = root / "grasp-retrieval"
    snapshot_download(args.retrieval_repo, repo_type="dataset", revision=args.retrieval_revision, local_dir=retrieval_root)
    if args.evidence_repo:
        snapshot_download(args.evidence_repo, repo_type="dataset", revision=args.evidence_revision, local_dir=root / "grasp-gold-evidence")

    config = {}
    for corpus in CORPORA:
        corpus_root = retrieval_root / corpus
        index = unique(corpus_root, "**/index.faiss")
        passages = unique(corpus_root, "**/passages.sqlite")
        triples = sorted(corpus_root.glob("**/triples.sqlite"))
        if not triples and args.triple_root:
            triples = sorted((args.triple_root / corpus).glob("**/triples.sqlite"))
        if len(triples) != 1:
            raise RuntimeError(f"{corpus}: one triples.sqlite is required for KI; fetch the Project A triple cache or pass --triple-root")
        config[corpus] = {"index_dir": str(index.parent), "triple_cache": str(triples[0])}
        assert passages.is_file()
    (root / "corpus.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(root / "corpus.json")


if __name__ == "__main__":
    main()
