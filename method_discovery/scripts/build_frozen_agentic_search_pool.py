"""Phase 5 (F1): build the frozen multi-hop QA pool manifest."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from agentcontrol_method.agentic_corpus import get_corpus, get_questions, retrieve_topk  # noqa: E402


def main() -> int:
    corpus = get_corpus()
    questions = get_questions()
    diff = Counter(q["difficulty"] for q in questions)

    # Compute, for each multi-hop question, what fraction of its supporting docs
    # are NOT retrieved by naive top-3 RAG. This validates that distractors do
    # cause linking-doc misses on at least some questions.
    miss_stats = {"all": 0, "any_miss": 0, "full_miss": 0, "examples": []}
    for q in questions:
        if not q["multihop"]:
            continue
        top3 = [doc_id for doc_id, _, _ in retrieve_topk(q["question"], k=3)]
        cit = set(q["citations"])
        retrieved = cit & set(top3)
        missed = cit - set(top3)
        miss_stats["all"] += 1
        if missed:
            miss_stats["any_miss"] += 1
            if not retrieved:
                miss_stats["full_miss"] += 1
            if len(miss_stats["examples"]) < 5:
                miss_stats["examples"].append({
                    "task_id": q["task_id"],
                    "question": q["question"],
                    "citations": q["citations"],
                    "top3": top3,
                    "missed": list(missed),
                })

    manifest = {
        "n_total": len(questions),
        "difficulty_counts": dict(diff),
        "n_corpus_docs": len(corpus),
        "naive_top3_miss_stats": miss_stats,
        "questions": [{"task_id": q["task_id"], "difficulty": q["difficulty"],
                       "multihop": q["multihop"], "n_supporting_docs": len(q["citations"])}
                      for q in questions],
        "honesty": (
            "Frozen multi-hop QA corpus with deliberate distractors. Naive top-3 "
            "RAG misses at least one supporting doc on a calibrated fraction of "
            "multi-hop questions. Auditable in src/agentcontrol_method/agentic_corpus.py. "
            "Not a benchmark download; not live web."
        ),
    }
    out_json = HERE / "experiments" / "agentic_search_pool_manifest.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md = ["# AGENTIC_SEARCH_POOL\n",
          f"\n- total questions: **{len(questions)}**\n",
          f"- difficulty: {dict(diff)}\n",
          f"- corpus docs: {len(corpus)}\n",
          f"- multi-hop questions whose top-3 RAG misses ≥1 supporting doc: "
          f"**{miss_stats['any_miss']}/{miss_stats['all']}**\n",
          f"- multi-hop questions whose top-3 RAG misses ALL supporting docs: "
          f"**{miss_stats['full_miss']}/{miss_stats['all']}**\n",
          "\n## Examples of naive-RAG-miss\n\n"]
    for ex in miss_stats["examples"]:
        md.append(f"- `{ex['task_id']}`: {ex['question']}\n")
        md.append(f"  - supports: {ex['citations']}\n")
        md.append(f"  - top3:    {ex['top3']}\n")
        md.append(f"  - missed:  {ex['missed']}\n")
    md.append("\n## Honesty\n\n" + manifest["honesty"] + "\n")

    (HERE / "reports" / "AGENTIC_SEARCH_POOL.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote {out_json.relative_to(REPO)} and reports/AGENTIC_SEARCH_POOL.md")
    print(f"any-miss: {miss_stats['any_miss']}/{miss_stats['all']}; full-miss: {miss_stats['full_miss']}/{miss_stats['all']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
