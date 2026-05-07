"""A-GPU Phase 2 — rebuild control labels with leakage-resistant splits.

Produces 4 split kinds:
  1. random        : standard 5-fold CV
  2. family_held_out : 1 family held out at a time (leave-one-family-out)
  3. sprint_held_out : 1 source sprint held out at a time
  4. hard_label_only : drop trivial family-tag labels (need_code, need_retrieve);
                       focus on labels not predictable from family alone.

For each (label, split) pair, report:
  - n_train / n_test
  - class balance pos_rate
  - whether the label is "trivial" (predictable from family) or "hard"
  - admissibility for Main claims
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APOLLO = REPO / "main_apollo"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "main_rescue_gpu" / "src"))
sys.path.insert(0, str(REPO / "method_discovery" / "src"))


def _load_prompt_lookup() -> dict[str, str]:
    """Reconstruct task_id → prompt-text map from all known task pool modules."""
    lookup: dict[str, str] = {}
    try:
        from agentcontrol.tasks_math import MATH_TASKS
        for t in MATH_TASKS: lookup[t["id"]] = t.get("question", "")
    except Exception: pass
    try:
        from agentcontrol.tasks_code import CODE_TASKS
        for t in CODE_TASKS: lookup[t["id"]] = t.get("prompt", "")
    except Exception: pass
    try:
        from agentcontrol.tasks_evidence import EVIDENCE_QA
        for t in EVIDENCE_QA: lookup[t["id"]] = t.get("question", "")
    except Exception: pass
    try:
        from agentcontrol.rescue_tasks import get_pool as _rp
        for tid, t in _rp().items(): lookup[tid] = t.get("question", t.get("prompt", ""))
    except Exception: pass
    try:
        from agentcontrol.hard_regime_tasks import get_pool as _hp
        for tid, t in _hp().items(): lookup[tid] = t.get("question", t.get("prompt", ""))
    except Exception: pass
    try:
        from agentcontrol_main_rescue.interactive_tasks import get_pool as _ip
        for tid, t in _ip().items(): lookup[tid] = t.get("question", t.get("prompt", ""))
    except Exception: pass
    try:
        from agentcontrol_method.agentic_corpus import get_questions as _aq
        for q in _aq(): lookup[q["task_id"]] = q.get("question", "")
    except Exception: pass
    try:
        from agentcontrol_method.f2_data_analysis_tasks import get_pool as _f2
        for tid, t in _f2().items(): lookup[tid] = t.get("question", t.get("prompt", ""))
    except Exception: pass
    return lookup


def main() -> int:
    labels_path = APOLLO / "kv_readout" / "experiments" / "control_labels.json"
    if not labels_path.exists():
        print(f"missing {labels_path}; run build_control_labels.py first")
        return 2
    base = json.loads(labels_path.read_text(encoding="utf-8"))
    text_lookup = _load_prompt_lookup()

    rows = []
    n_missing_text = 0
    for r in base["rows"]:
        text = text_lookup.get(r["task_id"], "")
        if not text:
            n_missing_text += 1
            continue
        rows.append({
            "task_id": r["task_id"],
            "family": r["family"],
            "source": r["source"],
            "text": text,
            "labels": r["labels"],
        })
    n = len(rows)

    # Splits
    import random
    rng = random.Random(42)
    perm = list(range(n))
    rng.shuffle(perm)
    random_folds = [[] for _ in range(5)]
    for i, idx in enumerate(perm):
        random_folds[i % 5].append(idx)

    families_present = sorted({r["family"] for r in rows})
    family_held_out = {fam: [i for i, r in enumerate(rows) if r["family"] == fam]
                       for fam in families_present}
    sources_present = sorted({r["source"] for r in rows})
    sprint_held_out = {src: [i for i, r in enumerate(rows) if r["source"] == src]
                       for src in sources_present}

    # Hard-label admissibility heuristic. need_code/need_retrieve perfectly track family →
    # trivial. need_strong/escalate/graph_candidate/verifier_risk are NOT predictable from
    # family alone (per Apollo external baseline AUROC 0.704–0.838); these are admissible.
    label_admissibility = {
        "need_code": "trivial_family_tag",
        "need_retrieve": "trivial_family_tag",
        "verifier_risk": "admissible_hard",
        "need_strong": "admissible_hard",
        "escalate": "admissible_hard",
        "graph_candidate": "admissible_hard",
    }

    # Per-label class balance / per-split sizes
    label_balance = {}
    for lab in label_admissibility:
        pos = sum(1 for r in rows if r["labels"].get(lab) == 1)
        label_balance[lab] = {"pos": pos, "neg": n - pos, "pos_rate": pos / max(1, n)}

    split_summary = {
        "random_5fold": [{"n_test": len(f)} for f in random_folds],
        "family_held_out": {
            fam: {"n_held": len(idxs),
                  "n_train": n - len(idxs),
                  "label_pos_rates_in_held": {
                      lab: sum(rows[i]["labels"][lab] for i in idxs) / max(1, len(idxs))
                      for lab in label_admissibility
                  }}
            for fam, idxs in family_held_out.items()
        },
        "sprint_held_out": {
            src: {"n_held": len(idxs),
                  "n_train": n - len(idxs),
                  "label_pos_rates_in_held": {
                      lab: sum(rows[i]["labels"][lab] for i in idxs) / max(1, len(idxs))
                      for lab in label_admissibility
                  }}
            for src, idxs in sprint_held_out.items()
        },
        "hard_label_only_set": [lab for lab, k in label_admissibility.items()
                                  if k == "admissible_hard"],
    }

    out = {
        "n_total": n,
        "n_missing_text": n_missing_text,
        "label_admissibility": label_admissibility,
        "label_balance": label_balance,
        "splits": split_summary,
        "random_folds_indices": random_folds,
        "family_held_out_indices": family_held_out,
        "sprint_held_out_indices": sprint_held_out,
        "rows": rows,  # full data, embedded for reproducibility
        "honesty": (
            "Labels mined from cached oracle outcomes; texts joined from task pool "
            "modules. Trivial family-tag labels (need_code, need_retrieve) excluded "
            "from Main claims because TF-IDF + LR achieves AUROC ≥ 0.99 on them — "
            "any GPU probe winning on these is uninformative. Admissible hard labels: "
            "need_strong, escalate, verifier_risk, graph_candidate."
        ),
    }
    out_path = APOLLO / "kv_readout" / "experiments" / "a_gpu_readout_dataset.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# A_GPU_READOUT_DATASET\n",
          f"\n- n_total: {n} (missing prompt text dropped: {n_missing_text})\n",
          f"- families: {families_present}\n",
          f"- sprints (sources): {sources_present}\n",
          "\n## Label admissibility for Main claims\n\n",
          "| label | category | pos | neg | pos_rate |\n|---|---|---|---|---|\n"]
    for lab, k in label_admissibility.items():
        b = label_balance[lab]
        md.append(f"| {lab} | {k} | {b['pos']} | {b['neg']} | {b['pos_rate']:.3f} |\n")
    md.append("\n## Random 5-fold split sizes\n\n")
    md.append("| fold | n_test |\n|---|---|\n")
    for i, f in enumerate(random_folds):
        md.append(f"| {i} | {len(f)} |\n")
    md.append("\n## Family-held-out splits\n\n")
    md.append("| held-out family | n_held | n_train |\n|---|---|---|\n")
    for fam, info in split_summary["family_held_out"].items():
        md.append(f"| {fam} | {info['n_held']} | {info['n_train']} |\n")
    md.append("\n## Sprint-held-out splits\n\n")
    md.append("| held-out sprint | n_held | n_train |\n|---|---|---|\n")
    for src, info in split_summary["sprint_held_out"].items():
        md.append(f"| {src} | {info['n_held']} | {info['n_train']} |\n")
    md.append("\n## Honesty\n\n" + out["honesty"] + "\n")
    (APOLLO / "kv_readout" / "reports" / "A_GPU_READOUT_DATASET.md").write_text(
        "".join(md), encoding="utf-8")
    print(f"n={n}; admissible labels: {[l for l, k in label_admissibility.items() if k == 'admissible_hard']}")
    print(f"families: {families_present}")
    print(f"sprints: {sources_present}")
    print(f"wrote {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
