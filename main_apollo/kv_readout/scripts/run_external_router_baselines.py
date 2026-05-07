"""KV-readout Phase 3A — external-router baseline (CPU only).

Joins control labels with prompt text and runs:
  1. Random / majority baselines
  2. TF-IDF + Logistic Regression on prompt text
  3. Family-only baseline (predict based on task family alone)

Reports per-label F1 / AUROC / accuracy under 5-fold cross-validation.
This is the strict baseline that any GPU hidden-state probe MUST beat to
earn a Main Track claim.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APOLLO = REPO / "main_apollo"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "main_rescue_gpu" / "src"))
sys.path.insert(0, str(REPO / "method_discovery" / "src"))


def _load_prompt_lookup() -> dict[str, str]:
    """Reconstruct task_id → prompt-text map from all known task pool modules."""
    lookup: dict[str, str] = {}

    # Smoke / rescue / hard-regime task modules.
    try:
        from agentcontrol.tasks_math import MATH_TASKS
        for t in MATH_TASKS:
            lookup[t["id"]] = t.get("question", "")
    except Exception:
        pass
    try:
        from agentcontrol.tasks_code import CODE_TASKS
        for t in CODE_TASKS:
            lookup[t["id"]] = t.get("prompt", "")
    except Exception:
        pass
    try:
        from agentcontrol.tasks_evidence import EVIDENCE_QA
        for t in EVIDENCE_QA:
            lookup[t["id"]] = t.get("question", "")
    except Exception:
        pass
    try:
        from agentcontrol.rescue_tasks import get_pool as _rp
        for tid, t in _rp().items():
            lookup[tid] = t.get("question", t.get("prompt", ""))
    except Exception:
        pass
    try:
        from agentcontrol.hard_regime_tasks import get_pool as _hp
        for tid, t in _hp().items():
            lookup[tid] = t.get("question", t.get("prompt", ""))
    except Exception:
        pass
    try:
        from agentcontrol_main_rescue.interactive_tasks import get_pool as _ip
        for tid, t in _ip().items():
            lookup[tid] = t.get("question", t.get("prompt", ""))
    except Exception:
        pass
    try:
        from agentcontrol_method.agentic_corpus import get_questions as _aq
        for q in _aq():
            lookup[q["task_id"]] = q.get("question", "")
    except Exception:
        pass
    try:
        from agentcontrol_method.f2_data_analysis_tasks import get_pool as _f2
        for tid, t in _f2().items():
            lookup[tid] = t.get("question", t.get("prompt", ""))
    except Exception:
        pass
    return lookup


def _join_labels_with_text() -> list[dict]:
    labels_path = APOLLO / "kv_readout" / "experiments" / "control_labels.json"
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    text_lookup = _load_prompt_lookup()
    rows = []
    n_missing = 0
    for r in data["rows"]:
        text = text_lookup.get(r["task_id"], "")
        if not text:
            n_missing += 1
            continue
        rows.append({
            "task_id": r["task_id"],
            "family": r["family"],
            "text": text,
            "labels": r["labels"],
        })
    print(f"joined {len(rows)} rows; {n_missing} task_ids missing prompt text")
    return rows


def _stratified_kfold(n: int, k: int = 5, seed: int = 42) -> list[list[int]]:
    import random
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    folds = [[] for _ in range(k)]
    for i, x in enumerate(idx):
        folds[i % k].append(x)
    return folds


def _train_eval(X_train, y_train, X_test, y_test, model_kind: str):
    """Returns (preds, probs) on X_test."""
    import numpy as np
    if model_kind == "majority":
        from collections import Counter
        c = Counter(y_train)
        pred = c.most_common(1)[0][0]
        preds = np.full(len(y_test), pred)
        probs = np.full(len(y_test), c[pred] / len(y_train))
        return preds, probs
    if model_kind == "random":
        rng = np.random.default_rng(0)
        preds = rng.integers(0, 2, len(y_test))
        probs = rng.random(len(y_test))
        return preds, probs
    if model_kind == "tfidf_lr":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
        Xtr = vec.fit_transform(X_train)
        Xte = vec.transform(X_test)
        if len(set(y_train)) < 2:
            # Degenerate fold: only one class.
            preds = np.full(len(y_test), y_train[0])
            probs = np.full(len(y_test), 1.0)
            return preds, probs
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        clf.fit(Xtr, y_train)
        preds = clf.predict(Xte)
        probs = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)] if 1 in clf.classes_ else np.zeros(len(y_test))
        return preds, probs
    if model_kind == "family_only":
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        # Family is encoded as the only feature in X_train (we'll pass it as a list of dicts).
        # X is family string.
        vec = DictVectorizer()
        Xtr = vec.fit_transform([{"family": x} for x in X_train])
        Xte = vec.transform([{"family": x} for x in X_test])
        if len(set(y_train)) < 2:
            preds = np.full(len(y_test), y_train[0])
            probs = np.full(len(y_test), 1.0)
            return preds, probs
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        clf.fit(Xtr, y_train)
        preds = clf.predict(Xte)
        probs = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)] if 1 in clf.classes_ else np.zeros(len(y_test))
        return preds, probs
    raise ValueError(model_kind)


def _eval_metrics(y_true, y_pred, y_prob) -> dict:
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_pos": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        if len(set(y_true)) >= 2:
            out["auroc"] = float(roc_auc_score(y_true, y_prob))
        else:
            out["auroc"] = float("nan")
    except Exception:
        out["auroc"] = float("nan")
    return out


def main() -> int:
    rows = _join_labels_with_text()
    if not rows:
        print("no rows; cannot run")
        return 2
    labels = ("need_code", "need_retrieve", "verifier_risk", "need_strong",
              "escalate", "graph_candidate")
    model_kinds = ("majority", "random", "family_only", "tfidf_lr")
    n = len(rows)
    folds = _stratified_kfold(n, k=5, seed=42)

    out = {"n": n, "labels": {}, "k_folds": 5}
    for label in labels:
        out["labels"][label] = {}
        for kind in model_kinds:
            metric_runs = []
            for fold_id, test_idx in enumerate(folds):
                test_set = set(test_idx)
                train_idx = [i for i in range(n) if i not in test_set]
                X_tr_text = [rows[i]["text"] for i in train_idx]
                X_te_text = [rows[i]["text"] for i in test_idx]
                X_tr_fam = [rows[i]["family"] for i in train_idx]
                X_te_fam = [rows[i]["family"] for i in test_idx]
                y_tr = [rows[i]["labels"][label] for i in train_idx]
                y_te = [rows[i]["labels"][label] for i in test_idx]
                if kind in ("tfidf_lr",):
                    Xtr, Xte = X_tr_text, X_te_text
                elif kind == "family_only":
                    Xtr, Xte = X_tr_fam, X_te_fam
                else:
                    Xtr, Xte = X_tr_text, X_te_text
                preds, probs = _train_eval(Xtr, y_tr, Xte, y_te, kind)
                metric_runs.append(_eval_metrics(y_te, preds, probs))
            agg = {
                k: sum(m[k] for m in metric_runs) / len(metric_runs)
                for k in metric_runs[0].keys() if not (isinstance(metric_runs[0][k], float) and metric_runs[0][k] != metric_runs[0][k])
            }
            # accuracy / f1 may yield NaN for AUROC under degenerate folds; handle.
            out["labels"][label][kind] = agg
    out_path = APOLLO / "kv_readout" / "experiments" / "external_router_baselines.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# KV_READOUT_EXTERNAL_ROUTER_BASELINES\n",
          f"\n- n total: {n}, 5-fold CV, deterministic seed 42\n\n",
          "## Per-label, per-baseline metrics (5-fold mean)\n\n",
          "| label | model | accuracy | f1_pos | f1_macro | auroc |\n|---|---|---|---|---|---|\n"]
    for label in labels:
        for kind in model_kinds:
            r = out["labels"][label].get(kind, {})
            md.append(f"| {label} | {kind} | {r.get('accuracy', float('nan')):.3f} | "
                      f"{r.get('f1_pos', float('nan')):.3f} | "
                      f"{r.get('f1_macro', float('nan')):.3f} | "
                      f"{r.get('auroc', float('nan')):.3f} |\n")
    md.append("\n## Reading\n\n")
    md.append("`tfidf_lr` is the **strict external-router baseline**. Any GPU hidden-state probe must beat this on the same label, with bootstrap CI strictly above. `family_only` quantifies how much of the label is just family discrimination (an unfair shortcut for production routers).\n")
    (APOLLO / "kv_readout" / "reports" / "KV_READOUT_EXTERNAL_BASELINE.md").write_text(
        "".join(md), encoding="utf-8")

    print(f"wrote {out_path.relative_to(REPO)} and reports/KV_READOUT_EXTERNAL_BASELINE.md")
    print("Headline (tfidf_lr):")
    for label in labels:
        r = out["labels"][label].get("tfidf_lr", {})
        print(f"  {label}: acc={r.get('accuracy', float('nan')):.3f} f1_pos={r.get('f1_pos', float('nan')):.3f} auroc={r.get('auroc', float('nan')):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
