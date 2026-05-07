"""A-GPU Phase 3 — strongest-admissible CPU baselines under each split.

Baselines:
  random / majority / family_only / tfidf_lr / tfidf_svm.

For each (label, split) compute AUROC + F1_pos. Report per-label
strongest-admissible bar = max over admissible (non-leaky) baselines that any
GPU hidden-state probe must beat to clear the Main gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APOLLO = REPO / "main_apollo"

DATASET = APOLLO / "kv_readout" / "experiments" / "a_gpu_readout_dataset.json"


def _train_eval(X_train, y_train, X_test, y_test, kind, X_train_fam=None, X_test_fam=None):
    import numpy as np
    if kind == "majority":
        from collections import Counter
        c = Counter(y_train); pred = c.most_common(1)[0][0]
        preds = np.full(len(y_test), pred); probs = np.full(len(y_test), c[pred] / len(y_train))
        return preds, probs
    if kind == "random":
        rng = np.random.default_rng(0)
        return rng.integers(0, 2, len(y_test)), rng.random(len(y_test))
    if kind == "family_only":
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        vec = DictVectorizer()
        Xtr = vec.fit_transform([{"family": x} for x in X_train_fam])
        Xte = vec.transform([{"family": x} for x in X_test_fam])
        if len(set(y_train)) < 2:
            return np.full(len(y_test), y_train[0]), np.full(len(y_test), 1.0)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        clf.fit(Xtr, y_train)
        probs = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)] if 1 in clf.classes_ else np.zeros(len(y_test))
        return clf.predict(Xte), probs
    if kind == "tfidf_lr":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
        Xtr = vec.fit_transform(X_train); Xte = vec.transform(X_test)
        if len(set(y_train)) < 2:
            return np.full(len(y_test), y_train[0]), np.full(len(y_test), 1.0)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        clf.fit(Xtr, y_train)
        probs = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)] if 1 in clf.classes_ else np.zeros(len(y_test))
        return clf.predict(Xte), probs
    if kind == "tfidf_svm":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
        Xtr = vec.fit_transform(X_train); Xte = vec.transform(X_test)
        if len(set(y_train)) < 2:
            return np.full(len(y_test), y_train[0]), np.full(len(y_test), 1.0)
        # Calibrate to get probabilities for AUROC; small CV inside.
        clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=2000),
                                      cv=3, method="sigmoid")
        try:
            clf.fit(Xtr, y_train)
            probs = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)] if 1 in clf.classes_ else np.zeros(len(y_test))
            return clf.predict(Xte), probs
        except Exception:
            return np.full(len(y_test), 0), np.full(len(y_test), 0.5)
    raise ValueError(kind)


def _metrics(y_true, y_pred, y_prob):
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    out = {"accuracy": float(accuracy_score(y_true, y_pred)),
           "f1_pos": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))}
    try:
        if len(set(y_true)) >= 2:
            out["auroc"] = float(roc_auc_score(y_true, y_prob))
        else:
            out["auroc"] = float("nan")
    except Exception:
        out["auroc"] = float("nan")
    return out


def main() -> int:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    rows = data["rows"]
    n = len(rows)
    labels = ("need_code", "need_retrieve", "verifier_risk", "need_strong",
              "escalate", "graph_candidate")
    admissible_hard = {l for l, k in data["label_admissibility"].items() if k == "admissible_hard"}
    BASELINES = ("random", "majority", "family_only", "tfidf_lr", "tfidf_svm")

    out = {"label_admissibility": data["label_admissibility"], "results": {}}

    # 1. random 5-fold
    folds = data["random_folds_indices"]
    for label in labels:
        out["results"].setdefault(label, {}).setdefault("random_5fold", {})
        for kind in BASELINES:
            metrics_runs = []
            for fold_id, test_idx in enumerate(folds):
                tset = set(test_idx)
                tr_idx = [i for i in range(n) if i not in tset]
                X_tr_text = [rows[i]["text"] for i in tr_idx]
                X_te_text = [rows[i]["text"] for i in test_idx]
                X_tr_fam = [rows[i]["family"] for i in tr_idx]
                X_te_fam = [rows[i]["family"] for i in test_idx]
                y_tr = [rows[i]["labels"][label] for i in tr_idx]
                y_te = [rows[i]["labels"][label] for i in test_idx]
                if kind in ("tfidf_lr", "tfidf_svm"):
                    Xtr, Xte = X_tr_text, X_te_text
                else:
                    Xtr, Xte = X_tr_text, X_te_text
                preds, probs = _train_eval(Xtr, y_tr, Xte, y_te, kind, X_tr_fam, X_te_fam)
                metrics_runs.append(_metrics(y_te, preds, probs))
            agg = {k: sum(m[k] for m in metrics_runs) / len(metrics_runs)
                    if isinstance(metrics_runs[0][k], float) and not (metrics_runs[0][k] != metrics_runs[0][k])
                    else float("nan")
                    for k in metrics_runs[0]}
            # Robust nan handling: use nanmean where applicable.
            import math
            def _nanmean(vs):
                vs = [v for v in vs if not (isinstance(v, float) and math.isnan(v))]
                return sum(vs) / len(vs) if vs else float("nan")
            agg = {k: _nanmean([m[k] for m in metrics_runs]) for k in metrics_runs[0]}
            out["results"][label]["random_5fold"][kind] = agg

    # 2. family-held-out (LOFO)
    for label in labels:
        out["results"][label]["family_held_out"] = {}
        for fam, idxs in data["family_held_out_indices"].items():
            tset = set(idxs)
            tr_idx = [i for i in range(n) if i not in tset]
            X_tr_text = [rows[i]["text"] for i in tr_idx]
            X_te_text = [rows[i]["text"] for i in idxs]
            X_tr_fam = [rows[i]["family"] for i in tr_idx]
            X_te_fam = [rows[i]["family"] for i in idxs]
            y_tr = [rows[i]["labels"][label] for i in tr_idx]
            y_te = [rows[i]["labels"][label] for i in idxs]
            for kind in BASELINES:
                Xtr, Xte = X_tr_text, X_te_text
                preds, probs = _train_eval(Xtr, y_tr, Xte, y_te, kind, X_tr_fam, X_te_fam)
                m = _metrics(y_te, preds, probs)
                out["results"][label]["family_held_out"].setdefault(fam, {})[kind] = m

    # Compute strongest-admissible baseline per label per split.
    # Production-admissible baselines exclude `family_only` (peeks at family at decision time).
    PROD_ADMISSIBLE = ("random", "majority", "tfidf_lr", "tfidf_svm")
    summary_strongest = {}
    for label in labels:
        # Random 5-fold strongest admissible
        best_random = max(
            (out["results"][label]["random_5fold"][k].get("auroc", 0.0) or 0.0
             for k in PROD_ADMISSIBLE),
            default=0.0)
        # Family-held-out: macro-average over held-out families
        fam_results = out["results"][label]["family_held_out"]
        avgs = {}
        for kind in PROD_ADMISSIBLE:
            vals = []
            for fam, m in fam_results.items():
                v = m.get(kind, {}).get("auroc")
                if v is not None and not (isinstance(v, float) and v != v):
                    vals.append(v)
            avgs[kind] = sum(vals) / len(vals) if vals else float("nan")
        best_fam = max((v for v in avgs.values() if isinstance(v, float) and not (v != v)), default=float("nan"))
        summary_strongest[label] = {
            "random_5fold_strongest_auroc": best_random,
            "family_held_out_macro_strongest_auroc": best_fam,
            "is_admissible_for_main_claims": data["label_admissibility"][label] == "admissible_hard",
        }
    out["strongest_admissible_per_label"] = summary_strongest

    out_path = APOLLO / "kv_readout" / "experiments" / "a_gpu_baselines.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# A_GPU_BASELINES\n",
          "\n5-fold random CV + leave-one-family-out under all 5 baselines. Production-admissible "
          "baselines = {random, majority, tfidf_lr, tfidf_svm} (`family_only` excluded as unfair "
          "production shortcut).\n\n",
          "## Strongest-admissible AUROC per label\n\n",
          "| label | admissible | random 5-fold | family-held-out (macro) |\n|---|---|---|---|\n"]
    for label in labels:
        s = summary_strongest[label]
        adm = "✅" if s["is_admissible_for_main_claims"] else "trivial"
        md.append(f"| {label} | {adm} | {s['random_5fold_strongest_auroc']:.3f} | "
                  f"{s['family_held_out_macro_strongest_auroc']:.3f} |\n")
    md.append("\n## Random 5-fold detail (AUROC)\n\n")
    md.append("| label | random | majority | family_only | tfidf_lr | tfidf_svm |\n|---|---|---|---|---|---|\n")
    for label in labels:
        cells = []
        for k in ("random", "majority", "family_only", "tfidf_lr", "tfidf_svm"):
            m = out["results"][label]["random_5fold"].get(k, {})
            v = m.get("auroc", float("nan"))
            cells.append(f"{v:.3f}" if isinstance(v, float) and not (v != v) else "—")
        md.append(f"| {label} | " + " | ".join(cells) + " |\n")
    md.append("\n## Headline implication\n\n")
    md.append("Any GPU hidden-state probe must beat the **strongest admissible** AUROC per label "
              "with bootstrap CI strictly above. On family-held-out splits this is the harder bar "
              "(families seen at training disappear at test time). Trivial labels "
              "(`need_code`, `need_retrieve`) are excluded from Main claims regardless of probe wins.\n")
    (APOLLO / "kv_readout" / "reports" / "A_GPU_BASELINES.md").write_text("".join(md), encoding="utf-8")
    print("Strongest admissible AUROC per label:")
    for label in labels:
        s = summary_strongest[label]
        print(f"  {label}: random_5fold={s['random_5fold_strongest_auroc']:.3f} "
              f"family_LOFO_macro={s['family_held_out_macro_strongest_auroc']:.3f} "
              f"admissible={s['is_admissible_for_main_claims']}")
    print(f"wrote {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
