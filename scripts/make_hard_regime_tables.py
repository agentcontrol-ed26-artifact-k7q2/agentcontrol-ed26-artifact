"""Generate paper-ready markdown tables for the hard-regime evaluation."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper"
PAPER.mkdir(parents=True, exist_ok=True)


def _load(name):
    p = REPO / "experiments" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def hard_regime_results():
    j = _load("hard_regime_summary_joint.json")
    if not j:
        return ""
    md = ["# Table — Hard-Regime Real-Model Results (n=90/provider)\n\n",
          "| provider | cheap | strong | router cost | graph cost | saving | success delta | router succ | graph succ | unsup risk |\n",
          "|---|---|---|---|---|---|---|---|---|---|\n"]
    PROV = {"deepseek": ("deepseek-chat (V4 non-thinking)", "deepseek-reasoner (V4 thinking / R1)"),
            "together": ("Qwen2.5-7B-Instruct-Turbo", "Llama-3.3-70B-Instruct-Turbo")}
    for p, d in j.items():
        c, s = PROV.get(p, (p, p))
        md.append(f"| {p} | {c} | {s} | {d['router_avg_cost']:.3f} | {d['graph_avg_cost']:.3f} | "
                  f"{d['graph_query_cost_saving_pct']:.2f}% | {d['graph_query_success_delta_pp']:.2f} pp | "
                  f"{d['router_success']:.3f} | {d['graph_success']:.3f} | {d['unsupported_evidence_risk']:.4f} |\n")
    return "".join(md)


def regime_map():
    j = _load("hard_regime_summary_joint.json")
    if not j:
        return ""
    md = ["# Table — Regime Map (per provider × family × difficulty regime)\n\n"]
    for p, d in j.items():
        md.append(f"## {p}\n\n")
        md.append("| family | regime | n | cheap | strong | router_cost | graph_cost | saving | unsup | label |\n")
        md.append("|---|---|---|---|---|---|---|---|---|---|\n")
        for c in sorted(d["regime_map"], key=lambda c: (c["family"], c["regime"])):
            md.append(f"| {c['family']} | {c['regime']} | {c['n']} | "
                      f"{c['cheap_success']:.3f} | {c['strong_success']:.3f} | "
                      f"{c['router_cost']:.3f} | {c['graph_cost']:.3f} | "
                      f"{c['graph_saving_pct']:.2f}% | {c['unsupported_risk']:.3f} | "
                      f"**{c['interpretation']}** |\n")
        md.append("\n")
    return "".join(md)


def provider_comparison():
    j = _load("hard_regime_summary_joint.json")
    if not j:
        return ""
    md = ["# Table — Provider Comparison (per family, hard regime)\n\n",
          "| family | DeepSeek cheap | DeepSeek strong | Together cheap | Together strong |\n",
          "|---|---|---|---|---|\n"]
    fams = ("math", "code", "evidence")
    rows = {}
    for p in ("deepseek", "together"):
        for c in j[p]["regime_map"]:
            key = (c["family"], p)
            rows.setdefault(key, []).append((c["cheap_success"], c["strong_success"], c["n"]))
    for fam in fams:
        cells = []
        for p in ("deepseek", "together"):
            data = rows.get((fam, p), [])
            n = sum(d[2] for d in data)
            if n == 0:
                cells.extend(["—", "—"])
                continue
            avg_c = sum(d[0] * d[2] for d in data) / n
            avg_s = sum(d[1] * d[2] for d in data) / n
            cells.append(f"{avg_c:.3f}")
            cells.append(f"{avg_s:.3f}")
        md.append(f"| {fam} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |\n")
    return "".join(md)


def verifier_ablation():
    md = ["# Table — Verifier Pareto Ablation (across budget tiers)\n\n"]
    for p in ("deepseek", "together"):
        d = _load(f"hard_regime_summary_{p}.json")
        if not d:
            continue
        md.append(f"## {p}\n\n")
        md.append("| budget | va succ | va cost | nv succ | nv cost | nv cost premium |\n|---|---|---|---|---|---|\n")
        for r in d["verifier_ablation"]:
            md.append(f"| {r['budget']} | {r['va_succ']:.3f} | {r['va_cost']:.3f} | "
                      f"{r['nv_succ']:.3f} | {r['nv_cost']:.3f} | "
                      f"{r['no_verifier_cost_premium_pct']:.2f}% |\n")
        md.append("\n")
    return "".join(md)


def partial_strong():
    md = ["# Table — Partial-Strong Ablation\n\n"]
    for p in ("deepseek", "together"):
        d = _load(f"hard_regime_summary_{p}.json")
        if not d:
            continue
        md.append(f"## {p}\n\n")
        md.append("| variant | success | avg_cost |\n|---|---|---|\n")
        for k, v in d["partial_strong"].items():
            ag = v["aggregate"]
            md.append(f"| {k} | {ag['success_rate']:.3f} | {ag['avg_cost']:.3f} |\n")
        md.append("\n")
    return "".join(md)


def bootstrap_table():
    b = _load("hard_regime_bootstrap.json")
    if not b:
        return ""
    md = ["# Table — Bootstrap CI (95%, hard regime)\n\n"]
    for p, d in b.items():
        md.append(f"## {p}\n\n")
        md.append("### Graph-query saving by stratification\n\n")
        md.append("| strat | mean | ci_lo | ci_hi |\n|---|---|---|---|\n")
        for s, name in (("unstratified", "unstratified"),
                        ("family_stratified", "family-strat"),
                        ("regime_stratified", "regime-strat"),
                        ("family_regime_stratified", "family×regime")):
            ci = d["graph_query_cost_saving_pct"][s]
            md.append(f"| {name} | {ci['mean']:.3f} | {ci['ci_lo']:.3f} | {ci['ci_hi']:.3f} |\n")
        md.append("\n### Cheap-vs-strong success gap (pp, family-strat)\n\n")
        ci = d["cheap_vs_strong_success_gap_pp"]
        md.append(f"mean = {ci['mean']:.3f}, 95% CI = [{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]\n\n")
    return "".join(md)


def main() -> int:
    (PAPER / "table_hard_regime_results.md").write_text(hard_regime_results(), encoding="utf-8")
    (PAPER / "table_regime_map.md").write_text(regime_map(), encoding="utf-8")
    (PAPER / "table_provider_comparison.md").write_text(provider_comparison(), encoding="utf-8")
    (PAPER / "table_verifier_ablation.md").write_text(verifier_ablation(), encoding="utf-8")
    (PAPER / "table_partial_strong.md").write_text(partial_strong(), encoding="utf-8")
    (PAPER / "table_bootstrap_hard_regime.md").write_text(bootstrap_table(), encoding="utf-8")
    print("wrote 6 paper/table_*.md files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
