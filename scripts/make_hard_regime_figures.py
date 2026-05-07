"""Make publication-quality figures from hard-regime real outcomes."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PROVIDERS = ("deepseek", "together")
LABEL_COLORS = {
    "saturation": "#2ca02c",          # green: cheap suffices
    "router-sufficient": "#ff7f0e",   # orange: cascade router enough
    "graph-headroom": "#d62728",      # red: graph value
    "verifier-risk": "#9467bd",       # purple
    "no-signal": "#7f7f7f",           # gray
}


def _load_summary():
    p = REPO / "experiments" / "hard_regime_summary_joint.json"
    return json.loads(p.read_text(encoding="utf-8"))


def fig_regime_map(joint):
    """Grid heatmap of interpretation labels per (provider, family, regime)."""
    families = ("math", "code", "evidence")
    regimes = ("easy_saturation", "medium_headroom", "hard_strong_gap",
               "weak_verifier_risk", "evidence_support_risk")
    titles = {"deepseek": "DeepSeek\n(chat V4 non-thinking / reasoner V4 thinking)",
              "together": "Together AI\n(Qwen2.5-7B / Llama-3.3-70B)"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, prov in zip(axes, PROVIDERS):
        cells = {(c["family"], c["regime"]): c for c in joint[prov]["regime_map"]}
        for i, fam in enumerate(families):
            for j, reg in enumerate(regimes):
                cell = cells.get((fam, reg))
                if not cell:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor="white",
                                               edgecolor="lightgray"))
                    continue
                color = LABEL_COLORS.get(cell["interpretation"], "white")
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="black"))
                txt = f"{cell['interpretation']}\nn={cell['n']}\nc={cell['cheap_success']:.2f} s={cell['strong_success']:.2f}\nsave={cell['graph_saving_pct']:.0f}%"
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=7)
        ax.set_xlim(0, len(regimes)); ax.set_ylim(0, len(families))
        ax.set_xticks([j + 0.5 for j in range(len(regimes))])
        ax.set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=8)
        ax.set_yticks([i + 0.5 for i in range(len(families))])
        ax.set_yticklabels(families)
        ax.set_title(titles.get(prov, prov), fontsize=10)
        ax.invert_yaxis()
    fig.suptitle("Regime map: per (provider, family, regime) AgentControl label", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "regime_map.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def fig_oracle_gap_by_regime(joint):
    """Cheap vs strong success per (family, regime), side by side per provider."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, prov in zip(axes, PROVIDERS):
        cells = sorted(joint[prov]["regime_map"], key=lambda c: (c["family"], c["regime"]))
        labels = [f"{c['family']}\n/{c['regime']}" for c in cells]
        cheap = [c["cheap_success"] for c in cells]
        strong = [c["strong_success"] for c in cells]
        x = list(range(len(cells)))
        ax.bar([xi - 0.2 for xi in x], cheap, width=0.4, label="cheap", color="#1f77b4")
        ax.bar([xi + 0.2 for xi in x], strong, width=0.4, label="strong", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("success rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(prov)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Cheap-vs-strong success per (family, regime) — hard regime real outcomes\n(no oracle gap because cascade router subsumes graph routes; see Section 4)", fontsize=11)
    plt.tight_layout()
    # Filename kept for backward-compat; figure now correctly titled cheap-vs-strong success.
    out = FIG_DIR / "cheap_vs_strong_by_regime.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def fig_provider_comparison(joint):
    """Per-family success of cheap vs strong, both providers."""
    families = ("math", "code", "evidence")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.2
    x = list(range(len(families)))
    cheap_ds = []
    strong_ds = []
    cheap_tg = []
    strong_tg = []
    for fam in families:
        cs_ds, ss_ds, cs_tg, ss_tg = 0, 0, 0, 0
        n_ds, n_tg = 0, 0
        for prov, lists in (("deepseek", (cheap_ds, strong_ds)),
                            ("together", (cheap_tg, strong_tg))):
            cells = [c for c in joint[prov]["regime_map"] if c["family"] == fam]
            n = sum(c["n"] for c in cells)
            if n == 0:
                lists[0].append(0); lists[1].append(0)
                continue
            avg_c = sum(c["cheap_success"] * c["n"] for c in cells) / n
            avg_s = sum(c["strong_success"] * c["n"] for c in cells) / n
            lists[0].append(avg_c); lists[1].append(avg_s)
    ax.bar([xi - 1.5 * width for xi in x], cheap_ds, width, label="DeepSeek cheap", color="#9ecae1")
    ax.bar([xi - 0.5 * width for xi in x], strong_ds, width, label="DeepSeek strong", color="#3182bd")
    ax.bar([xi + 0.5 * width for xi in x], cheap_tg, width, label="Together cheap", color="#fdae6b")
    ax.bar([xi + 1.5 * width for xi in x], strong_tg, width, label="Together strong", color="#e6550d")
    ax.set_xticks(x); ax.set_xticklabels(families)
    ax.set_ylim(0, 1.05); ax.set_ylabel("success rate")
    ax.set_title("Provider comparison — hard regime real outcomes (n=90/provider)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "provider_comparison_hard_regime.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def fig_verifier_ablation():
    """Verifier-aware vs no-verifier across budget tiers, per provider."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, prov in zip(axes, PROVIDERS):
        p = REPO / f"experiments/hard_regime_summary_{prov}.json"
        if not p.exists():
            ax.set_visible(False); continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = d["verifier_ablation"]
        budgets = [r["budget"] for r in rows]
        va_cost = [r["va_cost"] for r in rows]
        nv_cost = [r["nv_cost"] for r in rows]
        va_succ = [r["va_succ"] for r in rows]
        nv_succ = [r["nv_succ"] for r in rows]
        ax.plot(va_cost, va_succ, "o-", label="verifier-aware", color="#2ca02c")
        ax.plot(nv_cost, nv_succ, "s--", label="no-verifier", color="#d62728")
        for i, b in enumerate(budgets):
            ax.annotate(f"b={int(b)}", (va_cost[i], va_succ[i]), fontsize=7,
                        textcoords="offset points", xytext=(3, 3))
        ax.set_xlabel("avg cost (units)")
        ax.set_ylabel("success rate")
        ax.set_title(prov)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Verifier ablation Pareto frontier — hard regime", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "verifier_ablation_frontier.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def main() -> int:
    joint = _load_summary()
    paths = []
    paths.append(fig_regime_map(joint))
    paths.append(fig_oracle_gap_by_regime(joint))
    paths.append(fig_provider_comparison(joint))
    paths.append(fig_verifier_ablation())
    for p in paths:
        print(f"wrote {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
