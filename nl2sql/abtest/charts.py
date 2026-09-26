"""Deterministic charts for the readout report — matplotlib, no LLM."""
from __future__ import annotations

import base64
import io
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def ramp_chart(ramp: Dict[str, Any], metric: str = "gmv_per_user") -> str:
    """Weekly lift lines: one line per variant, control implicit at 0."""
    weeks: Dict[int, Dict[str, float]] = {}
    for row in ramp["per_week"]:
        weeks.setdefault(row["week"], {})[row["variant_id"]] = row["lift"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    xs = sorted(weeks)
    for vid in sorted({r["variant_id"] for r in ramp["per_week"]}):
        ys = [weeks[w].get(vid) for w in xs]
        ax.plot(xs, ys, marker="o", label=vid)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("test week")
    ax.set_ylabel("weekly lift vs control")
    ax.set_title(f"Ramp — {metric}")
    ax.legend()
    return _b64(fig)


def cut_bars(cuts: List[Dict[str, Any]], metric: str, variant: str, dim: str = "slice_country") -> str:
    """Lift with CI whiskers per slice value (Overall included)."""
    rows = [c for c in cuts if c["metric"] == metric and c["variant_id"] == variant
            and c["slice_dim"] == dim and c.get("lift") is not None]
    if not rows:
        return ""
    rows = rows[::-1]  # horizontal bars top-down
    labels = [c["slice_value"] for c in rows]
    vals = [c["lift"] for c in rows]
    err_lo = [max(0.0, c["lift"] - c["ci"][0]) if c.get("ci") else 0 for c in rows]
    err_hi = [max(0.0, c["ci"][1] - c["lift"]) if c.get("ci") else 0 for c in rows]
    colors = ["#1a7f37" if c.get("significant") and c["lift"] > 0 else
              "#c93c37" if c.get("significant") and c["lift"] < 0 else "#8b949e" for c in rows]
    fig, ax = plt.subplots(figsize=(7, max(1.6, 0.5 * len(rows))))
    ax.barh(labels, vals, xerr=[err_lo, err_hi], color=colors, height=0.55,
            error_kw={"ecolor": "#444", "capsize": 3})
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("lift vs control (CI 95%)")
    ax.set_title(f"{metric} — {variant} by {dim}")
    return _b64(fig)
