"""Lift-only visualizations, deterministic like charts.py.

No absolute metrics are drawn — every tile/label is a cut from the readout
dict (lift %, CI, significance), so the faithfulness gate (numbers string-
matched against the dict) extends to these charts.

Charts:
- lift_tiles: finviz-style treemap. Size = sample-days behind the cell
  (n_days, the only exposure the scoresheet holds), color = lift, washed
  out when the CI overlaps 0. Fixed [-50%, +50%] domain.
- verdict_grid: metric x variant grid of decision labels, gray = not
  significant, green/red = significant lift, SRM-HOLD blanks the board.
"""
from __future__ import annotations

import io
import base64
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

# Fixed divergence domain: a cell's color is comparable across reports/days.
_LIFT_LIMIT = 0.50
_NORM = TwoSlopeNorm(vmin=-_LIFT_LIMIT, vcenter=0.0, vmax=_LIFT_LIMIT)


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _srm_hold(readout: Dict[str, Any]) -> bool:
    return ((readout.get("srm") or {}).get("status") == "SRM-HOLD"
            or (readout.get("decision", {}).get("verdict") == "SRM-HOLD"))


def _cut_color(cut: Dict[str, Any], hold: bool) -> Any:
    if hold:
        return "#d3d3d3"
    lift = cut.get("lift")
    if lift is None:
        return "#f2f2f2"
    cmap = plt.get_cmap("RdYlGn")  # red = negative lift, green = positive
    rgba = cmap(_NORM(lift))
    if not (cut.get("significant") and not cut.get("underpowered_days")):
        rgba = (rgba[0], rgba[1], rgba[2], 0.38)  # wash out: CI overlaps 0
    return rgba


def _sort_key(c: Dict[str, Any]):
    return (c["variant_id"], -abs(c.get("lift") or 0.0))


def cut_tiles(readout: Dict[str, Any], metric: str) -> Optional[str]:
    """Finviz-style lift treemap for one metric.

    One row band per slice_dim (e.g. slice_country, slice_page), bands split
    into variant tiles (t1, t2). Tile size = n_days (assigned sample-days —
    the only exposure the contract holds). Color = lift on a fixed
    [-50%, +50%] scale; alpha washed out when not significant.
    """
    cuts = [c for c in readout.get("cuts", [])
            if c["metric"] == metric and c.get("lift") is not None]
    if not cuts:
        return None
    try:
        import squarify  # type: ignore
    except ImportError:
        return None

    hold = _srm_hold(readout)
    dims = sorted({c["slice_dim"] for c in cuts})
    variants = sorted({c["variant_id"] for c in cuts})
    n_bands = len(dims)

    fig, ax = plt.subplots(figsize=(7.6, 2.0 * n_bands))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    for band, dim in enumerate(dims):
        band_h = 100.0 / n_bands
        band_y0 = 100 - (band + 1) * band_h  # first dim on top
        bcuts = [c for c in cuts if c["slice_dim"] == dim]
        weights = [float(max(1, min(c.get("n_days", [1])))) for c in bcuts]
        rects = squarify.squarify(
            squarify.normalize_sizes(weights, 100, band_h),
            0, band_y0, 100, band_h)
        for c, r in zip(bcuts, rects):
            rgba = _cut_color(c, hold)
            ax.add_patch(plt.Rectangle(
                (r["x"], r["y"]), r["dx"], r["dy"],
                facecolor=rgba, edgecolor="white", linewidth=1.2))
            days = min(c.get("n_days", [0]))
            label = f'{c["variant_id"]} · {c["slice_value"]}\n{_pct(c.get("lift"))}\nn={days}d'
            ax.text(r["x"] + r["dx"] / 2, r["y"] + r["dy"] / 2, label,
                    ha="center", va="center", fontsize=8, color="black")
        ax.text(1, band_y0 + band_h - 3, dim, fontsize=7,
                color="#555", ha="left", va="top")
    ax.set_title(f"Lift tiles — {metric}"
                 + ("   [SRM-HOLD: no color]" if hold else ""))
    return _b64(fig)


def _pct(x: Any) -> str:
    return "n/a" if x is None else f"{x * 100:+.1f}%"


def verdict_grid(readout: Dict[str, Any]) -> Optional[str]:
    """Metric x variant grid; cell = lift + significance color decision."""
    cuts = [c for c in readout.get("cuts", []) if c.get("lift") is not None]
    if not cuts:
        return None
    hold = _srm_hold(readout)
    metrics = sorted({c["metric"] for c in cuts})
    variants = sorted({c["variant_id"] for c in cuts})
    lookup = {(c["metric"], c["variant_id"]): c
              for c in cuts if c["slice_dim"] == "Overall"}
    # fall back to per-slice cut if no Overall cut for the pair
    if missing := [k for k in [(m, v) for m in metrics for v in variants] if k not in lookup]:
        for m, v in missing:
            for c in cuts:
                if c["metric"] == m and c["variant_id"] == v:
                    lookup.setdefault((m, v), c)
                    break
    fig, ax = plt.subplots(figsize=(0.4 + 2.2 * len(variants), 0.4 + 0.9 * len(metrics)))
    norm = TwoSlopeNorm(vmin=-_LIFT_LIMIT, vcenter=0.0, vmax=_LIFT_LIMIT)
    for row, metric in enumerate(metrics):
        for col, vid in enumerate(variants):
            c = lookup.get((metric, vid))
            if not c:
                continue
            ax.add_patch(Rectangle((col, len(metrics) - 1 - row), 0.92, 0.84,
                                   facecolor=_cut_color(c, hold),
                                   edgecolor="#999", linewidth=0.6))
            sig = "•" if (c.get("significant") and not c.get("underpowered_days") and not hold) else " "
            ax.text(col + 0.46, len(metrics) - 1 - row + 0.42,
                    f'{_pct(c.get("lift"))} {sig}\nCI [{_pct(c["ci"][0])}, {_pct(c["ci"][1])}]',
                    ha="center", va="center", fontsize=8, color="black")
    ax.set_xticks([i + 0.46 for i in range(len(variants))])
    ax.set_xticklabels(variants)
    ax.set_yticks([len(metrics) - 1 - i + 0.42 for i in range(len(metrics))])
    ax.set_yticklabels(metrics)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlim(0, len(variants))
    ax.set_ylim(0, len(metrics))
    verdict = (readout.get("decision") or {}).get("verdict", "")
    ax.set_title(f"Verdict: {verdict}" + ("  [SRM-HOLD: gray]" if hold else ""))
    return _b64(fig)
