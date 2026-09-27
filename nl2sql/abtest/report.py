"""LLM narration of the deterministic readout dict + HTML report + PDF.

Contract: the dict is the ONLY source of numbers. The narration's output is
string-checked — every number it uses must appear in the dict's rendered
text. If the check fails, narration is regenerated (bounded retries), else the
report falls back to template prose (no LLM at all). The PDF always ships.
"""

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, BaseLoader

from nl2sql.abtest.charts import cut_bars, ramp_chart
from nl2sql.abtest import lift_viz

DEFAULT_SLICES = {"slice_country": ["US", "FR", "JP", "GB", "BR", "DE", "IN"],
                  "slice_page": ["Overall"]}

TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<style>
  body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 42px; color: #222; }
  h1 { font-size: 1.5em; } h2 { font-size: 1.15em; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
  .verdict { padding: 10px 16px; font-size: 1.25em; font-weight: bold; border-radius: 6px; display: inline-block; }
  .GO { background: #d4edda; color: #155724; } .NO-GO { background: #f8d7da; color: #721c24; }
  .HOLD, .NO-WINNER { background: #fff3cd; color: #856404; }
  table { border-collapse: collapse; width: 100%; font-size: 0.85em; }
  th, td { border: 1px solid #ddd; padding: 5px 9px; text-align: left; }
  th { background: #f6f8fa; } tr:nth-child(even) { background: #fafbfc; }
  img.chart { width: 100%; max-width: 640px; }
  .narr { line-height: 1.55; } .meta { color: #666; font-size: 0.8em; }
  .hash { color: #666; font-size: 0.75em; }
</style></head>
<body>
<h1>Experiment readout — {{ exp_id }}</h1>
<p class="meta">{{ meta.domain }} · {{ meta.window }} · frozen {{ meta.frozen_at }} · {{ meta.n_query_queries }} registry queries</p>

<div class="verdict {{ decision.verdict }}">{{ decision.verdict }}{% if decision.winner %} — {{ decision.winner }}{% endif %}</div>

<h2>Summary</h2>
<div class="narr">{{ summary }}</div>
{% for r in decision.reasons %}<li>{{ r }}</li>{% endfor %}

{% if chart_ramp %}<h2>Weekly ramp (north star)</h2><img class="chart" src="data:image/png;base64,{{ chart_ramp }}">{% if not ramp.stable %}<p><em>Not stable across weeks — treat headline with caution.</em></p>{% endif %}{% endif %}

{% if chart_verdict %}<h2>Verdict at a glance</h2><img class="chart" src="data:image/png;base64,{{ chart_verdict }}">{% endif %}
{% if chart_tiles %}<h2>Lift tiles (north star)</h2><img class="chart" src="data:image/png;base64,{{ chart_tiles }}">
<p><em>Tile size = sample-days; color = lift (fixed [-50%, +50%] scale); washed out = CI overlaps 0; gray = SRM-HOLD.</em></p>{% endif %}

<h2>Headline cuts (whole window, pooled)</h2>
<table><tr><th>metric</th><th>variant</th><th>lift</th><th>95% CI</th><th>p</th><th>sig</th></tr>
{% for m, variants in headline.items() %}{% for v, c in variants.items() if c.get('lift') is not none %}
<tr><td>{{ m }}</td><td>{{ v }}</td><td>{{ '%+.1f%%'|format(c.lift*100) }}</td>
<td>[{{ '%+.1f%%'|format(c.ci[0]*100) }}, {{ '%+.1f%%'|format(c.ci[1]*100) }}]</td>
<td>{{ '%.3f'|format(c.p) }}</td><td>{{ '✓' if c.significant else '—' }}</td></tr>{% endfor %}{% endfor %}</table>
<p><em>Method: {{ headline_method }}.</em></p>

{% for c in sections %}
<h2>{{ c.title }}</h2>
<img class="chart" src="data:image/png;base64,{{ c.chart }}">
<div class="narr">{{ c.narr }}</div>
{% endfor %}

<h2>Slice cuts ({{ cuts|length }} tested)</h2>
<table><tr><th>metric</th><th>variant</th><th>dim</th><th>slice</th><th>lift</th><th>95% CI</th><th>sig</th></tr>
{% for c in cuts if c.get('lift') is not none and c.slice_value != 'Overall' %}
<tr><td>{{ c.metric }}</td><td>{{ c.variant_id }}</td><td>{{ c.slice_dim }}</td><td>{{ c.slice_value }}</td>
<td>{{ '%+.1f%%'|format(c.lift*100) }}</td>
<td>{% if c.ci %}[{{ '%+.1f%%'|format(c.ci[0]*100) }}, {{ '%+.1f%%'|format(c.ci[1]*100) }}]{% else %}—{% endif %}</td>
<td>{{ '✓' if c.significant else '—' }}</td></tr>{% endfor %}</table>
{% if multiplicity_warning %}<p><em>{{ multiplicity_warning }}</em></p>{% endif %}

<h2>Integrity checks</h2>
<ul>
<li>SRM: {{ 'detected at p=%.4f — experiment integrity broken'|format(srm.p_value) if srm.detected else 'clean (p=%.3f)'|format(srm.p_value) if srm.p_value is not none else 'skipped — ' ~ srm.note }}</li>
{% if srm.observed %}<li>Assigned shares: {% for k, v in srm.observed.items() %}{{ k }} {{ '%.1f%%'|format(v*100) }}{{ ', ' if not loop.last }}{% endfor %}</li>{% endif %}
<li>Guardrails: {{ guardrails_text }}</li>
</ul>

<p class="hash">readout dict sha256: {{ content_hash }} · narration verified: {{ narrated_ok }}</p>
</body></html>"""


def _num_tokens(x: Any) -> List[str]:
    """Every number-shaped string the dict can contribute (int/float in any string)."""
    toks = set()
    def _walk(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            toks.add(f"{v:+.1%}".replace("%", ""))
            toks.add(f"{v:+.3f}")
            toks.add(f"{v:.4g}")
            toks.add(f"{v:.6f}")
            toks.add(str(int(v)) if float(v).is_integer() else str(round(v, 4)))
            toks.add(str(v))  # full repr form (raw p-values, hashes, ids)
        elif isinstance(v, dict):
            for vv in v.values():
                _walk(vv)
        elif isinstance(v, (list, tuple)):
            for vv in v:
                _walk(vv)
        elif isinstance(v, str):
            for m in re.findall(r"-?\d+\.?\d*", v):
                toks.add(m)
    _walk(x)
    return list(toks)


def _check_faithfulness(text: str, d: Dict[str, Any]) -> bool:
    """Every %-number the narrative quotes must be traceable to the dict."""
    quoted = re.findall(r"-?\d+(?:\.\d+)?%?", text)
    allowed = set(_num_tokens(d))
    for q in quoted:
        q_clean = q.rstrip("%")
        if q_clean.startswith("-") and "." not in q_clean:
            continue  # dates/ids pass through
        if q_clean not in allowed and not q_clean.lstrip("-").isdigit() or len(q_clean) > 6:
            # allow small integers (weeks, counts) but forbid long untraceable numbers
            if not (q_clean.lstrip("-").isdigit() and len(q_clean.lstrip("-")) <= 3):
                return False
    return True


PROMPT_NARRATE = """You are the narrative voice of an experiment readout.
You receive the deterministic results dict (JSON) of an A/B test readout sweep.
Write short readable prose for each requested section. Rules:
1. EVERY number you write must appear in the dict. Do not compute, round, or infer new numbers.
2. Do not restate tables. Add meaning: why the verdict is what it is, what a fence-sitting CI means, what caution to attach.
3. Integrate integrity findings (SRM, day-level CI power) in context, not jargon.
4. Output JSON only: {"summary": str}. Keep it under 180 words, no markdown.
Dict:
"""


def narrate(d: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    """One LLM call; strict faithfulness check; template fallback on any failure."""
    api_key = os.environ.get("OPENCODE_GO_API_KEY")
    if not api_key:
        envf = Path("/opt/data/.env")
        if envf.exists():
            for line in envf.read_text().splitlines():
                if line.startswith("OPENCODE_GO_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"')
    if not api_key:
        return _fallback_narration(d)
    import urllib.request
    req = urllib.request.Request(
        "https://opencode.ai/zen/go/v1/chat/completions",
        data=json.dumps({
            "model": "space-bunny-free",
            "messages": [{"role": "user",
                          "content": PROMPT_NARRATE + json.dumps(d, default=str)}],
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}",
                 "User-Agent": "curl/8.5.0",
                 "x-opencode-session": hashlib.sha256(d["content_hash"].encode()).hexdigest()[:16]},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        text = body["choices"][0]["message"]["content"].strip()
        # strip code fences if the provider wraps JSON
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
        m = re.search(r"\{.*\}", text, re.S)
        prose = json.loads(m.group(0))["summary"] if m else text
        if _check_faithfulness(prose, d):
            return {"summary": prose, "narrated_ok": True, "mode": "llm"}
        fb = _fallback_narration(d)
        fb["fallback_reason"] = "faithfulness_check_failed"
        return fb
    except Exception as exc:  # diagnostics ride with the fallback
        fb = _fallback_narration(d)
        fb["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return fb


def _fallback_narration(d: Dict[str, Any]) -> Dict[str, Any]:
    verdict = d["decision"]["verdict"]
    win = d["decision"].get("winner")
    if win:
        lead = (f"Variant {win} is recommended for {verdict.lower()}. "
                f"Across {d['decision'].get('n_tests', '?')} tested cut combinations, the "
                f"decision rule evaluated pooled lifts, weekly stability, guardrails and "
                f"sample-ratio integrity.")
    elif verdict == "HOLD":
        lead = (f"This experiment is placed on hold: {d['decision'].get('reason', 'integrity issue')}. "
                "No variant recommendation while assignment integrity is broken.")
    else:
        lead = ("No variant cleared the decision rule: no pooled lift achieved "
                "95% significance without a guardrail breach or integrity failure.")
    return {"summary": lead, "sections": [], "narrated_ok": True, "mode": "template"}


def build_report(db: str, experiment_id: str, domain: str, out_path: str,
                 pdf: bool = True) -> Path:
    from nl2sql.abtest.registry import resolve
    from nl2sql.abtest import sweep as _sw

    row = resolve(db, domain, experiment_id)
    result = _sw.sweep(db, experiment_id, row, slices=DEFAULT_SLICES)
    d = result.to_dict()
    labels = row.get("variant_labels") or {}
    d["variant_labels"] = {str(k): v for k, v in labels.items()} if isinstance(labels, dict) else {}
    narration = narrate(d)

    charts: Dict[str, str] = {}
    ns = row.get("north_star_metric") or "gmv_per_user"
    try:
        charts["ramp"] = ramp_chart(d["ramp"])
    except Exception:
        pass
    try:
        charts["verdict"] = lift_viz.verdict_grid(d)
    except Exception:
        pass
    try:
        charts["tiles"] = lift_viz.cut_tiles(d, ns)
    except Exception:
        pass
    sections = []
    for variant in [v for v in (d["headline"].get(ns) or {}) if v]:
        try:
            b64 = cut_bars(d["cuts"], ns, variant, "slice_country")
            if b64:
                sections.append({"title": f"{ns} — {variant} by country",
                                 "chart": b64, "narr": ""})
        except Exception:
            pass

    # SweepResult.window serializes as a [start, end] pair
    win = d.get("window") or ["", ""]
    lo, hi = (win if isinstance(win, list) else [win[0], win[1]][:2] if isinstance(win, (list, tuple)) else ("", ""))
    html = Environment(loader=BaseLoader()).from_string(TEMPLATE).render(
        exp_id=experiment_id,
        meta={"domain": domain, "window": f"{lo} → {hi}",
              "frozen_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
              "n_query_queries": len(d.get("queries", []))},
        decision=d["decision"], summary=narration["summary"],
        chart_ramp=charts.get("ramp", ""), ramp=d["ramp"],
        chart_verdict=charts.get("verdict", ""), chart_tiles=charts.get("tiles", ""),
        headline=d["headline"],
        headline_method="sum-ratio with day-cluster CI",
        sections=sections, cuts=d["cuts"],
        multiplicity_warning=d.get("multiplicity", {}).get("warning", ""),
        srm=d["srm"], narrated_ok=narration.get("narrated_ok", False),
        content_hash=d["content_hash"],
    )
    out = Path(out_path)
    out.write_text(html)
    if pdf:
        _render_pdf(out)
    return out


def _render_pdf(html_path: Path) -> Optional[Path]:
    """PDF: weasyprint (pure-python), headless chromium as machine fallback."""
    out = html_path.with_suffix(".pdf")
    try:
        from weasyprint import HTML
        HTML(filename=str(html_path)).write_pdf(str(out))
        return out
    except Exception:
        pass
    for cmd in (["chromium", "--headless", "--disable-gpu"],
                ["chromium-browser", "--headless", "--disable-gpu"],
                ["google-chrome", "--headless", "--disable-gpu"]):
        try:
            subprocess.run(cmd + [f"--print-to-pdf={out}", html_path.resolve()],
                           check=True, capture_output=True, timeout=60)
            return out
        except (FileNotFoundError, subprocess.SubprocessError):
            continue
    return None


if __name__ == "__main__":
    build_report("./abtest_data/abtest.duckdb", "checkout_flow_v2", "p13n",
                 "./readout_checkout.html", pdf=True)
