// readout viz — deterministic interactions over the embedded readout dict.
// Every number rendered here comes from the sweep dict; no external data.
// No dependencies; works offline; the dict itself carries the content hash.
function vizInit(DATA) {
  const $ = (id) => document.getElementById(id);
  const fmt = (x, d = 1) => (x === null || x === undefined) ? "—" : (x >= 0 ? "+" : "") + (x * 100).toFixed(d) + "%";

  // ---------- state ----------
  const metrics = [...new Set(DATA.cuts.map(c => c.metric))].sort();
  const variants = [...new Set(DATA.cuts.map(c => c.variant_id))].sort();
  const dims = [...new Set(DATA.cuts.map(c => c.slice_dim))].sort();
  let visibleDims = new Set(dims);
  let sel = null; // {metric, variant}

  const srmDetected = (DATA.srm && DATA.srm.detected === true) || (DATA.decision.verdict === "HOLD");
  const hold = srmDetected;

  const L = DATA.variant_labels || {};
  const label = (v) => L[v] ? `${v} · ${L[v]}` : v;

  // ---------- verdict pivot grid ----------
  function cellColor(c) {
    if (hold) return "#c9c9c9";
    if (!c) return "#f4f4f4";
    const lo = DATA.cuts.length ? c.lift : 0;
    // fixed domain [-50%, +50%], diverging through white
    const t = Math.max(-1, Math.min(1, c.lift / 0.5));
    if (!c.significant || c.underpowered_days) { // washed pastel
      return c.lift >= 0 ? "rgba(92,160,92,0.25)" : "rgba(200,90,80,0.25)";
    }
    // significant: stronger hue by |lift|
    const a = Math.min(1, 0.45 + Math.abs(c.lift) * 2);
    return c.lift >= 0 ? `rgba(34,139,74,${a})` : `rgba(205,60,50,${a})`;
  }

  function renderGrid() {
    const wrap = $("vz-grid");
    wrap.innerHTML = "";
    const table = document.createElement("table");
    table.className = "vz-grid";
    const head = `<tr><th></th>${variants.map(v => `<th>${label(v)}</th>`).join("")}</tr>`;
    let body = "";
    for (const m of metrics) {
      body += `<tr><th class="met">${m}</th>`;
      for (const v of variants) {
        const c = DATA.cuts.find(x => x.metric === m && x.variant_id === v && visibleDims.has(x.slice_dim));
        if (!c) { body += `<td class="vz-cell empty"></td>`; continue; }
        const bullet = (c.significant && !c.underpowered_days && !hold) ? " •" : "";
        const classes = ["vz-cell", sel && sel.metric === m && sel.variant_id === v ? "sel" : ""].filter(Boolean).join(" ");
        body += `<td class="${classes}" data-m="${m}" data-v="${v}" style="background:${cellColor(c)}">` +
                `<span class="lift">${fmt(c.lift)}${bullet}</span>` +
                `<span class="ci">CI [${fmt(c.ci[0])}, ${fmt(c.ci[1])}]</span></td>`;
      }
      body += "</tr>";
    }
    table.innerHTML = head + body;
    wrap.appendChild(table);
    table.querySelectorAll(".vz-cell[data-m]").forEach(td => {
      td.onclick = () => {
        const m = td.dataset.m, v = td.dataset.v;
        sel = (sel && sel.metric === m && sel.variant === v) ? null : { metric: m, variant_id: v };
        render(); // re-render everything on selection
      };
    });
  }

  // ---------- forest plot (cut CI coherence per selected cell or metric) ----------
  function forestCuts() {
    let cuts = DATA.cuts.filter(c => visibleDims.has(c.slice_dim) && c.lift !== null);
    if (sel) cuts = cuts.filter(c => c.metric === sel.metric && c.variant_id === sel.variant_id);
    return cuts.sort((a, b) => b.lift - a.lift);
  }

  function renderForest() {
    const wrap = $("vz-forest");
    const cuts = forestCuts().filter(c => c.ci);
    const title = sel ? `${sel.metric} — ${sel.variant_id}` : "All selected cuts";
    wrap.innerHTML = `<div class="vz-title">${title}</div>`;
    if (!cuts.length) { wrap.innerHTML += "<p>No cuts to plot.</p>"; return; }
    const lo = Math.min(-0.1, ...cuts.map(c => c.ci[0]));
    const hi = Math.max(0.1, ...cuts.map(c => c.ci[1]));
    const scale = x => ((x - lo) / (hi - lo)) * 100;
    let html = "";
    for (const c of cuts) {
      const l = scale(c.ci[0]), r = scale(c.ci[1]), pt = scale(c.lift);
      const sig = c.significant && !c.underpowered_days && !hold;
      html += `<div class="frow">
        <div class="flabel">${c.slice_dim}: ${c.slice_value}${sel ? " · " + label(sel.variant_id) : ""}</div>
        <div class="fbar">
          <div class="fband ${sig ? "sig" : "ns"}" style="left:${l}%;width:${Math.max(0.5, r - l)}%"></div>
          <div class="fdot" style="left:${pt}%"></div>
          <div class="fzero" style="left:${scale(0)}%"></div>
        </div>
        <div class="fval ${sig ? "sigv" : ""}">${fmt(c.lift)}</div>
      </div>`;
    }
    wrap.innerHTML += html;
  }

  // ---------- SRM explainer ----------
  function renderSrm() {
    const s = DATA.srm || {};
    if (!s.detected && DATA.decision.verdict !== "SRM-HOLD" && DATA.decision.verdict !== "HOLD") {
      $("vz-srm").innerHTML = `<p class="ok">✓ SRM clean${s.p_value !== undefined && s.p_value !== null ? ` (p=${s.p_value.toFixed(3)})` : ""}</p>`;
      return;
    }
    const rows = Object.keys(s.observed || {}).map(k =>
      `<div>${k}: observed <b>${(s.observed[k] * 100).toFixed(1)}%</b> / intended <b>${((s.expected[k] || 0) * 100).toFixed(1)}%</b></div>`).join("");
    $("vz-srm").innerHTML = `<p class="alert">⚠ SRM detected — sample-ratio mismatch breaks experiment integrity (χ²=${(s.chi_square || 0).toFixed(0)}, p=${s.p_value === undefined || s.p_value === null || s.p_value === 0 ? "??" : s.p_value.toExponential(1)}). No color is meaningful above this line.</p>${rows}`;
  }

  // ---------- meta row ----------
  function renderMeta() {
    $("vz-meta").textContent = `experiment ${DATA.experiment_id} · decision: ${DATA.decision.verdict}` +
      (DATA.decision.winner ? ` (winner: ${DATA.decision.winner})` : "") +
      ` · readout sha256 ${DATA.content_hash}`;
    const ctlL = L.control || L.ctl;
    $("vz-ctl").textContent = ctlL
      ? `control: ${ctlL} (baseline; every cell shows a variant's lift vs it)`
      : "control: baseline; every cell shows a variant's lift vs it";
  }

  function render() {
    renderMeta();
    renderGrid();
    renderForest();
    renderSrm();
  }

  // dim toggles
  const tgl = $("vz-dims");
  tgl.innerHTML = dims.map(d =>
    `<button class="chip on" data-d="${d}">${d}</button>`).join("");
  tgl.querySelectorAll("button").forEach(b => {
    b.onclick = () => {
      if (visibleDims.has(b.dataset.d)) visibleDims.delete(b.dataset.d);
      else visibleDims.add(b.dataset.d);
      b.classList.toggle("on");
      sel = null;
      render();
    };
  });

  render();
}
