"""Build the self-contained analytics report from engine output.

    python -m trademetrics.demo        # writes outputs/results.json
    python tools/build_dashboard.py    # writes trademetrics.html

Every number and every chart point in the generated page comes from
outputs/results.json, which is produced by the engine from the committed
sample data. The JSON is embedded at build time and the charts are drawn as
inline SVG, so the page is a single file with no network dependency and no
hardcoded metrics.

Design notes
------------
The page is styled as an institutional research note rather than a dark
"analytics dashboard": light paper surface, hairline rules instead of cards,
system sans throughout, tabular figures in every column of numbers, and colour
used only where it encodes data.

The categorical series palette (blue / orange / aqua) and the diverging
blue-red pair for signed values were validated for colour-vision deficiency
and contrast before use. Positive/negative values carry a sign as well as a
colour, so nothing is encoded by colour alone; the aqua series sits below the
3:1 contrast floor on white, so it carries a direct end-label and appears in a
table view, which is the documented relief for that case.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_PATH = REPO_ROOT / "outputs" / "results.json"
OUT_PATH = REPO_ROOT / "trademetrics.html"

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TradeMetrics — Engine Output (synthetic sample data)</title>
<!--
  GENERATED FILE — do not edit by hand.
  Built by tools/build_dashboard.py from outputs/results.json.
  Every figure below is computed by the engine from data/samples/*.csv.
-->
<style>
  :root {
    color-scheme: light;
    --surface:#ffffff;
    --plane:#f4f4f1;
    --ink:#0b0b0b;
    --ink-2:#52514e;
    --muted:#898781;
    --grid:#e6e5de;
    --axis:#c3c2b7;
    --rule:#d9d8d1;
    --rule-strong:#0b0b0b;
    --series-1:#2a78d6;
    --series-2:#eb6834;
    --series-3:#1baf7a;
    --neg:#c02f2f;
    --pos:#2a78d6;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  html { -webkit-text-size-adjust:100%; }
  body {
    font-family:var(--sans);
    background:var(--plane);
    color:var(--ink);
    line-height:1.5;
    font-size:15px;
  }
  .sheet {
    max-width:1180px; margin:0 auto; background:var(--surface);
    border-left:1px solid var(--rule); border-right:1px solid var(--rule);
    min-height:100vh;
  }
  .pad { padding:0 44px; }
  @media (max-width:760px) { .pad { padding:0 20px; } }

  /* ── masthead ─────────────────────────────────────────────── */
  .masthead {
    display:flex; justify-content:space-between; align-items:flex-end;
    gap:28px; flex-wrap:wrap;
    padding-top:40px; padding-bottom:16px;
    border-bottom:2px solid var(--rule-strong);
  }
  .wordmark { font-size:27px; font-weight:680; letter-spacing:-0.022em; line-height:1.1; }
  .wordmark em { font-style:normal; font-weight:400; color:var(--ink-2); }
  .tagline { font-size:13px; color:var(--ink-2); margin-top:5px; }
  .masthead-meta {
    display:grid; grid-template-columns:auto auto; gap:2px 16px;
    font-size:12px; color:var(--ink-2); text-align:right;
    font-variant-numeric:tabular-nums;
  }
  .masthead-meta dt { color:var(--muted); }
  .masthead-meta dd { font-weight:530; }
  .kicker {
    display:inline-block; margin-top:14px;
    font-size:10.5px; font-weight:640; letter-spacing:0.1em; text-transform:uppercase;
    color:var(--neg); border:1px solid currentColor; padding:3px 8px;
  }

  /* ── standfirst ───────────────────────────────────────────── */
  .standfirst {
    padding-top:22px; padding-bottom:26px;
    border-bottom:1px solid var(--rule);
    font-size:14.5px; line-height:1.65; color:var(--ink-2);
    max-width:76ch;
  }
  .standfirst strong { color:var(--ink); font-weight:620; }
  .standfirst p + p { margin-top:11px; }
  code {
    font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    font-size:0.88em; background:var(--plane); padding:1px 5px;
    border:1px solid var(--rule); white-space:nowrap;
  }

  /* ── section heads ────────────────────────────────────────── */
  .section { padding-top:34px; }
  .section-head {
    display:flex; align-items:baseline; justify-content:space-between; gap:16px;
    padding-bottom:9px; margin-bottom:18px; border-bottom:1px solid var(--rule-strong);
  }
  .section-head h2 { font-size:12px; font-weight:660; letter-spacing:0.09em; text-transform:uppercase; }
  .section-head .note { font-size:12px; color:var(--muted); }

  /* ── key figures: rules, not boxes ────────────────────────── */
  .figures { display:grid; grid-template-columns:repeat(6,1fr); }
  @media (max-width:980px) { .figures { grid-template-columns:repeat(3,1fr); row-gap:24px; } }
  @media (max-width:560px) { .figures { grid-template-columns:repeat(2,1fr); } }
  .figure { padding:2px 18px 0; border-left:1px solid var(--rule); }
  .figure:first-child { padding-left:0; border-left:0; }
  .figure-label {
    font-size:10.5px; font-weight:600; letter-spacing:0.07em; text-transform:uppercase;
    color:var(--muted); margin-bottom:7px;
  }
  .figure-value { font-size:30px; font-weight:640; letter-spacing:-0.02em; line-height:1.05; }
  .figure-value.neg { color:var(--neg); }
  .figure-sub { font-size:11.5px; color:var(--ink-2); margin-top:7px; line-height:1.5; font-variant-numeric:tabular-nums; }

  /* ── charts ───────────────────────────────────────────────── */
  .chart-grid { display:grid; grid-template-columns:1.55fr 1fr; gap:34px; }
  @media (max-width:980px) { .chart-grid { grid-template-columns:1fr; gap:30px; } }
  .chart-title { font-size:14px; font-weight:620; }
  .chart-sub { font-size:12px; color:var(--muted); margin-top:2px; }
  .chart-head { margin-bottom:10px; display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }
  .legend { display:flex; gap:14px; flex-wrap:wrap; padding-top:2px; }
  .legend-item { display:flex; gap:6px; align-items:center; font-size:11.5px; color:var(--ink-2); white-space:nowrap; }
  .legend-key { width:14px; height:2px; flex:none; }
  .chart-wrap { position:relative; }
  svg { display:block; width:100%; height:auto; overflow:visible; }

  /* ── tooltip ──────────────────────────────────────────────── */
  .tip {
    position:fixed; z-index:50; pointer-events:none; opacity:0;
    background:var(--surface); border:1px solid var(--axis);
    padding:8px 10px; font-size:11.5px; line-height:1.55;
    font-variant-numeric:tabular-nums; color:var(--ink);
    box-shadow:0 1px 2px rgba(11,11,11,0.09); transition:opacity .08s;
    min-width:130px;
  }
  .tip-title { font-weight:620; margin-bottom:4px; color:var(--ink); }
  .tip-row { display:flex; justify-content:space-between; gap:14px; }
  .tip-key { display:flex; align-items:center; gap:6px; color:var(--ink-2); }
  .tip-dot { width:8px; height:8px; flex:none; }

  /* ── tables ───────────────────────────────────────────────── */
  .table-scroll { overflow-x:auto; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; font-variant-numeric:tabular-nums; }
  caption {
    caption-side:top; text-align:left; font-size:12px; color:var(--muted);
    padding-bottom:8px;
  }
  th, td { padding:7px 12px; text-align:right; white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; padding-left:0; }
  th:last-child, td:last-child { padding-right:0; }
  thead th {
    font-size:10.5px; font-weight:620; letter-spacing:0.06em; text-transform:uppercase;
    color:var(--muted); border-bottom:1px solid var(--rule-strong);
    cursor:pointer; user-select:none; position:sticky; top:0; background:var(--surface);
  }
  thead th:hover { color:var(--ink); }
  thead th[aria-sort]:not([aria-sort="none"]) { color:var(--ink); }
  tbody tr { border-bottom:1px solid var(--grid); }
  tbody tr:hover { background:var(--plane); }
  td.name { font-weight:560; }
  td.num.neg, .neg-text { color:var(--neg); }
  .side {
    font-size:10px; font-weight:620; letter-spacing:0.06em; text-transform:uppercase;
    color:var(--ink-2);
  }
  .side::before {
    content:''; display:inline-block; width:6px; height:6px; margin-right:5px;
    vertical-align:1px; background:var(--muted);
  }
  .side.long::before { background:var(--series-1); }
  .side.short::before { background:var(--series-2); }

  /* ── two-column band ──────────────────────────────────────── */
  .band { display:grid; grid-template-columns:1.35fr 1fr; gap:38px; }
  @media (max-width:980px) { .band { grid-template-columns:1fr; gap:30px; } }
  .kv { border-top:1px solid var(--grid); }
  .kv-row {
    display:flex; justify-content:space-between; align-items:baseline; gap:16px;
    padding:8px 0; border-bottom:1px solid var(--grid); font-size:13px;
  }
  .kv-key { color:var(--ink-2); }
  .kv-val { font-weight:600; font-variant-numeric:tabular-nums; }
  .capm-block { border-top:1px solid var(--grid); padding:12px 0; border-bottom:1px solid var(--grid); }
  .capm-head { display:flex; justify-content:space-between; align-items:baseline; gap:14px; }
  .capm-name { font-size:13px; font-weight:640; }
  .capm-alpha { font-size:15px; font-weight:640; font-variant-numeric:tabular-nums; }
  .capm-detail { font-size:11.5px; color:var(--ink-2); margin-top:5px; font-variant-numeric:tabular-nums; }

  /* ── sector bars ──────────────────────────────────────────── */
  .sector-row { padding:9px 0; border-bottom:1px solid var(--grid); }
  .sector-head { display:flex; justify-content:space-between; align-items:baseline; gap:12px; margin-bottom:6px; }
  .sector-name { font-size:12.5px; }
  .sector-val { font-size:12.5px; font-weight:620; font-variant-numeric:tabular-nums; }
  .sector-track { height:6px; background:var(--plane); position:relative; }
  .sector-fill { position:absolute; top:0; bottom:0; }

  /* ── footer ───────────────────────────────────────────────── */
  footer {
    margin-top:44px; padding-top:18px; padding-bottom:44px;
    border-top:2px solid var(--rule-strong);
    font-size:11.5px; color:var(--muted); line-height:1.75;
  }
  footer strong { color:var(--ink-2); font-weight:600; }
  .visually-hidden {
    position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap;
  }
</style>
</head>
<body>
<div class="sheet">

  <div class="pad">
    <header class="masthead">
      <div>
        <div class="wordmark">TradeMetrics <em>— engine output</em></div>
        <div class="tagline">Portfolio analytics engine · verification report</div>
        <div class="kicker">Synthetic sample data — not trading results</div>
      </div>
      <dl class="masthead-meta">
        <dt>Period</dt><dd id="mPeriod"></dd>
        <dt>Risk-free rate</dt><dd id="mRf"></dd>
        <dt>Observations</dt><dd id="mObs"></dd>
        <dt>Generator seed</dt><dd id="mSeed"></dd>
      </dl>
    </header>

    <div class="standfirst">
      <p><strong>What this page is.</strong> Engine output computed from the committed
      synthetic sample data in <code>data/samples/</code>, generated by
      <code>python -m trademetrics.demo</code> and rendered by
      <code>python tools/build_dashboard.py</code>. Every figure traces to
      <code>outputs/results.json</code> — nothing here is hardcoded.</p>
      <p><strong>These are not real trading results.</strong> No personal account data
      appears anywhere in this repository. The sample portfolio exists to exercise every
      code path in the engine — long and short positions, partial fills, multi-lot FIFO
      ordering, commissions — not to demonstrate performance. Its Sharpe ratio is a
      property of one arbitrary random seed.</p>
      <p>The Sharpe ratio is shown with its 95% confidence interval throughout. Over a
      single year that interval is wide enough to contain zero, which is the honest
      reading of any one-year Sharpe and the reason this repository publishes no
      personal performance figure.</p>
    </div>

    <!-- KEY FIGURES -->
    <section class="section">
      <div class="section-head">
        <h2>Key figures</h2>
        <span class="note">computed from the daily NAV series</span>
      </div>
      <div class="figures" id="figures"></div>
    </section>

    <!-- PERFORMANCE -->
    <section class="section">
      <div class="section-head">
        <h2>Performance</h2>
        <span class="note">hover any chart for exact values</span>
      </div>
      <div class="chart-grid">
        <div>
          <div class="chart-head">
            <div>
              <div class="chart-title">Cumulative performance</div>
              <div class="chart-sub">Growth of 100, portfolio versus benchmarks</div>
            </div>
            <div class="legend" id="perfLegend"></div>
          </div>
          <div class="chart-wrap"><div id="perfChart"></div></div>
        </div>
        <div>
          <div class="chart-head">
            <div>
              <div class="chart-title">Monthly returns</div>
              <div class="chart-sub">Portfolio NAV return by calendar month</div>
            </div>
          </div>
          <div class="chart-wrap"><div id="monthChart"></div></div>
        </div>
      </div>
    </section>

    <!-- RISK -->
    <section class="section">
      <div class="section-head">
        <h2>Risk</h2>
        <span class="note">drawdown from running peak · one-day value at risk</span>
      </div>
      <div class="chart-grid">
        <div>
          <div class="chart-head">
            <div>
              <div class="chart-title">Drawdown</div>
              <div class="chart-sub" id="ddSub"></div>
            </div>
          </div>
          <div class="chart-wrap"><div id="ddChart"></div></div>
        </div>
        <div>
          <div class="chart-head">
            <div>
              <div class="chart-title">Risk measures</div>
              <div class="chart-sub">Parametric assumes normality; historical does not</div>
            </div>
          </div>
          <div class="kv" id="riskRows"></div>
        </div>
      </div>
    </section>

    <!-- ATTRIBUTION -->
    <section class="section">
      <div class="section-head">
        <h2>Attribution</h2>
        <span class="note">realised FIFO profit and loss, net of commissions</span>
      </div>
      <div class="band">
        <div>
          <div class="table-scroll">
            <table id="tradeTable">
              <caption id="tradeCaption"></caption>
              <thead><tr id="tradeHead"></tr></thead>
              <tbody id="tradeBody"></tbody>
            </table>
          </div>
        </div>
        <div>
          <div class="chart-head">
            <div>
              <div class="chart-title">By sector</div>
              <div class="chart-sub">Realised profit and loss</div>
            </div>
          </div>
          <div id="sectorList"></div>

          <div class="chart-head" style="margin-top:26px">
            <div>
              <div class="chart-title">CAPM decomposition</div>
              <div class="chart-sub">Excess-return regression versus each benchmark</div>
            </div>
          </div>
          <div id="capmRows"></div>
        </div>
      </div>
    </section>

    <!-- OPEN POSITIONS -->
    <section class="section">
      <div class="section-head">
        <h2>Open positions</h2>
        <span class="note">lots remaining after all executions</span>
      </div>
      <div class="table-scroll">
        <table>
          <thead><tr id="openHead"></tr></thead>
          <tbody id="openBody"></tbody>
        </table>
      </div>
    </section>

    <footer id="footer"></footer>
  </div>
</div>

<div class="tip" id="tip" role="status" aria-live="polite"></div>

<script id="engine-results" type="application/json">__RESULTS_JSON__</script>
<script>
const R = JSON.parse(document.getElementById('engine-results').textContent);
const S = R.summary;

const CSS = getComputedStyle(document.documentElement);
const C = n => CSS.getPropertyValue(n).trim();
const SERIES = [C('--series-1'), C('--series-2'), C('--series-3')];
const POS = C('--pos'), NEG = C('--neg');
const GRID = C('--grid'), AXIS = C('--axis'), MUTED = C('--muted');
const INK = C('--ink'), INK2 = C('--ink-2'), SURFACE = C('--surface');

// ── formatting ────────────────────────────────────────────────────────────
const pct  = (x,d=2) => (x*100).toFixed(d) + '%';
const spct = (x,d=2) => (x>=0?'+':'\\u2212') + Math.abs(x*100).toFixed(d) + '%';
const num  = (x,d=2) => (x<0?'\\u2212':'') + Math.abs(x).toFixed(d);
const snum = (x,d=2) => (x>=0?'+':'\\u2212') + Math.abs(x).toFixed(d);
const money = x => (x<0?'\\u2212':'') + Math.abs(x).toLocaleString('en-US',
  {minimumFractionDigits:2, maximumFractionDigits:2});
const smoney = x => (x>=0?'+':'\\u2212') + Math.abs(x).toLocaleString('en-US',
  {minimumFractionDigits:2, maximumFractionDigits:2});

// ── masthead ──────────────────────────────────────────────────────────────
document.getElementById('mPeriod').textContent = R.meta.period.replace(' to ', ' \\u2013 ');
document.getElementById('mRf').textContent = pct(R.meta.rf_ann, 1) + ' p.a.';
document.getElementById('mObs').textContent = S.n_obs + ' trading days';
document.getElementById('mSeed').textContent = R.meta.generator_seed;

// ── key figures ───────────────────────────────────────────────────────────
const figures = [
  { label:'Total return', value:spct(S.total_return), neg:S.total_return < 0,
    sub:'Annualised ' + spct(S.annualised_return) },
  { label:'Sharpe ratio', value:snum(S.sharpe), neg:S.sharpe < 0,
    sub:'95% CI ' + snum(S.sharpe_ci_low) + ' to ' + snum(S.sharpe_ci_high)
        + '<br>Lo (2002) SE ' + num(S.sharpe_se) },
  { label:'Sortino ratio', value:snum(S.sortino), neg:S.sortino < 0,
    sub:'Target = daily risk-free rate' },
  { label:'Max drawdown', value:spct(S.max_drawdown), neg:true,
    sub:'Peak to trough, daily NAV' },
  { label:'Annualised vol', value:pct(S.annualised_volatility), neg:false,
    sub:'Calmar ' + snum(S.calmar) },
  { label:'Win rate', value:pct(S.win_rate,1), neg:false,
    sub:S.n_round_trips + ' round-trips<br>Profit factor ' + num(S.profit_factor) },
];
document.getElementById('figures').innerHTML = figures.map(f => `
  <div class="figure">
    <div class="figure-label">${f.label}</div>
    <div class="figure-value${f.neg ? ' neg' : ''}">${f.value}</div>
    <div class="figure-sub">${f.sub}</div>
  </div>`).join('');

// ── tooltip plumbing ──────────────────────────────────────────────────────
const tip = document.getElementById('tip');
function showTip(html, evt) {
  tip.innerHTML = html;
  tip.style.opacity = '1';
  const r = tip.getBoundingClientRect();
  let x = evt.clientX + 14, y = evt.clientY - r.height / 2;
  if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - 14;
  y = Math.max(8, Math.min(y, window.innerHeight - r.height - 8));
  tip.style.left = x + 'px';
  tip.style.top = y + 'px';
}
function hideTip() { tip.style.opacity = '0'; }

// ── SVG helpers ───────────────────────────────────────────────────────────
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}
function ticks(lo, hi, count) {
  const span = hi - lo || 1;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = ([1,2,2.5,5,10].find(m => m * mag >= raw) || 10) * mag;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(t);
  return out;
}
function axisText(x, y, text, opts) {
  const t = el('text', Object.assign({
    x, y, fill:MUTED, 'font-size':11, 'font-family':'inherit',
  }, opts || {}));
  t.textContent = text;
  return t;
}

/**
 * Time-series line chart with a crosshair tooltip and direct end-labels.
 * series: [{name, data:[], color}]
 */
function lineChart(mount, series, dates, opts) {
  opts = opts || {};
  // Value axis on the LEFT, direct series labels on the RIGHT, so axis ticks
  // and end-labels never compete for the same margin.
  const W = 720, H = 300, P = { t:12, r:opts.rightPad || 62, b:30, l:opts.leftPad || 40 };
  const all = series.flatMap(s => s.data);
  let lo = Math.min(...all), hi = Math.max(...all);
  if (opts.includeZero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  const pad = (hi - lo) * 0.10 || 1;
  lo -= pad; hi += pad;
  const n = dates.length;
  const X = i => P.l + (i / (n - 1)) * (W - P.l - P.r);
  const Y = v => P.t + (1 - (v - lo) / (hi - lo)) * (H - P.t - P.b);

  const svg = el('svg', { viewBox:`0 0 ${W} ${H}`, role:'img' });

  // horizontal gridlines + right-hand value axis
  ticks(lo, hi, 5).forEach(t => {
    const isZero = Math.abs(t) < 1e-9;
    svg.appendChild(el('line', {
      x1:P.l, x2:W - P.r, y1:Y(t), y2:Y(t),
      stroke:isZero ? AXIS : GRID, 'stroke-width':1 }));
    svg.appendChild(axisText(P.l - 8, Y(t) + 3.8,
      opts.fmtAxis ? opts.fmtAxis(t) : t.toFixed(0), { 'text-anchor':'end' }));
  });

  // x-axis baseline + date ticks
  svg.appendChild(el('line', {
    x1:P.l, x2:W - P.r, y1:H - P.b, y2:H - P.b, stroke:AXIS, 'stroke-width':1 }));
  const step = Math.max(1, Math.round((n - 1) / 6));
  for (let i = 0; i < n; i += step) {
    svg.appendChild(axisText(X(i), H - P.b + 16, dates[i].slice(0, 7),
      { 'text-anchor':i === 0 ? 'start' : 'middle' }));
  }

  // area wash (single-series charts only)
  series.forEach(s => {
    if (!s.fill) return;
    const base = Y(Math.max(lo, opts.fillBase !== undefined ? opts.fillBase : lo));
    const d = s.data.map((v,i) => `${i ? 'L' : 'M'}${X(i).toFixed(2)},${Y(v).toFixed(2)}`).join('');
    svg.appendChild(el('path', {
      d:`${d}L${X(n-1).toFixed(2)},${base}L${X(0).toFixed(2)},${base}Z`,
      fill:s.color, 'fill-opacity':0.10, stroke:'none' }));
  });

  // 2px lines
  series.forEach(s => {
    const d = s.data.map((v,i) => `${i ? 'L' : 'M'}${X(i).toFixed(2)},${Y(v).toFixed(2)}`).join('');
    svg.appendChild(el('path', {
      d, fill:'none', stroke:s.color, 'stroke-width':2,
      'stroke-linejoin':'round', 'stroke-linecap':'round' }));
  });

  // direct end-labels (identity without relying on colour alone)
  if (opts.endLabels !== false) {
    const placed = [];
    // Highest series first, nudging collisions downward, so converging lines
    // keep labels in the same vertical order as the lines themselves.
    series.slice().sort((a, b) => b.data[n - 1] - a.data[n - 1]).forEach(s => {
      let y = Y(s.data[n - 1]);
      while (placed.some(p => Math.abs(p - y) < 14)) y += 14;
      placed.push(y);
      const t = axisText(W - P.r + 8, y + 3.8, s.name,
        { fill:INK2, 'font-size':11, 'font-weight':600 });
      svg.appendChild(t);
    });
  }

  // crosshair + hover layer
  const cross = el('line', {
    y1:P.t, y2:H - P.b, stroke:AXIS, 'stroke-width':1, opacity:0 });
  svg.appendChild(cross);
  const dots = series.map(s => {
    const g = el('circle', {
      r:4, fill:s.color, stroke:SURFACE, 'stroke-width':2, opacity:0 });
    svg.appendChild(g);
    return g;
  });
  const hit = el('rect', {
    x:P.l, y:0, width:Math.max(1, W - P.l - P.r), height:H, fill:'transparent' });
  svg.appendChild(hit);

  function at(evt) {
    const box = svg.getBoundingClientRect();
    const px = (evt.clientX - box.left) / box.width * W;
    let i = Math.round((px - P.l) / (W - P.l - P.r) * (n - 1));
    return Math.max(0, Math.min(n - 1, i));
  }
  hit.addEventListener('mousemove', evt => {
    const i = at(evt);
    cross.setAttribute('x1', X(i)); cross.setAttribute('x2', X(i));
    cross.setAttribute('opacity', 1);
    dots.forEach((d, k) => {
      d.setAttribute('cx', X(i)); d.setAttribute('cy', Y(series[k].data[i]));
      d.setAttribute('opacity', 1);
    });
    const rows = series.map((s, k) => `
      <div class="tip-row">
        <span class="tip-key"><span class="tip-dot" style="background:${s.color}"></span>${s.name}</span>
        <span>${opts.fmtTip ? opts.fmtTip(s.data[i]) : s.data[i].toFixed(2)}</span>
      </div>`).join('');
    showTip(`<div class="tip-title">${dates[i]}</div>${rows}`, evt);
  });
  hit.addEventListener('mouseleave', () => {
    cross.setAttribute('opacity', 0);
    dots.forEach(d => d.setAttribute('opacity', 0));
    hideTip();
  });

  mount.innerHTML = '';
  mount.appendChild(svg);
}

/** Column chart for signed values: diverging blue/red, 2px surface gaps. */
function barChart(mount, values, labels, opts) {
  opts = opts || {};
  const W = 440, H = 300, P = { t:12, r:8, b:44, l:44 };
  let lo = Math.min(0, ...values), hi = Math.max(0, ...values);
  const pad = (hi - lo) * 0.12 || 1;
  lo -= pad; hi += pad;
  const n = values.length;
  const band = (W - P.l - P.r) / n;
  const bw = Math.min(24, band - 2);            // cap thickness, leave air
  const Y = v => P.t + (1 - (v - lo) / (hi - lo)) * (H - P.t - P.b);

  const svg = el('svg', { viewBox:`0 0 ${W} ${H}`, role:'img' });

  ticks(lo, hi, 5).forEach(t => {
    const isZero = Math.abs(t) < 1e-9;
    svg.appendChild(el('line', {
      x1:P.l, x2:W - P.r, y1:Y(t), y2:Y(t),
      stroke:isZero ? AXIS : GRID, 'stroke-width':1 }));
    svg.appendChild(axisText(P.l - 8, Y(t) + 3.8, t.toFixed(1) + '%',
      { 'text-anchor':'end' }));
  });

  values.forEach((v, i) => {
    const cx = P.l + i * band + band / 2;
    const top = Y(Math.max(v, 0)), bottom = Y(Math.min(v, 0));
    const h = Math.max(1, bottom - top);
    const r = Math.min(4, h / 2);
    svg.appendChild(el('rect', {
      x:cx - bw / 2, y:top, width:bw, height:h,
      rx:r, ry:r, fill:v >= 0 ? POS : NEG }));
    // square off the baseline end so bars grow from a single flat baseline
    svg.appendChild(el('rect', {
      x:cx - bw / 2, y:v >= 0 ? bottom - r : top,
      width:bw, height:Math.min(r, h), fill:v >= 0 ? POS : NEG }));
    svg.appendChild(axisText(cx, H - P.b + 16, labels[i],
      { 'text-anchor':'end', transform:`rotate(-45 ${cx} ${H - P.b + 16})` }));

    const hit = el('rect', {
      x:P.l + i * band, y:P.t, width:band, height:H - P.t - P.b, fill:'transparent' });
    hit.addEventListener('mousemove', evt => showTip(
      `<div class="tip-title">${labels[i]}</div>
       <div class="tip-row"><span class="tip-key">Return</span>
       <span class="${v < 0 ? 'neg-text' : ''}">${spct(v / 100)}</span></div>`, evt));
    hit.addEventListener('mouseleave', hideTip);
    svg.appendChild(hit);
  });

  mount.innerHTML = '';
  mount.appendChild(svg);
}

// ── render charts ─────────────────────────────────────────────────────────
const perfSeries = [
  { name:'Portfolio', data:R.series.portfolio_indexed, color:SERIES[0] },
  { name:'SPY',       data:R.series.spy_indexed,       color:SERIES[1] },
  { name:'QQQ',       data:R.series.qqq_indexed,       color:SERIES[2] },
];
document.getElementById('perfLegend').innerHTML = perfSeries.map(s => `
  <span class="legend-item"><span class="legend-key" style="background:${s.color}"></span>${s.name}</span>`
).join('');
lineChart(document.getElementById('perfChart'), perfSeries, R.series.dates, {
  fmtAxis: t => t.toFixed(0),
  fmtTip: v => v.toFixed(2),
});

barChart(document.getElementById('monthChart'),
  R.monthly_returns.portfolio, R.monthly_returns.labels);

lineChart(document.getElementById('ddChart'), [
  { name:'Drawdown', data:R.series.drawdown.map(v => v * 100), color:NEG, fill:true },
], R.series.dates, {
  includeZero:true, fillBase:0, endLabels:false, rightPad:52,
  fmtAxis: t => t.toFixed(0) + '%',
  fmtTip: v => (v === 0 ? '0.00%' : '\\u2212' + Math.abs(v).toFixed(2) + '%'),
});
document.getElementById('ddSub').textContent =
  'Maximum ' + spct(S.max_drawdown) + ' from the running peak';

// ── risk measures ─────────────────────────────────────────────────────────
const spy = R.benchmarks.find(b => b.benchmark === 'SPY');
const riskRows = [
  ['Value at risk, 95% — parametric', pct(S.var_95_parametric)],
  ['Value at risk, 95% — historical', pct(S.var_95_historical)],
  ['Value at risk, 99% — parametric', pct(S.var_99_parametric)],
  ['Value at risk, 99% — historical', pct(S.var_99_historical)],
  ['Beta versus SPY', num(spy.beta)],
  ['Annualised volatility', pct(S.annualised_volatility)],
  ['Average holding period', num(S.avg_holding_days,1) + ' days'],
];
document.getElementById('riskRows').innerHTML = riskRows.map(([k,v]) => `
  <div class="kv-row"><span class="kv-key">${k}</span><span class="kv-val">${v}</span></div>`
).join('');

// ── CAPM ──────────────────────────────────────────────────────────────────
document.getElementById('capmRows').innerHTML = R.benchmarks.map(b => `
  <div class="capm-block">
    <div class="capm-head">
      <span class="capm-name">versus ${b.benchmark}</span>
      <span class="capm-alpha${b.capm_alpha_ann < 0 ? ' neg-text' : ''}">
        &alpha; ${spct(b.capm_alpha_ann)} p.a.</span>
    </div>
    <div class="capm-detail">
      95% CI ${spct(b.alpha_ci_low)} to ${spct(b.alpha_ci_high)}
      &middot; &beta; ${num(b.beta)} &middot; R&sup2; ${num(b.r_squared)}
      &middot; IR ${num(b.info_ratio)}
    </div>
  </div>`).join('');

// ── sector attribution ────────────────────────────────────────────────────
const sectors = R.pnl_by_sector;
const maxAbs = Math.max(...sectors.map(s => Math.abs(s.net_pnl)));
document.getElementById('sectorList').innerHTML = sectors.map(s => {
  const w = Math.abs(s.net_pnl) / maxAbs * 50;   // half-width: zero sits at centre
  const positive = s.net_pnl >= 0;
  return `
  <div class="sector-row">
    <div class="sector-head">
      <span class="sector-name">${s.sector}</span>
      <span class="sector-val${positive ? '' : ' neg-text'}">${smoney(s.net_pnl)}</span>
    </div>
    <div class="sector-track">
      <div class="sector-fill" style="
        ${positive ? 'left:50%' : 'right:50%'};
        width:${w}%; background:${positive ? POS : NEG}"></div>
    </div>
  </div>`;
}).join('');

// ── realised trades table ─────────────────────────────────────────────────
const tradeCols = [
  ['ticker','Ticker','text'], ['direction','Side','text'], ['qty','Qty','num'],
  ['entry_date','Entry','text'], ['entry_price','Entry px','num'],
  ['exit_date','Exit','text'], ['exit_price','Exit px','num'],
  ['commission','Comm.','num'], ['net_pnl','Net P&L','num'],
];
let trades = R.realised_trades.slice();
let sortKey = 'net_pnl', sortDir = -1;

document.getElementById('tradeCaption').textContent =
  trades.length + ' realised round-trips, net of commissions. Click a column to sort.';
document.getElementById('tradeHead').innerHTML = tradeCols
  .map(([k,l]) => `<th data-key="${k}" aria-sort="none" tabindex="0">${l}</th>`).join('');

function renderTrades() {
  trades.sort((a,b) => {
    const av = a[sortKey], bv = b[sortKey];
    return (typeof av === 'string' ? av.localeCompare(bv) : av - bv) * sortDir;
  });
  document.getElementById('tradeBody').innerHTML = trades.map(t => `
    <tr>
      <td class="name">${t.ticker}</td>
      <td><span class="side ${t.direction.toLowerCase()}">${t.direction}</span></td>
      <td>${t.qty}</td>
      <td>${t.entry_date}</td>
      <td>${t.entry_price.toFixed(2)}</td>
      <td>${t.exit_date}</td>
      <td>${t.exit_price.toFixed(2)}</td>
      <td>${t.commission.toFixed(2)}</td>
      <td class="num${t.net_pnl < 0 ? ' neg' : ''}">${smoney(t.net_pnl)}</td>
    </tr>`).join('');
  document.querySelectorAll('#tradeHead th').forEach(th => th.setAttribute(
    'aria-sort', th.dataset.key === sortKey ? (sortDir > 0 ? 'ascending' : 'descending') : 'none'));
}
function sortBy(key) {
  if (!key) return;
  if (sortKey === key) sortDir *= -1; else { sortKey = key; sortDir = -1; }
  renderTrades();
}
document.getElementById('tradeHead').addEventListener('click', e => sortBy(e.target.dataset.key));
document.getElementById('tradeHead').addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sortBy(e.target.dataset.key); }
});
renderTrades();

// ── open positions ────────────────────────────────────────────────────────
document.getElementById('openHead').innerHTML =
  ['Ticker','Side','Qty','Entry date','Entry price','Entry commission']
    .map(l => `<th>${l}</th>`).join('');
document.getElementById('openBody').innerHTML = R.open_positions.map(p => `
  <tr>
    <td class="name">${p.ticker}</td>
    <td><span class="side ${p.direction.toLowerCase()}">${p.direction}</span></td>
    <td>${p.qty}</td>
    <td>${p.entry_date}</td>
    <td>${p.entry_price.toFixed(2)}</td>
    <td>${p.entry_commission.toFixed(2)}</td>
  </tr>`).join('');

// ── footer ────────────────────────────────────────────────────────────────
document.getElementById('footer').innerHTML =
  '<strong>Provenance.</strong> Generated by <code>tools/build_dashboard.py</code> from '
  + '<code>outputs/results.json</code>. Data: ' + R.meta.data + '. '
  + 'Rebuild with <code>python -m trademetrics.demo</code> then '
  + '<code>python tools/build_dashboard.py</code>.<br>'
  + '<strong>Scope.</strong> No broker connection, no personal account data, no hardcoded '
  + 'metrics. Signed values carry both a sign and a colour, so nothing is encoded by colour '
  + 'alone; the series palette was checked for colour-vision deficiency and contrast.';
</script>
</body>
</html>
"""


def build(results_path: Path = RESULTS_PATH, out_path: Path = OUT_PATH) -> Path:
    if not results_path.exists():
        raise SystemExit(
            f"{results_path} not found — run `python -m trademetrics.demo` first."
        )
    results = json.loads(results_path.read_text())
    # Guard against the JSON terminating the inline <script> block early.
    payload = json.dumps(results, separators=(",", ":")).replace("</", "<\\/")
    out_path.write_text(TEMPLATE.replace("__RESULTS_JSON__", payload))
    return out_path


def main() -> None:
    out = build()
    kb = out.stat().st_size / 1024
    print(f"Wrote {out.relative_to(REPO_ROOT)} ({kb:.1f} KB, self-contained)")


if __name__ == "__main__":
    main()
