"""Build the self-contained dashboard from engine output.

    python -m trademetrics.demo        # writes outputs/results.json
    python tools/build_dashboard.py    # writes trademetrics.html

Every number and every chart point in the generated page comes from
outputs/results.json, which is produced by the engine from the committed
sample data. The JSON is embedded at build time and the charts are drawn as
inline SVG, so the page is a single file with no network dependency and no
hardcoded metrics.
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
    --bg:#050810; --bg2:#090d1a; --surface:#111827; --border:#1e2d45;
    --border2:#253550; --accent:#00d4aa; --accent2:#0099ff; --accent3:#ff6b35;
    --red:#ff4757; --green:#2ed573; --text:#e8edf5; --text2:#8fa3c0; --text3:#4a6080;
    --gold:#ffd700;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
    background:var(--bg); color:var(--text); min-height:100vh;
  }
  header {
    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;
    padding:18px 32px; border-bottom:1px solid var(--border); background:var(--bg2);
  }
  .logo-text { font-size:20px; font-weight:800; letter-spacing:-0.5px; }
  .logo-text span { color:var(--accent); }
  .logo-sub { font-size:10px; color:var(--text3); letter-spacing:2px; text-transform:uppercase; }
  .data-badge {
    padding:5px 12px; border-radius:4px; border:1px solid var(--accent3);
    background:rgba(255,107,53,0.1); font-size:10px; color:var(--accent3); letter-spacing:1.5px;
  }
  .meta-line { font-size:11px; color:var(--text3); margin-top:4px; }
  main { max-width:1500px; margin:0 auto; padding:24px 32px; display:flex; flex-direction:column; gap:20px; }
  .notice {
    padding:14px 18px; border-radius:8px; border:1px solid var(--border2);
    background:var(--surface); font-size:12px; line-height:1.7; color:var(--text2);
  }
  .notice strong { color:var(--text); }
  .kpi-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }
  .kpi-card {
    background:var(--surface); border:1px solid var(--border); border-radius:8px;
    padding:16px 18px; position:relative; overflow:hidden;
  }
  .kpi-card::after { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
  .kpi-card.green::after  { background:linear-gradient(90deg,var(--green),transparent); }
  .kpi-card.red::after    { background:linear-gradient(90deg,var(--red),transparent); }
  .kpi-card.blue::after   { background:linear-gradient(90deg,var(--accent2),transparent); }
  .kpi-card.gold::after   { background:linear-gradient(90deg,var(--gold),transparent); }
  .kpi-card.teal::after   { background:linear-gradient(90deg,var(--accent),transparent); }
  .kpi-card.orange::after { background:linear-gradient(90deg,var(--accent3),transparent); }
  .kpi-label { font-size:9px; color:var(--text3); letter-spacing:2px; text-transform:uppercase; margin-bottom:8px; }
  .kpi-value { font-size:22px; font-weight:700; line-height:1.1; }
  .kpi-value.up { color:var(--green); } .kpi-value.down { color:var(--red); }
  .kpi-value.blue { color:var(--accent2); } .kpi-value.gold { color:var(--gold); }
  .kpi-value.orange { color:var(--accent3); }
  .kpi-sub { font-size:10px; color:var(--text3); margin-top:6px; line-height:1.5; }
  .grid-2 { display:grid; grid-template-columns:2fr 1fr; gap:16px; }
  .grid-2b { display:grid; grid-template-columns:1.4fr 1fr; gap:16px; }
  @media (max-width:1100px) { .grid-2, .grid-2b { grid-template-columns:1fr; } }
  .card { background:var(--surface); border:1px solid var(--border); border-radius:8px; overflow:hidden; }
  .card-header {
    display:flex; align-items:center; justify-content:space-between; gap:12px;
    padding:14px 18px; border-bottom:1px solid var(--border); background:var(--bg2);
  }
  .card-title { font-size:13px; font-weight:700; }
  .card-title span { color:var(--text3); font-size:11px; font-weight:400; margin-left:8px; }
  .card-badge { font-size:9px; padding:3px 8px; border-radius:3px; letter-spacing:1px;
    text-transform:uppercase; background:rgba(0,212,170,0.15); color:var(--accent);
    border:1px solid rgba(0,212,170,0.3); white-space:nowrap; }
  .chart-body { padding:12px 8px 4px; }
  svg { display:block; width:100%; height:auto; }
  .legend { display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  .leg-item { display:flex; gap:6px; align-items:center; font-size:10px; color:var(--text3); }
  .leg-line { width:18px; height:2px; border-radius:1px; }
  .table-wrap { overflow:auto; max-height:420px; }
  table { width:100%; border-collapse:collapse; font-size:11px; }
  thead th {
    position:sticky; top:0; z-index:2; padding:10px 14px; text-align:left;
    font-size:9px; letter-spacing:1.5px; text-transform:uppercase; color:var(--text3);
    background:var(--bg2); border-bottom:1px solid var(--border); white-space:nowrap;
    cursor:pointer; user-select:none;
  }
  thead th:hover, thead th.sorted { color:var(--accent); }
  tbody tr { border-bottom:1px solid rgba(30,45,69,0.5); }
  tbody tr:hover { background:rgba(0,212,170,0.04); }
  td { padding:9px 14px; white-space:nowrap; }
  .ticker { color:var(--accent); font-weight:600; }
  .tag { font-size:9px; padding:2px 7px; border-radius:3px; letter-spacing:1px; }
  .tag.long  { background:rgba(46,213,115,0.1); color:var(--green); border:1px solid rgba(46,213,115,0.2); }
  .tag.short { background:rgba(255,71,87,0.1); color:var(--red); border:1px solid rgba(255,71,87,0.2); }
  .pos { color:var(--green); } .neg { color:var(--red); }
  .rows { padding:16px; display:flex; flex-direction:column; gap:10px; }
  .row {
    display:flex; align-items:center; justify-content:space-between; gap:10px;
    padding:10px 14px; background:var(--bg2); border:1px solid var(--border); border-radius:6px;
  }
  .row-label { font-size:10px; color:var(--text3); letter-spacing:1px; text-transform:uppercase; }
  .row-val { font-size:13px; font-weight:600; }
  .attrib { padding:16px; display:flex; flex-direction:column; gap:12px; }
  .attrib-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:5px; }
  .attrib-name { font-size:11px; color:var(--text2); }
  .attrib-val { font-size:12px; font-weight:600; }
  .bar-bg { height:6px; background:var(--border); border-radius:3px; overflow:hidden; }
  .bar { height:100%; border-radius:3px; }
  footer {
    padding:16px 32px; border-top:1px solid var(--border); background:var(--bg2);
    font-size:10px; color:var(--text3); line-height:1.8;
  }
  code { background:var(--bg); padding:1px 5px; border-radius:3px; color:var(--text2); }
</style>
</head>
<body>

<header>
  <div>
    <div class="logo-text">Trade<span>Metrics</span></div>
    <div class="logo-sub">Portfolio Analytics Engine</div>
  </div>
  <div style="text-align:right">
    <div class="data-badge">SYNTHETIC SAMPLE DATA</div>
    <div class="meta-line" id="metaLine"></div>
  </div>
</header>

<main>

  <div class="notice">
    <strong>What this page is.</strong> Engine output computed from the committed synthetic
    sample data in <code>data/samples/</code>, generated by
    <code>python -m trademetrics.demo</code> and rendered by
    <code>python tools/build_dashboard.py</code>. Every figure traces to
    <code>outputs/results.json</code> — nothing on this page is hardcoded.
    <strong>These are not real trading results</strong>, and no personal account data
    is present anywhere in this repository. The purpose of the sample portfolio is to
    exercise the engine, not to demonstrate performance. Sharpe is shown with its
    95% confidence interval because, over a single simulated year, the interval is wide
    enough to contain zero — which is the point.
  </div>

  <div class="kpi-row" id="kpiRow"></div>

  <div class="grid-2">
    <div class="card">
      <div class="card-header">
        <div class="card-title">Cumulative Performance <span>growth of 100</span></div>
        <div class="legend">
          <div class="leg-item"><div class="leg-line" style="background:var(--accent)"></div>Portfolio</div>
          <div class="leg-item"><div class="leg-line" style="background:var(--accent2)"></div>SPY</div>
          <div class="leg-item"><div class="leg-line" style="background:var(--accent3)"></div>QQQ</div>
        </div>
      </div>
      <div class="chart-body"><div id="perfChart"></div></div>
    </div>
    <div class="card">
      <div class="card-header">
        <div class="card-title">Monthly Returns</div>
        <div class="card-badge" id="monthBadge"></div>
      </div>
      <div class="chart-body"><div id="monthChart"></div></div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-header">
        <div class="card-title">Drawdown <span>from running peak</span></div>
        <div class="card-badge" id="ddBadge"></div>
      </div>
      <div class="chart-body"><div id="ddChart"></div></div>
    </div>
    <div class="card">
      <div class="card-header">
        <div class="card-title">Risk Metrics</div>
        <div class="card-badge">one-day VaR</div>
      </div>
      <div class="rows" id="riskRows"></div>
    </div>
  </div>

  <div class="grid-2b">
    <div class="card">
      <div class="card-header">
        <div class="card-title">Realised FIFO P&amp;L <span>net of commissions</span></div>
        <div class="card-badge" id="tradeBadge"></div>
      </div>
      <div class="table-wrap">
        <table id="tradeTable">
          <thead><tr id="tradeHead"></tr></thead>
          <tbody id="tradeBody"></tbody>
        </table>
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:16px;">
      <div class="card">
        <div class="card-header">
          <div class="card-title">P&amp;L by Sector <span>realised</span></div>
        </div>
        <div class="attrib" id="attribList"></div>
      </div>
      <div class="card">
        <div class="card-header">
          <div class="card-title">CAPM Decomposition <span>excess returns</span></div>
        </div>
        <div class="rows" id="capmRows"></div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-header">
      <div class="card-title">Open Positions <span>at period end</span></div>
      <div class="card-badge" id="openBadge"></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr id="openHead"></tr></thead>
        <tbody id="openBody"></tbody>
      </table>
    </div>
  </div>

</main>

<footer id="footer"></footer>

<script id="engine-results" type="application/json">__RESULTS_JSON__</script>
<script>
const R = JSON.parse(document.getElementById('engine-results').textContent);

// ── formatting helpers ────────────────────────────────────────────────────
const pct  = (x, d=2) => (x*100).toFixed(d) + '%';
const spct = (x, d=2) => (x>=0?'+':'') + (x*100).toFixed(d) + '%';
const num  = (x, d=2) => x.toFixed(d);
const money = x => (x<0?'-':'') + Math.abs(x).toLocaleString('en-US',
  {minimumFractionDigits:2, maximumFractionDigits:2});

const S = R.summary;

// ── header meta ───────────────────────────────────────────────────────────
document.getElementById('metaLine').textContent =
  R.meta.period + '  ·  rf ' + pct(R.meta.rf_ann,1) + '  ·  seed ' + R.meta.generator_seed;

// ── KPI cards ─────────────────────────────────────────────────────────────
const kpis = [
  { cls:'green', label:'Total Return', value:spct(S.total_return),
    valueCls:S.total_return>=0?'up':'down',
    sub:'annualised ' + spct(S.annualised_return) },
  { cls:'blue', label:'Sharpe Ratio', value:num(S.sharpe), valueCls:'blue',
    sub:'95% CI ' + num(S.sharpe_ci_low) + ' to ' + num(S.sharpe_ci_high)
        + '<br>Lo SE ' + num(S.sharpe_se) + ' · n=' + S.n_obs },
  { cls:'orange', label:'Sortino Ratio', value:num(S.sortino), valueCls:'orange',
    sub:'target = daily risk-free rate' },
  { cls:'red', label:'Max Drawdown', value:spct(S.max_drawdown), valueCls:'down',
    sub:'peak to trough, daily NAV' },
  { cls:'teal', label:'Annualised Vol', value:pct(S.annualised_volatility), valueCls:'',
    sub:'Calmar ' + num(S.calmar) },
  { cls:'gold', label:'Win Rate', value:pct(S.win_rate,1), valueCls:'gold',
    sub:S.n_round_trips + ' round-trips · profit factor ' + num(S.profit_factor) },
];
document.getElementById('kpiRow').innerHTML = kpis.map(k => `
  <div class="kpi-card ${k.cls}">
    <div class="kpi-label">${k.label}</div>
    <div class="kpi-value ${k.valueCls}">${k.value}</div>
    <div class="kpi-sub">${k.sub}</div>
  </div>`).join('');

// ── minimal SVG chart helpers (no external library) ───────────────────────
const SVG_NS = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}
function niceTicks(lo, hi, count) {
  const span = hi - lo || 1;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1,2,2.5,5,10].find(m => m*mag >= raw) * mag;
  const out = [];
  for (let t = Math.ceil(lo/step)*step; t <= hi + 1e-9; t += step) out.push(t);
  return out;
}

function lineChart(mount, series, labels, opts) {
  opts = opts || {};
  const W = 760, H = 260, P = { t:14, r:48, b:28, l:34 };
  const all = series.flatMap(s => s.data);
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.08 || 1;
  lo -= pad; hi += pad;
  const n = labels.length;
  const x = i => P.l + (i / (n - 1)) * (W - P.l - P.r);
  const y = v => P.t + (1 - (v - lo) / (hi - lo)) * (H - P.t - P.b);

  const svg = svgEl('svg', { viewBox:`0 0 ${W} ${H}` });

  niceTicks(lo, hi, 5).forEach(t => {
    svg.appendChild(svgEl('line', {
      x1:P.l, x2:W-P.r, y1:y(t), y2:y(t), stroke:'rgba(30,45,69,0.7)', 'stroke-width':1 }));
    const lab = svgEl('text', {
      x:W-P.r+6, y:y(t)+3.5, fill:'#4a6080', 'font-size':9, 'font-family':'monospace' });
    lab.textContent = opts.fmt ? opts.fmt(t) : t.toFixed(0);
    svg.appendChild(lab);
  });

  const step = Math.max(1, Math.floor(n / 7));
  for (let i = 0; i < n; i += step) {
    const lab = svgEl('text', {
      x:x(i), y:H-8, fill:'#4a6080', 'font-size':9,
      'font-family':'monospace', 'text-anchor':'middle' });
    lab.textContent = labels[i].slice(0, 7);
    svg.appendChild(lab);
  }

  series.forEach(s => {
    const d = s.data.map((v,i) => `${i?'L':'M'}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join('');
    if (s.fill) {
      const base = y(Math.max(lo, opts.fillBase !== undefined ? opts.fillBase : lo));
      svg.appendChild(svgEl('path', {
        d:`${d}L${x(n-1).toFixed(2)},${base}L${x(0).toFixed(2)},${base}Z`,
        fill:s.fill, stroke:'none' }));
    }
    svg.appendChild(svgEl('path', {
      d, fill:'none', stroke:s.color, 'stroke-width':s.width || 1.6,
      'stroke-dasharray':s.dash || 'none',
      'stroke-linejoin':'round' }));
  });

  mount.innerHTML = '';
  mount.appendChild(svg);
}

function barChart(mount, values, labels) {
  const W = 440, H = 260, P = { t:14, r:46, b:52, l:14 };
  let lo = Math.min(0, ...values), hi = Math.max(0, ...values);
  const pad = (hi - lo) * 0.12 || 1;
  lo -= pad; hi += pad;
  const n = values.length;
  const bw = (W - P.l - P.r) / n;
  const y = v => P.t + (1 - (v - lo) / (hi - lo)) * (H - P.t - P.b);

  const svg = svgEl('svg', { viewBox:`0 0 ${W} ${H}` });
  niceTicks(lo, hi, 5).forEach(t => {
    svg.appendChild(svgEl('line', {
      x1:P.l, x2:W-P.r, y1:y(t), y2:y(t),
      stroke: Math.abs(t) < 1e-9 ? 'rgba(143,163,192,0.5)' : 'rgba(30,45,69,0.7)',
      'stroke-width':1 }));
    const lab = svgEl('text', {
      x:W-P.r+6, y:y(t)+3.5, fill:'#4a6080', 'font-size':9, 'font-family':'monospace' });
    lab.textContent = t.toFixed(1) + '%';
    svg.appendChild(lab);
  });

  values.forEach((v, i) => {
    const top = y(Math.max(v, 0)), bottom = y(Math.min(v, 0));
    svg.appendChild(svgEl('rect', {
      x:P.l + i*bw + bw*0.18, y:top, width:bw*0.64,
      height:Math.max(1, bottom - top), rx:2,
      fill: v >= 0 ? 'rgba(46,213,115,0.75)' : 'rgba(255,71,87,0.75)' }));
    const cx = P.l + i*bw + bw/2, cy = H - P.b + 16;
    const lab = svgEl('text', {
      x:cx, y:cy, fill:'#4a6080', 'font-size':9,
      'font-family':'monospace', 'text-anchor':'end',
      transform:`rotate(-45 ${cx} ${cy})` });
    lab.textContent = labels[i];
    svg.appendChild(lab);
  });

  mount.innerHTML = '';
  mount.appendChild(svg);
}

// ── charts ────────────────────────────────────────────────────────────────
lineChart(document.getElementById('perfChart'), [
  { data:R.series.portfolio_indexed, color:'#00d4aa', width:2 },
  { data:R.series.spy_indexed, color:'rgba(0,153,255,0.8)', dash:'4,4' },
  { data:R.series.qqq_indexed, color:'rgba(255,107,53,0.7)', dash:'2,4' },
], R.series.dates, { fmt: t => t.toFixed(0) });

barChart(document.getElementById('monthChart'),
  R.monthly_returns.portfolio, R.monthly_returns.labels);
document.getElementById('monthBadge').textContent =
  R.monthly_returns.labels.length + ' months';

lineChart(document.getElementById('ddChart'), [
  { data:R.series.drawdown.map(v => v*100), color:'#ff4757',
    width:1.6, fill:'rgba(255,71,87,0.18)' },
], R.series.dates, { fmt: t => t.toFixed(0) + '%', fillBase: 0 });
document.getElementById('ddBadge').textContent = 'max ' + spct(S.max_drawdown);

// ── risk rows ─────────────────────────────────────────────────────────────
const spy = R.benchmarks.find(b => b.benchmark === 'SPY');
const riskRows = [
  ['VaR 95% — parametric', pct(S.var_95_parametric), 'var(--red)'],
  ['VaR 95% — historical', pct(S.var_95_historical), 'var(--red)'],
  ['VaR 99% — parametric', pct(S.var_99_parametric), 'var(--red)'],
  ['VaR 99% — historical', pct(S.var_99_historical), 'var(--red)'],
  ['Beta vs SPY', num(spy.beta), 'var(--accent2)'],
  ['Annualised volatility', pct(S.annualised_volatility), 'var(--text)'],
  ['Avg holding period', num(S.avg_holding_days,1) + ' days', 'var(--text)'],
];
document.getElementById('riskRows').innerHTML = riskRows.map(([l,v,c]) => `
  <div class="row"><span class="row-label">${l}</span>
  <span class="row-val" style="color:${c}">${v}</span></div>`).join('');

// ── CAPM rows ─────────────────────────────────────────────────────────────
document.getElementById('capmRows').innerHTML = R.benchmarks.map(b => `
  <div class="row" style="flex-direction:column;align-items:stretch;gap:6px">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <span class="row-label">${b.benchmark}</span>
      <span class="row-val" style="color:${b.capm_alpha_ann>=0?'var(--green)':'var(--red)'}">
        alpha ${spct(b.capm_alpha_ann)}/yr</span>
    </div>
    <div style="font-size:10px;color:var(--text3);line-height:1.7">
      95% CI ${spct(b.alpha_ci_low)} to ${spct(b.alpha_ci_high)}<br>
      beta ${num(b.beta)} · R² ${num(b.r_squared)} · IR ${num(b.info_ratio)}
    </div>
  </div>`).join('');

// ── sector attribution ────────────────────────────────────────────────────
const sectors = R.pnl_by_sector;
const maxAbs = Math.max(...sectors.map(s => Math.abs(s.net_pnl)));
document.getElementById('attribList').innerHTML = sectors.map(s => `
  <div>
    <div class="attrib-head">
      <span class="attrib-name">${s.sector}</span>
      <span class="attrib-val" style="color:${s.net_pnl>=0?'var(--green)':'var(--red)'}">
        ${s.net_pnl>=0?'+':''}${money(s.net_pnl)}</span>
    </div>
    <div class="bar-bg"><div class="bar" style="width:${(Math.abs(s.net_pnl)/maxAbs*100).toFixed(0)}%;
      background:${s.net_pnl>=0?'var(--accent)':'var(--red)'}"></div></div>
  </div>`).join('');

// ── realised trades table (sortable) ──────────────────────────────────────
const tradeCols = [
  ['ticker','Ticker'], ['direction','Side'], ['qty','Qty'],
  ['entry_date','Entry'], ['entry_price','Entry px'],
  ['exit_date','Exit'], ['exit_price','Exit px'],
  ['commission','Comm'], ['net_pnl','Net P&L'],
];
let trades = R.realised_trades.slice();
let sortKey = 'net_pnl', sortDir = -1;

document.getElementById('tradeHead').innerHTML =
  tradeCols.map(([k,l]) => `<th data-key="${k}">${l}</th>`).join('');
document.getElementById('tradeBadge').textContent =
  trades.length + ' round-trips';

function renderTrades() {
  trades.sort((a,b) => {
    const av = a[sortKey], bv = b[sortKey];
    return (typeof av === 'string' ? av.localeCompare(bv) : av - bv) * sortDir;
  });
  document.getElementById('tradeBody').innerHTML = trades.map(t => `
    <tr>
      <td class="ticker">${t.ticker}</td>
      <td><span class="tag ${t.direction.toLowerCase()}">${t.direction}</span></td>
      <td>${t.qty}</td>
      <td>${t.entry_date}</td>
      <td>${t.entry_price.toFixed(2)}</td>
      <td>${t.exit_date}</td>
      <td>${t.exit_price.toFixed(2)}</td>
      <td>${t.commission.toFixed(2)}</td>
      <td class="${t.net_pnl>=0?'pos':'neg'}">${t.net_pnl>=0?'+':''}${money(t.net_pnl)}</td>
    </tr>`).join('');
  document.querySelectorAll('#tradeHead th').forEach(th =>
    th.classList.toggle('sorted', th.dataset.key === sortKey));
}
document.getElementById('tradeHead').addEventListener('click', e => {
  const key = e.target.dataset.key;
  if (!key) return;
  if (sortKey === key) sortDir *= -1; else { sortKey = key; sortDir = -1; }
  renderTrades();
});
renderTrades();

// ── open positions ────────────────────────────────────────────────────────
const openCols = [['ticker','Ticker'],['direction','Side'],['qty','Qty'],
                  ['entry_date','Entry date'],['entry_price','Entry price']];
document.getElementById('openHead').innerHTML =
  openCols.map(([,l]) => `<th>${l}</th>`).join('');
document.getElementById('openBody').innerHTML = R.open_positions.map(p => `
  <tr>
    <td class="ticker">${p.ticker}</td>
    <td><span class="tag ${p.direction.toLowerCase()}">${p.direction}</span></td>
    <td>${p.qty}</td>
    <td>${p.entry_date}</td>
    <td>${p.entry_price.toFixed(2)}</td>
  </tr>`).join('');
document.getElementById('openBadge').textContent =
  R.open_positions.length + ' open lots';

// ── footer ────────────────────────────────────────────────────────────────
document.getElementById('footer').innerHTML =
  'Generated by <code>tools/build_dashboard.py</code> from <code>outputs/results.json</code>. '
  + 'Data: ' + R.meta.data + '.<br>'
  + 'Rebuild: <code>python -m trademetrics.demo &amp;&amp; python tools/build_dashboard.py</code>. '
  + 'No broker connection, no personal account data, no hardcoded metrics.';
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
