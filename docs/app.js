/* eve-money-button Emperor Edition UI
 *
 * - Confidence filter (default >= 70)
 * - Instant vs Patient pricing toggle
 * - Recommended runs / batches from depth validation
 * - Plan Builder: aggregate buy/sell lists, slot + capital summary, auto-plan optimizer
 */

const DATA_URL = 'data/rankings.json';

const els = {
  status: document.getElementById('status'),
  runBtn: document.getElementById('runBtn'),
  reloadBtn: document.getElementById('reloadBtn'),

  modeSelect: document.getElementById('modeSelect'),
  confidenceMin: document.getElementById('confidenceMin'),
  confidenceMinLabel: document.getElementById('confidenceMinLabel'),
  validatedOnly: document.getElementById('validatedOnly'),
  positiveOnly: document.getElementById('positiveOnly'),
  searchBox: document.getElementById('searchBox'),

  tabs: Array.from(document.querySelectorAll('.tab')),
  panels: {
    manufacturing: document.getElementById('mfgPanel'),
    reactions: document.getElementById('rxPanel'),
    refining: document.getElementById('refPanel'),
    t2: document.getElementById('t2Panel'),
    plan: document.getElementById('planPanel'),
  },

  mfgSort: document.getElementById('mfgSort'),
  rxSort: document.getElementById('rxSort'),
  refSort: document.getElementById('refSort'),
  t2Sort: document.getElementById('t2Sort'),

  mfgTable: document.getElementById('mfgTable'),
  rxTable: document.getElementById('rxTable'),
  refTable: document.getElementById('refTable'),
  t2Table: document.getElementById('t2Table'),

  // Plan
  planItems: document.getElementById('planItems'),
  planSummary: document.getElementById('planSummary'),
  buyList: document.getElementById('buyList'),
  sellList: document.getElementById('sellList'),
  copyBuy: document.getElementById('copyBuy'),
  copySell: document.getElementById('copySell'),
  dlBuyCsv: document.getElementById('dlBuyCsv'),
  dlSellCsv: document.getElementById('dlSellCsv'),
  clearPlan: document.getElementById('clearPlan'),

  budgetISK: document.getElementById('budgetISK'),
  horizonHrs: document.getElementById('horizonHrs'),
  mfgSlots: document.getElementById('mfgSlots'),
  rxSlots: document.getElementById('rxSlots'),
  invSlots: document.getElementById('invSlots'),
  maxHaulM3: document.getElementById('maxHaulM3'),
  autoPlan: document.getElementById('autoPlan'),

  includeMfg: document.getElementById('includeMfg'),
  includeRx: document.getElementById('includeRx'),
  includeT2: document.getElementById('includeT2'),
  includeRef: document.getElementById('includeRef'),

  inventoryFile: document.getElementById('inventoryFile'),
  inventoryStatus: document.getElementById('inventoryStatus'),
};

let DATA = null;
let MODE = loadLocal('mode', 'instant');
let PLAN = loadLocal('plan', []); // [{key, runs}]
let INVENTORY = null; // {nameLower: qty}

function loadLocal(key, fallback) {
  try {
    const v = localStorage.getItem('eveMoney_' + key);
    if (!v) return fallback;
    return JSON.parse(v);
  } catch (e) {
    return fallback;
  }
}
function saveLocal(key, value) {
  try {
    localStorage.setItem('eveMoney_' + key, JSON.stringify(value));
  } catch (e) {}
}

function fmtISK(x) {
  if (x === null || x === undefined || isNaN(x)) return '—';
  const abs = Math.abs(x);
  const sign = x < 0 ? '-' : '';
  const n = Math.round(abs);
  if (n >= 1e12) return sign + (n / 1e12).toFixed(2) + 'T';
  if (n >= 1e9)  return sign + (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6)  return sign + (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3)  return sign + (n / 1e3).toFixed(1) + 'K';
  return sign + n.toString();
}

function fmtPct(x) {
  if (x === null || x === undefined || isNaN(x)) return '—';
  return (x * 100).toFixed(1) + '%';
}

function fmtTime(s) {
  if (!s || isNaN(s)) return '—';
  const sec = Math.floor(s);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  if (h <= 0) return m + 'm';
  return h + 'h ' + m + 'm';
}

function badgeConfidence(c) {
  const v = Number(c || 0);
  let cls = 'conf-low';
  if (v >= 85) cls = 'conf-high';
  else if (v >= 70) cls = 'conf-med';
  return `<span class="conf ${cls}" title="Confidence ${v}/100">${v}</span>`;
}

function hasDepth(row) {
  const d = row.depth || null;
  if (!d) return false;
  if (row.category === 'refining') return !!d.recommended_batches;
  return !!d.recommended_runs;
}

function sellMarketKey(row) {
  if (!row) return '';
  if (row.category === 'refining') return row.best_market || '';
  const bm = row.best_market || {};
  return bm[MODE] || bm.instant || bm.patient || '';
}

function sellMarketName(row) {
  if (!row) return '';
  if (row.category === 'refining') return row.best_market_name || (row.best_market || '');
  const bn = row.best_market_name || {};
  const k = sellMarketKey(row);
  return bn[MODE] || bn.instant || bn.patient || k || '';
}

function ttlText(row) {
  const t = row?.ttl;
  if (!t) return '—';
  if (t.bucket && t.bucket !== 'unknown') return t.bucket;
  return '—';
}

function totalM3PerRun(row) {
  const h = row?.hauling;
  if (!h) return 0;
  return h.total_m3_per_run ?? 0;
}

function profitPerM3(row) {
  const m = getModeBlock(row);
  const m3 = totalM3PerRun(row);
  if (!m || !m3) return null;
  return (m.profit ?? 0) / m3;
}

function badgeGuaranteed(row) {
  const d = row?.depth;
  if (!d) return '';
  // Depth is computed for a specific sell market. Only show the badge if it matches the
  // currently-selected destination market (MODE).
  const curSell = sellMarketKey(row);
  const depthSell = d.sell_market || '';
  if (depthSell && curSell && depthSell !== curSell) return '';

  if (d.guaranteed) return `<span class="conf conf-high" title="Depth-validated and profitable at recommended runs">GUAR</span>`;
  return `<span class="conf conf-mid" title="Depth computed but not guaranteed profitable at recommended runs">DEPTH</span>`;
}

function fmtM3(x) {
  if (x === null || x === undefined) return '—';
  const v = Number(x);
  if (!isFinite(v)) return '—';
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 }) + ' m³';
  return v.toLocaleString(undefined, { maximumFractionDigits: 1 }) + ' m³';
}


function rowKey(row) {
  // Unique key across categories
  if (row.category === 'refining') return `ref:${row.input_type_id}`;
  if (row.category === 't2') return `t2:${row.product_type_id}`;
  if (row.category === 'reactions') return `rx:${row.product_type_id}`;
  return `mfg:${row.product_type_id}`;
}

function lookupRowByKey(key) {
  if (!DATA) return null;
  const [prefix, idStr] = key.split(':');
  const id = Number(idStr);
  const lists = {
    'mfg': DATA.manufacturing || [],
    'rx': DATA.reactions || [],
    'ref': DATA.refining || [],
    't2': DATA.t2 || [],
  };
  const arr = lists[prefix] || [];
  if (prefix === 'ref') return arr.find(r => Number(r.input_type_id) === id) || null;
  return arr.find(r => Number(r.product_type_id) === id) || null;
}

function getModeBlock(row) {
  if (!row) return null;
  // Most rows carry both `instant` and `patient`. Prefer current MODE, then fall back.
  if (row.category === 'refining') return null;
  const mb = row[MODE];
  return mb || row.instant || row.patient || null;
}

function getProfit(row) {
  if (row.category === 'refining') return Number(row.profit || 0);
  const m = getModeBlock(row);
  return Number((m && m.profit) || row.profit || 0);
}
function getROI(row) {
  if (row.category === 'refining') return row.roi;
  const m = getModeBlock(row);
  return (m && m.roi !== undefined) ? m.roi : row.roi;
}
function getCost(row) {
  if (row.category === 'refining') return Number(row.cost || 0);
  if (row.category === 't2') {
    // Emperor edition stores total cost in top-level cost
    return Number(row.cost || 0);
  }
  const m = getModeBlock(row);
  return Number((m && m.cost) || row.cost || 0);
}
function getRevenue(row) {
  if (row.category === 'refining') return Number(row.revenue || 0);
  const m = getModeBlock(row);
  return Number((m && m.revenue) || row.revenue || 0);
}
function getFees(row) {
  if (row.category === 'refining') return Number(row.fees || 0);
  const m = getModeBlock(row);
  return Number((m && m.fees) || row.fees || 0);
}
function getProfitPerHour(row) {
  if (row.category === 'refining') return row.profit_per_m3; // special
  if (row.category === 't2') return row.profit_per_hour;
  const m = getModeBlock(row);
  // manufacturing/reactions expose profit_per_hour top-level (instant) but we want mode-aware:
  if (m && row.time_s) return (m.profit / (row.time_s / 3600));
  return row.profit_per_hour;
}

function recommendedRuns(row) {
  if (!row.depth) return null;
  if (row.category === 'refining') return row.depth.recommended_batches || null;
  return row.depth.recommended_runs || null;
}

function expectedAtRecommended(row) {
  const d = row.depth || null;
  if (!d || !d.expected) return null;
  return d.expected;
}

function passesFilters(row) {
  const minC = Number(els.confidenceMin.value || 0);
  const conf = Number(row.confidence || 0);
  if (conf < minC) return false;

  if (els.validatedOnly.checked && !(row?.depth?.guaranteed && (!row.depth.sell_market || sellMarketKey(row) === row.depth.sell_market))) return false;

  if (els.positiveOnly.checked && getProfit(row) <= 0) return false;

  const q = (els.searchBox.value || '').trim().toLowerCase();
  if (q) {
    const hay = [
      row.product_name || '',
      row.blueprint_name || '',
      row.t2_blueprint_name || '',
      row.input_name || '',
    ].join(' ').toLowerCase();
    if (!hay.includes(q)) return false;
  }

  return true;
}

function renderDetails(row) {
  const m = getModeBlock(row);
  const feeBreak = (m && m.fee_breakdown) ? m.fee_breakdown : null;
  const mats = row.category === 'refining'
    ? (row.materials || [])
    : ((m && m.materials) || row.materials || []);

  const lines = [];
  if (row.category !== 'refining') {
    lines.push(`<div class="detail-line"><b>Output qty/run:</b> ${row.output_qty}</div>`);
    lines.push(`<div class="detail-line"><b>Time/run:</b> ${fmtTime(row.time_s)}</div>`);
  }

  if (row.category === 'manufacturing' || row.category === 'reactions') {
    lines.push(`<div class="detail-line"><b>Blueprint cost:</b> ${fmtISK(row.blueprint_cost)} (${row.blueprint_sell_orders} sell orders)</div>`);
    if (row.payback_runs) {
      lines.push(`<div class="detail-line"><b>Payback:</b> ~${row.payback_runs.toFixed(1)} runs (instant)</div>`);
    }
  }

  if (row.category === 't2') {
    const inv = row.invention || {};
    lines.push(`<div class="detail-line"><b>Invention:</b> ${inv.t1_blueprint_name || '—'} → ${row.t2_blueprint_name}</div>`);
    lines.push(`<div class="detail-line"><b>Success chance (base):</b> ${fmtPct(inv.probability)} | <b>Runs/success:</b> ${inv.runs_per_success}</div>`);
    lines.push(`<div class="detail-line"><b>Invention cost/run:</b> ${fmtISK(inv.cost_per_run)} | <b>Invention time/run:</b> ${fmtTime(inv.time_per_run_s)}</div>`);
  }

  if (feeBreak) {
    const parts = Object.entries(feeBreak).map(([k,v]) => `${k}: ${fmtISK(v)}`).join(' | ');
    lines.push(`<div class="detail-line"><b>Fees breakdown:</b> ${parts}</div>`);
  }

  // Depth
  if (row.depth) {
    if (row.category === 'refining') {
      const d = row.depth;
      lines.push(`<div class="detail-line"><b>Depth:</b> Recommend ${d.recommended_batches || '—'} batches (max input batches ${d.max_batches_input || '—'})</div>`);
      if (d.expected) {
        lines.push(`<div class="detail-line"><b>Expected profit at rec:</b> ${fmtISK(d.expected.profit_total)} (${fmtISK(d.expected.profit_total / d.expected.batches)} per batch)</div>`);
      }
    } else {
      const d = row.depth;
      const rec = d.recommended_runs || '—';
      const mi = d.max_runs_input || '—';
      const mo = d.max_runs_output || '—';
      const lim = d.limiting_input ? ` | limiting input: ${d.limiting_input}` : '';
      lines.push(`<div class="detail-line"><b>Depth:</b> Recommend ${rec} runs (max input ${mi}, max output ${mo})${lim}</div>`);
      if (d.expected) {
        lines.push(`<div class="detail-line"><b>Expected profit at rec:</b> ${fmtISK(d.expected.profit_total)} (${fmtISK(d.expected.profit_per_run)} per run)</div>`);
      }
    }
  }

  // Materials / outputs breakdown
  const matLines = mats.map(m => {
    const qty = m.qty;
    const up = m.unit_price_used !== undefined ? m.unit_price_used : m.unit_price;
    const ext = m.extended_used !== undefined ? m.extended_used : m.extended;
    return `<li>${m.name} × ${qty} @ ${fmtISK(up)} = ${fmtISK(ext)}</li>`;
  }).join('');
  lines.push(`<div class="detail-line"><b>${row.category === 'refining' ? 'Outputs' : 'Inputs'}:</b><ul class="mat-list">${matLines}</ul></div>`);

  return `<details><summary>Details</summary><div class="details">${lines.join('')}</div></details>`;
}

function renderTable(rows, kind) {
  const hdr = (() => {
    if (kind === 'refining') {
      return `<tr>
        <th>Refine</th>
        <th>Sell @</th>
        <th>Confidence</th>
        <th>Profit/batch</th>
        <th>ISK/m³</th>
        <th>ROI</th>
        <th>TTL</th>
        <th>Plan</th>
      </tr>`;
    }
    if (kind === 't2') {
      return `<tr>
        <th>Build</th>
        <th>Sell @</th>
        <th>Confidence</th>
        <th>Profit/run</th>
        <th>Profit/hr</th>
        <th>ROI</th>
        <th>ISK/m³</th>
        <th>TTL</th>
        <th>Time/run</th>
        <th>Rec runs</th>
        <th>Expected profit</th>
        <th>Plan</th>
      </tr>`;
    }
    return `<tr>
      <th>Build</th>
      <th>Sell @</th>
      <th>Confidence</th>
      <th>Profit/run</th>
      <th>Profit/hr</th>
      <th>ROI</th>
      <th>ISK/m³</th>
      <th>TTL</th>
      <th>Time/run</th>
      <th>BPO</th>
      <th>Payback</th>
      <th>Rec runs</th>
      <th>Expected profit</th>
      <th>Plan</th>
    </tr>`;
  })();

  const body = rows.map(row => {
    const key = rowKey(row);
    const profit = getProfit(row);
    const roi = getROI(row);
    const pph = getProfitPerHour(row);
    const rec = recommendedRuns(row);

    let expected = '—';
    const exp = expectedAtRecommended(row);
    if (exp) {
      expected = fmtISK(exp.profit_total);
    }

    if (kind === 'refining') {
      const iskpm3 = row.profit_per_m3;
      const name = `${row.input_name} → minerals`;
      return `<tr data-key="${key}">
        <td>
          <div class="title">${name}</div>
          ${renderDetails(row)}
        </td>
        <td>${escapeHtml(sellMarketName(row))}</td>
        <td>${badgeConfidence(row.confidence)}</td>
        <td class="${profit>=0?'pos':'neg'}">${fmtISK(profit)}</td>
        <td class="${iskpm3>=0?'pos':'neg'}">${fmtISK(iskpm3)}</td>
        <td>${fmtPct(roi)}</td>
        <td>${ttlText(row)}</td>
        <td><button class="addBtn" data-key="${key}">Add</button></td>
      </tr>`;
    }

    const title = `${row.product_name}`;
    const sub = (kind === 't2')
      ? `${row.t2_blueprint_name}`
      : `${row.blueprint_name}`;
    const time = fmtTime(row.time_s);

    const bpo = (kind === 't2') ? '—' : fmtISK(row.blueprint_cost);
    const pay = (kind === 't2') ? '—' : (row.payback_runs ? row.payback_runs.toFixed(1)+' runs' : '—');

    // Profit/hr for refining is isk/m3; for others show ISK/hr
    const pphCell = (pph === null || pph === undefined) ? '—' : fmtISK(pph);
    const iskpm3 = profitPerM3(row);
    const iskpm3Cell = (iskpm3 === null || iskpm3 === undefined) ? '—' : fmtISK(iskpm3);
    const ttlCell = ttlText(row);
    const sellCell = `${escapeHtml(sellMarketName(row))} ${badgeGuaranteed(row)}`;

    return `<tr data-key="${key}">
      <td>
        <div class="title">${title}</div>
        <div class="sub">${sub}</div>
        ${renderDetails(row)}
      </td>
      <td>${sellCell}</td>
      <td>${badgeConfidence(row.confidence)}</td>
      <td class="${profit>=0?'pos':'neg'}">${fmtISK(profit)}</td>
      <td class="${(pph||0)>=0?'pos':'neg'}">${pphCell}</td>
      <td>${fmtPct(roi)}</td>
      <td class="${(iskpm3||0)>=0?'pos':'neg'}">${iskpm3Cell}</td>
      <td>${ttlCell}</td>
      <td>${time}</td>
      ${kind === 't2' ? '' : `<td>${bpo}</td><td>${pay}</td>`}
      <td>${rec || '—'}</td>
      <td>${expected}</td>
      <td><button class="addBtn" data-key="${key}">Add</button></td>
    </tr>`;
  }).join('');

  return `<table>
    <thead>${hdr}</thead>
    <tbody>${body}</tbody>
  </table>`;
}

function sortRows(rows, kind, sortKey) {
  const copy = rows.slice();
  const cmp = (a,b) => {
    // Prefer depth-*guaranteed* rows first (works for manufacturing/reactions/t2).
    const ga = (a?.depth?.guaranteed && (!a.depth.sell_market || sellMarketKey(a) === a.depth.sell_market)) ? 1 : 0;
    const gb = (b?.depth?.guaranteed && (!b.depth.sell_market || sellMarketKey(b) === b.depth.sell_market)) ? 1 : 0;
    if (ga !== gb) return gb - ga;

    const va = sortValue(a, kind, sortKey);
    const vb = sortValue(b, kind, sortKey);
    return vb - va;
  };
  copy.sort(cmp);
  return copy;
}

function sortValue(row, kind, sortKey) {
  if (kind === 'refining') {
    if (sortKey === 'profit') return row.profit;
    if (sortKey === 'iskpm3') return row.profit_per_m3 || 0;
    if (sortKey === 'roi') return (row.roi || 0);
    if (sortKey === 'confidence') return row.confidence || 0;
    return row.profit_per_m3 || 0;
  }
  if (sortKey === 'profit') return getProfit(row);
  if (sortKey === 'pph') return getProfitPerHour(row) || 0;
  if (sortKey === 'roi') return getROI(row) || 0;
  if (sortKey === 'confidence') return row.confidence || 0;
  if (sortKey === 'rec') return recommendedRuns(row) || 0;
  return getProfitPerHour(row) || 0;
}

function renderAll() {
  if (!DATA) return;

  els.confidenceMinLabel.textContent = els.confidenceMin.value;

  const mfg = (DATA.manufacturing || []).filter(passesFilters);
  const rx  = (DATA.reactions || []).filter(passesFilters);
  const ref = (DATA.refining || []).filter(passesFilters);
  const t2  = (DATA.t2 || []).filter(passesFilters);

  const mfgSorted = sortRows(mfg, 'manufacturing', els.mfgSort.value);
  const rxSorted  = sortRows(rx, 'reactions', els.rxSort.value);
  const refSorted = sortRows(ref, 'refining', els.refSort.value);
  const t2Sorted  = sortRows(t2, 't2', els.t2Sort.value);

  els.mfgTable.innerHTML = renderTable(mfgSorted, 'manufacturing');
  els.rxTable.innerHTML  = renderTable(rxSorted, 'reactions');
  els.refTable.innerHTML = renderTable(refSorted, 'refining');
  els.t2Table.innerHTML  = renderTable(t2Sorted, 't2');

  renderPlan();
}

function setStatus(msg) {
  els.status.textContent = msg;
}

async function loadData() {
  setStatus('Loading rankings…');
  const r = await fetch(DATA_URL, {cache: 'no-store'});
  if (!r.ok) throw new Error('Failed to load rankings.json');
  DATA = await r.json();
  const buyName = DATA?.market?.buy_market?.name || (DATA?.market?.region_id ? `Region ${DATA.market.region_id}` : '—');
  const sellNames = (DATA?.market?.sell_markets || []).map(m => m.name || m.key);
  const sellShort = sellNames.slice(0, 6).join(', ') + (sellNames.length > 6 ? '…' : '');
  const charName = DATA?.character?.name ? ` | Char ${DATA.character.name}` : '';
  setStatus(`Updated ${DATA.generated_at} | Buy: ${buyName} | Sells: ${sellShort} | Mode: ${MODE}${charName}`);
}

function switchTab(tabName) {
  els.tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === tabName));
  Object.entries(els.panels).forEach(([k, el]) => {
    el.style.display = (k === tabName) ? '' : 'none';
  });
}

function addToPlan(key, defaultRuns = null) {
  const row = lookupRowByKey(key);
  if (!row) return;

  // default runs: depth rec if present, else 1
  let runs = defaultRuns;
  if (runs === null || runs === undefined) {
    runs = recommendedRuns(row) || 1;
  }
  runs = Math.max(1, Math.floor(Number(runs) || 1));

  const idx = PLAN.findIndex(p => p.key === key);
  if (idx >= 0) {
    PLAN[idx].runs += runs;
  } else {
    PLAN.push({key, runs});
  }
  saveLocal('plan', PLAN);
  switchTab('plan');
  renderPlan();
}

function removeFromPlan(key) {
  PLAN = PLAN.filter(p => p.key !== key);
  saveLocal('plan', PLAN);
  renderPlan();
}

function updatePlanRuns(key, runs) {
  const v = Math.max(1, Math.floor(Number(runs) || 1));
  const idx = PLAN.findIndex(p => p.key === key);
  if (idx >= 0) {
    PLAN[idx].runs = v;
    saveLocal('plan', PLAN);
    renderPlan();
  }
}

function getTypeVolumeByName(name) {
  if (!DATA || !DATA.type_info) return 0;
  // We'll compute by matching name in type_info (slow, but fine for small plan)
  // Better: build a reverse map once
  return 0;
}

let TYPE_BY_NAME = null;
function buildTypeByName() {
  TYPE_BY_NAME = {};
  const info = (DATA && DATA.type_info) ? DATA.type_info : {};
  for (const [tid, meta] of Object.entries(info)) {
    const nm = (meta.name || '').toLowerCase();
    if (nm) TYPE_BY_NAME[nm] = {type_id: tid, ...meta};
  }
}

function typeVolume(name) {
  if (!TYPE_BY_NAME) buildTypeByName();
  const meta = TYPE_BY_NAME[(name || '').toLowerCase()];
  return meta ? Number(meta.volume || 0) : 0;
}

function planCompute() {
  const items = [];
  let totalCost = 0;
  let totalRevenue = 0;
  let totalProfit = 0;

  let mfgHours = 0;
  let rxHours = 0;
  let invHours = 0;
  let haulM3 = 0;

  const buyMap = new Map(); // name -> qty
  const sellMap = new Map(); // name -> qty

  for (const p of PLAN) {
    const row = lookupRowByKey(p.key);
    if (!row) continue;

    const runs = Number(p.runs || 1);

    let cost = 0;
    let revenue = 0;
    let profit = 0;

    // Use depth expected when runs == recommended
    const exp = expectedAtRecommended(row);
    if (exp && row.category !== 'refining' && runs === Number(exp.runs || 0)) {
      cost = Number(exp.cost_total || 0);
      revenue = Number(exp.revenue_total || 0);
      profit = Number(exp.profit_total || 0);
    } else if (exp && row.category === 'refining' && runs === Number(exp.batches || 0)) {
      cost = Number(exp.cost_total || 0);
      revenue = Number(exp.revenue_total || 0);
      profit = Number(exp.profit_total || 0);
    } else {
      // baseline linear
      const pr = getProfit(row);
      profit = pr * runs;
      cost = getCost(row) * runs;
      revenue = getRevenue(row) * runs;
    }

    totalCost += cost;
    totalRevenue += revenue;
    totalProfit += profit;

    // Slot hours
    if (row.category === 'manufacturing') {
      mfgHours += (Number(row.time_s || 0) * runs) / 3600;
    } else if (row.category === 'reactions') {
      rxHours += (Number(row.time_s || 0) * runs) / 3600;
    } else if (row.category === 't2') {
      mfgHours += (Number(row.time_s || 0) * runs) / 3600;
      const inv = row.invention || {};
      invHours += (Number(inv.time_per_run_s || 0) * runs) / 3600;
    }

    // Buy & sell aggregation + hauling m3 (buy side only)
    if (row.category === 'refining') {
      // buy ore
      const oreQty = Number(row.batch_units || 0) * runs;
      bumpMap(buyMap, row.input_name, oreQty);
      haulM3 += oreQty * typeVolume(row.input_name);

      // sell outputs (tagged with destination market)
      const sellTag = `[${sellMarketName(row)}] `;
      for (const out of (row.outputs || [])) {
        const outQty = Number(out.qty || 0) * runs;
        bumpMap(sellMap, sellTag + out.name, outQty);
        haulM3 += outQty * typeVolume(out.name);
      }
    } else {
      // buy inputs
      const m = getModeBlock(row);
      const mats = (m && m.materials) ? m.materials : (row.materials || []);
      for (const mat of mats) {
        bumpMap(buyMap, mat.name, Number(mat.qty || 0) * runs);
        haulM3 += Number(mat.qty || 0) * runs * typeVolume(mat.name);
      }
      // sell output (tagged with destination market)
      bumpMap(sellMap, `[${sellMarketName(row)}] ` + row.product_name, Number(row.output_qty || 1) * runs);

      // for T2, also buy invention mats
      if (row.category === 't2') {
        const inv = row.invention || {};
        for (const mat of (inv.materials || [])) {
          bumpMap(buyMap, mat.name, Number(mat.qty || 0) * runs);
          haulM3 += Number(mat.qty || 0) * runs * typeVolume(mat.name);
        }
      }
    }

    items.push({
      key: p.key,
      runs,
      row,
      cost,
      revenue,
      profit,
    });
  }

  return {items, totalCost, totalRevenue, totalProfit, mfgHours, rxHours, invHours, haulM3, buyMap, sellMap};
}

function bumpMap(map, name, qty) {
  const k = name || '—';
  map.set(k, (map.get(k) || 0) + qty);
}

function mapToList(map) {
  return Array.from(map.entries())
    .sort((a,b) => a[0].localeCompare(b[0]))
    .map(([name, qty]) => ({name, qty}));
}

function renderPlan() {
  if (!DATA) return;
  if (!TYPE_BY_NAME) buildTypeByName();

  const c = planCompute();

  // Plan items table
  const rows = c.items.map(it => {
    const row = it.row;
    const name = row.category === 'refining'
      ? row.input_name + ' → refine'
      : row.product_name;
    const rec = recommendedRuns(row);
    const exp = expectedAtRecommended(row);

    let hint = '';
    if (rec && it.runs === rec && exp) {
      hint = `<span class="hint">using depth-expected</span>`;
    } else if (rec) {
      hint = `<span class="hint">rec: ${rec}</span>`;
    }

    return `<tr>
      <td>${name}<div class="sub">${row.category}</div>${hint}</td>
      <td><input class="planRuns" data-key="${it.key}" type="number" min="1" value="${it.runs}"></td>
      <td class="${it.profit>=0?'pos':'neg'}">${fmtISK(it.profit)}</td>
      <td>${fmtISK(it.cost)}</td>
      <td><button class="rmBtn" data-key="${it.key}">Remove</button></td>
    </tr>`;
  }).join('');

  els.planItems.innerHTML = `<table>
    <thead><tr><th>Item</th><th>Runs</th><th>Profit</th><th>Input ISK</th><th></th></tr></thead>
    <tbody>${rows || '<tr><td colspan="5">Plan is empty. Click “Add” on an item.</td></tr>'}</tbody>
  </table>`;

  // Summary
  const horizon = Number(els.horizonHrs.value || 24);
  const mfgSlots = Math.max(0, Number(els.mfgSlots.value || 0));
  const rxSlots = Math.max(0, Number(els.rxSlots.value || 0));
  const invSlots = Math.max(0, Number(els.invSlots.value || 0));
  const capBudget = Number(els.budgetISK.value || 0);
  const maxHaul = Number(els.maxHaulM3.value || 0);

  const mfgNeed = horizon > 0 ? c.mfgHours / horizon : 0;
  const rxNeed  = horizon > 0 ? c.rxHours / horizon : 0;
  const invNeed = horizon > 0 ? c.invHours / horizon : 0;

  const budgetOk = capBudget <= 0 ? true : c.totalCost <= capBudget;
  const haulOk = maxHaul <= 0 ? true : c.haulM3 <= maxHaul;
  const mfgOk = mfgSlots <= 0 ? true : c.mfgHours <= mfgSlots * horizon;
  const rxOk  = rxSlots <= 0 ? true : c.rxHours <= rxSlots * horizon;
  const invOk = invSlots <= 0 ? true : c.invHours <= invSlots * horizon;

  els.planSummary.innerHTML = `
    <div class="sum-grid">
      <div><b>Total input ISK:</b> ${fmtISK(c.totalCost)} ${budgetOk ? '' : '<span class="warn">over budget</span>'}</div>
      <div><b>Total expected output ISK:</b> ${fmtISK(c.totalRevenue)}</div>
      <div><b>Total expected profit:</b> <span class="${c.totalProfit>=0?'pos':'neg'}">${fmtISK(c.totalProfit)}</span></div>

      <div><b>MFG slot-hours:</b> ${c.mfgHours.toFixed(1)} (needs ~${mfgNeed.toFixed(1)} slots) ${mfgOk ? '' : '<span class="warn">over</span>'}</div>
      <div><b>RX slot-hours:</b> ${c.rxHours.toFixed(1)} (needs ~${rxNeed.toFixed(1)} slots) ${rxOk ? '' : '<span class="warn">over</span>'}</div>
      <div><b>INV slot-hours:</b> ${c.invHours.toFixed(1)} (needs ~${invNeed.toFixed(1)} slots) ${invOk ? '' : '<span class="warn">over</span>'}</div>

      <div><b>Estimated hauling (buy-side) m³:</b> ${c.haulM3.toFixed(1)} ${haulOk ? '' : '<span class="warn">over</span>'}</div>
    </div>
  `;

  // Buy/Sell lists
  const buyList = mapToList(c.buyMap);
  const sellList = mapToList(c.sellMap);

  const buyText = buyList.map(it => `${it.name}\t${Math.round(it.qty)}`).join('\n');
  const sellText = sellList.map(it => `${it.name}\t${Math.round(it.qty)}`).join('\n');

  els.buyList.value = buyText;
  els.sellList.value = sellText;

  // Inventory awareness (optional)
  if (INVENTORY) {
    // Update status with covered count
    let covered = 0;
    for (const it of buyList) {
      const have = INVENTORY[it.name.toLowerCase()] || 0;
      if (have >= it.qty) covered++;
    }
    els.inventoryStatus.textContent = `Inventory loaded. Covered lines: ${covered}/${buyList.length}`;
  } else {
    els.inventoryStatus.textContent = '';
  }

  // Attach plan events
  els.planItems.querySelectorAll('.rmBtn').forEach(btn => {
    btn.addEventListener('click', () => removeFromPlan(btn.dataset.key));
  });
  els.planItems.querySelectorAll('.planRuns').forEach(inp => {
    inp.addEventListener('change', () => updatePlanRuns(inp.dataset.key, inp.value));
  });

  // CSV downloads
  els.dlBuyCsv.onclick = () => downloadCSV('buy_list.csv', buyList);
  els.dlSellCsv.onclick = () => downloadCSV('sell_list.csv', sellList);
}

function downloadCSV(filename, rows) {
  const header = 'name,qty\n';
  const body = rows.map(r => `"${(r.name||'').replaceAll('"','""')}",${Math.round(r.qty)}`).join('\n');
  const blob = new Blob([header + body + '\n'], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function copyText(text) {
  navigator.clipboard.writeText(text).catch(() => {
    // fallback
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
  });
}

// Auto-plan: greedy fill under constraints
function autoPlan() {
  if (!DATA) return;

  const minC = Number(els.confidenceMin.value || 0);
  const horizon = Math.max(1, Number(els.horizonHrs.value || 24));
  const budget = Math.max(0, Number(els.budgetISK.value || 0));
  const mfgSlots = Math.max(0, Number(els.mfgSlots.value || 0));
  const rxSlots = Math.max(0, Number(els.rxSlots.value || 0));
  const invSlots = Math.max(0, Number(els.invSlots.value || 0));
  const maxHaul = Math.max(0, Number(els.maxHaulM3.value || 0));

  const capMfg = mfgSlots > 0 ? mfgSlots * horizon : Infinity;
  const capRx  = rxSlots > 0 ? rxSlots * horizon : Infinity;
  const capInv = invSlots > 0 ? invSlots * horizon : Infinity;

  let usedMfg = 0;
  let usedRx = 0;
  let usedInv = 0;
  let usedISK = 0;
  let usedM3 = 0;

  const candidates = [];

  function addCandidates(arr) {
    for (const row of arr) {
      if (Number(row.confidence || 0) < minC) continue;
      if (getProfit(row) <= 0) continue;

      const rec = recommendedRuns(row) || 1;
      const exp = expectedAtRecommended(row);
      const runs = rec;

      // Use depth expected where possible, else linear
      let profit = getProfit(row) * runs;
      let cost = getCost(row) * runs;
      let mfgH = 0, rxH = 0, invH = 0;

      if (exp && row.category !== 'refining' && runs === Number(exp.runs || 0)) {
        profit = Number(exp.profit_total || 0);
        cost = Number(exp.cost_total || 0);
      } else if (exp && row.category === 'refining' && runs === Number(exp.batches || 0)) {
        profit = Number(exp.profit_total || 0);
        cost = Number(exp.cost_total || 0);
      }

      if (row.category === 'manufacturing') mfgH = (Number(row.time_s || 0) * runs) / 3600;
      if (row.category === 'reactions') rxH = (Number(row.time_s || 0) * runs) / 3600;
      if (row.category === 't2') {
        mfgH = (Number(row.time_s || 0) * runs) / 3600;
        invH = (Number((row.invention||{}).time_per_run_s || 0) * runs) / 3600;
      }

      // Haul m3 estimate
      let haul = 0;
      if (!TYPE_BY_NAME) buildTypeByName();
      if (row.category === 'refining') {
        haul += (Number(row.batch_units || 0) * runs) * typeVolume(row.input_name);
      } else {
        const m = getModeBlock(row);
        const mats = (m && m.materials) ? m.materials : (row.materials || []);
        for (const mat of mats) {
          haul += Number(mat.qty || 0) * runs * typeVolume(mat.name);
        }
        if (row.category === 't2') {
          for (const mat of ((row.invention||{}).materials || [])) {
            haul += Number(mat.qty || 0) * runs * typeVolume(mat.name);
          }
        }
      }

      // Score: profit per limiting slot-hour (rough) and confidence.
      const denom = Math.max(0.01, Math.max(mfgH, rxH, invH, 0.01));
      const score = profit / denom;

      candidates.push({row, runs, profit, cost, mfgH, rxH, invH, haul, score});
    }
  }

  if (els.includeMfg.checked) addCandidates(DATA.manufacturing || []);
  if (els.includeRx.checked) addCandidates(DATA.reactions || []);
  if (els.includeT2.checked) addCandidates(DATA.t2 || []);
  if (els.includeRef.checked) addCandidates(DATA.refining || []);

  candidates.sort((a,b) => b.score - a.score);

  PLAN = [];
  for (const c of candidates) {
    if (budget > 0 && usedISK + c.cost > budget) continue;
    if (usedMfg + c.mfgH > capMfg) continue;
    if (usedRx + c.rxH > capRx) continue;
    if (usedInv + c.invH > capInv) continue;
    if (maxHaul > 0 && usedM3 + c.haul > maxHaul) continue;

    const key = rowKey(c.row);
    PLAN.push({key, runs: c.runs});

    usedISK += c.cost;
    usedMfg += c.mfgH;
    usedRx += c.rxH;
    usedInv += c.invH;
    usedM3 += c.haul;

    // stop if plan is getting too long
    if (PLAN.length >= 10) break;
  }

  saveLocal('plan', PLAN);
  switchTab('plan');
  renderPlan();
}

function parseInventoryCSV(text) {
  // Very forgiving: expects "name,qty" or tab-separated.
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const map = {};
  for (const line of lines) {
    const parts = line.includes(',') ? line.split(',') : line.split(/\t+/);
    if (parts.length < 2) continue;
    const name = parts[0].trim();
    const qty = Number(parts[1].trim());
    if (!name || !isFinite(qty)) continue;
    map[name.toLowerCase()] = (map[name.toLowerCase()] || 0) + qty;
  }
  return map;
}

async function wireEvents() {
  els.modeSelect.value = MODE;
  els.confidenceMin.value = loadLocal('confidenceMin', 70);
  els.validatedOnly.checked = loadLocal('validatedOnly', true);
  els.positiveOnly.checked = loadLocal('positiveOnly', true);
  els.searchBox.value = loadLocal('search', '');

  els.modeSelect.addEventListener('change', () => {
    MODE = els.modeSelect.value;
    saveLocal('mode', MODE);
    renderAll();
    setStatus(`Mode: ${MODE} | Updated ${DATA?.generated_at || ''}`);
  });

  els.confidenceMin.addEventListener('input', () => {
    saveLocal('confidenceMin', Number(els.confidenceMin.value));
    renderAll();
  });
  els.validatedOnly.addEventListener('change', () => {
    saveLocal('validatedOnly', els.validatedOnly.checked);
    renderAll();
  });
  els.positiveOnly.addEventListener('change', () => {
    saveLocal('positiveOnly', els.positiveOnly.checked);
    renderAll();
  });
  els.searchBox.addEventListener('input', () => {
    saveLocal('search', els.searchBox.value);
    renderAll();
  });

  els.mfgSort.addEventListener('change', renderAll);
  els.rxSort.addEventListener('change', renderAll);
  els.refSort.addEventListener('change', renderAll);
  els.t2Sort.addEventListener('change', renderAll);

  els.tabs.forEach(t => {
    t.addEventListener('click', () => switchTab(t.dataset.tab));
  });

  // Delegated click for Add buttons
  document.body.addEventListener('click', (ev) => {
    const btn = ev.target.closest('.addBtn');
    if (!btn) return;
    addToPlan(btn.dataset.key);
  });

  els.reloadBtn.addEventListener('click', async () => {
    try {
      await loadData();
      buildTypeByName();
      renderAll();
    } catch (e) {
      console.error(e);
      setStatus('Reload failed: ' + e.message);
    }
  });

  els.runBtn.addEventListener('click', async () => {
    // For GitHub Pages we can't trigger Actions without auth.
    // This button just reloads the latest rankings.json.
    els.reloadBtn.click();
  });

  els.copyBuy.addEventListener('click', () => copyText(els.buyList.value));
  els.copySell.addEventListener('click', () => copyText(els.sellList.value));

  els.clearPlan.addEventListener('click', () => {
    PLAN = [];
    saveLocal('plan', PLAN);
    renderPlan();
  });

  els.autoPlan.addEventListener('click', autoPlan);

  els.inventoryFile.addEventListener('change', async () => {
    const f = els.inventoryFile.files && els.inventoryFile.files[0];
    if (!f) return;
    const text = await f.text();
    INVENTORY = parseInventoryCSV(text);
    renderPlan();
  });
}

(async function init() {
  try {
    await wireEvents();
    await loadData();
    buildTypeByName();
    renderAll();
    switchTab('manufacturing');
  } catch (e) {
    console.error(e);
    setStatus('Error: ' + e.message);
  }
})();
