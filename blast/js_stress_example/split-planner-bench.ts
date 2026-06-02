/**
 * Split-Planner Benchmark — a live, in-browser A/B of the destruction split
 * planner: the shipping connected-component planner vs. the original dense
 * Hungarian it replaced. Pure CPU work (the topology-diff a fracture runs to
 * decide which Rapier bodies to reuse) — no physics, no WASM — so it runs
 * straight from the library source and isolates the optimization.
 *
 * It charts ms-per-plan on a log-log scale as the fragment count grows, so the
 * reference's O(N^3) cliff (rocketing past the 60 fps frame budget) and the new
 * planner's flat, near-linear line are visible side by side — with a per-size
 * check that both reach the *same* optimal assignment (the speedup is free).
 */
import {
  measureRow,
  type BenchRow,
  type PlannerScenario,
} from "../blast-stress-solver/src/rapier/splitPlannerBench";

const SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
// The reference is cubic; cap it so the heaviest row stays ~100ms (responsive).
// The shipping planner is measured across every size, far past where the
// reference is even feasible.
const REFERENCE_MAX_N = 256;
const FRAME_BUDGET_MS = 1000 / 60; // 16.67ms — one frame at 60fps

const PROD_COLOR = "#5ad19a";
const REF_COLOR = "#ffb454";
const BUDGET_COLOR = "rgba(255,120,120,0.7)";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const els = {
  scenario: $("scenario") as HTMLSelectElement,
  run: $("run") as HTMLButtonElement,
  status: $("status"),
  headline: $("headline"),
  chart: $("chart") as HTMLCanvasElement,
  tbody: $("tbody"),
};

let running = false;

function fmtMs(v: number | null): string {
  if (v == null) return "—";
  if (v >= 100) return v.toFixed(0);
  if (v >= 1) return v.toFixed(2);
  if (v >= 0.001) return v.toFixed(4);
  return v.toExponential(1);
}

function yieldToEventLoop(): Promise<void> {
  return new Promise((r) => requestAnimationFrame(() => r()));
}

async function run() {
  if (running) return;
  running = true;
  els.run.disabled = true;
  els.scenario.disabled = true;
  els.tbody.innerHTML = "";
  els.headline.innerHTML = "";

  const scenario = els.scenario.value as PlannerScenario;
  const rows: BenchRow[] = [];

  for (const n of SIZES) {
    els.status.textContent =
      `Measuring ${scenario} N=${n}…` +
      (n <= REFERENCE_MAX_N ? " (running both planners)" : " (reference omitted — would take seconds–minutes)");
    await yieldToEventLoop();

    const row = measureRow(scenario, n, {
      budgetMs: 30,
      maxIters: 400,
      referenceMaxN: REFERENCE_MAX_N,
    });
    rows.push(row);
    renderTable(rows);
    drawChart(rows);
    renderHeadline(rows);
    await yieldToEventLoop();
  }

  els.status.textContent = "Done.";
  els.run.disabled = false;
  els.scenario.disabled = false;
  running = false;
}

function renderHeadline(rows: BenchRow[]) {
  const withRef = rows.filter((r) => r.referenceMs != null && r.speedup != null);
  if (withRef.length === 0) return;
  const best = withRef.reduce((a, b) => (b.speedup! > a.speedup! ? b : a));
  const allEquiv = withRef.every((r) => r.optimalEquivalent);
  const maxProd = Math.max(...rows.map((r) => r.n));

  els.headline.innerHTML = `
    <div class="stat">
      <div class="stat-num">${Math.round(best.speedup!).toLocaleString()}×</div>
      <div class="stat-label">faster at ${best.scenario} N=${best.n}<br/>
        <span class="muted">${fmtMs(best.referenceMs)} ms → ${fmtMs(best.productionMs)} ms per plan</span>
      </div>
    </div>
    <div class="stat">
      <div class="stat-num">${allEquiv ? "✓" : "✗"}</div>
      <div class="stat-label">${allEquiv ? "same optimal assignment" : "ASSIGNMENT MISMATCH"}<br/>
        <span class="muted">${allEquiv ? "the speedup is free" : "regression!"}</span>
      </div>
    </div>
    <div class="stat">
      <div class="stat-num">N=${maxProd.toLocaleString()}</div>
      <div class="stat-label">shipping planner stays&nbsp;flat<br/>
        <span class="muted">where the old planner can't run</span>
      </div>
    </div>`;
}

function renderTable(rows: BenchRow[]) {
  els.tbody.innerHTML = rows
    .map((r) => {
      const speedup = r.speedup == null ? '<span class="muted">ref omitted</span>' : `${Math.round(r.speedup).toLocaleString()}×`;
      const equiv =
        r.optimalEquivalent == null
          ? '<span class="muted">—</span>'
          : r.optimalEquivalent
            ? '<span class="ok">✓</span>'
            : '<span class="bad">✗</span>';
      const refCell =
        r.referenceMs == null
          ? '<span class="muted">omitted</span>'
          : `${fmtMs(r.referenceMs)} ms`;
      const overBudget = (r.referenceMs ?? 0) > FRAME_BUDGET_MS;
      return `<tr>
        <td>${r.n.toLocaleString()}</td>
        <td>${r.bodyCount.toLocaleString()}</td>
        <td class="${overBudget ? "warn" : ""}">${refCell}</td>
        <td class="good">${fmtMs(r.productionMs)} ms</td>
        <td>${speedup}</td>
        <td style="text-align:center">${equiv}</td>
      </tr>`;
    })
    .join("");
}

// ── Log-log chart ────────────────────────────────────────────────────────────

function drawChart(rows: BenchRow[]) {
  const canvas = els.chart;
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 720;
  const cssH = canvas.clientHeight || 360;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const pad = { l: 56, r: 16, t: 18, b: 40 };
  const plotW = cssW - pad.l - pad.r;
  const plotH = cssH - pad.t - pad.b;

  // x: log2(N) over the full size range; y: log10(ms) over a fixed, readable band.
  const xs = SIZES.map((n) => Math.log2(n));
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMinMs = 0.005;
  const yMaxMs = 10000;
  const yMin = Math.log10(yMinMs);
  const yMax = Math.log10(yMaxMs);

  const xPix = (n: number) => pad.l + ((Math.log2(n) - xMin) / (xMax - xMin || 1)) * plotW;
  const yPix = (ms: number) => {
    const v = Math.log10(Math.max(ms, yMinMs));
    return pad.t + (1 - (v - yMin) / (yMax - yMin)) * plotH;
  };

  // y gridlines at powers of 10
  ctx.font = "11px ui-monospace, monospace";
  ctx.textBaseline = "middle";
  for (let e = Math.ceil(yMin); e <= Math.floor(yMax); e += 1) {
    const ms = Math.pow(10, e);
    const y = yPix(ms);
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(cssW - pad.r, y);
    ctx.stroke();
    ctx.fillStyle = "rgba(200,210,240,0.5)";
    ctx.textAlign = "right";
    const label = ms >= 1 ? `${ms} ms` : `${ms} ms`;
    ctx.fillText(label, pad.l - 8, y);
  }

  // x ticks at each N
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const n of SIZES) {
    const x = xPix(n);
    ctx.fillStyle = "rgba(200,210,240,0.5)";
    ctx.fillText(String(n), x, cssH - pad.b + 8);
  }
  ctx.fillStyle = "rgba(200,210,240,0.7)";
  ctx.fillText("fragments in one fracture (N)", pad.l + plotW / 2, cssH - 16);

  // 60fps frame-budget line
  const yb = yPix(FRAME_BUDGET_MS);
  ctx.strokeStyle = BUDGET_COLOR;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.l, yb);
  ctx.lineTo(cssW - pad.r, yb);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = BUDGET_COLOR;
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText("16.7 ms — 60 fps frame budget", pad.l + 6, yb - 3);

  const drawSeries = (
    getMs: (r: BenchRow) => number | null,
    color: string,
  ) => {
    const pts = rows
      .map((r) => ({ n: r.n, ms: getMs(r) }))
      .filter((p): p is { n: number; ms: number } => p.ms != null);
    if (pts.length === 0) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = xPix(p.n);
      const y = yPix(p.ms);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = color;
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(xPix(p.n), yPix(p.ms), 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  drawSeries((r) => r.referenceMs, REF_COLOR);
  drawSeries((r) => r.productionMs, PROD_COLOR);

  // legend
  const legend = [
    ["dense Hungarian (before)", REF_COLOR],
    ["connected components (after)", PROD_COLOR],
  ] as const;
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  let lx = pad.l + 8;
  const ly = pad.t + 8;
  for (const [label, color] of legend) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(lx, ly, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(220,228,255,0.8)";
    ctx.fillText(label, lx + 10, ly);
    lx += ctx.measureText(label).width + 38;
  }
}

els.run.addEventListener("click", () => void run());
// Kick off an initial run so the page is immediately illustrative.
void run();
