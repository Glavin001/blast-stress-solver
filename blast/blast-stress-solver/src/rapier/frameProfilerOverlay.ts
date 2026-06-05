/**
 * Reusable, self-contained live frame-profiler overlay for the destructible core.
 *
 * `createFrameProfilerOverlay()` builds its own DOM (a floating panel with the
 * per-phase chart, a spike-cause callout, a live legend, stats, and an A/B
 * toggle), injects its own styles once, owns a {@link FrameProfilerBuffer}, and
 * wires `core.setProfiler(...)`. A demo only needs:
 *
 * ```ts
 * const profiler = createFrameProfilerOverlay();   // self-mounts into .viewport
 * // …after building / resetting the core:
 * profiler.attach(core);
 * // …each frame, after core.step():
 * profiler.render();
 * ```
 *
 * It touches the DOM only when called (not at import), so importing the rapier
 * entry in Node stays safe; if there is no `document`, it returns a no-op handle.
 */
import {
  FrameProfilerBuffer,
  FRAME_PHASES,
  drawFrameProfilerChart,
  frameProfilerToCsv,
  type FrameProfilerExport,
} from "./frameProfiler";
import type { CoreProfilerConfig } from "./types";

/** Minimal shape of the destructible core this overlay needs. */
export type ProfilableCore = {
  setProfiler: (config: CoreProfilerConfig | null) => void;
};

export type FrameProfilerOverlayOptions = {
  /** Where to mount. Default: the first `.viewport` element, else `document.body`. */
  mount?: HTMLElement | null;
  /** Rolling window length (frames). Default 180. */
  capacity?: number;
  /** Frame budget line (ms). Default 1000/60. */
  budgetMs?: number;
  /** Panel title. */
  title?: string;
  /** Start with the chart body collapsed (header only). */
  startCollapsed?: boolean;
  /** Start with the A/B "old planner" measurement on. */
  measureOld?: boolean;
  /** Optional context merged into a data export (scenario, config, build info…). */
  getMeta?: () => Record<string, unknown>;
  /** Base filename for exported dumps (no extension). Default "frame-profile". */
  exportName?: string;
};

export type FrameProfilerOverlayHandle = {
  /** The underlying rolling buffer (e.g. to read stats elsewhere). */
  readonly buffer: FrameProfilerBuffer;
  /** Point the overlay at a core: clears the buffer and wires `setProfiler`. */
  attach(core: ProfilableCore | null): void;
  /** Draw the chart + refresh the text. Call once per rendered frame. */
  render(): void;
  /** Toggle the A/B old-planner measurement (re-wires the live core). */
  setMeasureOld(on: boolean): void;
  /** Build a full data dump of the captured window (frames + raw samples +
   *  stats + metadata) — e.g. to POST somewhere or save yourself. */
  exportData(): FrameProfilerExport | null;
  /** Trigger a browser download of the dump as JSON / CSV. */
  downloadJSON(): void;
  downloadCSV(): void;
  /** Show/hide the whole overlay. */
  setVisible(visible: boolean): void;
  /** The root DOM element (for custom placement). */
  readonly el: HTMLElement | null;
  /** Remove the DOM and detach the profiler. */
  destroy(): void;
};

const STYLE_ID = "bss-frame-profiler-style";
const STYLE_CSS = `
.bss-fp{position:absolute;left:12px;right:12px;bottom:12px;max-width:720px;
  background:rgba(8,11,20,.82);border:1px solid rgba(255,255,255,.1);border-radius:10px;
  padding:7px 10px 6px;backdrop-filter:blur(6px);color:#e7ecff;pointer-events:none;
  font-family:ui-monospace,"SF Mono",Menlo,monospace;z-index:20}
.bss-fp-head{display:flex;align-items:center;gap:10px;font-size:11px;margin-bottom:4px}
.bss-fp-title{color:rgba(200,210,240,.75);font-weight:600;white-space:nowrap}
.bss-fp-cause{margin-left:auto;font-size:11px;text-align:right}
.bss-fp-ab{display:inline-flex;align-items:center;gap:4px;color:rgba(220,228,255,.75);
  pointer-events:auto;cursor:pointer;white-space:nowrap}
.bss-fp-ab input{cursor:pointer}
.bss-fp-min{pointer-events:auto;cursor:pointer;background:rgba(255,255,255,.07);color:#cfd8ff;
  border:1px solid rgba(255,255,255,.14);border-radius:5px;width:20px;height:18px;line-height:1;
  font-size:11px;padding:0}
.bss-fp-btn{pointer-events:auto;cursor:pointer;background:rgba(107,140,255,.16);color:#cfd8ff;
  border:1px solid rgba(107,140,255,.4);border-radius:5px;height:18px;line-height:1;
  font-size:10px;padding:0 6px;white-space:nowrap}
.bss-fp-btn:hover{background:rgba(107,140,255,.3)}
.bss-fp-canvas{width:100%;height:108px;display:block;border-radius:6px}
.bss-fp-legend{display:flex;flex-wrap:wrap;gap:3px 12px;margin-top:5px;font-size:10.5px;
  color:rgba(220,228,255,.8)}
.bss-fp-legend .pl{display:inline-flex;align-items:center;gap:4px}
.bss-fp-legend i{width:9px;height:9px;border-radius:2px;display:inline-block}
.bss-fp-legend b{color:#fff;font-weight:600}
.bss-fp-legend .pk{color:rgba(255,170,170,.85);font-size:9.5px}
.bss-fp-stats{margin-top:4px;font-size:10.5px;color:rgba(200,210,240,.7)}
.bss-fp-stats b{color:#fff}
.bss-fp.collapsed .bss-fp-body{display:none}
/* Phones: shrink the panel, drop the secondary chrome, and keep it hugging the
   bottom edge so it stops covering the scene. Pairs with auto-collapse below. */
@media (max-width:768px){
  .bss-fp{left:8px;right:8px;bottom:8px;padding:6px 8px 5px;border-radius:9px}
  .bss-fp-head{flex-wrap:wrap;gap:6px 8px}
  .bss-fp-title{font-size:10.5px}
  .bss-fp-json,.bss-fp-csv{display:none}
  .bss-fp-canvas{height:84px}
  .bss-fp-legend{gap:2px 8px;font-size:9.5px}
  .bss-fp.collapsed .bss-fp-cause{flex-basis:100%;text-align:left;margin-left:0}
}
`;

/** True on phone-sized viewports (matches the demos' 768px breakpoint). */
function isSmallScreen(doc?: Document): boolean {
  try {
    return doc?.defaultView?.matchMedia("(max-width: 768px)").matches ?? false;
  } catch {
    return false;
  }
}

function ensureStyles(doc: Document) {
  if (doc.getElementById(STYLE_ID)) return;
  const style = doc.createElement("style");
  style.id = STYLE_ID;
  style.textContent = STYLE_CSS;
  (doc.head ?? doc.body ?? doc.documentElement).appendChild(style);
}

const NOOP_HANDLE: FrameProfilerOverlayHandle = {
  buffer: new FrameProfilerBuffer(1),
  attach() {},
  render() {},
  setMeasureOld() {},
  exportData: () => null,
  downloadJSON() {},
  downloadCSV() {},
  setVisible() {},
  el: null,
  destroy() {},
};

export function createFrameProfilerOverlay(
  options: FrameProfilerOverlayOptions = {},
): FrameProfilerOverlayHandle {
  const doc: Document | undefined = typeof document !== "undefined" ? document : undefined;
  if (!doc) return NOOP_HANDLE;

  const {
    capacity = 180,
    budgetMs = 1000 / 60,
    title = "⚡ Frame profiler — simulation cost / frame",
    // Default to collapsed on phones, where the panel otherwise buries the scene;
    // an explicit `startCollapsed` always wins.
    startCollapsed = isSmallScreen(doc),
    measureOld = false,
    getMeta,
    exportName = "frame-profile",
  } = options;

  const buffer = new FrameProfilerBuffer(capacity, budgetMs);
  let core: ProfilableCore | null = null;
  let measuringOld = measureOld;

  ensureStyles(doc);

  const root = doc.createElement("div");
  root.className = "bss-fp" + (startCollapsed ? " collapsed" : "");
  root.innerHTML = `
    <div class="bss-fp-head">
      <span class="bss-fp-title"></span>
      <label class="bss-fp-ab"><input type="checkbox" /> A/B old planner</label>
      <button class="bss-fp-btn bss-fp-json" title="Download full data dump (JSON: frames + raw samples + stats)">⬇ JSON</button>
      <button class="bss-fp-btn bss-fp-csv" title="Download per-frame breakdown (CSV)">⬇ CSV</button>
      <button class="bss-fp-min" title="Collapse / expand">▾</button>
      <span class="bss-fp-cause"></span>
    </div>
    <div class="bss-fp-body">
      <canvas class="bss-fp-canvas"></canvas>
      <div class="bss-fp-legend"></div>
      <div class="bss-fp-stats"></div>
    </div>`;
  (root.querySelector(".bss-fp-title") as HTMLElement).textContent = title;

  const canvas = root.querySelector(".bss-fp-canvas") as HTMLCanvasElement;
  const causeEl = root.querySelector(".bss-fp-cause") as HTMLElement;
  const legendEl = root.querySelector(".bss-fp-legend") as HTMLElement;
  const statsEl = root.querySelector(".bss-fp-stats") as HTMLElement;
  const abInput = root.querySelector(".bss-fp-ab input") as HTMLInputElement;
  const minBtn = root.querySelector(".bss-fp-min") as HTMLButtonElement;
  const jsonBtn = root.querySelector(".bss-fp-json") as HTMLButtonElement;
  const csvBtn = root.querySelector(".bss-fp-csv") as HTMLButtonElement;
  const ctx = canvas.getContext("2d");

  // Reflect the initial collapsed state in the caret glyph.
  minBtn.textContent = root.classList.contains("collapsed") ? "▴" : "▾";

  abInput.checked = measuringOld;
  abInput.addEventListener("change", () => setMeasureOld(abInput.checked));
  minBtn.addEventListener("click", () => {
    root.classList.toggle("collapsed");
    minBtn.textContent = root.classList.contains("collapsed") ? "▴" : "▾";
  });
  jsonBtn.addEventListener("click", () => downloadJSON());
  csvBtn.addEventListener("click", () => downloadCSV());

  const mount = options.mount ?? doc.querySelector(".viewport") ?? doc.body;
  // The viewport is usually position:relative; ensure absolute placement anchors to it.
  if (mount && mount !== doc.body && getComputedPosition(mount) === "static") {
    (mount as HTMLElement).style.position = "relative";
  }
  mount?.appendChild(root);

  function getComputedPosition(el: Element): string {
    try {
      return (doc!.defaultView?.getComputedStyle(el).position) ?? "static";
    } catch {
      return "static";
    }
  }

  function wireProfiler() {
    core?.setProfiler({
      enabled: true,
      onSample: (s) => buffer.push(s),
      measureReferencePlanner: measuringOld,
    });
  }

  function attach(next: ProfilableCore | null) {
    core = next;
    buffer.clear();
    wireProfiler();
  }

  function setMeasureOld(on: boolean) {
    measuringOld = on;
    if (abInput.checked !== on) abInput.checked = on;
    wireProfiler(); // re-apply the flag to the live core
  }

  function setVisible(visible: boolean) {
    root.style.display = visible ? "" : "none";
  }

  function exportData(): FrameProfilerExport {
    let meta: Record<string, unknown> | undefined;
    try {
      meta = getMeta?.();
    } catch {
      /* metadata is best-effort */
    }
    if (typeof navigator !== "undefined" && navigator.userAgent) {
      meta = { userAgent: navigator.userAgent, ...meta };
    }
    return buffer.export(meta);
  }

  function download(filename: string, mime: string, text: string) {
    try {
      const blob = new Blob([text], { type: mime });
      const url = URL.createObjectURL(blob);
      const a = doc!.createElement("a");
      a.href = url;
      a.download = filename;
      (doc!.body ?? doc!.documentElement).appendChild(a);
      a.click();
      a.remove();
      // Revoke on the next tick so the download has started.
      setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch {
      /* download unsupported in this environment — no-op */
    }
  }

  function stamp(): string {
    return new Date().toISOString().replace(/[:.]/g, "-");
  }

  function downloadJSON() {
    download(`${exportName}-${stamp()}.json`, "application/json", JSON.stringify(exportData(), null, 2));
  }

  function downloadCSV() {
    download(`${exportName}-${stamp()}.csv`, "text/csv", frameProfilerToCsv(exportData()));
  }

  function render() {
    if (root.style.display === "none" || root.classList.contains("collapsed") || !ctx) {
      // Still refresh the cause callout cheaply so the header stays informative.
      updateCause();
      return;
    }
    const dpr = Math.min((typeof devicePixelRatio !== "undefined" ? devicePixelRatio : 1) || 1, 2);
    const cssW = canvas.clientWidth || 360;
    const cssH = canvas.clientHeight || 108;
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const frames = buffer.frames();
    drawFrameProfilerChart(ctx, cssW, cssH, frames, { showProjectedOld: measuringOld, budgetMs });

    const stats = buffer.stats();
    updateCause();

    // Legend: each phase's window-average and (dim) its window peak, sorted by
    // peak so the biggest spike contributors lead.
    legendEl.innerHTML = FRAME_PHASES.map((p) => ({ p, ms: stats.perPhaseMean[p.key], peak: stats.perPhasePeak[p.key] }))
      .filter((x) => x.peak > 0.01)
      .sort((a, b) => b.peak - a.peak)
      .map(
        ({ p, ms, peak }) =>
          `<span class="pl"><i style="background:${p.color}"></i>${p.label} <b>${ms.toFixed(2)}</b><span class="pk" title="window peak">▲${peak.toFixed(1)}</span></span>`,
      )
      .join("");

    const latest = buffer.latest();
    let s =
      `sim <b>${stats.meanMs.toFixed(2)} ms</b> avg · p95 ${stats.p95Ms.toFixed(1)} · ` +
      `${stats.spikeCount} spike${stats.spikeCount === 1 ? "" : "s"}>${budgetMs.toFixed(1)}ms`;
    if (latest) s += ` · resim ${latest.resimPasses} · bodies ${latest.rigidBodies}`;
    // Attributed window peak — the worst frame's total, its cause, and how long
    // ago, so a spike that has scrolled off-screen stays explained.
    const worst = stats.worst;
    if (worst && worst.totalMs > 0) {
      const wdef = FRAME_PHASES.find((p) => p.key === worst.dominant);
      const ago = worst.timestamp > 0 ? `, ${((Date.now() - worst.timestamp) / 1000).toFixed(1)}s ago` : "";
      s += ` · <span style="color:${wdef?.color}">peak <b>${worst.totalMs.toFixed(1)} ms</b> ${wdef?.label}${ago}</span>`;
    }
    if (measuringOld) {
      let worstOld = 0;
      let worstNew = 0;
      for (const f of frames) {
        if (f.projectedOldTotalMs && f.projectedOldTotalMs > worstOld) {
          worstOld = f.projectedOldTotalMs;
          worstNew = f.totalMs;
        }
      }
      if (worstOld > 0) {
        s += ` · <span style="color:#ffb454">OLD planner peak ≈ ${worstOld.toFixed(1)} ms (now ${worstNew.toFixed(1)} ms)</span>`;
      }
    }
    statsEl.innerHTML = s;
  }

  function updateCause() {
    const latest = buffer.latest();
    if (!latest) {
      causeEl.textContent = "";
      return;
    }
    const def = FRAME_PHASES.find((p) => p.key === latest.dominant);
    causeEl.innerHTML =
      latest.totalMs > budgetMs
        ? `spike: <b style="color:${def?.color}">${def?.label}</b> (${latest.totalMs.toFixed(1)} ms)`
        : `<span style="opacity:.55">within budget (${latest.totalMs.toFixed(1)} ms)</span>`;
  }

  function destroy() {
    try {
      core?.setProfiler(null);
    } catch {
      /* ignore */
    }
    core = null;
    root.remove();
  }

  return {
    buffer,
    attach,
    render,
    setMeasureOld,
    exportData,
    downloadJSON,
    downloadCSV,
    setVisible,
    el: root,
    destroy,
  };
}
