/**
 * Self-contained **record / stop / download** overlay for the destructible core.
 *
 * Mirrors `createFrameProfilerOverlay`: it builds its own DOM (a small floating
 * panel with a ● Record button, a live readout, and a ⬇ Download button), injects
 * its styles once, owns a {@link SessionRecorderHandle}, and only touches the DOM
 * when called (importing the rapier entry in Node stays safe — it returns a no-op
 * handle when there is no `document`).
 *
 * A demo only needs:
 *
 * ```ts
 * const recorder = createRecordingOverlay({ getContext: () => ({ scenario, coreConfig, meta }) });
 * // …after building / rebuilding the core:
 * recorder.attach(core);
 * // …each frame (optional — only refreshes the readout text):
 * recorder.render();
 * ```
 *
 * Capture is automatic once you press ● Record: the recorder wraps the core's
 * `step` / input methods, so projectiles, forces, gravity changes, fractures and
 * every body's per-frame kinematics are recorded without any render-loop changes.
 * ⬇ Download produces a single gzipped JSON bundle (`*.sim.json.gz`) suitable for
 * attaching to a bug report or analysing offline.
 */
import {
  createSessionRecorder,
  gzipJson,
  type RecordableCore,
  type SessionRecorderContext,
  type SessionRecorderOptions,
  type SimRecordingExport,
} from './sessionRecorder';

export type RecordingOverlayOptions = {
  /** Where to mount. Default: the first `.viewport` element, else `document.body`. */
  mount?: HTMLElement | null;
  /** Panel title. */
  title?: string;
  /** Base filename for the downloaded bundle (no extension). Default "sim-recording". */
  exportName?: string;
  /** Recorder behaviour (frame cap, etc). */
  recorder?: SessionRecorderOptions;
  /**
   * Resolve the recording context (scenario / core config / metadata) lazily at
   * the moment recording starts, so the freshest scenario + UI config is captured
   * even after rebuilds. Re-evaluated on every ● Record press.
   */
  getContext?: () => SessionRecorderContext;
  /** Optional frame-profiler dump to embed in the bundle (e.g. a profiler
   *  overlay's `exportData`). Returns null/undefined to omit. */
  getProfilerExport?: () => unknown;
  /** Start with the panel collapsed to just the header. */
  startCollapsed?: boolean;
};

export type RecordingOverlayHandle = {
  /** Point the overlay at a core (wraps its step/input methods). Pass null to
   *  detach (e.g. before a rebuild). An optional context (scenario / core config /
   *  metadata) can be supplied here — useful when those are in scope at the call
   *  site; it takes effect unless `getContext` provides a fresher value at record
   *  time. */
  attach(core: RecordableCore | null, ctx?: SessionRecorderContext): void;
  /** Programmatic record/stop (same as the button). */
  start(): void;
  stop(): void;
  isRecording(): boolean;
  /** Refresh the live readout. Cheap; call once per frame if you want a smooth
   *  elapsed/size counter (not required for capture). */
  render(): void;
  /** Build the bundle without downloading (e.g. to POST it somewhere). */
  exportData(): SimRecordingExport | null;
  /** Trigger a browser download of the gzipped bundle. */
  download(): Promise<void>;
  setVisible(visible: boolean): void;
  readonly el: HTMLElement | null;
  destroy(): void;
};

const STYLE_ID = 'bss-recording-style';
const STYLE_CSS = `
.bss-rec{position:absolute;right:12px;top:12px;min-width:230px;
  background:rgba(8,11,20,.82);border:1px solid rgba(255,255,255,.1);border-radius:10px;
  padding:7px 10px 8px;backdrop-filter:blur(6px);color:#e7ecff;
  font-family:ui-monospace,"SF Mono",Menlo,monospace;z-index:21}
.bss-rec-head{display:flex;align-items:center;gap:8px;font-size:11px;margin-bottom:6px}
.bss-rec-title{color:rgba(200,210,240,.75);font-weight:600;white-space:nowrap}
.bss-rec-min{margin-left:auto;cursor:pointer;background:rgba(255,255,255,.07);color:#cfd8ff;
  border:1px solid rgba(255,255,255,.14);border-radius:5px;width:20px;height:18px;line-height:1;
  font-size:11px;padding:0}
.bss-rec-row{display:flex;align-items:center;gap:8px}
.bss-rec-btn{cursor:pointer;border-radius:6px;height:26px;line-height:1;font-size:12px;
  padding:0 12px;white-space:nowrap;font-weight:600;display:inline-flex;align-items:center;gap:6px}
.bss-rec-record{background:rgba(255,90,110,.16);color:#ff97a6;border:1px solid rgba(255,90,110,.5)}
.bss-rec-record:hover{background:rgba(255,90,110,.3)}
.bss-rec-record.active{background:rgba(255,60,80,.85);color:#fff;border-color:rgba(255,60,80,.9);
  animation:bss-rec-pulse 1.1s ease-in-out infinite}
.bss-rec-dl{background:rgba(107,140,255,.16);color:#cfd8ff;border:1px solid rgba(107,140,255,.4)}
.bss-rec-dl:hover{background:rgba(107,140,255,.3)}
.bss-rec-btn:disabled{opacity:.4;cursor:default}
.bss-rec-dot{width:9px;height:9px;border-radius:50%;background:currentColor;display:inline-block}
.bss-rec-stat{margin-top:6px;font-size:10.5px;color:rgba(200,210,240,.72);line-height:1.45}
.bss-rec-stat b{color:#fff;font-weight:600}
.bss-rec.collapsed .bss-rec-body{display:none}
@keyframes bss-rec-pulse{0%,100%{box-shadow:0 0 0 0 rgba(255,60,80,.5)}50%{box-shadow:0 0 0 5px rgba(255,60,80,0)}}
`;

function ensureStyles(doc: Document) {
  if (doc.getElementById(STYLE_ID)) return;
  const style = doc.createElement('style');
  style.id = STYLE_ID;
  style.textContent = STYLE_CSS;
  (doc.head ?? doc.body ?? doc.documentElement).appendChild(style);
}

const NOOP_HANDLE: RecordingOverlayHandle = {
  attach() {},
  start() {},
  stop() {},
  isRecording: () => false,
  render() {},
  exportData: () => null,
  async download() {},
  setVisible() {},
  el: null,
  destroy() {},
};

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return m > 0 ? `${m}:${sec.toFixed(1).padStart(4, '0')}` : `${sec.toFixed(1)}s`;
}

export function createRecordingOverlay(options: RecordingOverlayOptions = {}): RecordingOverlayHandle {
  const doc: Document | undefined = typeof document !== 'undefined' ? document : undefined;
  if (!doc) return NOOP_HANDLE;

  const {
    title = '⏺ Session recorder',
    exportName = 'sim-recording',
    startCollapsed = false,
    getContext,
    getProfilerExport,
  } = options;

  const recorder = createSessionRecorder({
    ...options.recorder,
    onAutoStop: () => {
      options.recorder?.onAutoStop?.();
      syncButtons();
      render();
    },
  });
  if (getProfilerExport) recorder.setProfilerExport(getProfilerExport);

  let hasRecording = false;

  ensureStyles(doc);

  const root = doc.createElement('div');
  root.className = 'bss-rec' + (startCollapsed ? ' collapsed' : '');
  root.innerHTML = `
    <div class="bss-rec-head">
      <span class="bss-rec-title"></span>
      <button class="bss-rec-min" title="Collapse / expand">▾</button>
    </div>
    <div class="bss-rec-body">
      <div class="bss-rec-row">
        <button class="bss-rec-btn bss-rec-record"><span class="bss-rec-dot"></span><span class="bss-rec-label">Record</span></button>
        <button class="bss-rec-btn bss-rec-dl" disabled title="Download recording as gzipped JSON">⬇ Save</button>
      </div>
      <div class="bss-rec-stat"></div>
    </div>`;
  (root.querySelector('.bss-rec-title') as HTMLElement).textContent = title;

  const recordBtn = root.querySelector('.bss-rec-record') as HTMLButtonElement;
  const recordLabel = root.querySelector('.bss-rec-label') as HTMLElement;
  const dlBtn = root.querySelector('.bss-rec-dl') as HTMLButtonElement;
  const minBtn = root.querySelector('.bss-rec-min') as HTMLButtonElement;
  const statEl = root.querySelector('.bss-rec-stat') as HTMLElement;

  recordBtn.addEventListener('click', () => {
    if (recorder.isRecording()) stop();
    else start();
  });
  dlBtn.addEventListener('click', () => void download());
  minBtn.addEventListener('click', () => {
    root.classList.toggle('collapsed');
    minBtn.textContent = root.classList.contains('collapsed') ? '▴' : '▾';
  });

  const mount = options.mount ?? doc.querySelector('.viewport') ?? doc.body;
  if (mount && mount !== doc.body && getComputedPosition(mount) === 'static') {
    (mount as HTMLElement).style.position = 'relative';
  }
  mount?.appendChild(root);

  function getComputedPosition(el: Element): string {
    try {
      return doc!.defaultView?.getComputedStyle(el).position ?? 'static';
    } catch {
      return 'static';
    }
  }

  let currentCore: RecordableCore | null = null;
  let pinnedCtx: SessionRecorderContext | undefined;

  function resolveContext(): SessionRecorderContext | undefined {
    try {
      // A live `getContext` wins (freshest); otherwise fall back to whatever was
      // pinned at the last attach() call.
      return getContext?.() ?? pinnedCtx;
    } catch {
      return pinnedCtx; // context is best-effort
    }
  }

  function attach(core: RecordableCore | null, ctx?: SessionRecorderContext) {
    currentCore = core;
    pinnedCtx = ctx;
    recorder.attach(core, resolveContext());
    hasRecording = false;
    syncButtons();
    render();
  }

  function start() {
    if (!recorder.isAttached()) return;
    // Re-attach with fresh context so the newest scenario / UI config / meta is
    // captured even if it changed since the last attach. Re-attaching also resets
    // the buffers, which is exactly what we want before a new take.
    if (currentCore) recorder.attach(currentCore, resolveContext());
    recorder.start();
    hasRecording = true;
    syncButtons();
    render();
  }

  function stop() {
    recorder.stop();
    syncButtons();
    render();
  }

  function syncButtons() {
    const rec = recorder.isRecording();
    recordBtn.classList.toggle('active', rec);
    recordLabel.textContent = rec ? 'Stop' : 'Record';
    recordBtn.title = rec ? 'Stop recording' : 'Start recording the simulation';
    dlBtn.disabled = rec || !hasRecording || recorder.frameCount() === 0;
  }

  function render() {
    const rec = recorder.isRecording();
    const frames = recorder.frameCount();
    const bytes = recorder.estimatedBytes();
    if (rec) {
      statEl.innerHTML = `<b style="color:#ff8b98">● REC</b> · <b>${frames}</b> frames · ~<b>${fmtBytes(bytes)}</b> raw`;
    } else if (hasRecording && frames > 0) {
      statEl.innerHTML = `captured <b>${frames}</b> frames · ~<b>${fmtBytes(bytes)}</b> raw · press ⬇ Save`;
    } else if (!recorder.isAttached()) {
      statEl.innerHTML = `<span style="opacity:.6">no core attached</span>`;
    } else {
      statEl.innerHTML = `<span style="opacity:.6">press ● Record to capture a session</span>`;
    }
  }

  function exportData(): SimRecordingExport | null {
    return recorder.export();
  }

  function stamp(): string {
    return new Date().toISOString().replace(/[:.]/g, '-');
  }

  async function download() {
    const data = recorder.export();
    if (!data) return;
    dlBtn.disabled = true;
    const prevLabel = dlBtn.textContent;
    dlBtn.textContent = '… packing';
    try {
      const { blob, gzipped } = await gzipJson(data);
      const ext = gzipped ? 'sim.json.gz' : 'sim.json';
      triggerDownload(blob, `${exportName}-${stamp()}.${ext}`);
    } catch {
      /* download unsupported — no-op */
    } finally {
      dlBtn.textContent = prevLabel;
      syncButtons();
    }
  }

  function triggerDownload(blob: Blob, filename: string) {
    try {
      const url = URL.createObjectURL(blob);
      const a = doc!.createElement('a');
      a.href = url;
      a.download = filename;
      (doc!.body ?? doc!.documentElement).appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch {
      /* ignore */
    }
  }

  function setVisible(visible: boolean) {
    root.style.display = visible ? '' : 'none';
  }

  function destroy() {
    try {
      recorder.detach();
    } catch {
      /* ignore */
    }
    root.remove();
  }

  syncButtons();
  render();

  return {
    attach,
    start,
    stop,
    isRecording: () => recorder.isRecording(),
    render,
    exportData,
    download,
    setVisible,
    el: root,
    destroy,
  };
}
