/**
 * Tests for the reusable frame-profiler overlay widget. Runs against a stub DOM
 * (no jsdom) so it exercises the real widget code path: DOM construction, the
 * `core.setProfiler` wiring (incl. the A/B re-wire), and a render pass that
 * updates the cause/legend/stats from a fabricated sample stream — without a
 * browser or WASM.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createFrameProfilerOverlay } from "../rapier/frameProfilerOverlay";
import type { CoreProfilerSample } from "../rapier/types";

// ── Minimal DOM stub ─────────────────────────────────────────────────────────

function makeCtx() {
  return new Proxy(
    {},
    {
      get(_t, p: string) {
        if (p === "measureText") return () => ({ width: 8 });
        return () => {};
      },
      set() {
        return true;
      },
    },
  ) as unknown as CanvasRenderingContext2D;
}

class StubEl {
  className = "";
  id = "";
  textContent = "";
  style: Record<string, string> = {};
  children: StubEl[] = [];
  tagName: string;
  checked = false;
  type = "";
  private listeners: Record<string, Array<() => void>> = {};
  private _html = "";
  width = 0;
  height = 0;
  clientWidth = 360;
  clientHeight = 108;

  constructor(tag: string) {
    this.tagName = tag.toLowerCase();
  }
  set innerHTML(v: string) {
    this._html = v;
    this.children = parseControls(v);
  }
  get innerHTML() {
    return this._html;
  }
  appendChild(c: StubEl) {
    this.children.push(c);
    return c;
  }
  remove() {}
  addEventListener(type: string, cb: () => void) {
    (this.listeners[type] ??= []).push(cb);
  }
  dispatch(type: string) {
    (this.listeners[type] ?? []).forEach((cb) => cb());
  }
  classList = {
    _set: new Set<string>(),
    add: (c: string) => this.classList._set.add(c),
    remove: (c: string) => this.classList._set.delete(c),
    toggle: (c: string) => (this.classList._set.has(c) ? (this.classList._set.delete(c), false) : (this.classList._set.add(c), true)),
    contains: (c: string) => this.classList._set.has(c),
  };
  getContext() {
    return makeCtx();
  }
  // querySelector resolves the handful of class selectors the widget uses.
  querySelector(sel: string): StubEl | null {
    return this.findByClass(sel.replace(/^\./, "").split(" ")[0]) ?? new StubEl("div");
  }
  private findByClass(cls: string): StubEl | null {
    if (this.className.split(" ").includes(cls)) return this;
    for (const c of this.children) {
      const hit = c.findByClass(cls);
      if (hit) return hit;
    }
    return null;
  }
}

// The widget sets innerHTML once with a known template; create the sub-elements
// it then querySelector()s for, so the references are stable and inspectable.
function parseControls(html: string): StubEl[] {
  if (!html.includes("bss-fp-canvas")) return [];
  const mk = (cls: string, tag = "div") => {
    const e = new StubEl(tag);
    e.className = cls;
    return e;
  };
  const abLabel = mk("bss-fp-ab", "label");
  const abInput = new StubEl("input");
  abInput.type = "checkbox";
  abLabel.children.push(abInput);
  return [mk("bss-fp-title", "span"), abLabel, mk("bss-fp-min", "button"), mk("bss-fp-cause", "span"), mk("bss-fp-canvas", "canvas"), mk("bss-fp-legend"), mk("bss-fp-stats")];
}

function installStubDom() {
  const head = new StubEl("head");
  const body = new StubEl("body");
  const created: StubEl[] = [];
  const doc = {
    head,
    body,
    documentElement: new StubEl("html"),
    defaultView: { getComputedStyle: () => ({ position: "relative" }) },
    getElementById: (id: string) => created.find((e) => e.id === id) ?? null,
    querySelector: (_sel: string) => null, // no .viewport -> mounts to body
    createElement: (tag: string) => {
      const e = new StubEl(tag);
      created.push(e);
      return e;
    },
  };
  (globalThis as any).document = doc;
  (globalThis as any).devicePixelRatio = 1;
  return { doc, body };
}

function sample(fields: Partial<Record<keyof CoreProfilerSample, number>>): CoreProfilerSample {
  return fields as unknown as CoreProfilerSample;
}

afterEach(() => {
  delete (globalThis as any).document;
  delete (globalThis as any).devicePixelRatio;
});

describe("createFrameProfilerOverlay", () => {
  it("returns a no-op handle when there is no document (Node-safe)", () => {
    expect((globalThis as any).document).toBeUndefined();
    const h = createFrameProfilerOverlay();
    expect(h.el).toBeNull();
    expect(() => {
      h.attach({ setProfiler() {} });
      h.render();
      h.destroy();
    }).not.toThrow();
  });

  it("mounts, wires core.setProfiler on attach, and feeds the buffer", () => {
    installStubDom();
    const overlay = createFrameProfilerOverlay();
    expect(overlay.el).not.toBeNull();

    let captured: any = null;
    const core = { setProfiler: vi.fn((cfg: any) => (captured = cfg)) };
    overlay.attach(core as any);

    expect(core.setProfiler).toHaveBeenCalledTimes(1);
    expect(captured.enabled).toBe(true);
    expect(captured.measureReferencePlanner).toBe(false);

    // Drive samples through the wired callback -> the overlay's buffer fills.
    captured.onSample(sample({ frameIndex: 0, totalMs: 5, rapierStepMs: 5 }));
    captured.onSample(sample({ frameIndex: 1, totalMs: 40, fractureMs: 36, rapierStepMs: 4 }));
    expect(overlay.buffer.frames().length).toBe(2);
    expect(overlay.buffer.latest()?.dominant).toBe("fracture");
  });

  it("re-wires the live core with the A/B flag via setMeasureOld", () => {
    installStubDom();
    const overlay = createFrameProfilerOverlay();
    const core = { setProfiler: vi.fn() };
    overlay.attach(core as any);
    overlay.setMeasureOld(true);

    expect(core.setProfiler).toHaveBeenCalledTimes(2); // attach + re-wire
    const last = (core.setProfiler as any).mock.calls.at(-1)[0];
    expect(last.measureReferencePlanner).toBe(true);
  });

  it("render() updates cause/legend/stats from the sample stream without throwing", () => {
    installStubDom();
    const overlay = createFrameProfilerOverlay();
    let cfg: any = null;
    overlay.attach({ setProfiler: (c: any) => (cfg = c) } as any);
    for (let i = 0; i < 30; i += 1) cfg.onSample(sample({ frameIndex: i, totalMs: 6, rapierStepMs: 4, solverUpdateMs: 2 }));
    cfg.onSample(sample({ frameIndex: 99, totalMs: 50, fractureMs: 44, rapierStepMs: 6 }));

    expect(() => overlay.render()).not.toThrow();

    const root = overlay.el as any;
    const cause = root.querySelector(".bss-fp-cause");
    const legend = root.querySelector(".bss-fp-legend");
    const stats = root.querySelector(".bss-fp-stats");
    expect(cause.innerHTML).toContain("spike"); // last frame was a 50ms spike
    expect(legend.innerHTML).toContain("Physics step");
    expect(stats.innerHTML).toContain("sim");
  });

  it("destroy() detaches the profiler", () => {
    installStubDom();
    const overlay = createFrameProfilerOverlay();
    const core = { setProfiler: vi.fn() };
    overlay.attach(core as any);
    overlay.destroy();
    const last = (core.setProfiler as any).mock.calls.at(-1)[0];
    expect(last).toBeNull(); // setProfiler(null) on destroy
  });
});
