/**
 * Smoke tests for the record/stop/download overlay against a stub DOM (no jsdom,
 * no WASM, no browser): construction, the Record button toggling capture on the
 * underlying recorder, attach passing context, and exportData producing a bundle.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { createRecordingOverlay } from '../rapier/recordingOverlay';
import type { ChunkData, RecordableCore } from '../rapier/sessionRecorder';

class StubEl {
  className = '';
  id = '';
  textContent = '';
  title = '';
  disabled = false;
  style: Record<string, string> = {};
  children: StubEl[] = [];
  tagName: string;
  private listeners: Record<string, Array<() => void>> = {};
  private _html = '';
  href = '';
  download = '';

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
  click() {}
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
    toggle: (c: string) =>
      this.classList._set.has(c)
        ? (this.classList._set.delete(c), false)
        : (this.classList._set.add(c), true),
    contains: (c: string) => this.classList._set.has(c),
  };
  querySelector(sel: string): StubEl | null {
    return this.findByClass(sel.replace(/^\./, '').split(' ')[0]) ?? new StubEl('div');
  }
  private findByClass(cls: string): StubEl | null {
    if (this.className.split(' ').includes(cls)) return this;
    for (const c of this.children) {
      const hit = c.findByClass(cls);
      if (hit) return hit;
    }
    return null;
  }
}

function parseControls(html: string): StubEl[] {
  if (!html.includes('bss-rec-record')) return [];
  const mk = (cls: string, tag = 'div') => {
    const e = new StubEl(tag);
    e.className = cls;
    return e;
  };
  const recordBtn = mk('bss-rec-btn bss-rec-record', 'button');
  recordBtn.children.push(mk('bss-rec-dot', 'span'), mk('bss-rec-label', 'span'));
  return [
    mk('bss-rec-title', 'span'),
    mk('bss-rec-min', 'button'),
    recordBtn,
    mk('bss-rec-btn bss-rec-dl', 'button'),
    mk('bss-rec-stat'),
  ];
}

function installStubDom() {
  const head = new StubEl('head');
  const body = new StubEl('body');
  const created: StubEl[] = [];
  const doc = {
    head,
    body,
    documentElement: new StubEl('html'),
    defaultView: { getComputedStyle: () => ({ position: 'relative' }) },
    getElementById: (id: string) => created.find((e) => e.id === id) ?? null,
    querySelector: (_sel: string) => null,
    createElement: (tag: string) => {
      const e = new StubEl(tag);
      created.push(e);
      return e;
    },
  };
  (globalThis as any).document = doc;
  return { doc, body, created };
}

function newChunk(nodeIndex: number, bodyHandle: number | null): ChunkData {
  return {
    nodeIndex,
    size: { x: 1, y: 1, z: 1 },
    isSupport: false,
    baseLocalOffset: { x: 0, y: 0, z: 0 },
    localOffset: { x: 0, y: 0, z: 0 },
    colliderHandle: null,
    bodyHandle,
    active: true,
    detached: false,
  };
}

function makeCore(): { core: RecordableCore } {
  const chunks: ChunkData[] = [newChunk(0, 1)];
  const core: RecordableCore = {
    world: {
      forEachRigidBody: (cb) =>
        cb({
          handle: 1,
          isFixed: () => false,
          translation: () => ({ x: 0, y: 0, z: 0 }),
          rotation: () => ({ x: 0, y: 0, z: 0, w: 1 }),
          linvel: () => ({ x: 0, y: 0, z: 0 }),
          angvel: () => ({ x: 0, y: 0, z: 0 }),
        }),
    },
    chunks,
    getActiveBondsCount: () => 10,
    getRigidBodyCount: () => 1,
    projectiles: { length: 0 },
    step: () => {},
    enqueueProjectile: () => {},
    applyExternalForce: () => {},
    setGravity: () => {},
  };
  return { core };
}

afterEach(() => {
  delete (globalThis as any).document;
});

describe('createRecordingOverlay', () => {
  it('returns a no-op handle without a document', () => {
    const h = createRecordingOverlay();
    expect(h.el).toBeNull();
    expect(() => {
      h.attach(null);
      h.start();
      h.render();
      h.destroy();
    }).not.toThrow();
    expect(h.isRecording()).toBe(false);
  });

  it('the Record button toggles capture and captures a session', () => {
    installStubDom();
    let metaProvided = false;
    const overlay = createRecordingOverlay({
      getContext: () => {
        metaProvided = true;
        return { meta: { page: 'test' } };
      },
    });
    expect(overlay.el).not.toBeNull();

    const { core } = makeCore();
    overlay.attach(core);
    expect(metaProvided).toBe(true);

    const root = overlay.el as any;
    const recordBtn = root.querySelector('.bss-rec-record');

    recordBtn.dispatch('click'); // start
    expect(overlay.isRecording()).toBe(true);

    core.step(1 / 60);
    core.step(1 / 60);

    recordBtn.dispatch('click'); // stop
    expect(overlay.isRecording()).toBe(false);

    const data = overlay.exportData();
    expect(data).not.toBeNull();
    expect(data!.durationFrames).toBe(2);
    expect(data!.meta).toMatchObject({ page: 'test' });
  });

  it('detaches the core on destroy', () => {
    installStubDom();
    const overlay = createRecordingOverlay();
    const { core } = makeCore();
    const origStep = core.step;
    overlay.attach(core);
    expect(core.step).not.toBe(origStep);
    overlay.destroy();
    expect(core.step).toBe(origStep);
  });
});
