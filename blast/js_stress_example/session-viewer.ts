/**
 * Session Recording Viewer
 * ========================
 *
 * A standalone web page that **imports** a `*.sim.json[.gz]` bundle produced by
 * the in-app ⏺ Session recorder (see `SESSION_RECORDER.md`) and **plays it back
 * visually** in a Three.js scene — no physics engine, no WASM, no re-simulation.
 * It decodes the recorded per-frame body kinematics + the baseline node→body map
 * + the topology event stream and reconstructs each chunk's world transform every
 * frame, so you can scrub through a destruction, watch fragments split off, and
 * read the live scalars (bonds, bodies, projectiles, per-frame cost).
 *
 * Reconstruction model
 * --------------------
 * The recording stores, per frame, every *dynamic rigid body's* world transform
 * (position + quaternion), keyed by a dense session body-id. It also stores, once,
 * which body each chunk belonged to at the start (`initialBodyByNode`) and the
 * chunk→node map (`nodeIndices`), plus a `migrate`/`detach`/`destroy` event stream.
 *
 * A chunk is rigidly attached to its current body, so its world transform is
 * `bodyTransform ∘ localOffset`, where `localOffset` is fixed for as long as the
 * chunk stays on that body. We precompute, per chunk, a list of *segments*
 * `{ startFrame, body, localOffset }`: the first comes from the chunk's rest
 * centroid relative to its initial body at frame 0, and each `migrate` event opens
 * a new segment (recomputing the offset so the chunk stays put across the split).
 * Rendering a frame is then: find each chunk's active segment, look up its body's
 * recorded transform that frame, compose, and write an instanced matrix.
 *
 * If a recording carries no scenario (older/geometry-stripped bundles) we fall back
 * to a per-body view: every recorded body is drawn as a unit box at its transform.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  decodeSimRecording,
  decodeTyped,
  resolveScenarioNodeSize,
} from 'blast-stress-solver/rapier';

// ── Small typed helpers ───────────────────────────────────────────────────────
type Vec3 = { x: number; y: number; z: number };

type Segment = {
  /** First frame this segment is in effect. */
  f: number;
  /** Session body-id this chunk is attached to (−1 = static / no body). */
  body: number;
  /** Chunk offset within `body` (or world transform when body === −1). */
  lp: THREE.Vector3;
  lq: THREE.Quaternion;
};

type ChunkModel = {
  node: number;
  size: THREE.Vector3;
  segments: Segment[];
  /** First frame the chunk is destroyed (hidden from here on). */
  destroyFrame: number;
};

type ViewerModel = {
  data: any;
  dec: ReturnType<typeof decodeSimRecording>;
  frames: number;
  /** Per-frame chunk reconstruction (empty in per-body fallback mode). */
  chunks: ChunkModel[];
  perBodyMode: boolean;
  /** Capacity for the box instanced mesh. */
  boxCapacity: number;
  /** Capacity for the extras (projectiles / unowned bodies) instanced mesh. */
  extraCapacity: number;
  bounds: THREE.Box3;
};

// ── DOM ───────────────────────────────────────────────────────────────────────
const canvas = document.getElementById('viewer-canvas') as HTMLCanvasElement;
const viewport = document.querySelector('.viewport') as HTMLElement;
const dropOverlay = document.getElementById('drop-overlay') as HTMLElement;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const transport = document.getElementById('transport') as HTMLElement;
const playBtn = document.getElementById('play-btn') as HTMLButtonElement;
const scrub = document.getElementById('scrub') as HTMLInputElement;
const timeLabel = document.getElementById('time-label') as HTMLElement;
const speedSel = document.getElementById('speed-sel') as HTMLSelectElement;
const loopChk = document.getElementById('loop-chk') as HTMLInputElement;
const eventStrip = document.getElementById('event-strip') as HTMLCanvasElement;
const metaEl = document.getElementById('meta') as HTMLElement;
const statsEl = document.getElementById('stats') as HTMLElement;
const errorEl = document.getElementById('load-error') as HTMLElement;

// ── Three.js scene ──────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setClearColor(0x05070c, 1);
renderer.shadowMap.enabled = false;

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x05070c, 60, 320);

const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 2000);
camera.position.set(16, 12, 24);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.HemisphereLight(0xbfd0ff, 0x202838, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(30, 50, 20);
scene.add(sun);
const fill = new THREE.DirectionalLight(0x8fb0ff, 0.5);
fill.position.set(-25, 20, -15);
scene.add(fill);

const grid = new THREE.GridHelper(200, 80, 0x2a3550, 0x141c30);
(grid.material as THREE.Material).transparent = true;
(grid.material as THREE.Material).opacity = 0.55;
scene.add(grid);

// Instanced meshes (allocated per-recording in setupSceneMeshes).
const boxGeo = new THREE.BoxGeometry(1, 1, 1);
const sphereGeo = new THREE.SphereGeometry(1, 16, 12);
const chunkMat = new THREE.MeshStandardMaterial({ metalness: 0.05, roughness: 0.85 });
const extraMat = new THREE.MeshStandardMaterial({ color: 0xff5566, metalness: 0.1, roughness: 0.6, emissive: 0x441018 });
let chunkMesh: THREE.InstancedMesh | null = null;
let extraMesh: THREE.InstancedMesh | null = null;

// ── Reusable math scratch (no per-frame allocation) ─────────────────────────────
const _bodyPos = new THREE.Vector3();
const _bodyQuat = new THREE.Quaternion();
const _worldPos = new THREE.Vector3();
const _worldQuat = new THREE.Quaternion();
const _invQ = new THREE.Quaternion();
const _mat = new THREE.Matrix4();
const _zero = new THREE.Vector3(0, 0, 0);
const _one = new THREE.Vector3(1, 1, 1);
const _col = new THREE.Color();
const IDENTITY_Q = new THREE.Quaternion();

// ── Playback state ──────────────────────────────────────────────────────────────
let model: ViewerModel | null = null;
let playing = false;
let playhead = 0; // float frame index
let lastTick = 0;
/** Per-chunk last-rendered body-id, so instance colours only update on change. */
let lastBody: Int32Array = new Int32Array(0);

// ── File loading ────────────────────────────────────────────────────────────────
function isGzip(buf: ArrayBuffer): boolean {
  const b = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
  return b.length >= 2 && b[0] === 0x1f && b[1] === 0x8b;
}

async function readRecordingFile(file: File): Promise<any> {
  const buf = await file.arrayBuffer();
  let text: string;
  const gz = file.name.endsWith('.gz') || isGzip(buf);
  if (gz && typeof (globalThis as any).DecompressionStream === 'function') {
    const ds = new (globalThis as any).DecompressionStream('gzip');
    const stream = new Blob([buf]).stream().pipeThrough(ds);
    text = await new Response(stream).text();
  } else if (gz) {
    throw new Error('This browser cannot gunzip — re-save the recording uncompressed (*.sim.json).');
  } else {
    text = new TextDecoder().decode(buf);
  }
  return JSON.parse(text);
}

async function loadFile(file: File) {
  errorEl.style.display = 'none';
  try {
    const data = await readRecordingFile(file);
    if (!data || typeof data !== 'object' || !data.bodies || !data.columns) {
      throw new Error('Not a session recording (missing body/column data).');
    }
    if (typeof data.schema === 'string' && !data.schema.startsWith('blast-sim-recording/')) {
      throw new Error(`Unexpected schema "${data.schema}".`);
    }
    buildModel(data, file.name);
  } catch (err) {
    console.error(err);
    errorEl.textContent = `Could not load “${file.name}”: ${(err as Error).message ?? err}`;
    errorEl.style.display = 'block';
  }
}

// ── Model construction ──────────────────────────────────────────────────────────
function buildModel(data: any, fileName: string) {
  const dec = decodeSimRecording(data);
  const frames = dec.durationFrames;
  const scenario = data.scenario;
  const nodes: Array<{ centroid?: Vec3 }> = scenario?.nodes ?? [];

  // Decode the two baseline columns the package decoder leaves encoded.
  const nodeIndices = data.nodeIndices ? (decodeTyped(data.nodeIndices) as Int32Array) : new Int32Array(0);
  const initialBodyByNode = data.initialBodyByNode
    ? (decodeTyped(data.initialBodyByNode) as Int32Array)
    : new Int32Array(0);

  const perBodyMode = !scenario || nodes.length === 0 || nodeIndices.length === 0;

  const bounds = new THREE.Box3();
  const chunks: ChunkModel[] = [];

  if (!perBodyMode) {
    // node → chunk index (for routing topology events, keyed by node).
    const nodeToChunk = new Map<number, number>();
    for (let c = 0; c < nodeIndices.length; c += 1) nodeToChunk.set(nodeIndices[c], c);

    for (let c = 0; c < nodeIndices.length; c += 1) {
      const node = nodeIndices[c];
      const centroid = nodes[node]?.centroid ?? { x: 0, y: 0, z: 0 };
      const size = scenario ? resolveScenarioNodeSize(node, scenario) : { x: 0.5, y: 0.5, z: 0.5 };
      const sizeV = new THREE.Vector3(size.x, size.y, size.z);
      bounds.expandByPoint(_worldPos.set(centroid.x, centroid.y, centroid.z).clone());

      const initBody = initialBodyByNode[c] ?? -1;
      const seg = makeInitialSegment(dec, initBody, centroid);
      chunks.push({ node, size: sizeV, segments: [seg], destroyFrame: Infinity });
    }

    // Replay topology events to split segments at migrations and mark destroys.
    for (const e of dec.events as any[]) {
      if (e.type === 'migrate') {
        const c = nodeToChunk.get(e.node);
        if (c == null) continue;
        const ch = chunks[c];
        const cur = ch.segments[ch.segments.length - 1];
        worldFromSegment(dec, cur, e.f, _worldPos, _worldQuat);
        ch.segments.push(makeMigratedSegment(dec, e.to, e.f, _worldPos, _worldQuat));
      } else if (e.type === 'destroy') {
        const c = nodeToChunk.get(e.node);
        if (c != null) chunks[c].destroyFrame = Math.min(chunks[c].destroyFrame, e.f);
      }
    }
  } else {
    // Per-body fallback: bounds from the (sub-sampled) body trace.
    growBoundsFromTrace(dec, bounds);
  }

  if (bounds.isEmpty()) bounds.set(new THREE.Vector3(-5, 0, -5), new THREE.Vector3(5, 10, 5));

  let peakBodies = 0;
  for (let i = 0; i < dec.columns.bodyCount.length; i += 1) {
    peakBodies = Math.max(peakBodies, dec.columns.bodyCount[i]);
  }

  const boxCapacity = perBodyMode ? peakBodies : chunks.length;
  const extraCapacity = perBodyMode ? 0 : peakBodies;

  model = { data, dec, frames, chunks, perBodyMode, boxCapacity, extraCapacity, bounds };
  lastBody = new Int32Array(boxCapacity).fill(-2);

  setupSceneMeshes(model);
  fitCamera(bounds);
  renderMeta(data, dec, fileName, peakBodies);

  // Reset transport (show it first so the event strip can size to its real width).
  scrub.max = String(Math.max(0, frames - 1));
  scrub.value = '0';
  playhead = 0;
  setPlaying(frames > 1);
  transport.style.display = 'flex';
  dropOverlay.classList.add('hidden');
  drawEventStrip();
  renderFrame(0);
}

function makeInitialSegment(dec: ViewerModel['dec'], initBody: number, centroid: Vec3): Segment {
  if (initBody == null || initBody < 0) {
    // Static support chunk: store its rest world transform directly.
    return { f: 0, body: -1, lp: new THREE.Vector3(centroid.x, centroid.y, centroid.z), lq: new THREE.Quaternion() };
  }
  const row = dec.bodyInFrame(0, initBody);
  if (!row) {
    return { f: 0, body: -1, lp: new THREE.Vector3(centroid.x, centroid.y, centroid.z), lq: new THREE.Quaternion() };
  }
  readBody(row, _bodyPos, _bodyQuat);
  // localOffset = body0⁻¹ ∘ (centroid, identity)
  _invQ.copy(_bodyQuat).invert();
  const lp = new THREE.Vector3(centroid.x, centroid.y, centroid.z).sub(_bodyPos).applyQuaternion(_invQ);
  const lq = _invQ.clone(); // identity worldQuat ⇒ lq = body0⁻¹
  return { f: 0, body: initBody, lp, lq };
}

function makeMigratedSegment(
  dec: ViewerModel['dec'],
  toBody: number,
  f: number,
  worldPos: THREE.Vector3,
  worldQuat: THREE.Quaternion,
): Segment {
  const row = dec.bodyInFrame(f, toBody);
  if (!row) {
    // Target body not present this frame — keep the world transform; it'll hide
    // until the body appears. Offsets are placeholders.
    return { f, body: toBody, lp: worldPos.clone(), lq: worldQuat.clone() };
  }
  readBody(row, _bodyPos, _bodyQuat);
  _invQ.copy(_bodyQuat).invert();
  const lp = worldPos.clone().sub(_bodyPos).applyQuaternion(_invQ);
  const lq = _invQ.clone().multiply(worldQuat);
  return { f, body: toBody, lp, lq };
}

/** World transform of a chunk given its segment at frame f → out params. */
function worldFromSegment(
  dec: ViewerModel['dec'],
  seg: Segment,
  f: number,
  outPos: THREE.Vector3,
  outQuat: THREE.Quaternion,
): boolean {
  if (seg.body < 0) {
    outPos.copy(seg.lp);
    outQuat.copy(seg.lq);
    return true;
  }
  const row = dec.bodyInFrame(f, seg.body);
  if (!row) return false;
  readBody(row, _bodyPos, _bodyQuat);
  outQuat.copy(_bodyQuat).multiply(seg.lq);
  outPos.copy(seg.lp).applyQuaternion(_bodyQuat).add(_bodyPos);
  return true;
}

function readBody(row: Float32Array, pos: THREE.Vector3, quat: THREE.Quaternion) {
  pos.set(row[1], row[2], row[3]);
  quat.set(row[4], row[5], row[6], row[7]);
}

/** Last-segment-at-or-before f (segments are short, so a tail scan is cheapest). */
function segmentAt(ch: ChunkModel, f: number): Segment {
  const segs = ch.segments;
  for (let i = segs.length - 1; i >= 0; i -= 1) {
    if (segs[i].f <= f) return segs[i];
  }
  return segs[0];
}

function growBoundsFromTrace(dec: ViewerModel['dec'], bounds: THREE.Box3) {
  const step = Math.max(1, Math.floor(dec.durationFrames / 30));
  for (let f = 0; f < dec.durationFrames; f += step) {
    const rows = dec.frame(f);
    for (let off = 0; off < rows.length; off += dec.bodyStride) {
      bounds.expandByPoint(_worldPos.set(rows[off + 1], rows[off + 2], rows[off + 3]).clone());
    }
  }
}

// ── Scene meshes ────────────────────────────────────────────────────────────────
function setupSceneMeshes(m: ViewerModel) {
  if (chunkMesh) {
    scene.remove(chunkMesh);
    chunkMesh.dispose();
    chunkMesh = null;
  }
  if (extraMesh) {
    scene.remove(extraMesh);
    extraMesh.dispose();
    extraMesh = null;
  }

  chunkMesh = new THREE.InstancedMesh(boxGeo, chunkMat, Math.max(1, m.boxCapacity));
  chunkMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  chunkMesh.frustumCulled = false;
  chunkMesh.count = 0;
  // Allocate the per-instance colour buffer.
  chunkMesh.setColorAt(0, _col.set(0x6a8cff));
  if (chunkMesh.instanceColor) chunkMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
  scene.add(chunkMesh);

  if (m.extraCapacity > 0) {
    extraMesh = new THREE.InstancedMesh(sphereGeo, extraMat, m.extraCapacity);
    extraMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    extraMesh.frustumCulled = false;
    extraMesh.count = 0;
    scene.add(extraMesh);
  }
}

// ── Per-frame rendering ─────────────────────────────────────────────────────────
/** Reusable map: session body-id → byte offset of its row in the current frame. */
const frameMap = new Map<number, number>();
const owned = new Set<number>();

function buildFrameMap(dec: ViewerModel['dec'], f: number): Float32Array {
  const rows = dec.frame(f);
  frameMap.clear();
  for (let off = 0; off < rows.length; off += dec.bodyStride) {
    frameMap.set(rows[off], off);
  }
  return rows;
}

function bodyColor(body: number, out: THREE.Color): THREE.Color {
  if (body < 0) return out.set(0x4a5168); // static support: slate grey
  // Deterministic hue from the body id so fragments that move together share a hue.
  const h = ((body * 2654435761) >>> 0) / 0xffffffff;
  return out.setHSL(h, 0.55, 0.58);
}

function renderFrame(fFloat: number) {
  if (!model || !chunkMesh) return;
  const dec = model.dec;
  const f = Math.max(0, Math.min(model.frames - 1, Math.round(fFloat)));
  const rows = buildFrameMap(dec, f);

  if (model.perBodyMode) {
    renderPerBody(rows, dec);
  } else {
    renderChunks(f, rows, dec);
  }
  updateStats(f, dec);
  timeLabel.textContent = `${f} · ${fmtTime(dec.columns.simTime[f] ?? 0)}`;
}

function renderChunks(f: number, rows: Float32Array, dec: ViewerModel['dec']) {
  const mesh = chunkMesh!;
  const chunks = model!.chunks;
  owned.clear();
  let visible = 0;
  let colorDirty = false;

  for (let c = 0; c < chunks.length; c += 1) {
    const ch = chunks[c];
    let show = f < ch.destroyFrame;
    if (show) {
      const seg = segmentAt(ch, f);
      if (worldFromSegment(dec, seg, f, _worldPos, _worldQuat)) {
        _mat.compose(_worldPos, _worldQuat, ch.size);
        owned.add(seg.body);
        if (lastBody[visible] !== seg.body) {
          bodyColor(seg.body, _col);
          mesh.setColorAt(visible, _col);
          lastBody[visible] = seg.body;
          colorDirty = true;
        }
      } else {
        show = false;
      }
    }
    if (!show) {
      _mat.compose(_zero, IDENTITY_Q, _zero); // zero-scale ⇒ invisible
    }
    mesh.setMatrixAt(visible, _mat);
    visible += 1;
  }

  mesh.count = visible;
  mesh.instanceMatrix.needsUpdate = true;
  if (colorDirty && mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

  // Extras = recorded bodies not owned by any chunk this frame (projectiles, etc.).
  if (extraMesh) {
    let e = 0;
    const cap = model!.extraCapacity;
    for (const [body, off] of frameMap) {
      if (owned.has(body) || e >= cap) continue;
      _bodyPos.set(rows[off + 1], rows[off + 2], rows[off + 3]);
      _bodyQuat.set(rows[off + 4], rows[off + 5], rows[off + 6], rows[off + 7]);
      _mat.compose(_bodyPos, _bodyQuat, _one.clone().multiplyScalar(0.4));
      extraMesh.setMatrixAt(e, _mat);
      e += 1;
    }
    extraMesh.count = e;
    extraMesh.instanceMatrix.needsUpdate = true;
  }
}

function renderPerBody(rows: Float32Array, dec: ViewerModel['dec']) {
  const mesh = chunkMesh!;
  const stride = dec.bodyStride;
  let i = 0;
  let colorDirty = false;
  for (let off = 0; off < rows.length && i < model!.boxCapacity; off += stride) {
    const body = rows[off];
    _bodyPos.set(rows[off + 1], rows[off + 2], rows[off + 3]);
    _bodyQuat.set(rows[off + 4], rows[off + 5], rows[off + 6], rows[off + 7]);
    _mat.compose(_bodyPos, _bodyQuat, _one.clone().multiplyScalar(0.5));
    mesh.setMatrixAt(i, _mat);
    if (lastBody[i] !== body) {
      bodyColor(body, _col);
      mesh.setColorAt(i, _col);
      lastBody[i] = body;
      colorDirty = true;
    }
    i += 1;
  }
  mesh.count = i;
  mesh.instanceMatrix.needsUpdate = true;
  if (colorDirty && mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}

// ── HUD ─────────────────────────────────────────────────────────────────────────
function fmtTime(s: number): string {
  if (!Number.isFinite(s)) return '0.0s';
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return m > 0 ? `${m}:${sec.toFixed(1).padStart(4, '0')}` : `${sec.toFixed(2)}s`;
}

function renderMeta(data: any, dec: ViewerModel['dec'], fileName: string, peakBodies: number) {
  const env = data.environment ?? {};
  const meta = data.meta ?? {};
  const nodes = data.scenario?.nodes?.length ?? '—';
  const bonds = data.scenario?.bonds?.length ?? '—';
  const hasTiming = dec.timing && dec.timing.totalMs && dec.timing.totalMs.length === dec.durationFrames;
  const rows: Array<[string, string]> = [
    ['File', fileName],
    ['Demo', String(meta.demo ?? meta.page ?? env.page ?? '—')],
    ['Recorded', data.generatedAt ? new Date(data.generatedAt).toLocaleString() : '—'],
    ['Frames', `${dec.durationFrames} (${fmtTime(dec.durationSeconds)})`],
    ['Nodes · Bonds', `${nodes} · ${bonds}`],
    ['Peak bodies', String(peakBodies)],
    ['Timing', hasTiming ? 'full-session ✓' : 'none'],
    ['Mode', model?.perBodyMode ? 'per-body (no scenario)' : 'per-chunk reconstruction'],
  ];
  metaEl.innerHTML = rows
    .map(([k, v]) => `<div class="kv"><span>${k}</span><b>${escapeHtml(String(v))}</b></div>`)
    .join('');
}

function updateStats(f: number, dec: ViewerModel['dec']) {
  const col = dec.columns;
  const hasTiming = dec.timing && dec.timing.totalMs;
  const items: Array<[string, string]> = [
    ['Frame', `${f} / ${dec.durationFrames - 1}`],
    ['Time', fmtTime(col.simTime[f] ?? 0)],
    ['Bodies', String(col.bodyCount[f] ?? 0)],
    ['Active bonds', String(col.activeBonds[f] ?? 0)],
    ['Rigid bodies', String(col.rigidBodies[f] ?? 0)],
    ['Projectiles', String(col.projectiles[f] ?? 0)],
  ];
  if (col.islandCount && col.islandCount[f] > 0) {
    items.push(['Islands', `${col.islandCount[f]} (${col.islandsSkipped?.[f] ?? 0} skipped)`]);
  }
  if (hasTiming) {
    const ms = dec.timing.totalMs[f] ?? 0;
    items.push(['Frame cost', `${ms.toFixed(2)} ms`]);
  }
  statsEl.innerHTML = items
    .map(([k, v]) => `<div class="kv"><span>${k}</span><b>${escapeHtml(v)}</b></div>`)
    .join('');
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch] as string));
}

// ── Event timeline strip ─────────────────────────────────────────────────────────
const EVENT_STYLE: Record<string, { color: string; h: number }> = {
  projectile: { color: '#5fe0a0', h: 1.0 },
  force: { color: '#ffd060', h: 0.8 },
  gravity: { color: '#6cb6ff', h: 0.8 },
  destroy: { color: '#ff4d5e', h: 0.7 },
  detach: { color: '#ff9a3d', h: 0.55 },
  migrate: { color: 'rgba(150,170,255,0.5)', h: 0.4 },
  bodyRemoved: { color: 'rgba(200,120,120,0.5)', h: 0.4 },
};

function drawEventStrip() {
  if (!model) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = eventStrip.clientWidth || 600;
  const h = eventStrip.clientHeight || 18;
  eventStrip.width = Math.floor(w * dpr);
  eventStrip.height = Math.floor(h * dpr);
  const ctx = eventStrip.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const frames = Math.max(1, model.frames - 1);
  // Draw faint ticks first, prominent ones last, so impacts stay visible.
  const order = ['migrate', 'bodyRemoved', 'gravity', 'force', 'detach', 'destroy', 'projectile'];
  const byType: Record<string, any[]> = {};
  for (const e of model.dec.events as any[]) (byType[e.type] ??= []).push(e);
  for (const type of order) {
    const style = EVENT_STYLE[type];
    if (!style || !byType[type]) continue;
    ctx.fillStyle = style.color;
    for (const e of byType[type]) {
      const x = (e.f / frames) * w;
      const bh = h * style.h;
      ctx.fillRect(x, h - bh, Math.max(1, dpr), bh);
    }
  }
}

// ── Camera framing ──────────────────────────────────────────────────────────────
function fitCamera(bounds: THREE.Box3) {
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const radius = Math.max(size.length() * 0.5, 4);
  controls.target.copy(center);
  const dir = new THREE.Vector3(0.8, 0.55, 1).normalize();
  camera.position.copy(center).addScaledVector(dir, radius * 2.2);
  camera.near = Math.max(0.05, radius / 200);
  camera.far = radius * 40;
  camera.updateProjectionMatrix();
  controls.update();
  // Scale fog to the model so large scenes (mini-city) aren't fogged out.
  if (scene.fog instanceof THREE.Fog) {
    scene.fog.near = radius * 2.5;
    scene.fog.far = radius * 9;
  }
  // Re-seat the grid under the model.
  grid.position.set(center.x, bounds.min.y, center.z);
}

// ── Playback / transport ─────────────────────────────────────────────────────────
function setPlaying(v: boolean) {
  if (!model) v = false;
  if (v && model && playhead >= model.frames - 1) playhead = 0;
  playing = v;
  playBtn.textContent = v ? '❚❚' : '▶';
  playBtn.title = v ? 'Pause' : 'Play';
}

function advance(now: number) {
  if (!playing || !model) {
    lastTick = now;
    return;
  }
  const dtReal = Math.min(0.1, (now - lastTick) / 1000);
  lastTick = now;
  const speed = parseFloat(speedSel.value) || 1;
  const avgDt = model.dec.durationSeconds / Math.max(1, model.frames) || 1 / 60;
  playhead += (dtReal * speed) / avgDt;
  if (playhead >= model.frames - 1) {
    if (loopChk.checked) {
      playhead = 0;
    } else {
      playhead = model.frames - 1;
      setPlaying(false);
    }
  }
  scrub.value = String(Math.round(playhead));
}

function animate(now: number) {
  requestAnimationFrame(animate);
  advance(now);
  if (model) renderFrame(playhead);
  controls.update();
  renderer.render(scene, camera);
}

// ── Resize ──────────────────────────────────────────────────────────────────────
function resize() {
  const w = viewport.clientWidth;
  const h = viewport.clientHeight;
  if (w === 0 || h === 0) return;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  drawEventStrip();
}

// ── Wiring ──────────────────────────────────────────────────────────────────────
playBtn.addEventListener('click', () => setPlaying(!playing));
scrub.addEventListener('input', () => {
  playhead = parseFloat(scrub.value);
  setPlaying(false);
  if (model) renderFrame(playhead);
});
fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) void loadFile(file);
});

['dragenter', 'dragover'].forEach((ev) =>
  window.addEventListener(ev, (e) => {
    e.preventDefault();
    dropOverlay.classList.remove('hidden');
    dropOverlay.classList.add('drag');
  }),
);
['dragleave', 'drop'].forEach((ev) =>
  window.addEventListener(ev, (e) => {
    e.preventDefault();
    dropOverlay.classList.remove('drag');
    if (ev === 'dragleave' && model) dropOverlay.classList.add('hidden');
  }),
);
window.addEventListener('drop', (e) => {
  const file = (e as DragEvent).dataTransfer?.files?.[0];
  if (file) void loadFile(file);
});

window.addEventListener('keydown', (e) => {
  if (!model) return;
  if (e.code === 'Space') {
    e.preventDefault();
    setPlaying(!playing);
  } else if (e.code === 'ArrowRight') {
    setPlaying(false);
    playhead = Math.min(model.frames - 1, playhead + (e.shiftKey ? 10 : 1));
    scrub.value = String(Math.round(playhead));
    renderFrame(playhead);
  } else if (e.code === 'ArrowLeft') {
    setPlaying(false);
    playhead = Math.max(0, playhead - (e.shiftKey ? 10 : 1));
    scrub.value = String(Math.round(playhead));
    renderFrame(playhead);
  }
});

window.addEventListener('resize', resize);
resize();
requestAnimationFrame(animate);
