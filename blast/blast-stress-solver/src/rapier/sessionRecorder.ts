/**
 * Allocation-free per-frame **session recorder** for the destructible core.
 *
 * The goal is a single, self-contained recording you can attach to a bug report
 * and analyse offline — rich enough to answer "where was body 47 / node 47 at
 * frame 120, and with what linear/angular velocity?", "which bonds broke between
 * frames 90–110?", "when did this chunk detach and which body did it migrate
 * to?" — without re-running the simulation, and ideally enough to reproduce the
 * scenario from scratch (initial structure + config + the exact ordered input
 * stream).
 *
 * Design constraints (in priority order):
 *
 *  1. **No measurable per-frame distortion.** The hot path is pure typed-array
 *     writes: every dynamic body's `[handle, px,py,pz, qx,qy,qz,qw, lvx,lvy,lvz,
 *     avx,avy,avz]` (14 floats) is written straight into a growable
 *     `Float32Array`. There is **no object allocation, no JSON, no stringify**
 *     while recording, so there are no GC pauses to perturb the very frame times
 *     you are trying to capture. When *not* recording the cost is a single
 *     boolean check in the wrapped `step`.
 *  2. **Drop-in.** `attach(core)` monkey-patches `core.step` /
 *     `enqueueProjectile` / `applyExternalForce` / `setGravity` so inputs and
 *     per-frame state are captured automatically regardless of which page or
 *     which button (click, meteor storm, detonate, …) produced them. A page only
 *     needs `recorder.attach(core)`; nothing in its render loop changes.
 *  3. **Rich, but compact on disk.** Heavy data is stored columnar in typed
 *     arrays and serialised as base64; topology (the body↔node hierarchy and what
 *     fractured when) is stored as a baseline snapshot + a delta event stream
 *     rather than a per-frame dump. The whole thing gzips extremely well because
 *     motion is smooth (see {@link gzipJson} / the overlay's download).
 *
 * The capture engine here is environment-agnostic (no DOM); the record/stop/
 * download UI lives in `recordingOverlay.ts`.
 */
import type { ChunkData, ProjectileSpawn, Vec3 } from './types';

export const SIM_RECORDING_SCHEMA = 'blast-sim-recording/v1' as const;

/** Floats written per body, per frame, in this order. */
export const BODY_STRIDE = 14;
export const BODY_LAYOUT = [
  'handle',
  'px',
  'py',
  'pz',
  'qx',
  'qy',
  'qz',
  'qw',
  'lvx',
  'lvy',
  'lvz',
  'avx',
  'avy',
  'avz',
] as const;

// Rapier's `@dimforge/rapier3d-compat` returns `RigidBody.handle` (a u64) as a
// JS `number` whose *raw 64-bit pattern* is reinterpreted as a float64 — so
// handle 1 arrives as the denormal 5e-324 (bits 0x…01), handle 37 as 1.83e-322,
// etc. Stored naively those underflow Float32 to 0 and truncate Int32 to 0. We
// reinterpret the bits back into the real integer index (low 32 bits) + any
// generation (high 32 bits) so the handle is a stable, comparable integer.
const _handleF64 = new Float64Array(1);
const _handleU32 = new Uint32Array(_handleF64.buffer); // shares bytes (LE in browsers)
const MIN_NORMAL_F64 = 2.2250738585072014e-308;

export function decodeRapierHandle(h: number): number {
  if (typeof h !== 'number' || !Number.isFinite(h)) return -1;
  if (h === 0) return 0;
  // Subnormal magnitude ⇒ this is a bit-encoded handle; decode the u64.
  if (Math.abs(h) < MIN_NORMAL_F64) {
    _handleF64[0] = h;
    return _handleU32[0] + _handleU32[1] * 0x1_0000_0000;
  }
  // Already a plain integer handle (other Rapier builds / our test stubs).
  return h;
}

/** Minimal Rapier body shape we read each frame (avoids a hard Rapier import). */
type BodyLike = {
  handle: number;
  isFixed: () => boolean;
  translation: () => Vec3;
  rotation: () => { x: number; y: number; z: number; w: number };
  linvel: () => Vec3;
  angvel: () => Vec3;
};

type WorldLike = {
  forEachRigidBody: (cb: (body: BodyLike) => void) => void;
};

/** A per-frame profiler sample (the core's `CoreProfilerSample`), read loosely so
 *  we don't couple to its exact shape — we copy whatever numeric `*Ms` / count
 *  fields are present. */
type ProfilerSampleLike = Record<string, unknown>;
type ProfilerConfigLike = {
  enabled: boolean;
  onSample?: (s: ProfilerSampleLike) => void;
  measureReferencePlanner?: boolean;
};

/**
 * Per-frame timing fields copied from the core's profiler sample into the
 * full-session timing stream. These are the leaf phase timers (≈ mutually
 * exclusive — they sum to ~`totalMs`), the wrapper totals, and a few useful
 * counts. Captured every frame for the whole recording (unlike the frame-profiler
 * overlay, which only retains a rolling window).
 */
export const TIMING_FIELDS = [
  'totalMs',
  // physics + solver (solver* sub-phases break down solverUpdateMs: JS gravity
  // fill, JS contact/splash injection, vs. the WASM CGNR solve)
  'rapierStepMs',
  'solverUpdateMs',
  'solverGravityInjectMs',
  'solverContactInjectMs',
  // contactInject* break solverContactInjectMs into resolve (Rapier round-trips +
  // force rotation), splash-grid rebuild, splash neighbour search, and submit
  // (the WASM addForce FFI + C++ per-force work). They sum to solverContactInjectMs.
  'contactInjectResolveMs',
  'contactInjectGridMs',
  'contactInjectSplashMs',
  'contactInjectSubmitMs',
  'solverSolveMs',
  // contacts / forces
  'contactDrainMs',
  'externalForceMs',
  'preStepSweepMs',
  // fracture (+ sub-phases)
  'fractureMs',
  'fractureGenerateMs',
  'fractureApplyMs',
  // split planning / topology edits
  'splitPlannerMs',
  'splitQueueMs',
  'bodyCreateMs',
  'colliderRebuildMs',
  'rebuildColliderMapMs',
  'cleanupDisabledMs',
  // snapshots (resim rollback)
  'snapshotCaptureMs',
  'snapshotRestoreMs',
  // damage subsystem
  'damageTickMs',
  'damageReplayMs',
  'damagePreviewMs',
  'damageRestoreMs',
  'damageSnapshotMs',
  'damagePreDestroyMs',
  'damageFlushMs',
  // spawn / cleanup
  'spawnMs',
  'projectileCleanupMs',
  // wrapper totals (contain leaf work — for cross-checking, not summing)
  'initialPassMs',
  'resimMs',
  // counts
  'resimPasses',
  'rigidBodies',
] as const;

/** The slice of the destructible core the recorder reads from / wraps. */
export type RecordableCore = {
  world: WorldLike;
  chunks: ChunkData[];
  getActiveBondsCount: () => number;
  getRigidBodyCount: () => number;
  /** Optional island-aware solver stats (Stage 4); recorded per frame when present. */
  getIslandSolverStats?: () => { enabled: boolean; skipSettled: boolean; islandCount: number; islandsSkipped: number };
  projectiles: { length: number };
  step: (dt?: number) => void;
  stepEventful?: (dt?: number) => void;
  stepSafe?: (dt?: number) => void;
  enqueueProjectile: (s: ProjectileSpawn) => void;
  applyExternalForce: (nodeIndex: number, worldPoint: Vec3, worldForce: Vec3) => void;
  setGravity: (g: number) => void;
  /** Optional — when present, the recorder multiplexes onto it to capture the
   *  full-session per-frame timing stream without disturbing any other consumer
   *  (e.g. the frame-profiler overlay). */
  setProfiler?: (config: ProfilerConfigLike | null) => void;
};

/** Context describing what is being recorded — captured once at `attach`. */
export type SessionRecorderContext = {
  /** The scenario the core was built from (serialised into the bundle). */
  scenario?: unknown;
  /** The options the core was built with (gravity, materialScale, friction…). */
  coreConfig?: Record<string, unknown>;
  /** Free-form metadata (page name, UI config snapshot, build SHA…). */
  meta?: Record<string, unknown>;
  /** Serialise `scenario.parameters.fragmentGeometries` into the bundle so the
   *  exact fractured geometry can be reproduced. Default true. These are static
   *  (captured once, not per-frame) but can be large; gzip handles it well. */
  includeScenarioGeometry?: boolean;
};

export type SessionRecorderOptions = {
  /**
   * Soft cap on the number of recorded frames. When reached, recording stops
   * automatically (and `onAutoStop` fires) so a forgotten recording can't grow
   * without bound. Default 36000 (~10 min at 60 fps). Set 0 to disable.
   */
  maxFrames?: number;
  /** Called once if recording auto-stops at {@link maxFrames}. */
  onAutoStop?: () => void;
};

/** A single discrete event on the timeline (input or topology change). */
export type SimRecordingEvent =
  | { f: number; t: number; type: 'start' }
  | { f: number; t: number; type: 'stop' }
  | { f: number; t: number; type: 'projectile'; spawn: ProjectileSpawn }
  | { f: number; t: number; type: 'force'; node: number; point: Vec3; force: Vec3 }
  | { f: number; t: number; type: 'gravity'; value: number }
  /** A chunk moved from one rigid body to another (split / collider migration). */
  | { f: number; t: number; type: 'migrate'; node: number; from: number; to: number }
  /** A chunk detached from the structure (became free debris). */
  | { f: number; t: number; type: 'detach'; node: number; body: number }
  /** A chunk was destroyed (no longer simulated). */
  | { f: number; t: number; type: 'destroy'; node: number }
  /** A rigid body that previously held chunks disappeared from the world. */
  | { f: number; t: number; type: 'bodyRemoved'; body: number };

/** A typed array serialised as little-endian base64 (compact + gzip-friendly). */
export type EncodedTypedArray = {
  type: 'f64' | 'f32' | 'u32' | 'i32';
  /** Element count (not byte count). */
  length: number;
  /** base64 of the array's raw little-endian bytes. */
  data: string;
};

export type SimRecordingExport = {
  schema: typeof SIM_RECORDING_SCHEMA;
  generatedAt: string;
  /** Where the recording was taken (best effort). */
  environment: {
    page?: string;
    href?: string;
    userAgent?: string;
    viewport?: { width: number; height: number };
    devicePixelRatio?: number;
  };
  durationFrames: number;
  durationSeconds: number;
  meta?: Record<string, unknown>;
  coreConfig?: Record<string, unknown>;
  scenario?: unknown;
  /** node → rigid-body handle at the moment recording started (the baseline the
   *  `migrate`/`detach`/`destroy` events mutate). Parallel to {@link nodeIndices}. */
  initialBodyByNode: EncodedTypedArray;
  /** The node index of each chunk, in chunk order (so the baseline can be keyed
   *  back to scenario nodes). */
  nodeIndices: EncodedTypedArray;
  bodyStride: number;
  bodyLayout: readonly string[];
  /**
   * Maps the dense session body-id used in the `bodies` handle column and the
   * topology events back to the raw (decoded) Rapier handle. Index = session id,
   * value = raw handle. Interning keeps ids small so they stay exact in the
   * Float32 trace and Int32 diff state — raw Rapier handles can carry a
   * generation in the high 32 bits (e.g. 2³²+2 after index reuse), which would
   * otherwise overflow/round and fabricate per-frame "migrations".
   */
  handleTable: EncodedTypedArray;
  /** Per-frame columnar scalars (all parallel, length === durationFrames). */
  columns: {
    /** Accumulated simulation time (s). */
    simTime: EncodedTypedArray;
    /** Frame delta time (s). */
    dt: EncodedTypedArray;
    /** Dynamic bodies captured this frame (rows in {@link bodies}). */
    bodyCount: EncodedTypedArray;
    activeBonds: EncodedTypedArray;
    rigidBodies: EncodedTypedArray;
    projectiles: EncodedTypedArray;
    /** Connected components ("islands") in the solver graph this frame (Stage 2a partition). */
    islandCount: EncodedTypedArray;
    /** Islands skipped as settled this frame (0 unless island-aware skipping is enabled). */
    islandsSkipped: EncodedTypedArray;
  };
  /**
   * The flat body trace: `Σ bodyCount` rows of {@link BODY_STRIDE} floats. The
   * rows for frame `i` start at `(Σ_{j<i} bodyCount[j]) * BODY_STRIDE`.
   */
  bodies: EncodedTypedArray;
  events: SimRecordingEvent[];
  /**
   * Full-session per-frame timing stream — one column per {@link TIMING_FIELDS}
   * entry, each parallel to the kinematic frames (length === durationFrames). This
   * is the "where did every millisecond go" data: the leaf phase columns sum to
   * ~`totalMs` each frame. Present when the core exposed `setProfiler`.
   */
  timing?: {
    fields: readonly string[];
    columns: Record<string, EncodedTypedArray>;
  };
  /**
   * Sparse per-frame resim-pass breakdown — one entry only for frames that
   * resimulated (a fracture frame). Each lists every pass that ran that frame
   * (index 0 = the base step, 1+ = resim re-steps) with its leaf costs and the
   * reasons it fired, so the cost of resimulation is fully attributable.
   */
  resimLog?: Array<{
    f: number;
    passes: Array<{
      index: number;
      solverMs: number;
      fractureMs: number;
      bodyCreateMs: number;
      totalMs: number;
      reasons: string[];
    }>;
  }>;
  /** Optional rolling frame-profiler dump (stats + legend) if a profiler export
   *  was linked via {@link SessionRecorderHandle.setProfilerExport}. */
  profiler?: unknown;
};

// ── Growable Float32 buffer ───────────────────────────────────────────────────
// Doubles on demand; after warm-up there is no further allocation, so steady-state
// recording produces zero GC pressure.
class GrowableF32 {
  buf: Float32Array;
  len = 0;
  constructor(initial = 1 << 16) {
    this.buf = new Float32Array(Math.max(BODY_STRIDE, initial));
  }
  private ensure(extra: number) {
    const need = this.len + extra;
    if (need <= this.buf.length) return;
    let cap = this.buf.length;
    while (cap < need) cap *= 2;
    const next = new Float32Array(cap);
    next.set(this.buf.subarray(0, this.len));
    this.buf = next;
  }
  pushBody(
    handle: number,
    t: Vec3,
    r: { x: number; y: number; z: number; w: number },
    lv: Vec3,
    av: Vec3,
  ) {
    this.ensure(BODY_STRIDE);
    const b = this.buf;
    let i = this.len;
    b[i++] = handle;
    b[i++] = t.x;
    b[i++] = t.y;
    b[i++] = t.z;
    b[i++] = r.x;
    b[i++] = r.y;
    b[i++] = r.z;
    b[i++] = r.w;
    b[i++] = lv.x;
    b[i++] = lv.y;
    b[i++] = lv.z;
    b[i++] = av.x;
    b[i++] = av.y;
    b[i++] = av.z;
    this.len = i;
  }
  clear() {
    this.len = 0;
  }
  view(): Float32Array {
    return this.buf.subarray(0, this.len);
  }
}

// Growable scalar columns, kept as plain arrays during capture (cheap pushes,
// no per-frame typed-array reallocation) and packed to typed arrays on export.
class FrameColumns {
  simTime: number[] = [];
  dt: number[] = [];
  bodyCount: number[] = [];
  activeBonds: number[] = [];
  rigidBodies: number[] = [];
  projectiles: number[] = [];
  islandCount: number[] = [];
  islandsSkipped: number[] = [];
  clear() {
    this.simTime.length = 0;
    this.dt.length = 0;
    this.bodyCount.length = 0;
    this.activeBonds.length = 0;
    this.rigidBodies.length = 0;
    this.projectiles.length = 0;
    this.islandCount.length = 0;
    this.islandsSkipped.length = 0;
  }
}

// One growable column per TIMING_FIELDS entry, appended once per captured frame
// so they stay parallel to FrameColumns. A frame with no profiler sample pushes 0.
class TimingColumns {
  readonly cols: Record<string, number[]> = {};
  constructor() {
    for (const f of TIMING_FIELDS) this.cols[f] = [];
  }
  push(sample: ProfilerSampleLike | null) {
    for (const f of TIMING_FIELDS) {
      const v = sample ? sample[f] : 0;
      this.cols[f].push(typeof v === 'number' && Number.isFinite(v) ? v : 0);
    }
  }
  clear() {
    for (const f of TIMING_FIELDS) this.cols[f].length = 0;
  }
  get length() {
    return this.cols[TIMING_FIELDS[0]].length;
  }
}

// ── base64 helpers (browser + node) ───────────────────────────────────────────
function bytesToBase64(bytes: Uint8Array): string {
  if (typeof btoa !== 'undefined') {
    let binary = '';
    const CHUNK = 0x8000;
    for (let i = 0; i < bytes.length; i += CHUNK) {
      binary += String.fromCharCode(...bytes.subarray(i, Math.min(i + CHUNK, bytes.length)));
    }
    return btoa(binary);
  }
  // Node fallback.
  return Buffer.from(bytes).toString('base64');
}

function base64ToBytes(b64: string): Uint8Array {
  if (typeof atob !== 'undefined') {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i += 1) out[i] = bin.charCodeAt(i);
    return out;
  }
  return new Uint8Array(Buffer.from(b64, 'base64'));
}

type AnyTypedArray = Float64Array | Float32Array | Uint32Array | Int32Array;

function encodeTyped(arr: AnyTypedArray): EncodedTypedArray {
  const type =
    arr instanceof Float64Array
      ? 'f64'
      : arr instanceof Float32Array
        ? 'f32'
        : arr instanceof Uint32Array
          ? 'u32'
          : 'i32';
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  return { type, length: arr.length, data: bytesToBase64(bytes) };
}

/** Decode an {@link EncodedTypedArray} back into the matching typed array. */
export function decodeTyped(enc: EncodedTypedArray): AnyTypedArray {
  const bytes = base64ToBytes(enc.data);
  // Copy into an aligned buffer (base64 decode may not be 8-byte aligned).
  const aligned = bytes.slice().buffer;
  switch (enc.type) {
    case 'f64':
      return new Float64Array(aligned, 0, enc.length);
    case 'f32':
      return new Float32Array(aligned, 0, enc.length);
    case 'u32':
      return new Uint32Array(aligned, 0, enc.length);
    case 'i32':
      return new Int32Array(aligned, 0, enc.length);
    default:
      throw new Error(`unknown encoded type: ${(enc as EncodedTypedArray).type}`);
  }
}

// ── Scenario serialisation (duck-typed; no THREE import) ──────────────────────
// Replaces THREE.BufferGeometry instances with plain {position, index} typed
// arrays so the bundle stays JSON-serialisable without a three dependency here.
function serialiseScenario(scenario: unknown, includeGeometry: boolean): unknown {
  if (!scenario || typeof scenario !== 'object') return scenario;
  const s = scenario as Record<string, unknown>;
  const params = s.parameters as Record<string, unknown> | undefined;
  if (!params || !Array.isArray(params.fragmentGeometries)) return scenario;

  const geometries = params.fragmentGeometries as unknown[];
  const encodedGeos = includeGeometry
    ? geometries.map((g) => encodeGeometry(g))
    : geometries.map(() => null);

  return {
    ...s,
    parameters: { ...params, fragmentGeometries: encodedGeos },
  };
}

function encodeGeometry(geo: unknown): unknown {
  if (!geo || typeof geo !== 'object') return null;
  const g = geo as {
    attributes?: { position?: { array?: ArrayLike<number> } };
    index?: { array?: ArrayLike<number> } | null;
  };
  const pos = g.attributes?.position?.array;
  if (!pos) return null;
  const position = encodeTyped(pos instanceof Float32Array ? pos : Float32Array.from(pos));
  const idxArr = g.index?.array;
  const index = idxArr
    ? encodeTyped(idxArr instanceof Uint32Array ? idxArr : Uint32Array.from(idxArr))
    : null;
  return { position, index };
}

// ──────────────────────────────────────────────────────────────────────────────

export type SessionRecorderHandle = {
  /** Wrap a core's step/input methods so capture is automatic. Re-attaching to a
   *  new core (e.g. after a scenario rebuild) detaches the previous one first. */
  attach(core: RecordableCore | null, ctx?: SessionRecorderContext): void;
  /** Restore the wrapped core methods. Safe to call multiple times. */
  detach(): void;
  /** Begin capturing from the current state (clears any prior buffer). */
  start(): void;
  /** Stop capturing (keeps the buffer so it can be exported). */
  stop(): void;
  isRecording(): boolean;
  isAttached(): boolean;
  frameCount(): number;
  /** Approximate retained bytes (body trace + columns) — for a live size readout. */
  estimatedBytes(): number;
  /** Link a function that returns the frame-profiler dump to embed in exports. */
  setProfilerExport(fn: (() => unknown) | null): void;
  /** Build the recording bundle (or null if nothing was captured). */
  export(): SimRecordingExport | null;
};

export function createSessionRecorder(options: SessionRecorderOptions = {}): SessionRecorderHandle {
  const { maxFrames = 36000, onAutoStop } = options;

  let core: RecordableCore | null = null;
  let ctx: SessionRecorderContext = {};
  let recording = false;
  let profilerExportFn: (() => unknown) | null = null;

  const bodies = new GrowableF32();
  const columns = new FrameColumns();
  const timing = new TimingColumns();
  // Intern raw (decoded) Rapier handles → small dense session ids, so the id is
  // exact in the Float32 trace and the Int32 topology-diff state even when the raw
  // handle carries a generation in its high 32 bits. handleList[id] = raw handle.
  const handleIds = new Map<number, number>();
  let handleList: number[] = [];
  const internHandle = (raw: number): number => {
    if (raw < 0) return -1;
    let id = handleIds.get(raw);
    if (id === undefined) {
      id = handleList.length;
      handleIds.set(raw, id);
      handleList.push(raw);
    }
    return id;
  };
  let events: SimRecordingEvent[] = [];
  let resimLog: NonNullable<SimRecordingExport['resimLog']> = [];
  // Latest profiler sample seen this frame (set in the multiplexed onSample,
  // consumed + cleared in captureFrame so timing stays aligned to body frames).
  let lastProfilerSample: ProfilerSampleLike | null = null;
  let sawProfiler = false; // any sample ever seen → emit the timing stream on export

  let localFrame = 0;
  let simTime = 0;

  // Topology baseline + per-frame diff state (allocated to chunk count at start).
  let nodeIndexByChunk: Int32Array = new Int32Array(0);
  let initialBodyByChunk: Int32Array = new Int32Array(0);
  let prevBody: Int32Array = new Int32Array(0);
  let prevFlags: Uint8Array = new Uint8Array(0); // bit0 active, bit1 detached, bit2 destroyed
  // Bodies that ever held a chunk — to detect when one disappears from the world.
  const liveBodies = new Set<number>();
  let seenBodies = new Set<number>();

  // Saved originals so detach() can restore them. Keyed to the patched core.
  type Patched = RecordableCore & { __bssRecorderPatched?: boolean };
  let patched: Patched | null = null;
  let origStep: RecordableCore['step'] | null = null;
  let origStepEventful: RecordableCore['stepEventful'] | null = null;
  let origStepSafe: RecordableCore['stepSafe'] | null = null;
  let origEnqueue: RecordableCore['enqueueProjectile'] | null = null;
  let origForce: RecordableCore['applyExternalForce'] | null = null;
  let origGravity: RecordableCore['setGravity'] | null = null;
  let origSetProfiler: RecordableCore['setProfiler'] | null = null;
  // The profiler config last requested by another consumer (e.g. the overlay), so
  // we can multiplex our capture onto it and forward samples to it unchanged.
  let userProfiler: ProfilerConfigLike | null = null;
  // Re-applies the merged profiler config to the live core (recomputes `enabled`).
  let reinstallProfiler: (() => void) | null = null;

  const FLAG_ACTIVE = 1;
  const FLAG_DETACHED = 2;
  const FLAG_DESTROYED = 4;

  function resetBuffers() {
    bodies.clear();
    columns.clear();
    timing.clear();
    events = [];
    resimLog = [];
    handleIds.clear();
    handleList = [];
    lastProfilerSample = null;
    localFrame = 0;
    simTime = 0;
    liveBodies.clear();
  }

  function snapshotBaseline() {
    const chunks = core?.chunks ?? [];
    const n = chunks.length;
    nodeIndexByChunk = new Int32Array(n);
    initialBodyByChunk = new Int32Array(n);
    prevBody = new Int32Array(n);
    prevFlags = new Uint8Array(n);
    liveBodies.clear();
    for (let i = 0; i < n; i += 1) {
      const c = chunks[i];
      const body = c.bodyHandle == null ? -1 : internHandle(decodeRapierHandle(c.bodyHandle));
      nodeIndexByChunk[i] = c.nodeIndex;
      initialBodyByChunk[i] = body;
      prevBody[i] = body;
      prevFlags[i] =
        (c.active ? FLAG_ACTIVE : 0) |
        (c.detached ? FLAG_DETACHED : 0) |
        (c.destroyed ? FLAG_DESTROYED : 0);
      if (body >= 0) liveBodies.add(body);
    }
  }

  // Diff the chunk→body hierarchy against the previous frame, emitting delta
  // events. O(chunks) with no allocation in steady state.
  function diffTopology() {
    if (!core) return;
    const chunks = core.chunks;
    const n = Math.min(chunks.length, prevBody.length);
    seenBodies.clear();
    for (let i = 0; i < n; i += 1) {
      const c = chunks[i];
      const node = nodeIndexByChunk[i];
      const body = c.bodyHandle == null ? -1 : internHandle(decodeRapierHandle(c.bodyHandle));
      if (body >= 0) seenBodies.add(body);

      const flags =
        (c.active ? FLAG_ACTIVE : 0) |
        (c.detached ? FLAG_DETACHED : 0) |
        (c.destroyed ? FLAG_DESTROYED : 0);

      if (body !== prevBody[i] && body >= 0 && prevBody[i] >= 0) {
        events.push({ f: localFrame, t: simTime, type: 'migrate', node, from: prevBody[i], to: body });
      }
      const wasDetached = (prevFlags[i] & FLAG_DETACHED) !== 0;
      if (!wasDetached && (flags & FLAG_DETACHED) !== 0) {
        events.push({ f: localFrame, t: simTime, type: 'detach', node, body });
      }
      const wasDestroyed = (prevFlags[i] & FLAG_DESTROYED) !== 0;
      if (!wasDestroyed && (flags & FLAG_DESTROYED) !== 0) {
        events.push({ f: localFrame, t: simTime, type: 'destroy', node });
      }
      prevBody[i] = body;
      prevFlags[i] = flags;
    }
    // Bodies that held chunks last frame but no chunk references this frame.
    if (seenBodies.size !== liveBodies.size || !isSuperset(seenBodies, liveBodies)) {
      for (const b of liveBodies) {
        if (!seenBodies.has(b)) {
          events.push({ f: localFrame, t: simTime, type: 'bodyRemoved', body: b });
        }
      }
    }
    // Swap roles to avoid reallocating sets.
    const tmp = liveBodies;
    // copy seen → live
    tmp.clear();
    for (const b of seenBodies) tmp.add(b);
  }

  function isSuperset(a: Set<number>, b: Set<number>): boolean {
    for (const x of b) if (!a.has(x)) return false;
    return true;
  }

  // The hot path: write every dynamic body's kinematics + the scalar columns.
  function captureFrame(dt: number) {
    if (!core) return;
    simTime += Number.isFinite(dt) ? dt : 0;

    let count = 0;
    core.world.forEachRigidBody((body) => {
      if (body.isFixed()) return;
      bodies.pushBody(
        internHandle(decodeRapierHandle(body.handle)),
        body.translation(),
        body.rotation(),
        body.linvel(),
        body.angvel(),
      );
      count += 1;
    });

    columns.simTime.push(simTime);
    columns.dt.push(Number.isFinite(dt) ? dt : 0);
    columns.bodyCount.push(count);
    columns.activeBonds.push(core.getActiveBondsCount());
    columns.rigidBodies.push(core.getRigidBodyCount());
    columns.projectiles.push(core.projectiles.length);
    const islandStats = core.getIslandSolverStats?.();
    columns.islandCount.push(islandStats ? islandStats.islandCount : 0);
    columns.islandsSkipped.push(islandStats ? islandStats.islandsSkipped : 0);

    // Full-session timing: the profiler sample for this frame fired during
    // orig(dt) (just before this); append it (or zeros) so timing stays aligned.
    timing.push(lastProfilerSample);
    // Sparse per-pass resim breakdown — only when the frame actually resimulated
    // (more than the base pass), so it's cheap and rare.
    const passes = lastProfilerSample?.passes as
      | Array<{ index?: number; solverMs?: number; fractureMs?: number; bodyCreateMs?: number; totalMs?: number; reasons?: string[] }>
      | undefined;
    if (Array.isArray(passes) && passes.length > 1) {
      resimLog.push({
        f: localFrame,
        passes: passes.map((p) => ({
          index: p.index ?? 0,
          solverMs: p.solverMs ?? 0,
          fractureMs: p.fractureMs ?? 0,
          bodyCreateMs: p.bodyCreateMs ?? 0,
          totalMs: p.totalMs ?? 0,
          reasons: Array.isArray(p.reasons) ? p.reasons.slice() : [],
        })),
      });
    }
    lastProfilerSample = null;

    diffTopology();

    localFrame += 1;
    if (maxFrames > 0 && localFrame >= maxFrames) {
      stop();
      onAutoStop?.();
    }
  }

  function patch(next: RecordableCore) {
    const p = next as Patched;
    // Save the raw originals so detach() can restore them by identity. We invoke
    // them with the core as receiver (`.call`) so an implementation that relies on
    // `this` keeps working, without changing the stored references.
    origStep = next.step;
    origEnqueue = next.enqueueProjectile;
    origForce = next.applyExternalForce;
    origGravity = next.setGravity;
    origStepEventful = next.stepEventful ?? null;
    origStepSafe = next.stepSafe ?? null;

    const wrapStep = (orig: (dt?: number) => void) => (dt?: number) => {
      orig.call(next, dt);
      if (recording) captureFrame(typeof dt === 'number' ? dt : NaN);
    };

    next.step = wrapStep(origStep);
    if (origStepEventful) next.stepEventful = wrapStep(origStepEventful);
    if (origStepSafe) next.stepSafe = wrapStep(origStepSafe);

    next.enqueueProjectile = (s: ProjectileSpawn) => {
      if (recording) {
        // Deep-clone the spawn so later mutation by the caller can't corrupt the
        // log. Spawns are infrequent, so the clone cost is irrelevant.
        events.push({ f: localFrame, t: simTime, type: 'projectile', spawn: cloneSpawn(s) });
      }
      origEnqueue!.call(next, s);
    };
    next.applyExternalForce = (node: number, point: Vec3, force: Vec3) => {
      if (recording) {
        events.push({
          f: localFrame,
          t: simTime,
          type: 'force',
          node,
          point: { x: point.x, y: point.y, z: point.z },
          force: { x: force.x, y: force.y, z: force.z },
        });
      }
      origForce!.call(next, node, point, force);
    };
    next.setGravity = (g: number) => {
      if (recording) events.push({ f: localFrame, t: simTime, type: 'gravity', value: g });
      origGravity!.call(next, g);
    };

    // Multiplex onto setProfiler so we capture the full-session per-frame timing
    // stream without disturbing any other consumer (the frame-profiler overlay):
    // every consumer's `onSample` is fanned out to, plus our own capture.
    if (next.setProfiler) {
      origSetProfiler = next.setProfiler;
      const installMerged = () => {
        // Keep the profiler enabled if another consumer wants it (overlay) or
        // while we're recording — otherwise off, so idle frames pay nothing.
        const enabled = (userProfiler?.enabled ?? false) || recording;
        origSetProfiler!.call(next, {
          enabled,
          measureReferencePlanner: userProfiler?.measureReferencePlanner,
          onSample: (s: ProfilerSampleLike) => {
            // Stash for captureFrame (keeps timing aligned to body frames).
            if (recording) {
              lastProfilerSample = s;
              sawProfiler = true;
            }
            userProfiler?.onSample?.(s);
          },
        });
      };
      reinstallProfiler = installMerged;
      next.setProfiler = (cfg: ProfilerConfigLike | null) => {
        userProfiler = cfg ?? null;
        installMerged();
      };
      installMerged();
    }

    p.__bssRecorderPatched = true;
    patched = p;
  }

  function unpatch() {
    if (!patched) return;
    if (origStep) patched.step = origStep;
    if (origEnqueue) patched.enqueueProjectile = origEnqueue;
    if (origForce) patched.applyExternalForce = origForce;
    if (origGravity) patched.setGravity = origGravity;
    if (origStepEventful && patched.stepEventful) patched.stepEventful = origStepEventful;
    if (origStepSafe && patched.stepSafe) patched.stepSafe = origStepSafe;
    if (origSetProfiler) {
      patched.setProfiler = origSetProfiler;
      // Restore whatever the other consumer last wanted directly on the core.
      try {
        origSetProfiler.call(patched, userProfiler);
      } catch {
        /* best-effort */
      }
    }
    delete patched.__bssRecorderPatched;
    patched = null;
    origStep = origEnqueue = origForce = origGravity = null;
    origStepEventful = origStepSafe = null;
    origSetProfiler = null;
    reinstallProfiler = null;
    userProfiler = null;
  }

  function cloneSpawn(s: ProjectileSpawn): ProjectileSpawn {
    return JSON.parse(JSON.stringify(s)) as ProjectileSpawn;
  }

  function attach(next: RecordableCore | null, nextCtx?: SessionRecorderContext) {
    // Stop & detach any existing recording (e.g. on scenario rebuild).
    if (recording) stop();
    unpatch();
    core = next;
    ctx = nextCtx ?? {};
    resetBuffers();
    if (core) patch(core);
  }

  function detach() {
    if (recording) stop();
    unpatch();
    core = null;
  }

  function start() {
    if (!core || recording) return;
    resetBuffers();
    snapshotBaseline();
    recording = true;
    reinstallProfiler?.(); // ensure the profiler is enabled for timing capture
    events.push({ f: 0, t: 0, type: 'start' });
  }

  function stop() {
    if (!recording) return;
    events.push({ f: localFrame, t: simTime, type: 'stop' });
    recording = false;
    reinstallProfiler?.(); // let the profiler turn back off if nothing else needs it
  }

  function estimatedBytes(): number {
    // Body trace dominates; + scalar columns + the per-frame timing columns.
    const timingCols = sawProfiler ? TIMING_FIELDS.length * 4 : 0;
    return bodies.len * 4 + columns.simTime.length * (8 + 4 + 4 * 4 + timingCols);
  }

  function buildExport(): SimRecordingExport | null {
    if (columns.simTime.length === 0) return null;

    const environment: SimRecordingExport['environment'] = {};
    if (typeof location !== 'undefined') {
      environment.page = location.pathname;
      environment.href = location.href;
    }
    if (typeof navigator !== 'undefined' && navigator.userAgent) {
      environment.userAgent = navigator.userAgent;
    }
    if (typeof window !== 'undefined') {
      environment.viewport = { width: window.innerWidth, height: window.innerHeight };
      environment.devicePixelRatio = window.devicePixelRatio;
    }

    let profiler: unknown;
    if (profilerExportFn) {
      try {
        profiler = profilerExportFn() ?? undefined;
      } catch {
        /* profiler dump is best-effort */
      }
    }

    let timingExport: SimRecordingExport['timing'];
    if (sawProfiler) {
      const cols: Record<string, EncodedTypedArray> = {};
      for (const f of TIMING_FIELDS) cols[f] = encodeTyped(Float32Array.from(timing.cols[f]));
      timingExport = { fields: TIMING_FIELDS as unknown as string[], columns: cols };
    }

    return {
      schema: SIM_RECORDING_SCHEMA,
      generatedAt: new Date().toISOString(),
      environment,
      durationFrames: columns.simTime.length,
      durationSeconds: simTime,
      meta: ctx.meta,
      coreConfig: ctx.coreConfig,
      scenario: ctx.scenario
        ? serialiseScenario(ctx.scenario, ctx.includeScenarioGeometry !== false)
        : undefined,
      initialBodyByNode: encodeTyped(initialBodyByChunk),
      nodeIndices: encodeTyped(nodeIndexByChunk),
      bodyStride: BODY_STRIDE,
      bodyLayout: BODY_LAYOUT,
      handleTable: encodeTyped(Float64Array.from(handleList)),
      columns: {
        simTime: encodeTyped(Float64Array.from(columns.simTime)),
        dt: encodeTyped(Float32Array.from(columns.dt)),
        bodyCount: encodeTyped(Uint32Array.from(columns.bodyCount)),
        activeBonds: encodeTyped(Uint32Array.from(columns.activeBonds)),
        rigidBodies: encodeTyped(Uint32Array.from(columns.rigidBodies)),
        projectiles: encodeTyped(Uint32Array.from(columns.projectiles)),
        islandCount: encodeTyped(Uint32Array.from(columns.islandCount)),
        islandsSkipped: encodeTyped(Uint32Array.from(columns.islandsSkipped)),
      },
      bodies: encodeTyped(bodies.view().slice()),
      events: events.slice(),
      timing: timingExport,
      resimLog: resimLog.length > 0 ? resimLog.slice() : undefined,
      profiler,
    };
  }

  return {
    attach,
    detach,
    start,
    stop,
    isRecording: () => recording,
    isAttached: () => core != null,
    frameCount: () => localFrame,
    estimatedBytes,
    setProfilerExport: (fn) => {
      profilerExportFn = fn;
    },
    export: buildExport,
  };
}

// ── Decoding / analysis helpers (for tooling + tests) ─────────────────────────

export type DecodedSimRecording = {
  durationFrames: number;
  durationSeconds: number;
  bodyStride: number;
  bodyLayout: readonly string[];
  /** Dense session body-id → raw Rapier handle (see SimRecordingExport). */
  handleTable: Float64Array;
  columns: {
    simTime: Float64Array;
    dt: Float32Array;
    bodyCount: Uint32Array;
    activeBonds: Uint32Array;
    rigidBodies: Uint32Array;
    projectiles: Uint32Array;
    islandCount: Uint32Array;
    islandsSkipped: Uint32Array;
  };
  bodies: Float32Array;
  /** Float offset (into `bodies`) where each frame's rows begin. */
  frameBodyOffset: Uint32Array;
  events: SimRecordingEvent[];
  /** Full-session per-frame timing columns (decoded), keyed by field name, each
   *  parallel to the frames. Empty object if the recording had no timing stream. */
  timing: Record<string, Float32Array>;
  /** Sparse per-frame resim-pass breakdown (only fracture/resim frames). */
  resimLog: NonNullable<SimRecordingExport['resimLog']>;
  /** Return the rows for a frame as `[{handle, px,…}]`-style flat slices. */
  frame(i: number): Float32Array;
  /** Find one body's 14-float row in a frame by Rapier handle, or null. */
  bodyInFrame(frameIndex: number, handle: number): Float32Array | null;
};

/** Rehydrate an exported bundle into typed arrays for offline analysis. */
export function decodeSimRecording(data: SimRecordingExport): DecodedSimRecording {
  const columns = {
    simTime: decodeTyped(data.columns.simTime) as Float64Array,
    dt: decodeTyped(data.columns.dt) as Float32Array,
    bodyCount: decodeTyped(data.columns.bodyCount) as Uint32Array,
    activeBonds: decodeTyped(data.columns.activeBonds) as Uint32Array,
    rigidBodies: decodeTyped(data.columns.rigidBodies) as Uint32Array,
    projectiles: decodeTyped(data.columns.projectiles) as Uint32Array,
    // Backward-compatible: recordings made before island instrumentation lack these columns.
    islandCount: (data.columns.islandCount ? decodeTyped(data.columns.islandCount) : new Uint32Array(data.durationFrames || 0)) as Uint32Array,
    islandsSkipped: (data.columns.islandsSkipped ? decodeTyped(data.columns.islandsSkipped) : new Uint32Array(data.durationFrames || 0)) as Uint32Array,
  };
  const bodies = decodeTyped(data.bodies) as Float32Array;
  const stride = data.bodyStride;
  const n = columns.bodyCount.length;
  const frameBodyOffset = new Uint32Array(n);
  let acc = 0;
  for (let i = 0; i < n; i += 1) {
    frameBodyOffset[i] = acc;
    acc += columns.bodyCount[i] * stride;
  }

  const frame = (i: number): Float32Array => {
    if (i < 0 || i >= n) return new Float32Array(0);
    const start = frameBodyOffset[i];
    return bodies.subarray(start, start + columns.bodyCount[i] * stride);
  };

  const bodyInFrame = (frameIndex: number, handle: number): Float32Array | null => {
    const rows = frame(frameIndex);
    for (let off = 0; off < rows.length; off += stride) {
      if (rows[off] === handle) return rows.subarray(off, off + stride);
    }
    return null;
  };

  const timing: Record<string, Float32Array> = {};
  if (data.timing) {
    for (const f of data.timing.fields) {
      const enc = data.timing.columns[f];
      if (enc) timing[f] = decodeTyped(enc) as Float32Array;
    }
  }

  return {
    durationFrames: data.durationFrames,
    durationSeconds: data.durationSeconds,
    bodyStride: stride,
    bodyLayout: data.bodyLayout,
    handleTable: (data.handleTable ? decodeTyped(data.handleTable) : new Float64Array(0)) as Float64Array,
    columns,
    bodies,
    frameBodyOffset,
    events: data.events,
    timing,
    resimLog: data.resimLog ?? [],
    frame,
    bodyInFrame,
  };
}

/** gzip a JSON-serialisable value to a Blob (browser `CompressionStream`), or
 *  fall back to an uncompressed JSON Blob where compression is unavailable.
 *  Returns the blob and whether it was actually gzipped. */
export async function gzipJson(
  value: unknown,
): Promise<{ blob: Blob; gzipped: boolean }> {
  const json = JSON.stringify(value);
  const CS = (globalThis as { CompressionStream?: typeof CompressionStream }).CompressionStream;
  if (typeof CS === 'function' && typeof Blob !== 'undefined' && typeof Response !== 'undefined') {
    try {
      const stream = new Blob([json]).stream().pipeThrough(new CS('gzip'));
      const blob = await new Response(stream).blob();
      return { blob, gzipped: true };
    } catch {
      /* fall through to uncompressed */
    }
  }
  return { blob: new Blob([json], { type: 'application/json' }), gzipped: false };
}
