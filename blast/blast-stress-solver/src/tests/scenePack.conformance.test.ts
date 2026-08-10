/**
 * Cross-runtime ScenePack conformance — TypeScript side.
 *
 * The ScenePack JSON is the contract that lets the same structure run under
 * TS/Rapier, Rust/Rapier and C++/PhysX so their APIs, behavior and performance
 * can be compared without the structure itself being a variable. That only
 * holds if all three loaders interpret the file identically.
 *
 * All three load the SAME fixture and assert the SAME digest. If a loader
 * drifts — a silently ignored field, an off-by-one index, a misread unit —
 * that runtime's own suite fails.
 *
 * The digest pins interpretation of the ASSET, not simulation results: Rapier
 * and PhysX legitimately produce different trajectories from identical input.
 *
 * See SCENE_PACK_FORMAT.md ("Conformance") and the sibling tests:
 *   demos/blast-stress-demo/tests/scene_pack_conformance_test.cpp
 *   blast/blast-stress-demo-rs/tests/scene_pack_conformance.rs
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import { parseScenePackJson } from '../rapier/scenePackLoader';

const here = path.dirname(fileURLToPath(import.meta.url));
const conformanceDir = path.resolve(here, '../../assets/conformance');
const fixturePath = path.join(conformanceDir, 'structure-conformance-v2.json');
const goldenPath = path.join(conformanceDir, 'structure-conformance-v2.digest.json');

type Digest = {
  version: number;
  nodeCount: number;
  bondCount: number;
  materialCount: number;
  materialNames: string[];
  supportNodeCount: number;
  totalMassKg: number;
  totalBondAreaM2: number;
  bondsPerMaterial: number[];
  bondsPerJointClass: Record<string, number>;
  gravity: number;
  contactForceScale: number;
};

describe('ScenePack cross-runtime conformance', () => {
  const golden: Digest = JSON.parse(readFileSync(goldenPath, 'utf8'));
  const pack = parseScenePackJson(readFileSync(fixturePath, 'utf8'));

  it('matches the golden digest', () => {
    const bondsPerMaterial = golden.bondsPerMaterial.map(() => 0);
    for (const m of pack.bondMaterials) bondsPerMaterial[m] += 1;

    const bondsPerJointClass: Record<string, number> = {};
    for (const b of pack.scenario.bonds) {
      const key = [pack.nodeTypes[b.node0], pack.nodeTypes[b.node1]].sort().join('~');
      bondsPerJointClass[key] = (bondsPerJointClass[key] ?? 0) + 1;
    }

    const digest: Digest = {
      version: pack.version,
      nodeCount: pack.scenario.nodes.length,
      bondCount: pack.scenario.bonds.length,
      materialCount: pack.materials.length,
      materialNames: pack.materials.map((m) => m.name),
      supportNodeCount: pack.scenario.nodes.filter((n) => n.mass === 0).length,
      totalMassKg: round6(pack.scenario.nodes.reduce((sum, n) => sum + n.mass, 0)),
      totalBondAreaM2: round6(pack.scenario.bonds.reduce((sum, b) => sum + b.area, 0)),
      bondsPerMaterial,
      bondsPerJointClass: sortKeys(bondsPerJointClass),
      gravity: pack.defaults.gravity,
      contactForceScale: pack.defaults.physics.contactForceScale,
    };

    expect(digest).toEqual({ ...golden, bondsPerJointClass: sortKeys(golden.bondsPerJointClass) });
  });

  it('routes material indices to the right bonds, not just the right counts', () => {
    // Footings use the frame default; the facade clips are drywall-track.
    expect(pack.materials[pack.bondMaterials[0]].name).toBe('reinforced-concrete');
    expect(pack.materials[pack.bondMaterials[4]].name).toBe('drywall-track');
  });

  it('treats an omitted `m` as material 0 rather than unset', () => {
    // Bonds 0 and 1 carry no `m` field at all in the fixture.
    expect(pack.bondMaterials[0]).toBe(0);
    expect(pack.bondMaterials[1]).toBe(0);
  });

  it('rejects a v2 pack with no materials', () => {
    const broken = JSON.parse(readFileSync(fixturePath, 'utf8'));
    delete broken.defaults.solver.materials;
    expect(() => parseScenePackJson(JSON.stringify(broken))).toThrow(/materials is required/);
  });

  it('rejects an out-of-range bond material rather than clamping it', () => {
    // A silent clamp to material 0 would turn an authoring typo into a
    // mysteriously strong joint — the exact class of bug this format exists
    // to make impossible.
    const broken = JSON.parse(readFileSync(fixturePath, 'utf8'));
    broken.scenario.bonds[2].m = 99;
    expect(() => parseScenePackJson(JSON.stringify(broken))).toThrow(/references material 99/);
  });

  it('still loads v1 packs by synthesizing a one-entry material table', () => {
    const v1 = JSON.parse(readFileSync(fixturePath, 'utf8'));
    v1.version = 1;
    v1.defaults.solver.limits = v1.defaults.solver.materials[1];
    delete v1.defaults.solver.materials;
    for (const b of v1.scenario.bonds) delete b.m;

    const loaded = parseScenePackJson(JSON.stringify(v1));
    expect(loaded.version).toBe(1);
    expect(loaded.materials).toHaveLength(1);
    expect(loaded.bondMaterials.every((m) => m === 0)).toBe(true);
    expect(loaded.defaults.solverSettings?.compressionElasticLimit).toBe(12e6);
  });
});

function round6(value: number): number {
  return Math.round(value * 1e6) / 1e6;
}

function sortKeys(record: Record<string, number>): Record<string, number> {
  return Object.fromEntries(Object.entries(record).sort(([a], [b]) => a.localeCompare(b)));
}
