use std::collections::{HashMap, HashSet};

use rapier3d::prelude::RigidBodyHandle;

use crate::types::SplitChild;

/// Body state before a split.
pub struct ExistingBodyState {
    pub handle: RigidBodyHandle,
    pub node_indices: HashSet<u32>,
    pub is_fixed: bool,
}

/// Plan for how to handle split children: reuse existing bodies or create new ones.
pub struct SplitMigrationPlan {
    /// Children that can reuse an existing body (same node set).
    pub reuse: Vec<ReuseEntry>,
    /// Children that need a new body created.
    pub create: Vec<CreateEntry>,
}

pub struct ReuseEntry {
    pub child_index: usize,
    pub body_handle: RigidBodyHandle,
}

pub struct CreateEntry {
    pub child_index: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PlannerChildSupport {
    pub is_support: bool,
}

/// Plan body reuse vs. creation for split children.
///
/// The planner optimizes for minimal edit distance on the Rapier side:
/// preserve exact matches first, then maximize node overlap with existing bodies.
/// Support status is handled during reconciliation by changing the body type
/// in-place if needed, so fixed bodies may be reused for dynamic children.
pub fn plan_split_migration(
    bodies: &[ExistingBodyState],
    children: &[SplitChild],
) -> SplitMigrationPlan {
    plan_split_migration_with_support(
        bodies,
        children,
        &vec![PlannerChildSupport::default(); children.len()],
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlannerMode {
    /// Shipping path: solve the assignment per connected component of the body↔child
    /// overlap graph, with argmax shortcuts for single-body / single-child components.
    Production,
    /// Reference path (test/bench only): one square-padded dense Hungarian over all
    /// unmatched bodies × children. Kept as an independent oracle for the tests.
    #[cfg(any(test, feature = "bench-support"))]
    ReferenceDense,
}

pub fn plan_split_migration_with_support(
    bodies: &[ExistingBodyState],
    children: &[SplitChild],
    child_support: &[PlannerChildSupport],
) -> SplitMigrationPlan {
    plan_split_migration_inner(bodies, children, child_support, PlannerMode::Production)
}

/// Test/bench-only entry point that forces the original square-padded dense Hungarian over
/// the whole unmatched set (no component decomposition, no fast path). Kept as an
/// independent reference to prove the shipping planner is equally optimal, and to A/B
/// their cost at scale. Never compiled into shipped builds.
#[cfg(any(test, feature = "bench-support"))]
pub fn plan_split_migration_reference(
    bodies: &[ExistingBodyState],
    children: &[SplitChild],
    child_support: &[PlannerChildSupport],
) -> SplitMigrationPlan {
    plan_split_migration_inner(bodies, children, child_support, PlannerMode::ReferenceDense)
}

fn plan_split_migration_inner(
    bodies: &[ExistingBodyState],
    children: &[SplitChild],
    _child_support: &[PlannerChildSupport],
    mode: PlannerMode,
) -> SplitMigrationPlan {
    if bodies.is_empty() || children.is_empty() {
        return SplitMigrationPlan {
            reuse: Vec::new(),
            create: (0..children.len())
                .map(|i| CreateEntry { child_index: i })
                .collect(),
        };
    }

    // Build hash index of existing bodies
    let mut body_hashes: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, body) in bodies.iter().enumerate() {
        let hash = hash_node_set(&body.node_indices);
        body_hashes.entry(hash).or_default().push(i);
    }

    let mut reuse = Vec::new();
    let mut assigned_bodies = HashSet::new();
    let mut unmatched_children = Vec::new();

    for (ci, child) in children.iter().enumerate() {
        let child_set: HashSet<u32> = child.nodes.iter().copied().collect();
        let hash = hash_node_set(&child_set);

        let mut matched = false;
        if let Some(candidates) = body_hashes.get(&hash) {
            for &bi in candidates {
                if assigned_bodies.contains(&bi) {
                    continue;
                }
                if bodies[bi].node_indices == child_set {
                    reuse.push(ReuseEntry {
                        child_index: ci,
                        body_handle: bodies[bi].handle,
                    });
                    assigned_bodies.insert(bi);
                    matched = true;
                    break;
                }
            }
        }
        if !matched {
            unmatched_children.push(ci);
        }
    }

    let unmatched_bodies: Vec<usize> = bodies
        .iter()
        .enumerate()
        .filter_map(|(idx, _)| (!assigned_bodies.contains(&idx)).then_some(idx))
        .collect();

    if !unmatched_bodies.is_empty() && !unmatched_children.is_empty() {
        // Assign existing bodies to children by maximum node overlap (reusing a body keeps
        // its handle + colliders — the cheapest Rapier edit). The shipping path decomposes
        // the body↔child overlap graph into connected components and solves each in
        // isolation, which is provably identical to one global assignment — cross-component
        // pairs share no node, so they never help — but avoids the O(max(M,N)^3) blow-up of
        // a single square-padded Hungarian over the whole (usually sparse) set.
        let reused_children = match mode {
            PlannerMode::Production => assign_by_components(
                bodies,
                &unmatched_bodies,
                children,
                &unmatched_children,
                &mut reuse,
            ),
            #[cfg(any(test, feature = "bench-support"))]
            PlannerMode::ReferenceDense => assign_dense_hungarian(
                bodies,
                &unmatched_bodies,
                children,
                &unmatched_children,
                &mut reuse,
            ),
        };

        let create = unmatched_children
            .into_iter()
            .filter(|child_index| !reused_children.contains(child_index))
            .map(|child_index| CreateEntry { child_index })
            .collect();
        return SplitMigrationPlan { reuse, create };
    }

    let create = unmatched_children
        .into_iter()
        .map(|child_index| CreateEntry { child_index })
        .collect();
    SplitMigrationPlan { reuse, create }
}

/// Maximum-overlap assignment via connected-component decomposition of the body↔child
/// overlap graph. Each component is solved independently: single-body and single-child
/// components by argmax (O(size)); only a component with multiple bodies AND multiple
/// mutually-overlapping children falls back to a Hungarian, and only over that small
/// component. This is equivalent to a single global maximum-overlap matching, because an
/// optimal matching never pairs a body with a 0-overlap child — so no beneficial edge ever
/// crosses a component boundary — while avoiding the O(max(M,N)^3) cost of one dense
/// square-padded Hungarian over the whole (usually sparse) unmatched set.
///
/// Pushes the chosen reuse pairs into `reuse` (ascending child index, for a deterministic
/// plan) and returns the set of reused global child indices.
fn assign_by_components(
    bodies: &[ExistingBodyState],
    unmatched_bodies: &[usize],
    children: &[SplitChild],
    unmatched_children: &[usize],
    reuse: &mut Vec<ReuseEntry>,
) -> HashSet<usize> {
    let nb = unmatched_bodies.len();
    let nc = unmatched_children.len();

    // node -> local body index. Existing bodies are disjoint node sets (each node lives on
    // exactly one rigid body), so a node maps to at most one unmatched body.
    let mut node_to_local_body: HashMap<u32, usize> = HashMap::new();
    for (bi, &gb) in unmatched_bodies.iter().enumerate() {
        for &node in &bodies[gb].node_indices {
            node_to_local_body.insert(node, bi);
        }
    }

    // Sparse overlaps (per child: body local index -> shared node count) + union the body
    // and child into one component for every positive overlap edge.
    let mut overlaps: Vec<HashMap<usize, usize>> = vec![HashMap::new(); nc];
    let mut uf = UnionFind::new(nb + nc); // bodies are [0, nb), children are [nb, nb + nc)
    for (ci, &gc) in unmatched_children.iter().enumerate() {
        for &node in &children[gc].nodes {
            if let Some(&bi) = node_to_local_body.get(&node) {
                *overlaps[ci].entry(bi).or_insert(0) += 1;
                uf.union(bi, nb + ci);
            }
        }
    }

    // Group bodies and (overlapping) children by component root. A child with no overlap is
    // left out — it is created, not reused; a body in no edge stays unmatched and is retired.
    let mut comp_bodies: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut comp_children: HashMap<usize, Vec<usize>> = HashMap::new();
    for bi in 0..nb {
        comp_bodies.entry(uf.find(bi)).or_default().push(bi);
    }
    for ci in 0..nc {
        if !overlaps[ci].is_empty() {
            comp_children.entry(uf.find(nb + ci)).or_default().push(ci);
        }
    }

    // Solve each component (deterministic order: ascending root; members are already in
    // ascending local-index order from the loops above).
    let mut roots: Vec<usize> = comp_children.keys().copied().collect();
    roots.sort_unstable();
    let mut pairs: Vec<(usize, usize)> = Vec::new(); // (local body, local child)
    for root in roots {
        // Every child in `comp_children` was unioned with a body, so `comp_bodies` has it.
        solve_component(&comp_bodies[&root], &comp_children[&root], &overlaps, &mut pairs);
    }

    pairs.sort_unstable_by_key(|&(_, ci)| ci);
    let mut reused_children = HashSet::with_capacity(pairs.len());
    for (bi, ci) in pairs {
        reuse.push(ReuseEntry {
            child_index: unmatched_children[ci],
            body_handle: bodies[unmatched_bodies[bi]].handle,
        });
        reused_children.insert(unmatched_children[ci]);
    }
    reused_children
}

/// Optimal maximum-overlap matching within one component, appended to `pairs` as
/// (local body, local child). Argmax for single-body / single-child components (covering
/// the 1×N cascade and the N×1 merge); a small dense Hungarian otherwise.
fn solve_component(
    comp_bodies: &[usize],
    comp_children: &[usize],
    overlaps: &[HashMap<usize, usize>],
    pairs: &mut Vec<(usize, usize)>,
) {
    let ov = |bi: usize, ci: usize| overlaps[ci].get(&bi).copied().unwrap_or(0);

    if comp_bodies.len() == 1 {
        let bi = comp_bodies[0];
        let mut best: Option<(usize, usize)> = None; // (local child, overlap)
        for &ci in comp_children {
            let o = ov(bi, ci);
            if o > 0 && best.map_or(true, |(_, b)| o > b) {
                best = Some((ci, o));
            }
        }
        if let Some((ci, _)) = best {
            pairs.push((bi, ci));
        }
        return;
    }
    if comp_children.len() == 1 {
        let ci = comp_children[0];
        let mut best: Option<(usize, usize)> = None; // (local body, overlap)
        for &bi in comp_bodies {
            let o = ov(bi, ci);
            if o > 0 && best.map_or(true, |(_, b)| o > b) {
                best = Some((bi, o));
            }
        }
        if let Some((bi, _)) = best {
            pairs.push((bi, ci));
        }
        return;
    }

    // Dense component (multiple bodies AND children): Hungarian over just these members.
    let matrix: Vec<Vec<usize>> = comp_bodies
        .iter()
        .map(|&bi| comp_children.iter().map(|&ci| ov(bi, ci)).collect())
        .collect();
    for (row, assignment) in hungarian_max(&matrix).into_iter().enumerate() {
        let Some(col) = assignment else { continue };
        if matrix[row][col] == 0 {
            continue;
        }
        pairs.push((comp_bodies[row], comp_children[col]));
    }
}

/// Minimal union-find with path compression; `union` attaches the larger root under the
/// smaller, so a component's root is its lowest member (keeps component order stable).
struct UnionFind {
    parent: Vec<usize>,
}
impl UnionFind {
    fn new(n: usize) -> Self {
        Self { parent: (0..n).collect() }
    }
    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut cur = x;
        while self.parent[cur] != cur {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra.max(rb)] = ra.min(rb);
        }
    }
}

/// Reference assignment (test/bench only): the original single square-padded dense
/// Hungarian over all unmatched bodies × children. Kept as an independent oracle so the
/// component-decomposed shipping path can be proven equally optimal.
#[cfg(any(test, feature = "bench-support"))]
fn assign_dense_hungarian(
    bodies: &[ExistingBodyState],
    unmatched_bodies: &[usize],
    children: &[SplitChild],
    unmatched_children: &[usize],
    reuse: &mut Vec<ReuseEntry>,
) -> HashSet<usize> {
    let overlap = build_overlap_matrix(bodies, unmatched_bodies, children, unmatched_children);
    let assignments = hungarian_max(&overlap);
    let mut reused_children = HashSet::new();
    for (row_idx, assignment) in assignments.into_iter().enumerate() {
        let Some(col_idx) = assignment else { continue };
        let score = overlap.get(row_idx).and_then(|r| r.get(col_idx)).copied().unwrap_or(0);
        if score == 0 {
            continue;
        }
        reuse.push(ReuseEntry {
            child_index: unmatched_children[col_idx],
            body_handle: bodies[unmatched_bodies[row_idx]].handle,
        });
        reused_children.insert(unmatched_children[col_idx]);
    }
    reused_children
}

fn hash_node_set(nodes: &HashSet<u32>) -> u64 {
    let mut sorted: Vec<u32> = nodes.iter().copied().collect();
    sorted.sort_unstable();
    let mut hash = 0u64;
    for n in sorted {
        // Simple FNV-1a-like hash
        hash ^= n as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(any(test, feature = "bench-support"))]
fn build_overlap_matrix(
    bodies: &[ExistingBodyState],
    unmatched_bodies: &[usize],
    children: &[SplitChild],
    unmatched_children: &[usize],
) -> Vec<Vec<usize>> {
    unmatched_bodies
        .iter()
        .map(|&body_idx| {
            let body = &bodies[body_idx];
            unmatched_children
                .iter()
                .map(|&child_idx| {
                    children[child_idx]
                        .nodes
                        .iter()
                        .filter(|node| body.node_indices.contains(node))
                        .count()
                })
                .collect()
        })
        .collect()
}

fn hungarian_max(matrix: &[Vec<usize>]) -> Vec<Option<usize>> {
    let rows = matrix.len();
    let cols = matrix.first().map(Vec::len).unwrap_or(0);
    if rows == 0 || cols == 0 {
        return vec![None; rows];
    }

    let size = rows.max(cols);
    let max_val = matrix
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .max()
        .unwrap_or(0) as i64;
    let mut cost = vec![vec![max_val; size]; size];
    for (i, row) in matrix.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            cost[i][j] = max_val - (*value as i64);
        }
    }

    let assignments = hungarian(&cost);
    assignments
        .into_iter()
        .take(rows)
        .map(|col| (col < cols).then_some(col))
        .collect()
}

fn hungarian(cost: &[Vec<i64>]) -> Vec<usize> {
    let size = cost.len();
    let mut u = vec![0i64; size + 1];
    let mut v = vec![0i64; size + 1];
    let mut p = vec![0usize; size + 1];
    let mut way = vec![0usize; size + 1];

    for i in 1..=size {
        p[0] = i;
        let mut minv = vec![i64::MAX; size + 1];
        let mut used = vec![false; size + 1];
        let mut j0 = 0usize;
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = i64::MAX;
            let mut j1 = 0usize;
            for j in 1..=size {
                if used[j] {
                    continue;
                }
                let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for j in 0..=size {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else if minv[j] != i64::MAX {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut result = vec![usize::MAX; size];
    for j in 1..=size {
        if p[j] > 0 {
            result[p[j] - 1] = j - 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    //! Focused, no-simulation tests for the split planner — the topology-diff that
    //! decides which existing Rapier bodies are *reused* (handle + colliders stay put,
    //! the cheapest edit) vs. which children get *new* bodies. We drive the planner
    //! directly with contrived fractures, so we can scale to large/intricate cases and,
    //! crucially, assert that the single-parent fast path is never bought at the expense
    //! of the genuinely hard case: many surviving multi-node bodies competing for many
    //! reparented children (a true M×N maximum-overlap assignment).
    //!
    //! Fractures are modelled realistically: every node belongs to exactly one existing
    //! body *and* exactly one child (a disjoint partition), so total reuse-overlap is a
    //! well-defined objective and an independent brute-force optimum is meaningful.

    use super::*;
    use rapier3d::prelude::{RigidBodyBuilder, RigidBodyHandle, RigidBodySet};

    fn mk_bodies(sets: &[Vec<u32>], rbs: &mut RigidBodySet) -> Vec<ExistingBodyState> {
        sets.iter()
            .map(|nodes| ExistingBodyState {
                handle: rbs.insert(RigidBodyBuilder::dynamic()),
                node_indices: nodes.iter().copied().collect(),
                is_fixed: false,
            })
            .collect()
    }

    fn mk_children(sets: &[Vec<u32>]) -> Vec<SplitChild> {
        sets.iter()
            .enumerate()
            .map(|(i, nodes)| SplitChild { actor_index: i as u32, nodes: nodes.clone() })
            .collect()
    }

    fn support(children: &[SplitChild]) -> Vec<PlannerChildSupport> {
        vec![PlannerChildSupport::default(); children.len()]
    }

    fn plan(bodies: &[ExistingBodyState], children: &[SplitChild]) -> SplitMigrationPlan {
        plan_split_migration_with_support(bodies, children, &support(children))
    }

    /// The shipping planner with the fast path bypassed — the reference algorithm.
    fn plan_ref(bodies: &[ExistingBodyState], children: &[SplitChild]) -> SplitMigrationPlan {
        plan_split_migration_reference(bodies, children, &support(children))
    }

    fn overlap(body: &ExistingBodyState, child: &SplitChild) -> usize {
        child.nodes.iter().filter(|n| body.node_indices.contains(n)).count()
    }

    /// Assert the plan is a valid partition of the children (each assigned exactly once,
    /// reuse XOR create; each body reused at most once; reused pairs share ≥1 node) and
    /// return the total reuse-overlap, which the planner maximizes — and which is the
    /// proxy for "colliders that stay attached to their body" (the cheapest Rapier edit).
    fn validate_and_score(
        p: &SplitMigrationPlan,
        bodies: &[ExistingBodyState],
        children: &[SplitChild],
    ) -> usize {
        let mut assigned = vec![0u32; children.len()];
        for r in &p.reuse {
            assigned[r.child_index] += 1;
        }
        for c in &p.create {
            assigned[c.child_index] += 1;
        }
        assert!(
            assigned.iter().all(|&t| t == 1),
            "each child must be assigned exactly once (reuse XOR create): {assigned:?}"
        );

        let mut reuse_count: std::collections::HashMap<RigidBodyHandle, u32> = Default::default();
        let mut total = 0usize;
        for r in &p.reuse {
            *reuse_count.entry(r.body_handle).or_insert(0) += 1;
            let body = bodies
                .iter()
                .find(|b| b.handle == r.body_handle)
                .expect("reused handle must be an existing body");
            let ov = overlap(body, &children[r.child_index]);
            assert!(ov > 0, "a reused (body,child) pair must share at least one node");
            total += ov;
        }
        assert!(
            reuse_count.values().all(|&c| c <= 1),
            "an existing body may be reused by at most one child"
        );
        total
    }

    /// Independent brute-force optimum: max total overlap over all body→distinct-child
    /// assignments (a body may also stay unmatched). Exponential — small instances only.
    fn brute_optimum(bodies: &[ExistingBodyState], children: &[SplitChild]) -> usize {
        fn rec(bi: usize, b: &[ExistingBodyState], c: &[SplitChild], used: &mut [bool]) -> usize {
            if bi == b.len() {
                return 0;
            }
            let mut best = rec(bi + 1, b, c, used); // body bi takes no child
            for ci in 0..c.len() {
                if used[ci] {
                    continue;
                }
                let ov = overlap(&b[bi], &c[ci]);
                if ov == 0 {
                    continue;
                }
                used[ci] = true;
                best = best.max(ov + rec(bi + 1, b, c, used));
                used[ci] = false;
            }
            best
        }
        let mut used = vec![false; children.len()];
        rec(0, bodies, children, &mut used)
    }

    fn reuse_pairs(p: &SplitMigrationPlan) -> Vec<(usize, RigidBodyHandle)> {
        let mut v: Vec<_> = p.reuse.iter().map(|r| (r.child_index, r.body_handle)).collect();
        v.sort_by_key(|(c, _)| *c);
        v
    }
    fn create_set(p: &SplitMigrationPlan) -> Vec<usize> {
        let mut v: Vec<_> = p.create.iter().map(|c| c.child_index).collect();
        v.sort_unstable();
        v
    }

    // === 1. Degenerate cascade (the fast path): one parent -> N children. =========
    #[test]
    fn cascade_fastpath_equals_reference_and_is_optimal() {
        let mut rbs = RigidBodySet::new();
        for n in [2u32, 5, 33, 256] {
            let bodies = mk_bodies(&[(0..n).collect()], &mut rbs);
            let children = mk_children(&(0..n).map(|k| vec![k]).collect::<Vec<_>>());

            let fast = plan(&bodies, &children);
            let reference = plan_ref(&bodies, &children);
            let s_fast = validate_and_score(&fast, &bodies, &children);
            let s_ref = validate_and_score(&reference, &bodies, &children);

            assert_eq!(fast.reuse.len(), 1, "n={n}: the one body reuses one child");
            assert_eq!(fast.create.len(), (n - 1) as usize, "n={n}: the rest are created");
            assert_eq!(s_fast, 1, "n={n}: best single-node overlap is 1");
            assert_eq!(s_fast, s_ref, "n={n}: fast path must be as optimal as the Hungarian");
            if n <= 8 {
                assert_eq!(s_fast, brute_optimum(&bodies, &children), "n={n}: globally optimal");
            }
        }
    }

    // === 2. Complex partial reparenting (the real worst case). ====================
    #[test]
    fn complex_reparenting_is_optimal_and_matches_reference() {
        let mut rbs = RigidBodySet::new();
        // 4 bodies partition nodes 0..16; a shear regroups them into 4 children that each
        // straddle two old bodies — a genuine 4×4 reparenting assignment (no exact match,
        // so the fast path must NOT fire; the Hungarian must be globally optimal).
        let bodies = mk_bodies(
            &[vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11], vec![12, 13, 14, 15]],
            &mut rbs,
        );
        let children = mk_children(&[
            vec![0, 1, 4],            // body0:2  body1:1  -> body0
            vec![2, 3, 5, 6, 7],      // body0:2  body1:3  -> body1
            vec![8, 9, 12],           // body2:2  body3:1  -> body2
            vec![10, 11, 13, 14, 15], // body2:2  body3:3  -> body3
        ]);

        let fast = plan(&bodies, &children);
        let reference = plan_ref(&bodies, &children);
        let s_fast = validate_and_score(&fast, &bodies, &children);
        let s_ref = validate_and_score(&reference, &bodies, &children);

        assert_eq!(s_fast, brute_optimum(&bodies, &children), "must be globally optimal");
        assert_eq!(s_fast, s_ref, "fast and reference agree on the multi-body case");
        assert_eq!(create_set(&fast), Vec::<usize>::new(), "all 4 chunks reuse a body");
        assert_eq!(reuse_pairs(&fast), reuse_pairs(&reference), "identical (no ties here)");
    }

    // === 3. Realistic mixed scene: most bodies survive, one interior body shatters. =
    #[test]
    fn mostly_persist_one_shatters_uses_fastpath_correctly() {
        let mut rbs = RigidBodySet::new();
        // bodies 0..4 persist unchanged; body 4 (nodes 100..108) shatters into 8 shards.
        let persist = [vec![0u32, 1], vec![2, 3], vec![4, 5], vec![6, 7]];
        let mut bsets = persist.to_vec();
        bsets.push((100..108).collect());
        let bodies = mk_bodies(&bsets, &mut rbs);

        let mut csets = persist.to_vec(); // the 4 survivors (exact matches)
        csets.extend((100..108u32).map(|k| vec![k])); // + 8 shards
        let children = mk_children(&csets);

        let fast = plan(&bodies, &children);
        let s_fast = validate_and_score(&fast, &bodies, &children);
        let s_ref = validate_and_score(&plan_ref(&bodies, &children), &bodies, &children);

        // Survivors must be reused with their *own* body (never needlessly recreated):
        for (ci, _) in persist.iter().enumerate() {
            let r = fast.reuse.iter().find(|r| r.child_index == ci).expect("survivor reused");
            assert_eq!(bodies[ci].handle, r.body_handle, "exact match reuses its own body");
        }
        // After exact matches drop out, only the shattered body is unmatched -> fast path
        // legitimately fires inside a complex scene: it reuses 1 shard, creates 7.
        assert_eq!(fast.create.len(), 7);
        assert_eq!(s_fast, s_ref, "fast (fast-path firing) equals the reference");
        assert_eq!(s_fast, brute_optimum(&bodies, &children), "globally optimal (= 8 + 1)");
    }

    // === 4. Ties: equal-overlap children -> equally optimal, identical edit count. =
    #[test]
    fn ties_are_resolved_equally_optimally() {
        let mut rbs = RigidBodySet::new();
        let bodies = mk_bodies(&[vec![0, 1, 2, 3]], &mut rbs);
        let children = mk_children(&[vec![0, 1], vec![2, 3]]); // both overlap the body by 2
        let fast = plan(&bodies, &children);
        let s_fast = validate_and_score(&fast, &bodies, &children);
        let s_ref = validate_and_score(&plan_ref(&bodies, &children), &bodies, &children);
        assert_eq!((fast.reuse.len(), fast.create.len()), (1, 1));
        assert_eq!(s_fast, 2);
        assert_eq!(s_fast, s_ref, "tie resolved with equal total overlap & edit count");
        assert_eq!(s_fast, brute_optimum(&bodies, &children));
    }

    // === 5. Exact node-set matches must always be reused (cheapest possible edit). =
    #[test]
    fn exact_matches_are_always_reused() {
        let mut rbs = RigidBodySet::new();
        let bodies = mk_bodies(&[vec![0, 1, 2], vec![3, 4], vec![5, 6, 7, 8]], &mut rbs);
        let children = mk_children(&[vec![5, 6, 7, 8], vec![0, 1, 2], vec![3, 4]]); // permuted
        let fast = plan(&bodies, &children);
        validate_and_score(&fast, &bodies, &children);
        assert!(fast.create.is_empty(), "all children are exact matches -> zero creates");
        assert_eq!(fast.reuse.len(), 3);
        for r in &fast.reuse {
            let body = bodies.iter().find(|b| b.handle == r.body_handle).unwrap();
            let cset: std::collections::HashSet<u32> =
                children[r.child_index].nodes.iter().copied().collect();
            assert_eq!(body.node_indices, cset, "exact match reused the identical body");
        }
    }

    // === 6. Randomized property sweep — the core anti-regression guard. ===========
    #[test]
    fn randomized_fractures_planner_equals_reference_and_optimum() {
        // dependency-free deterministic PRNG (SplitMix64).
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            }
            fn below(&mut self, n: u32) -> u32 {
                (self.next() % n as u64) as u32
            }
        }

        let mut rng = Rng(0xDEAD_BEEF);
        let mut checked = 0;
        for case in 0..400 {
            let node_count = 1 + rng.below(24);
            let nb = 1 + rng.below(6); // up to 6 bodies  } small enough for brute force,
            let nc = 1 + rng.below(7); // up to 7 children } spans degenerate..complex
            // Disjoint partition: each node gets one body label and one child label.
            let mut bn: Vec<Vec<u32>> = vec![Vec::new(); nb as usize];
            let mut cn: Vec<Vec<u32>> = vec![Vec::new(); nc as usize];
            for node in 0..node_count {
                bn[rng.below(nb) as usize].push(node);
                cn[rng.below(nc) as usize].push(node);
            }
            let bsets: Vec<Vec<u32>> = bn.into_iter().filter(|v| !v.is_empty()).collect();
            let csets: Vec<Vec<u32>> = cn.into_iter().filter(|v| !v.is_empty()).collect();
            if bsets.is_empty() || csets.is_empty() {
                continue;
            }
            let mut rbs = RigidBodySet::new();
            let bodies = mk_bodies(&bsets, &mut rbs);
            let children = mk_children(&csets);

            let fast = plan(&bodies, &children);
            let reference = plan_ref(&bodies, &children);
            let s_fast = validate_and_score(&fast, &bodies, &children);
            let s_ref = validate_and_score(&reference, &bodies, &children);
            let opt = brute_optimum(&bodies, &children);

            assert_eq!(
                s_fast, opt,
                "case {case}: shipping planner not globally optimal\n bodies={bsets:?}\n children={csets:?}"
            );
            assert_eq!(
                s_ref, opt,
                "case {case}: reference not globally optimal\n bodies={bsets:?}\n children={csets:?}"
            );
            assert_eq!(
                s_fast, s_ref,
                "case {case}: fast path traded optimality vs the Hungarian\n bodies={bsets:?}\n children={csets:?}"
            );
            // determinism: identical plan on a repeat run.
            let again = plan(&bodies, &children);
            assert_eq!(reuse_pairs(&fast), reuse_pairs(&again), "case {case}: nondeterministic reuse");
            assert_eq!(create_set(&fast), create_set(&again), "case {case}: nondeterministic create");
            checked += 1;
        }
        assert!(checked > 300, "expected most random cases to be checked, got {checked}");
    }

    // === 7. Scale: fast path stays O(N) & correct at huge N; large complex stays valid. =
    #[test]
    fn scales_to_large_inputs() {
        // 7a. one body -> 8192 shards (fast path). Reuse exactly one, create the rest.
        let mut rbs = RigidBodySet::new();
        let bodies = mk_bodies(&[(0..8192).collect()], &mut rbs);
        let children = mk_children(&(0..8192u32).map(|k| vec![k]).collect::<Vec<_>>());
        let p = plan(&bodies, &children);
        validate_and_score(&p, &bodies, &children);
        assert_eq!((p.reuse.len(), p.create.len()), (1, 8191));

        // 7b. 64 bodies: 32 persist exactly, 32 reshuffle into children that straddle two
        // adjacent old bodies. After exact matches drop out a 32×32 Hungarian remains
        // (fast path must NOT fire). Assert valid + survivors reused + fast == reference.
        let mut rbs2 = RigidBodySet::new();
        let bsets: Vec<Vec<u32>> = (0..64u32).map(|k| vec![k * 2, k * 2 + 1]).collect();
        let bodies2 = mk_bodies(&bsets, &mut rbs2);
        let mut csets: Vec<Vec<u32>> = (0..32u32).map(|k| vec![k * 2, k * 2 + 1]).collect();
        csets.extend((32..64u32).map(|k| vec![k * 2, ((k + 1) % 64) * 2 + 1]));
        let children2 = mk_children(&csets);
        let p2 = plan(&bodies2, &children2);
        let s2 = validate_and_score(&p2, &bodies2, &children2);
        let s_ref2 = validate_and_score(&plan_ref(&bodies2, &children2), &bodies2, &children2);
        assert_eq!(s2, s_ref2, "large complex case: fast and reference agree");
        for k in 0..32usize {
            let r = p2.reuse.iter().find(|r| r.child_index == k).expect("survivor reused");
            assert_eq!(bodies2[k].handle, r.body_handle, "survivor keeps its own body");
        }
    }
}
