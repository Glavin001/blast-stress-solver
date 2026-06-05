//! Boundary coverage for `FracturePolicy` — the per-frame budgets and child-admission
//! limits. These pure predicates gate every fracture/split, so an off-by-one (`>` vs `>=`)
//! or a mishandled "0 means unlimited?" case silently breaks the limiter. None of this was
//! tested before.

use blast_stress_solver::rapier::FracturePolicy;

fn policy() -> FracturePolicy {
    FracturePolicy::default()
}

#[test]
fn should_suppress_at_the_dynamic_body_cap() {
    let p = FracturePolicy { max_dynamic_bodies: 5, ..policy() };
    assert!(!p.should_suppress(4), "below cap: allow");
    assert!(p.should_suppress(5), "at cap: suppress (>=)");
    assert!(p.should_suppress(6), "above cap: suppress");
}

#[test]
fn dynamic_body_cap_of_zero_or_negative_is_unlimited() {
    // The guard is `max > 0`, so 0 and -1 both mean "no cap".
    assert!(!FracturePolicy { max_dynamic_bodies: 0, ..policy() }.should_suppress(1000));
    assert!(!FracturePolicy { max_dynamic_bodies: -1, ..policy() }.should_suppress(1000));
}

#[test]
fn clamp_fractures_respects_the_budget() {
    assert_eq!(FracturePolicy { max_fractures_per_frame: -1, ..policy() }.clamp_fractures(500), 500);
    assert_eq!(FracturePolicy { max_fractures_per_frame: 0, ..policy() }.clamp_fractures(500), 0);
    let p = FracturePolicy { max_fractures_per_frame: 3, ..policy() };
    assert_eq!(p.clamp_fractures(5), 3);
    assert_eq!(p.clamp_fractures(3), 3);
    assert_eq!(p.clamp_fractures(2), 2);
}

#[test]
fn clamp_new_bodies_and_migrations_respect_budgets() {
    let p = FracturePolicy { max_new_bodies_per_frame: 2, max_collider_migrations_per_frame: 4, ..policy() };
    assert_eq!(p.clamp_new_bodies(10), 2);
    assert_eq!(p.clamp_new_bodies(1), 1);
    assert_eq!(p.clamp_collider_migrations(10), 4);
    assert_eq!(FracturePolicy { max_new_bodies_per_frame: -1, ..policy() }.clamp_new_bodies(usize::MAX), usize::MAX);
}

#[test]
fn child_qualifies_at_the_min_node_count() {
    let p = FracturePolicy { min_child_node_count: 3, ..policy() };
    assert!(!p.child_qualifies(2), "below min: reject");
    assert!(p.child_qualifies(3), "at min: accept (>=)");
    assert!(p.child_qualifies(4), "above min: accept");
    // Default min is 1, so single-node children qualify but zero-node ones never do.
    assert!(policy().child_qualifies(1));
    assert!(!policy().child_qualifies(0));
}

#[test]
fn split_budgets_unlimited_only_when_both_unlimited() {
    assert!(FracturePolicy { max_new_bodies_per_frame: -1, max_collider_migrations_per_frame: -1, ..policy() }.split_budgets_unlimited());
    assert!(!FracturePolicy { max_new_bodies_per_frame: 0, max_collider_migrations_per_frame: -1, ..policy() }.split_budgets_unlimited());
    assert!(!FracturePolicy { max_new_bodies_per_frame: -1, max_collider_migrations_per_frame: 0, ..policy() }.split_budgets_unlimited());
}
