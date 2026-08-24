//! The process-wide PhysX runtime must survive concurrent world construction.
//!
//! `PxFoundation` and `PxPhysics` are created once per process and shared by
//! every world. The first version of that singleton checked `!foundation`
//! without a lock, so two threads constructing worlds at the same time both
//! took the create branch and the process died with SIGSEGV. It surfaced as an
//! intermittent failure of `byo_world_test` -- three worlds, three cargo test
//! threads -- and would hit any host that builds destructibles from a pool.
#![cfg(feature = "physx")]

use std::sync::{Arc, Barrier};

use blast_stress_solver::types::Vec3;
use blast_stress_solver::backends::PhysXWorld;

const G: Vec3 = Vec3 { x: 0.0, y: -9.81, z: 0.0 };

#[test]
fn worlds_can_be_created_concurrently_from_many_threads() {
    const THREADS: usize = 8;
    // A barrier, not just spawn order: without it the threads trickle in and
    // the very race we are testing for is unlikely to be hit.
    let gate = Arc::new(Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || {
                gate.wait();
                let world = PhysXWorld::new_cpu(G, 1);
                world.is_some()
            })
        })
        .collect();

    let built = handles
        .into_iter()
        .map(|h| h.join().expect("world-construction thread panicked or crashed"))
        .filter(|ok| *ok)
        .count();
    assert_eq!(built, THREADS, "every thread must get a usable world");
}

#[test]
fn worlds_can_be_created_and_dropped_concurrently() {
    // Teardown returns memory to the same shared PxPhysics, so it races
    // creation just as creation races itself. Interleave the two.
    const THREADS: usize = 8;
    let gate = Arc::new(Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
        .map(|i| {
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || {
                gate.wait();
                for _ in 0..4 {
                    let world = PhysXWorld::new_cpu(G, 1);
                    assert!(world.is_some(), "thread {i} failed to build a world");
                    drop(world);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("create/drop thread panicked or crashed");
    }
}
