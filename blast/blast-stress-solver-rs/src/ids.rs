//! Stable network id packing, parameterised on the host's entity namespace.
//!
//! These ids are the library's own concepts -- structures, nodes, bonds and
//! island serials -- so the algebra belongs here. Only the *namespace tag* is
//! the application's, because it partitions the app's global entity space
//! against namespaces the library knows nothing about.
//!
//! It had been written four times over (a Rust copy, a C++ copy, a TypeScript
//! copy and the wire encoder) and the copies drifted, which is not a
//! hypothetical: two of them packed the node index into a 12-bit field after it
//! had been widened to 16 elsewhere. Past that limit the index carried into the
//! structure field and came back out masked to `node % 4096`, so a promoted
//! island claimed chunks from a different building hundreds of metres away and
//! drew them there. 74% of one pack's chunks were affected, and it was silent
//! in release because the guard was a `debug_assert`.
//!
//! Two rules follow from that, and both are load-bearing:
//!
//! - **Bounds are hard asserts, never `debug_assert`.** The bounds are fixed
//!   the moment a scene pack loads, so a violation is a startup-time authoring
//!   error that should fail immediately and loudly. As `debug_assert` it
//!   vanished from release and corrupted island membership silently for a whole
//!   match, which is far worse than a crash on load.
//! - **One layout, checked against real extremes.** The conformance vectors at
//!   the bottom of this file use the actual figures from the shipped packs
//!   (node 15,918; bond 74,543), not round numbers.
//!
//! # Porting note for non-Rust consumers
//!
//! A JavaScript port must emit *arithmetic*, not bitwise operators. JS bitwise
//! ops coerce to int32, so `0x8000_0000 | x` is negative and every comparison
//! downstream is wrong. Use `+ structure * 0x40_0000` and `Math.floor(...)`.

/// Bits available below a top-nibble namespace tag.
pub const ID_BITS: u32 = 28;
/// Mask for the payload under a top-nibble namespace tag.
pub const ID_MASK: u32 = (1u32 << ID_BITS) - 1;
/// Selects the namespace nibble.
pub const NAMESPACE_MASK: u32 = 0xf000_0000;

/// How ids are packed. `Copy`, and const-constructible so a host can pin its
/// layout at compile time.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IdLayout {
    namespace: u32,
    node_bits: u32,
    bond_bits: u32,
    serial_bits: u32,
}

impl IdLayout {
    /// # Panics
    /// At const-eval time if the layout cannot hold at least one structure, or
    /// if `namespace` is not a bare top-nibble tag. Both are authoring errors
    /// in the host's id plan, so they should never reach a running program.
    pub const fn new(namespace: u32, node_bits: u32, bond_bits: u32, serial_bits: u32) -> Self {
        assert!(namespace & ID_MASK == 0, "namespace must occupy only the top nibble");
        assert!(namespace != 0, "namespace 0 collides with untagged ids");
        assert!(node_bits < ID_BITS, "node field must leave room for a structure id");
        assert!(bond_bits < ID_BITS, "bond field must leave room for a structure id");
        assert!(serial_bits < ID_BITS, "serial field must leave room for a structure id");
        Self { namespace, node_bits, bond_bits, serial_bits }
    }

    pub const fn namespace(&self) -> u32 {
        self.namespace
    }

    pub const fn max_nodes_per_structure(&self) -> u32 {
        1u32 << self.node_bits
    }

    pub const fn max_bonds_per_structure(&self) -> u32 {
        1u32 << self.bond_bits
    }

    pub const fn max_island_serials(&self) -> u32 {
        1u32 << self.serial_bits
    }

    /// Structures addressable by *every* packing.
    ///
    /// The three packings leave different amounts of room -- with the default
    /// layout, 12 bits of structure in a chunk id, 8 in a bond id, 6 in a body
    /// entity. The usable space is the minimum, because a structure id has to
    /// mean the same thing in all three. Taking the per-field maximum instead
    /// would let a structure id round-trip through a chunk id and alias
    /// through a body entity, which is the drift this module exists to stop.
    pub const fn max_structures(&self) -> u32 {
        let a = ID_BITS - self.node_bits;
        let b = ID_BITS - self.bond_bits;
        let c = ID_BITS - self.serial_bits;
        let m = if a < b { a } else { b };
        let m = if m < c { m } else { c };
        1u32 << m
    }

    #[inline]
    fn check_structure(&self, structure: u32) {
        assert!(
            structure < self.max_structures(),
            "structure {structure} exceeds the id space ({} structures)",
            self.max_structures()
        );
    }

    /// Chunk id for a node. Carries no namespace tag: chunk ids index into the
    /// destruction ledger, they are not entities.
    #[inline]
    pub fn chunk_id(&self, structure: u32, node_index: u32) -> u32 {
        self.check_structure(structure);
        assert!(
            node_index < self.max_nodes_per_structure(),
            "node {node_index} exceeds {} nodes/structure",
            self.max_nodes_per_structure()
        );
        (structure << self.node_bits) | node_index
    }

    #[inline]
    pub fn chunk_id_parts(&self, chunk_id: u32) -> (u32, u32) {
        (chunk_id >> self.node_bits, chunk_id & (self.max_nodes_per_structure() - 1))
    }

    #[inline]
    pub fn bond_id(&self, structure: u32, bond_index: u32) -> u32 {
        self.check_structure(structure);
        assert!(
            bond_index < self.max_bonds_per_structure(),
            "bond {bond_index} exceeds {} bonds/structure",
            self.max_bonds_per_structure()
        );
        (structure << self.bond_bits) | bond_index
    }

    #[inline]
    pub fn bond_id_parts(&self, bond_id: u32) -> (u32, u32) {
        (bond_id >> self.bond_bits, bond_id & (self.max_bonds_per_structure() - 1))
    }

    /// Namespaced entity id for an island's body.
    ///
    /// Hard asserts here too. The original had these as `debug_assert` while
    /// the chunk and bond packers were hard -- an inconsistency with no
    /// justification, and on the field most likely to overflow in practice:
    /// serials are never reused, so the space is consumed by *cumulative* body
    /// creation rather than by how many bodies are live.
    #[inline]
    pub fn body_entity(&self, structure: u32, island_serial: u32) -> u32 {
        self.check_structure(structure);
        assert!(
            island_serial < self.max_island_serials(),
            "island serial {island_serial} exceeds {} serials; serials are never \
             reused, so this is cumulative body creation, not live body count",
            self.max_island_serials()
        );
        self.namespace | (structure << self.serial_bits) | island_serial
    }

    #[inline]
    pub fn is_body_entity(&self, entity: u32) -> bool {
        entity & NAMESPACE_MASK == self.namespace
    }

    /// # Panics
    /// If `entity` is not in this layout's namespace -- unpacking a foreign
    /// entity yields a structure and serial that look valid and are not.
    #[inline]
    pub fn body_entity_parts(&self, entity: u32) -> (u32, u32) {
        assert!(
            self.is_body_entity(entity),
            "entity {entity:#010x} is not in namespace {:#010x}",
            self.namespace
        );
        (
            (entity & ID_MASK) >> self.serial_bits,
            entity & (self.max_island_serials() - 1),
        )
    }

    /// Does an island serial fit? For callers that would rather degrade than
    /// abort -- e.g. a server deciding to recycle a match before the serial
    /// space runs out.
    #[inline]
    pub fn serial_fits(&self, island_serial: u64) -> bool {
        island_serial < self.max_island_serials() as u64
    }
}

/// The layout `/city` runs, and the default for a host with no reason to differ.
///
/// Widths are not round numbers; each was set by a real pack and a real bug:
///
/// - **node 16 bits.** A structure is one scene-pack *instance*, and an
///   authored city district is one structure with 15,918 nodes. 12 bits (4,096)
///   truncated it.
/// - **bond 20 bits.** Bonds outnumber nodes roughly 3:1, so the bond field
///   fills first: a dense 27-building downtown is 24,105 nodes but 74,543
///   bonds, already past 16 bits.
/// - **serial 22 bits, leaving 6 for the structure.** Deliberately lopsided.
///   We place 16 structures and will never place 64, whereas serials are
///   consumed cumulatively and are genuinely unbounded; past a wrap, a new body
///   aliases onto a live one and the client draws two chunk sets with one pose.
pub const DEFAULT_LAYOUT: IdLayout = IdLayout::new(0x8000_0000, 16, 20, 22);

/// Serial reserved for the intact support actor, which is anchored and never
/// transmitted. See [`crate::pipeline::events::DestructionEvent::IslandPromoted`]
/// -- consumers should filter on `anchored`, not on this value.
pub const SUPPORT_ISLAND_SERIAL: u32 = 0;

#[cfg(test)]
mod tests {
    use super::*;

    const L: IdLayout = DEFAULT_LAYOUT;

    #[test]
    fn the_default_layout_matches_the_shipped_one() {
        // Pins the numbers `/city` and its client are already running. Changing
        // any of these renumbers every entity mid-match.
        assert_eq!(L.namespace(), 0x8000_0000);
        assert_eq!(L.max_nodes_per_structure(), 1 << 16);
        assert_eq!(L.max_bonds_per_structure(), 1 << 20);
        assert_eq!(L.max_island_serials(), 1 << 22);
        assert_eq!(L.max_structures(), 1 << 6);
    }

    #[test]
    fn structure_space_is_the_minimum_across_packings() {
        // Chunk ids alone would allow 4096 structures and bond ids 256, but a
        // body entity allows 64. Anything above 64 aliases in one of the three.
        assert_eq!(ID_BITS - 16, 12);
        assert_eq!(ID_BITS - 20, 8);
        assert_eq!(ID_BITS - 22, 6);
        assert_eq!(L.max_structures(), 1 << 6);
    }

    #[test]
    fn ids_round_trip() {
        assert_eq!(L.chunk_id_parts(L.chunk_id(15, 203)), (15, 203));
        assert_eq!(L.bond_id_parts(L.bond_id(15, 545)), (15, 545));
        let e = L.body_entity(15, 42);
        assert!(L.is_body_entity(e));
        assert_eq!(L.body_entity_parts(e), (15, 42));
    }

    #[test]
    fn other_namespaces_are_not_mistaken_for_ours() {
        for ns in [0x1000_0000u32, 0x2000_0000, 0x4000_0000, 0x6000_0000, 0x7000_0000] {
            assert!(!L.is_body_entity(ns | 42));
        }
    }

    /// Conformance vectors at the real extremes of the shipped packs.
    #[test]
    fn district_and_downtown_packs_round_trip() {
        // fractured-district.json: 15,918 nodes in one structure.
        for node in [0, 1, 4_095, 4_096, 9_594, 15_917, 15_918, L.max_nodes_per_structure() - 1] {
            assert_eq!(L.chunk_id_parts(L.chunk_id(0, node)), (0, node), "node {node}");
            for structure in [0, 1, 5, L.max_structures() - 1] {
                assert_eq!(
                    L.chunk_id_parts(L.chunk_id(structure, node)),
                    (structure, node),
                    "node {node} @ structure {structure}"
                );
            }
        }
        // A dense downtown: 24,105 nodes, 74,543 bonds.
        for bond in [65_535, 65_536, 74_542, 74_543, L.max_bonds_per_structure() - 1] {
            assert_eq!(L.bond_id_parts(L.bond_id(0, bond)), (0, bond), "bond {bond}");
            assert_eq!(L.bond_id_parts(L.bond_id(5, bond)), (5, bond), "bond {bond} @ 5");
        }
        assert_eq!(L.chunk_id_parts(L.chunk_id(3, 24_104)), (3, 24_104));
    }

    #[test]
    fn every_packed_id_stays_inside_the_namespace_payload() {
        // Otherwise a chunk or bond id ORed with a namespace would collide with
        // a different namespace entirely.
        assert!(L.chunk_id(L.max_structures() - 1, L.max_nodes_per_structure() - 1) <= ID_MASK);
        assert!(L.bond_id(L.max_structures() - 1, L.max_bonds_per_structure() - 1) <= ID_MASK);
        let e = L.body_entity(L.max_structures() - 1, L.max_island_serials() - 1);
        assert_eq!(e & NAMESPACE_MASK, L.namespace());
    }

    #[test]
    fn consecutive_chunks_stay_dense_for_gap_coding() {
        // Packet encoders LEB128 the gaps, so consecutive chunks within one
        // structure must differ by 1 to stay a single byte. Island membership
        // never mixes structures, so the wider node field costs nothing here.
        assert_eq!(L.chunk_id(3, 101) - L.chunk_id(3, 100), 1);
    }

    #[test]
    #[should_panic(expected = "exceeds")]
    fn a_node_past_the_field_is_loud() {
        L.chunk_id(0, L.max_nodes_per_structure());
    }

    #[test]
    #[should_panic(expected = "exceeds")]
    fn a_bond_past_the_field_is_loud() {
        L.bond_id(0, L.max_bonds_per_structure());
    }

    /// The regression that motivated hard asserts on the serial field.
    #[test]
    #[should_panic(expected = "cumulative body creation")]
    fn a_serial_past_the_field_is_loud() {
        L.body_entity(0, L.max_island_serials());
    }

    #[test]
    #[should_panic(expected = "exceeds the id space")]
    fn a_structure_past_the_field_is_loud() {
        L.chunk_id(L.max_structures(), 0);
    }

    #[test]
    #[should_panic(expected = "not in namespace")]
    fn unpacking_a_foreign_entity_is_loud() {
        L.body_entity_parts(0x2000_0000 | 7);
    }

    /// A host may pin its own widths; the algebra must not assume the default.
    #[test]
    fn an_alternate_layout_is_self_consistent() {
        const ALT: IdLayout = IdLayout::new(0x3000_0000, 14, 18, 20);
        assert_eq!(ALT.max_structures(), 1 << 8);
        assert_eq!(ALT.chunk_id_parts(ALT.chunk_id(200, 16_000)), (200, 16_000));
        assert_eq!(ALT.bond_id_parts(ALT.bond_id(200, 250_000)), (200, 250_000));
        let e = ALT.body_entity(200, 1_000_000);
        assert_eq!(ALT.body_entity_parts(e), (200, 1_000_000));
        assert!(ALT.is_body_entity(e));
        assert!(!DEFAULT_LAYOUT.is_body_entity(e));
    }
}
