// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESS OR
// IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

#ifndef NVBLASTEXTSTRESSPHYSX_H
#define NVBLASTEXTSTRESSPHYSX_H

#include "NvBlastTypes.h"
#include "PxPhysicsAPI.h"

#include <cstdint>

namespace physx
{
// PhysX 4 exposes this interface. PhysX 5 moved runtime cooking to the public
// PxCreate* functions, but retaining the forward declaration keeps one host
// descriptor source-compatible with either SDK generation.
class PxCooking;
}

namespace Nv
{
namespace Blast
{

using ExtStressPhysXId = uint64_t;

enum class ExtStressPhysXGeometryType : uint32_t
{
    Cuboid,
    Convex
};

enum class ExtStressPhysXError : uint32_t
{
    None,
    InvalidDescriptor,
    AllocationFailed,
    SolverCreationFailed,
    BodyCreationFailed,
    ShapeCreationFailed,
    ConvexPointLimitExceeded,
    ConvexCookingFailed,
    SceneMutationFailed,
    FractureBufferOverflow,
    MappingInvalid
};

struct ExtStressPhysXNodeGeometry
{
    ExtStressPhysXGeometryType type;
    physx::PxTransform localPose;
    physx::PxVec3 halfExtents;
    const physx::PxVec3* convexPoints;
    uint32_t convexPointCount;

    ExtStressPhysXNodeGeometry()
        : type(ExtStressPhysXGeometryType::Cuboid)
        , localPose(physx::PxIdentity)
        , halfExtents(0.5f)
        , convexPoints(nullptr)
        , convexPointCount(0)
    {
    }
};

/**
Node coordinates are in structure space. A mass of zero marks a support node;
every actor containing such a node is represented by a kinematic
PxRigidDynamic. Geometry point storage only needs to remain valid for create().
*/
struct ExtStressPhysXNodeDesc
{
    physx::PxVec3 centroid;
    float mass;
    float volume;
    ExtStressPhysXNodeGeometry geometry;
    // Index into ExtStressPhysXDesc::stressMaterials, selecting this chunk's
    // CRUSH properties. Independent of the materials on its bonds: a chunk's
    // own resistance to being ground up need not match the joints holding it.
    // Irrelevant unless that material sets crushCapPressure > 0.
    uint32_t material;

    ExtStressPhysXNodeDesc()
        : centroid(0.0f)
        , mass(1.0f)
        , volume(1.0f)
        , material(0)
    {
    }
};

struct ExtStressPhysXBondDesc
{
    physx::PxVec3 centroid;
    physx::PxVec3 normal;
    // Geometry only: the real contact patch (m^2), which is also the damage
    // pool. Strength is authored through `material`, never by scaling area.
    float area;
    uint32_t node0;
    uint32_t node1;
    // Index into ExtStressPhysXDesc::stressMaterials.
    uint32_t material;

    ExtStressPhysXBondDesc()
        : centroid(0.0f)
        , normal(0.0f, 1.0f, 0.0f)
        , area(1.0f)
        , node0(0)
        , node1(0)
        , material(0)
    {
    }
};

struct ExtStressPhysXSettings
{
    uint32_t maxSolverIterationsPerFrame;
    uint32_t graphReductionLevel;
    bool islandAware;
    bool skipSettledIslands;
    bool gpuStressSolver;
    uint32_t gpuStressMinimumBondCount;
    bool recordSplitContinuity;
    bool applyExcessForces;
    float excessForceScale;
    /**
    Feed each spinning body's omega-squared-r load to the solver.

    A free island receives gravity as a uniform per-node acceleration, which is
    a rigid translation and therefore produces no bond stress at all. Its spin
    is the one self-generated load it does have, so without this a tumbling
    slab is stress-free no matter how fast it turns.
    */
    bool applyCentrifugal;
    /**
    Charge comminution work to the crusher.

    The crush model's crushEnergy is the work per unit volume it takes to grind
    the material up. Without this flag that energy is only a damage-rate
    calibration: a chunk comminutes and whatever pressed on it keeps every
    joule of its kinetic energy, so a penetrator gets its hole for free. With
    it, each tick's damage increment dD on a chunk of volume V extracts
    dD * crushEnergy * V from the strongest external contactor's kinetic
    energy along the closing axis -- a resistive impulse, applied with its
    momentum-conserving reaction on the chunk's own body, clamped so it can
    slow the crusher to a stop but never reverse it.

    This is the crumple-zone half of the crush physics: material that is being
    ground up eats the impact instead of merely getting out of the way. It is
    not a velocity cap and never overrides a fracture verdict; it is the
    reaction force comminution exerts, sized by the same authored crushEnergy
    that drives the damage.
    */
    bool applyCrushResistance;
    // Opt-in hard stop only: when >0 and the structure already has this many
    // bodies, skip further fracture. Default 0 = unlimited. Do not use this as
    // a perf "budget" — capping fracture falsifies impact/resim behavior.
    uint32_t maximumBodies;
    // Opt-in per-actor bond-break cap for a single tick. Default 0 = unlimited.
    // Same warning as maximumBodies: artificial limits degrade simulation quality.
    uint32_t maximumFracturesPerActorPerTick;
    // NOTE: this adapter never overrides a stress-solver fracture verdict and
    // never applies persistent velocity caps to fracture bodies. Joint strength
    // is authored (bond area = geometry, material limits = strength); anything
    // the solver says breaks, breaks. Earlier revisions carried support-bond
    // protection, impact-bond fatalization, and per-body velocity clamps as
    // compensations for authoring the solver could not yet express — see
    // PHYSICS_ENGINE_CONTRACT.md for the invariant that replaced them.
    // Applied AT BODY CREATION, not by a host-side walk after the fact. The
    // walk variant left every split child's FIRST step unprotected: the walk
    // ran against a body cache refreshed before endTick created the children,
    // so an overlapping child's first depenetration step was unbounded --
    // >1000 m/s, through a 20 m ground slab in one tick. Measured: ~7 bodies
    // per heavy bombardment escaped below ground; the worst live case reached
    // y = -19.6 million metres and overflowed the GPU patch buffer.
    // PhysX per-body solver iteration counts (positions, velocities).
    // Defaults are the engine's own 4/1. Raising positions is the engine's
    // sanctioned lever for stack/contact robustness on a backend with no
    // sweep-based CCD -- more iterations distribute contact correction across
    // a pile instead of resolving it as a few violent pushes.
    // Extra solver update batches per tick while the stress solve has NOT
    // converged (each batch is maxSolverIterationsPerFrame iterations).
    //
    // A pristine downtown-scale structure is one ~24k-node island; 32 CG
    // iterations cannot reach equilibrium on it, and an unconverged island can
    // never earn the settled-island skip -- so a 4-structure city re-solved
    // ~296k bonds EVERY tick at rest (measured 37-50 ms of GPU stress with
    // zero awake bodies). Worse than the cost: the unconverged residual reads
    // as bond stress, which broke 20 bonds on an untouched city and inflated
    // utilisation to 8.97 at rest -- phantom damage, the exact failure the
    // docs warn about for under-solving.
    //
    // Pursuing convergence is therefore a CORRECTNESS setting, not tuning: it
    // finishes the equilibrium the quasi-static model is defined by. Bounded
    // per tick so a fracturing frame cannot stall; a structure typically
    // converges over the first second and then skips at ~zero cost.
    // See ExtStressGpuSolveParams::skipStableUnconverged.
    bool skipStableUnconverged;
    uint32_t unconvergedExtraUpdates;
    uint32_t bodyPositionIterations;
    uint32_t bodyVelocityIterations;
    bool enableSpeculativeCcd;
    // 0 = leave PhysX's (unbounded) default. Bounds only how fast the solver
    // corrects interpenetration, which is a numerical artifact of discrete
    // stepping -- it never overrides a fracture verdict or caps a trajectory.
    float maxDepenetrationVelocity;
    bool idleSkip;
    bool baseStepSleep;
    float settledLinearSpeed;
    float settledAngularSpeed;
    float minimumSeparationVelocity;
    // Damping applied to every body this adapter creates.
    //
    // PhysX defaults (linear 0, angular 0.05) suit gameplay objects that
    // should coast. Fracture debris is not that: with no linear damping a
    // rubble pile trades micro-contacts indefinitely, never falling below the
    // sleep threshold, and PhysX sleeps per contact island -- so one jittering
    // body holds every body it transitively touches awake. Measured on a
    // demolished city, that left ~93% of 20k bodies awake permanently.
    //
    // This is not a velocity cap and does not contradict the no-clamp
    // invariant below: damping is a continuous force opposing motion, the
    // physical stand-in for the air drag and micro-friction a real rubble
    // field has, and it never overrides a fracture verdict or bounds a
    // trajectory. The sibling Rapier implementation carries the same feature
    // as `smallBodyDamping`, and the shipped scene packs already request it.
    float linearDamping;
    float angularDamping;
    physx::PxShapeFlags shapeFlags;

    ExtStressPhysXSettings()
        : maxSolverIterationsPerFrame(25)
        , graphReductionLevel(0)
        , islandAware(true)
        , skipSettledIslands(true)
        , gpuStressSolver(false)
        , gpuStressMinimumBondCount(4096)
        , recordSplitContinuity(true)
        , applyExcessForces(true)
        , excessForceScale(1.0f)
        , applyCentrifugal(true)
        , applyCrushResistance(true)
        , maximumBodies(0)
        , maximumFracturesPerActorPerTick(0)
        , skipStableUnconverged(true)
        , unconvergedExtraUpdates(0)
        , bodyPositionIterations(4)
        , bodyVelocityIterations(1)
        , enableSpeculativeCcd(true)
        , maxDepenetrationVelocity(1.0f)
        , idleSkip(true)
        , baseStepSleep(false)
        , settledLinearSpeed(0.15f)
        , settledAngularSpeed(0.15f)
        , minimumSeparationVelocity(0.0f)
        , linearDamping(0.0f)
        , angularDamping(0.05f)
        , shapeFlags(physx::PxShapeFlag::eVISUALIZATION |
                     physx::PxShapeFlag::eSCENE_QUERY_SHAPE |
                     physx::PxShapeFlag::eSIMULATION_SHAPE)
    {
    }
};

struct ExtStressPhysXTelemetry
{
    uint64_t ticks;
    uint64_t contactsQueued;
    uint64_t contactsProcessed;
    uint64_t contactsDropped;
    uint64_t sleepingActorsSkipped;
    uint64_t splits;
    uint64_t bodiesCreated;
    uint64_t bodiesReused;
    uint64_t bodiesRecycled;
    uint64_t shapesMigrated;
    uint64_t convexPointLimitRejections;
    uint64_t convexCookingFailures;
    uint64_t mappingValidationFailures;
    /// Times the incrementally-maintained node lookups disagreed with a full
    /// rebuild. Only counted under BLAST_LOOKUP_VALIDATE=1; must be 0.
    uint64_t lookupTableDrifts;
    uint32_t bodyCount;
    /// Bodies made of exactly ONE node. Such a body has no internal bonds, so
    /// there is nothing in it that can break: the stress solve cannot produce
    /// a fracture there, and every contact routed into it is work whose only
    /// possible outcome is zero. Counted to size that waste before anything
    /// is built on the assumption.
    uint32_t singleNodeBodyCount{0};
    /// Contacts queued this tick whose target node belongs to a single-node
    /// body — the directly skippable share of the contact stream.
    uint32_t singleNodeContacts{0};
    /// Single-node bodies that are AWAKE. These can never break — no bonds —
    /// so they contribute nothing to the stress solve and are not even in its
    /// island set, yet each one is a full PhysX rigid body generating contact
    /// reports. If the awake population is mostly these, then retiring them
    /// is the lever, and it costs the stress path nothing at all.
    uint32_t singleNodeAwakeBodies{0};
    /// Contacts dropped because their body has no bonds and therefore nothing
    /// the contact could ever affect. Cumulative.
    uint64_t bondlessContactsSkipped{0};
    uint32_t awakeDynamicBodyCount;
    uint32_t overstressedBondCount;
    uint32_t solverIslandCount;
    uint32_t solverIslandsSkipped;
    float maxSplitWorldPositionDrift;
    float maxSplitPointVelocityDrift;
    double contactProcessingMilliseconds;
    double gravityMilliseconds;
    double stressSolveMilliseconds;
    double fractureTopologyMilliseconds;
    /// The interior of fractureTopologyMilliseconds. That phase is the largest
    /// in the tick during a collapse (9.8 ms at 7k awake, 0.000 at rest) and
    /// nobody had ever looked inside it -- it was one number covering a solver
    /// call, three whole-population rebuilds and a scene mutation loop.
    ///
    /// These are children of fractureTopology and must not be summed with it.
    /// mappingValidationMilliseconds is ALSO inside it (fracture() ends with
    /// `return validateMappings()`), so the unattributed remainder is
    /// fractureTopology - (these five) - mappingValidation.
    double fractureGenerateMilliseconds;  ///< generating fracture commands from the solver
    double fracturePrepMilliseconds;  ///< command sort/limit, node snapshot, parent motion capture
    double fractureApplyMilliseconds;  ///< the solver's own island split
    double fractureSceneMilliseconds;  ///< event sort plus the applySplit loop under the scene write lock
    double fractureRebuildMilliseconds;  ///< rebuildLookupTables: three whole-population hash maps
    double mappingValidationMilliseconds;
    double gpuStressSolveMilliseconds;
    /// Of the host wall inside the GPU solve: working, versus blocked on the
    /// device. Only the first can be reclaimed by faster host code, so the
    /// split is the ceiling on every host-side optimization of this path.
    double gpuStressHostWorkMilliseconds;
    /// Host walls around the GPU solve inside solveTick.
    double stressImpulseCopyMilliseconds;
    /// Audit of the flat bondless flags against the two-deref predicate they
    /// replace. Mismatches must be zero.
    /// Default-initialised explicitly: this struct is declared as a plain
    /// member with no initialiser, so a field without one starts
    /// indeterminate and `++` on it reads garbage. The first run of this
    /// audit reported 4.5e18 checks and "MISMATCH" for exactly that reason.
    uint64_t bondlessVerifyChecks{0};
    uint64_t bondlessVerifyMismatches{0};
    double stressInitializeMilliseconds;
    double stressCalcErrorMilliseconds;
    double gpuStressHostBlockedMilliseconds;
    uint64_t gpuStressHostToDeviceBytes;
    uint64_t gpuStressDeviceToHostBytes;
    uint64_t chunksCrushed;
    double crushedMassKg;
    double crushedVolumeM3;
    uint32_t nodesAtCrushYield;
    //! Highest crush utilisation any chunk has reached over this run. Below 1
    //! nothing ever came close; 1/this is the chunk's crush safety factor.
    float peakCrushUtilisation;
    uint64_t debrisBodiesSpawned;
    //! Total comminution work charged to crushing bodies (J), and how many
    //! resistive impulses carried it.
    double crushResistanceJoules;
    uint64_t crushResistanceImpulses;
    /// Extra solve batches spent pursuing convergence (see
    /// unconvergedExtraUpdates), and ticks that ended still unconverged.
    uint64_t extraSolveUpdates;
    uint64_t unconvergedTicks;
    uint64_t resimulationCaptures;
    uint64_t resimulationRestores;
    uint64_t resimulationBodiesRestored;
    uint64_t resimulationBodiesRederived;
    double resimulationCaptureMilliseconds;
    double resimulationRestoreMilliseconds;
    float resimulationMaxRederivedDriftMeters;
    ExtStressPhysXError lastError;
    uint32_t lastErrorNode;

    ExtStressPhysXTelemetry();
};

struct ExtStressPhysXBodySnapshot
{
    ExtStressPhysXId bodyId;
    uint32_t actorIndex;
    physx::PxRigidDynamic* body;
    physx::PxTransform globalPose;
    physx::PxTransform centerOfMassLocalPose;
    physx::PxVec3 linearVelocity;
    physx::PxVec3 angularVelocity;
    uint32_t nodeCount;
    /// Cached at mass-recompute time, not read live: consumers were issuing
    /// two PxRigidDynamic::getMass calls per contact PAIR (~52k per tick at a
    /// 7.9k-awake peak) for a value that only changes when a split recomputes
    /// mass properties.
    float mass;
    bool kinematic;
    bool sleeping;
};

struct ExtStressPhysXShapeSnapshot
{
    ExtStressPhysXId shapeId;
    ExtStressPhysXId bodyId;
    uint32_t nodeIndex;
    physx::PxShape* shape;
    physx::PxTransform worldPose;
    bool bodyKinematic;
    bool bodySleeping;
};

/**
Queue one contact for the adapter-owned shape. impulse is the world-space
impulse applied to that shape during the preceding simulation step. shapeId may
be used instead of shape; if both are supplied, shapeId takes precedence.
*/
struct ExtStressPhysXContact
{
    ExtStressPhysXId shapeId;
    const physx::PxShape* shape;
    physx::PxVec3 worldPosition;
    physx::PxVec3 worldImpulse;
    /**
    Relative velocity of the two bodies at the contact point, other minus this,
    in world space. Only its component along the contact normal matters and only
    when crushing is enabled: it supplies the compaction strain rate the crush
    work integral needs.

    The stress solver is quasi-static and produces forces, not strains, so it
    has no strain rate of its own. Leaving this at zero is safe -- the chunk
    simply accumulates no crush damage from this contact, which is the right
    answer for a resting contact that is not closing.
    */
    physx::PxVec3 worldRelativeVelocity;
    /**
    The body on the other side of the contact, when the host knows it. Optional
    and only consumed by crush resistance: comminution work is charged to this
    body's kinetic energy, so without it a crush dissipates topologically but
    the crusher sails on unslowed. Null is always safe.
    */
    physx::PxRigidActor* otherActor;
    // Whether this contact may wake a sleeping body. New impacts should;
    // PERSISTING resting contacts must not -- in a rubble pile every body has
    // one, and waking for them re-opens the whole contact island every tick,
    // which is indistinguishable from sleeping being broken. The load is fed
    // to the solver either way; if it overstresses a bond, the fracture
    // itself wakes the body.
    bool wake;

    /**
    Pre-resolved support-graph node, from nodeForShape() on the SAME
    destructible within the same tick. When set, queueContact skips its
    shapeId/shape map lookup entirely -- the point of the field: a PhysX
    manifold's points all share one shape, so the host can resolve the node
    once per manifold instead of paying the hash find per point. UINT32_MAX
    (the default) means unresolved; queueContact then falls back to the
    shapeId/shape paths, so leaving shape set alongside an unresolved
    nodeIndex is always safe and behaves exactly as before this field
    existed. Never carry a value across a tick boundary: fracture moves
    shapes between nodes at endTick.
    */
    uint32_t nodeIndex;

    ExtStressPhysXContact()
        : shapeId(0)
        , shape(nullptr)
        , worldPosition(0.0f)
        , worldImpulse(0.0f)
        , worldRelativeVelocity(0.0f)
        , otherActor(nullptr)
        , wake(true)
        , nodeIndex(0xFFFFFFFFu)
    {
    }
};

/**
One chunk was pulverized: its crush damage reached 1, it was severed from every
bond, and its shape has been removed from the scene. It no longer exists as a
rigid body.

Everything a consumer needs to spawn a momentum-matched dust cloud is here --
where it was, how fast it was going, and how much mass and volume left the
simulation. Without mass and momentum on the event a dust effect can only be
guessed at, and guessed dust is what makes destruction read as a cartoon.

Drained through drainChunkDestroyedEvents(), or pushed to
ExtStressPhysXFrameHooks::onChunkDestroyed() as it happens.
*/
struct ExtStressPhysXChunkDestroyed
{
    uint64_t sequence;                  //!< Monotonic, per destructible.
    uint32_t nodeIndex;                 //!< Authored node index of the pulverized chunk.
    uint32_t materialIndex;             //!< Its material, for per-material reporting.
    ExtStressPhysXId shapeId;           //!< The shape that was removed (now invalid).
    ExtStressPhysXId bodyId;            //!< The body it belonged to (may still exist).
    physx::PxTransform worldPose;       //!< Where the chunk was when it went.
    physx::PxVec3 linearVelocity;       //!< Of the chunk's centroid, world space.
    physx::PxVec3 angularVelocity;      //!< Of its body, world space.
    float mass;                         //!< kg leaving the rigid-body simulation.
    float volume;                       //!< m^3 leaving the rigid-body simulation.
    float peakPressure;                 //!< Pa. Confining pressure at the moment it failed.
    float peakDeviator;                 //!< Pa. Von Mises deviator at the moment it failed.
    //! Fraction of `mass` that was respawned as rigid debris rather than lost.
    //! The dust cloud should carry the remainder: mass * (1 - debrisMassFraction).
    float debrisMassFraction;
    uint32_t debrisBodiesSpawned;       //!< How many debris bodies were actually created.

    ExtStressPhysXChunkDestroyed()
        : sequence(0)
        , nodeIndex(0)
        , materialIndex(0)
        , shapeId(0)
        , bodyId(0)
        , worldPose(physx::PxIdentity)
        , linearVelocity(0.0f)
        , angularVelocity(0.0f)
        , mass(0.0f)
        , volume(0.0f)
        , peakPressure(0.0f)
        , peakDeviator(0.0f)
        , debrisMassFraction(0.0f)
        , debrisBodiesSpawned(0)
    {
    }
};

struct ExtStressPhysXSplitContinuity
{
    uint64_t splitSequence;
    ExtStressPhysXId parentBodyId;
    uint32_t parentActorIndex;
    uint32_t reusedChildActorIndex;
    float maxWorldPositionDrift;
    float maxPointVelocityDrift;
};

using ExtStressPhysXErrorCallback =
    void (*)(ExtStressPhysXError error, uint32_t nodeIndex, const char* message, void* userData);

/**
Per-material stress limits (Pa) for bonds. Negative tension/shear limits
inherit the corresponding compression limit. Ductility is the width of the
(fatal - elastic) band. Strength lives here and only here — see
ExtStressPhysXBondDesc::area.
*/
struct ExtStressPhysXMaterial
{
    float compressionElasticLimit;
    float compressionFatalLimit;
    float tensionElasticLimit;
    float tensionFatalLimit;
    float shearElasticLimit;
    float shearFatalLimit;

    /*
    Optional CHUNK crushing (comminution). The limits above decide whether a
    JOINT fails; these decide whether the CHUNK ITSELF is ground up and leaves
    the rigid-body simulation as dust. Both happen in the same impact: most of
    a wall separates along its joints while the small region under the hit is
    pulverized.

    Each solve builds a per-chunk Cauchy stress tensor from the forces acting
    on the chunk (Love-Weber virial sum), reduces it to pressure p and von
    Mises deviator q, and yields against a Drucker-Prager cone with a pressure
    cap. Damage accumulates as plastic work per unit volume normalized by
    crushEnergy; the chunk pulverizes at damage 1.

    Comminution is a COMPRESSIVE phenomenon, so a chunk in net tension is
    excluded outright: it fails by cracking, which the bond model already
    represents. With the cone, that is what keeps free-floating debris intact --
    it carries no confining pressure, so it tumbles rather than crumbling.

    DISABLED unless crushCapPressure > 0. A default-constructed material is
    bond-only and behaves exactly as it did before crushing existed.
    */
    float crushCapPressure;         //!< Pa. Hydrostatic cap. <= 0 disables crushing.
    float crushCohesion;            //!< Pa. Drucker-Prager deviatoric intercept at p = 0.
    float crushFrictionSlope;       //!< dq/dp of the cone. Dimensionless, >= 0.
    float crushEnergy;              //!< J/m^3. Plastic work per unit volume to fully comminute.
    //! Pa*s. Perzyna viscosity: how fast the material flows once past yield.
    //! Larger is more sluggish. Must be > 0 when crushing is enabled.
    float crushViscosity;
    float crushStrainRateExponent;  //!< CEB dynamic-increase-factor exponent. 0 disables rate hardening.
    float crushReferenceStrainRate; //!< 1/s. Strain rate at which the DIF is 1.
    //! Fraction of a pulverized chunk's mass respawned as rigid fragments.
    //! 0 (default) means all of its mass leaves the simulation; the
    //! chunk-destroyed event still reports that mass and momentum so a
    //! consumer can spawn a matching dust cloud.
    float crushDebrisMassFraction;
    uint32_t crushDebrisFragmentCount;

    ExtStressPhysXMaterial()
        : compressionElasticLimit(1.0f)
        , compressionFatalLimit(2.0f)
        , tensionElasticLimit(-1.0f)
        , tensionFatalLimit(-1.0f)
        , shearElasticLimit(-1.0f)
        , shearFatalLimit(-1.0f)
        , crushCapPressure(0.0f)
        , crushCohesion(0.0f)
        , crushFrictionSlope(0.0f)
        , crushEnergy(1.0f)
        , crushViscosity(1.0f)
        , crushStrainRateExponent(0.0f)
        , crushReferenceStrainRate(1.0f)
        , crushDebrisMassFraction(0.0f)
        , crushDebrisFragmentCount(0)
    {
    }
};

struct ExtStressPhysXDesc
{
    physx::PxPhysics* physics;
    physx::PxCooking* cooking;
    physx::PxScene* scene;
    physx::PxMaterial* material;
    const ExtStressPhysXNodeDesc* nodes;
    uint32_t nodeCount;
    const ExtStressPhysXBondDesc* bonds;
    uint32_t bondCount;
    // Required, >= 1 entry; index 0 is the structure default. Every bond's
    // material index must be inside this table (create fails otherwise).
    const ExtStressPhysXMaterial* stressMaterials;
    uint32_t stressMaterialCount;
    physx::PxTransform worldTransform;
    ExtStressPhysXSettings settings;
    ExtStressPhysXErrorCallback errorCallback;
    void* errorUserData;

    ExtStressPhysXDesc()
        : physics(nullptr)
        , cooking(nullptr)
        , scene(nullptr)
        , material(nullptr)
        , nodes(nullptr)
        , nodeCount(0)
        , bonds(nullptr)
        , bondCount(0)
        , stressMaterials(nullptr)
        , stressMaterialCount(0)
        , worldTransform(physx::PxIdentity)
        , errorCallback(nullptr)
        , errorUserData(nullptr)
    {
    }
};

/**
Owns the PhysX representation and C stress-bridge handle for one Blast family.
Call tick only after PxScene::fetchResults, so topology edits occur outside
simulation. The host retains ownership of all objects supplied in the descriptor.
*/
class NV_DLL_EXPORT ExtStressPhysXDestructible
{
public:
    static ExtStressPhysXDestructible* create(
        const ExtStressPhysXDesc& desc,
        ExtStressPhysXTelemetry* failureTelemetry = nullptr);

    virtual void release() = 0;
    virtual bool queueContact(const ExtStressPhysXContact& contact) = 0;
    virtual bool queueContact(
        const physx::PxShape& shape,
        const physx::PxVec3& worldPosition,
        const physx::PxVec3& worldImpulse) = 0;
    /**
     * Resolve a shape to its support-graph node for ExtStressPhysXContact::
     * nodeIndex. Exactly the lookup queueContact performs per contact, exposed
     * so a host queuing a whole manifold pays it once. Returns UINT32_MAX for
     * a shape this destructible does not own. Valid only until the next
     * endTick(), which may re-map shapes to nodes.
     */
    virtual uint32_t nodeForShape(const physx::PxShape* shape) const = 0;
    /**
     * Three-phase tick API for batching independent destructibles. beginTick()
     * and endTick() access PhysX and must run serially after fetchResults().
     * solveTick() only updates this destructible's stress solver and may run
     * concurrently with solveTick() on other destructibles.
     */
    virtual bool beginTick(float dt, const physx::PxVec3& worldGravity) = 0;

    /**
     * beginTick() driven entirely by a caller-supplied body snapshot.
     *
     * Identical solver inputs to beginTick(), with one difference: it makes no
     * PhysX calls at all, so it may run concurrently across destructibles.
     * beginTick() reads isSleeping() and getGlobalPose() for every body, which
     * is both a per-tick cost proportional to body count and unsynchronised
     * PxScene access if called off the main thread -- the latter races PhysX's
     * deferred shape/bounds sync and crashes inside SqBoundsManagerEx.
     *
     * Hosts that already read body state each tick (via getBodySnapshots)
     * therefore pay for those reads twice. Pass that snapshot here instead.
     *
     * Contacts on a sleeping body cannot be woken from here, since wakeUp() is
     * a scene write. Their ids are appended to `outWakeBodies` and the caller
     * must wake them before the next simulate(). If `wakeCapacity` is too
     * small the call still succeeds and `outWakeCount` reports the number that
     * would have been written, so the caller can detect truncation.
     *
     * `bodies` must cover every body of this destructible; entries for other
     * destructibles are ignored (matched by bodyId).
     */
    virtual bool beginTickFromSnapshot(
        float dt,
        const physx::PxVec3& worldGravity,
        const ExtStressPhysXBodySnapshot* bodies,
        uint32_t bodyCount,
        ExtStressPhysXId* outWakeBodies,
        uint32_t wakeCapacity,
        uint32_t* outWakeCount) = 0;
    virtual bool solveTick() = 0;
    virtual bool endTick() = 0;
    virtual bool tick(float dt, const physx::PxVec3& worldGravity) = 0;

    virtual bool validateMappings() = 0;
    virtual const ExtStressPhysXTelemetry& getTelemetry() const = 0;
    virtual bool usesGpuStressSolver() const = 0;

    virtual uint32_t getBodySnapshots(
        ExtStressPhysXBodySnapshot* snapshots,
        uint32_t capacity) const = 0;
    virtual uint32_t getShapeSnapshots(
        ExtStressPhysXShapeSnapshot* snapshots,
        uint32_t capacity) const = 0;
    /**
     * Returns shapes owned by awake dynamic bodies plus one final snapshot
     * when a body transitions to sleeping. Intended for delta state export.
     */
    virtual uint32_t getActiveShapeSnapshots(
        ExtStressPhysXShapeSnapshot* snapshots,
        uint32_t capacity) const = 0;
    virtual uint32_t getSplitContinuity(
        ExtStressPhysXSplitContinuity* records,
        uint32_t capacity) const = 0;

    /**
     * Drain chunk-destruction events recorded since the last call. Each chunk
     * is reported exactly once.
     *
     * This is a DRAIN, not a peek: anything returned is removed from the queue.
     * A host that never calls it and never installs the onChunkDestroyed hook
     * would otherwise grow the queue without bound over a long demolition.
     *
     * Returns the number written; call again while the return value equals
     * `capacity` to be sure the queue is empty.
     */
    virtual uint32_t drainChunkDestroyedEvents(
        ExtStressPhysXChunkDestroyed* records,
        uint32_t capacity) = 0;

    /**
     * Per-node accumulated crush damage in [0,1], indexed by authored node
     * index. 1 means the chunk pulverized. Nodes whose material has crushing
     * disabled always read 0.
     *
     * Sampling this is the crush analogue of getBondUtilisations: it says how
     * close each chunk is to being ground up, before anything visibly happens.
     */
    virtual uint32_t getNodeCrushDamage(
        float* damage,
        uint32_t capacity) const = 0;

    /**
     * Per-node stress invariants from the last solve, authored-node-indexed:
     * `pressure` is p (Pa, positive in compression) and `deviator` is the von
     * Mises equivalent q (Pa). Either pointer may be null.
     *
     * These are the two numbers the crush yield surface is evaluated on, so
     * they are what to look at when a structure crushes where it should not.
     * Populated only for nodes on a crush-enabled material.
     */
    virtual uint32_t getNodeStressInvariants(
        float* pressure,
        float* deviator,
        uint32_t capacity) const = 0;

    /**
     * Per-node crush utilisation, authored-node-indexed: max of
     * q/(cohesion + frictionSlope*p) and p/capPressure. 1 means the chunk is at
     * its crush yield surface; above 1 it is comminuting.
     *
     * This is to crushing what getBondUtilisations is to joints, and it is the
     * number to author against. Sampling it after a gravity settle says how
     * much of each chunk's crush capacity its own weight already consumes;
     * sampling the peak over an impact says how close that hit came. It reads
     * correctly even when nothing is moving, because it is a property of the
     * stress state and not of the damage rate.
     */
    virtual uint32_t getNodeCrushUtilisation(
        float* utilisation,
        uint32_t capacity) const = 0;

    /** Whether chunk crushing is active (a material enables it and the graph is unreduced). */
    virtual bool isCrushEnabled() const = 0;

    /**
     * Per-bond stress from the last solve, indexed by the authored bond index
     * (the order bonds were supplied in ExtStressPhysXDescriptor::bonds), so a
     * host can relate stress back to the joint it authored.
     *
     * Compression and tension are mutually exclusive; shear is independent.
     * Values are pressures directly comparable to the descriptor's stress
     * limits: limit / stress is that joint's safety factor. Broken bonds read 0.
     * Any output pointer may be null. Returns the number of entries written.
     *
     * Sampling this after gravity settle and before any impact is the load-path
     * check: it says how much of each joint's capacity the structure's own
     * weight already consumes.
     */
    virtual uint32_t getBondStresses(
        float* compression,
        float* tension,
        float* shear,
        uint32_t capacity) const = 0;

    /**
     * Per-bond utilisation from the last solve, authored-bond-indexed: the max
     * over stress modes of stress divided by THAT bond's own material elastic
     * limit. 1/utilisation is the joint's safety factor.
     *
     * Prefer this over dividing getBondStresses by hand — with a mixed-material
     * structure there is no single correct divisor, and using one silently
     * misreports every joint whose material differs from it.
     */
    virtual uint32_t getBondUtilisations(
        float* utilisation,
        uint32_t capacity) const = 0;

    /**
     * Fracture-frame resimulation (engine contract §2.8). Capture before
     * PxScene::simulate; if the following tick fractured, restore between
     * fetchResults and the re-run simulate, then step and tick again so
     * contacts resolve against the already-split pieces.
     *
     * Restore rewinds motion state only — fracture topology, masses, shapes,
     * and kinematic flags are kept. Bodies that existed at capture get their
     * pose, velocities, and sleep state back (with a COM-shift velocity
     * correction for bodies whose mass frame moved during a split); bodies
     * created since the capture are re-placed relative to their source
     * parent's restored state. Kinematic bodies are pose-only.
     *
     * Both calls require the Idle tick phase and must run outside
     * simulate/fetchResults. capture returns the number of bodies recorded;
     * restore returns false if no capture is held or the phase is wrong.
     */
    virtual uint32_t captureResimulationSnapshot() = 0;
    /**
     * Restore motion for captured bodies. When activeBodies is null / activeCount
     * is 0, every captured body is restored (full-scene §2.8). Otherwise only
     * bodies present in the active set are rewound; provenance children of
     * restored parents are still re-derived.
     */
    virtual bool restoreResimulationSnapshot(
        physx::PxRigidDynamic* const* activeBodies = nullptr,
        uint32_t activeCount = 0) = 0;

    virtual bool needsResimulationSnapshot() const = 0;
    virtual uint32_t getResimulationSeedBodies(
        physx::PxRigidDynamic** bodies,
        uint32_t capacity) const = 0;
    virtual uint32_t applyBaseStepSleep() = 0;

    virtual ExtStressPhysXId getBodyId(const physx::PxRigidDynamic* body) const = 0;
    virtual ExtStressPhysXId getShapeId(const physx::PxShape* shape) const = 0;

protected:
    virtual ~ExtStressPhysXDestructible() {}
};

} // namespace Blast
} // namespace Nv

#endif // NVBLASTEXTSTRESSPHYSX_H
