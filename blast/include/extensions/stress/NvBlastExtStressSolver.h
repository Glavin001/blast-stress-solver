// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2016-2024 NVIDIA Corporation. All rights reserved.

//! @file
//!
//! @brief NvBlastExtStressSolver blast extension, provides functionality to calculate stress on a destructible

#ifndef NVBLASTEXTSTRESSSOLVER_H
#define NVBLASTEXTSTRESSSOLVER_H

#include "NvBlastTypes.h"
#include "NvCTypes.h"


namespace Nv
{
namespace Blast
{

/**
Process-wide parallel-for hook.

This library has no thread pool; the caller owns one. Rather than thread a
dispatcher through three constructors, the caller installs it once and the
solver uses it where a walk is large enough to be worth splitting. Free
function plus a file static, because the pool it wraps is process-wide too.

`body` is invoked with (userData, index) for index in [0, count). The call
must not return until every index has run.
*/
typedef void (*ExtStressParallelForBody)(void* userData, uint32_t index);
typedef void (*ExtStressParallelFor)(void* ctx,
                                     uint32_t count,
                                     ExtStressParallelForBody body,
                                     void* bodyUserData);

NV_C_API void NvBlastExtStressSetParallelFor(ExtStressParallelFor fn, void* ctx);
NV_C_API void getExtStressParallelFor(ExtStressParallelFor& fn, void*& ctx);


/**
Stress Solver Settings

Stress on every bond is calculated with these components:
    compression/tension (parallel to bond normal)
    shear (perpendicular to bond normal)
Damage is done based on the limits defined in this structure to simulate micro bonds in the material breaking
Units for all limits are in pascals

Support graph reduction:
graphReductionLevel is the number of node merge passes.  The resulting graph will be
roughly 2^graphReductionLevel times smaller than the original.
NOTE: the reduction is currently fairly random and can lead to interlocked actors when solver bonds break.
If we are going to keep the feature, the algorithm for combining bonds should be revisited to take locality into account.
*/
struct ExtStressSolverSettings
{
    uint32_t    maxSolverIterationsPerFrame;//!<    the maximum number of iterations to perform per frame
    uint32_t    graphReductionLevel;        //!<    graph reduction level

    ExtStressSolverSettings() :
        maxSolverIterationsPerFrame(25),
        graphReductionLevel(0)
    {}
};

/**
A material for bonds: the stress limits (Pa) at which a joint made of this
material yields and breaks. Strength lives here and ONLY here — bond area is
geometry (the real contact patch, m^2) and doubles as the damage pool, so
authoring strength through area corrupts both the stress readout and the
effective toughness. Assign per-bond material indices instead.

Ductility is the width of the (fatal - elastic) band: a wide band takes
partial damage over many frames (concrete with rebar), a narrow band snaps
(glass, drywall tape).

Negative tension/shear limits inherit the corresponding compression limit,
resolved when the table is set on the solver.
*/
/**
Optional per-material CHUNK crushing (comminution) properties.

Bond limits decide whether a JOINT fails; these decide whether the CHUNK ITSELF
is ground up and leaves the simulation as dust. The two are independent: a wall
can shed a panel along its joints (bonds) while the small region under an impact
is pulverized (crush).

The model is standard granular/geomechanics. Each solve builds a per-chunk
Cauchy stress tensor by the Love-Weber (virial) sum over the forces acting on
the chunk, reduces it to the pressure/deviator invariants

    p = -trace(sigma)/3        (positive in compression)
    q = sqrt(1.5 * (s : s))    (von Mises equivalent, s = sigma + p*I)

and yields against a Drucker-Prager cone with a pressure cap:

    excess = max(q - (cohesion + frictionSlope*p),  p - capPressure)

Flow is Perzyna overstress viscoplasticity, the standard rate-dependent
plasticity law: the plastic strain rate is proportional to how far the stress
state sits outside the yield surface,

    epsdot_p = excess / crushViscosity

and crush damage accumulates as that plastic work per unit volume, normalized
by the material's specific comminution energy:

    D += excess * epsdot_p * dt / crushEnergy
       = excess^2 * dt / (crushViscosity * crushEnergy),   pulverized at D >= 1

Note the QUADRATIC dependence on overstress. That is what makes the behaviour
read correctly: a chunk just past yield takes a very long time to comminute,
while a chunk hit hard enough to sit far outside the surface goes almost at
once. It also means crushing needs no strain measurement, so a chunk loaded
through its BONDS -- deep inside a collapsing structure, with no contact of its
own -- comminutes just as a directly struck one does.

A chunk inside the yield surface has excess = 0 and accumulates nothing, so a
settled structure never grinds itself to dust no matter how long it stands.

Using a cap surface rather than a plain pressure threshold is what separates
CONFINED crushing from unconfined shear: free-floating debris has low p, stays
inside the cone, and tumbles intact instead of crumbling.

DISABLED BY DEFAULT. capPressure <= 0 means this material's chunks are
indestructible in the crush sense and no per-chunk stress work is done at all,
so existing assets are bit-identical until they opt in.
*/
struct ExtStressCrushProperties
{
    float capPressure;          //!< Pa. Hydrostatic cap. <= 0 disables crushing for this material.
    float cohesion;             //!< Pa. Drucker-Prager deviatoric intercept at p = 0.
    float frictionSlope;        //!< dq/dp of the Drucker-Prager cone. Dimensionless, >= 0.
    float crushEnergy;          //!< J/m^3 (== Pa). Plastic work per unit volume to fully comminute.
    //! Pa*s. Perzyna viscosity: how fast the material flows once past yield.
    //! Larger is more sluggish, so a chunk lingers past yield instead of
    //! comminuting. Must be > 0 when crushing is enabled.
    float crushViscosity;
    float strainRateExponent;   //!< CEB dynamic-increase-factor exponent. 0 disables rate hardening.
    float referenceStrainRate;  //!< 1/s. Strain rate at which the DIF is 1.
    float debrisMassFraction;   //!< [0,1] of the chunk's mass respawned as rigid fragments. 0 = all mass leaves.
    uint32_t debrisFragmentCount; //!< Number of fragments to respawn when debrisMassFraction > 0.

    ExtStressCrushProperties() :
        capPressure(0.0f),
        cohesion(0.0f),
        frictionSlope(0.0f),
        crushEnergy(1.0f),
        crushViscosity(1.0f),
        strainRateExponent(0.0f),
        referenceStrainRate(1.0f),
        debrisMassFraction(0.0f),
        debrisFragmentCount(0)
    {}

    bool enabled() const { return capPressure > 0.0f; }
};


struct ExtStressMaterial
{
    float compressionElasticLimit;  //!< below this compression pressure no damage occurs
    float compressionFatalLimit;    //!< above this compression pressure the bond breaks outright
    float tensionElasticLimit;      //!< < 0 inherits compression
    float tensionFatalLimit;        //!< < 0 inherits compression
    float shearElasticLimit;        //!< < 0 inherits compression
    float shearFatalLimit;          //!< < 0 inherits compression

    //! Chunk comminution. Disabled unless crush.capPressure > 0. @see ExtStressCrushProperties
    ExtStressCrushProperties crush;

    ExtStressMaterial() :
        compressionElasticLimit(1.0f),
        compressionFatalLimit(2.0f),
        tensionElasticLimit(-1.0f),
        tensionFatalLimit(-1.0f),
        shearElasticLimit(-1.0f),
        shearFatalLimit(-1.0f)
    {}
};


/**
Parameter to addForce() calls, determines the exact operation that is carried out.

@see ExtStressSolver.addForce()
*/
struct ExtForceMode
{
    enum Enum
    {
        FORCE,          //!< parameter has unit of mass * distance / time^2
        ACCELERATION,   //!< parameter has unit of distance / time^2, i.e. the effect is mass independent
    };
};


/**
Stress Solver.

Uses NvBlastFamily, allocates and prepares its graph once when it's created. Then it's being quickly updated on every
actor split.
It uses NvBlastAsset support graph, you can apply forces on nodes and stress on bonds will be calculated as the result.
When stress on bond exceeds it's health bond is considered broken (overstressed).
Basic usage:
1. Create it with create function once for family
2. Fill node info for every node in support graph or use setAllNodesInfoFromLL() function.
3. Use notifyActorCreated / notifyActorDestroyed whenever actors are created and destroyed in family.
4. Every frame: Apply forces (there are different functions for it see @addForce)
5. Every frame: Call update() for actual solver to process.
6. If getOverstressedBondCount() > 0 use generateFractureCommands() functions to get FractureCommands with bonds fractured
*/
class NV_DLL_EXPORT ExtStressSolver
{
public:
    //////// creation ////////

    /**
    Create a new ExtStressSolver.

    \param[in]  family          The NvBlastFamily instance to calculate stress on.
    \param[in]  settings        The settings to be set on ExtStressSolver.

    \return the new ExtStressSolver if successful, NULL otherwise.
    */
    static ExtStressSolver*                 create(const NvBlastFamily& family, const ExtStressSolverSettings& settings = ExtStressSolverSettings());


    //////// interface ////////

    /**
    Release this stress solver.
    */
    virtual void                            release() = 0;

    /**
    Set node info.

    All the required info per node for stress solver is set with this function. Call it for every node in graph or use setAllNodesInfoFromLL().

    \param[in]  graphNodeIndex  Index of the node in support graph. see NvBlastSupportGraph.
    \param[in]  mass            Node mass. For static node it is must be zero.
    \param[in]  volume          Node volume. For static node it is irrelevant.
    \param[in]  localPosition   Node local position.
    */
    virtual void                            setNodeInfo(uint32_t graphNodeIndex, float mass, float volume, NvcVec3 localPosition) = 0;

    /**
    Set all nodes info using low level NvBlastAsset data.
    Uses NvBlastChunk's centroid and volume.
    Uses 'external' node to mark nodes as static.

    \param[in]  density         Density. Used to convert volume to mass.
    */
    virtual void                            setAllNodesInfoFromLL(float density = 1.0f) = 0;

    /**
    Set stress solver settings.
    Changing graph reduction level will lead to graph being rebuilt (which is fast, but still not recommended).
    All other settings are applied instantly and can be changed every frame.

    \param[in]  settings        The settings to be set on ExtStressSolver.
    */
    virtual void                            setSettings(const ExtStressSolverSettings& settings) = 0;

    /**
    Get stress solver settings.

    \return the pointer to stress solver settings currently set.
    */
    virtual const ExtStressSolverSettings&  getSettings() const = 0;

    /**
    Set the material table. Negative tension/shear limits are resolved to the
    corresponding compression limit at set-time. A solver that never receives a
    table behaves as if it had a 1-entry table of default ExtStressMaterial.

    Replacing the table is cheap and can be done every frame (e.g. to sweep a
    global strength scale); it does not rebuild the graph.

    \param[in] materials    Array of materials. Index 0 is the scene default.
    \param[in] count        Number of materials, >= 1.
    */
    virtual void                            setMaterials(const ExtStressMaterial* materials, uint32_t count) = 0;

    /**
    Assign each ASSET bond (indexed as in NvBlastAssetGetBonds) a material from
    the table. Out-of-range indices clamp to 0. Passing null resets all bonds
    to material 0.

    \param[in] materialIndices  Array of at least `bondCount` indices, or null.
    \param[in] bondCount        Number of entries provided.
    */
    virtual void                            setBondMaterials(const uint32_t* materialIndices, uint32_t bondCount) = 0;

    /**
    Assign each graph NODE a material from the table, which is what selects its
    ExtStressCrushProperties. Out-of-range indices clamp to 0. Passing null
    resets all nodes to material 0.

    Node materials are independent of bond materials: a chunk's own crush
    resistance need not match the joints that hold it. Crush stays disabled
    unless the selected material has crush.capPressure > 0.

    \param[in] materialIndices  Array of at least `nodeCount` indices, or null.
    \param[in] nodeCount        Number of entries provided.
    */
    virtual void                            setNodeMaterials(const uint32_t* materialIndices, uint32_t nodeCount) = 0;

    /**
    Supply each graph node's external loading rate (1/s) for the coming
    update(), and the timestep. Entries persist until overwritten; passing null
    zeroes them.

    This feeds the OPTIONAL strain-rate hardening term only (the CEB dynamic
    increase factor), which is what makes a fast projectile spall where a slow
    press crushes. It is not required for crushing to work: flow is driven by
    overstress, not by this rate, so leaving every entry at zero simply means
    no rate hardening. The PhysX adapter derives it from contact closing rates.

    Materials with strainRateExponent == 0 ignore it entirely.

    \param[in] strainRates  Array of at least `nodeCount` rates (1/s), or null.
    \param[in] nodeCount    Number of entries provided.
    \param[in] deltaTime    Timestep (s) the next update() advances by. The
                            solver has no clock of its own; this is what turns a
                            rate into the work increment.
    */
    virtual void                            setNodeStrainRates(const float* strainRates, uint32_t nodeCount, float deltaTime) = 0;

    /**
    Read back per-node accumulated crush damage in [0, 1], indexed by graph
    node index. 1 means the chunk pulverized. Nodes on a material without
    crush enabled always read 0.

    \return entries written (min of capacity and the graph's node count).
    */
    virtual uint32_t                        getNodeCrushDamage(float* damage, uint32_t capacity) const = 0;

    /**
    Read back the per-node stress invariants from the last update(), indexed by
    graph node index: `pressure` is p = -trace(sigma)/3 in Pa (positive in
    compression) and `deviator` is the von Mises equivalent q in Pa. Either
    pointer may be null.

    Unlike the bond readbacks these are populated only for nodes whose material
    has crush enabled -- the virial sum is skipped entirely otherwise.

    \return entries written (min of capacity and the graph's node count).
    */
    virtual uint32_t                        getNodeStressInvariants(float* pressure, float* deviator, uint32_t capacity) const = 0;

    /**
    Read back how close each chunk is to its crush yield surface, indexed by
    graph node index: max of q/(cohesion + frictionSlope*p) and p/capPressure.
    1 means the chunk is exactly at yield; above 1 it is comminuting.

    This is the crush analogue of getBondUtilisations, and it is what makes
    crush authorable. Sample it after a gravity settle to see how much of each
    chunk's crush capacity its own structure already consumes, and during an
    impact to see how close a hit came. It is a property of the stress state
    alone, so it reads correctly even when nothing is moving and no damage is
    accumulating.

    \return entries written (min of capacity and the graph's node count).
    */
    virtual uint32_t                        getNodeCrushUtilisation(float* utilisation, uint32_t capacity) const = 0;

    /**
    Drain the list of nodes that reached full crush damage since the last call.
    Each node is reported exactly once. The caller is responsible for removing
    the corresponding body/shape from its physics scene -- the solver only
    severs the chunk structurally.

    \return entries written (the drained count, capped at capacity).
    */
    virtual uint32_t                        getCrushedNodes(uint32_t* nodeIndices, uint32_t capacity) = 0;

    /**
    Whether chunk crushing is active: at least one material has
    crush.capPressure > 0 AND graphReductionLevel is 0.

    Crush requires unreduced nodes. Graph reduction merges support nodes into
    one solver node, so a per-node stress tensor would describe an aggregate
    rather than a chunk. Rather than report a plausible wrong number, the
    solver refuses: with reduction > 0 this returns false, no per-chunk stress
    is computed, and nothing is ever crushed.
    */
    virtual bool                            isCrushEnabled() const = 0;

    /**
    Read back per-bond utilisation from the last update(), indexed by ASSET
    bond index: max over stress modes of (stress / that bond's own material
    ELASTIC limit). 1/utilisation is the joint's safety factor. Using this
    instead of dividing getBondStresses by hand guarantees the division uses
    the bond's material rather than any global value. Broken bonds read 0.

    \param[out] utilisation  Array of at least `capacity` floats.
    \param[in]  capacity     Entries available.

    \return entries written (min of capacity and the asset's bond count).
    */
    virtual uint32_t                        getBondUtilisations(float* utilisation, uint32_t capacity) const = 0;

    /**
    Notify stress solver on newly created actor.

    Call this function for all initial actors present in family and later upon every actor split.

    \param[in]  actor           The actor created.

    \return true if actor will take part in stress solver process.  false if actor doesn't contain any bonds.
    */
    virtual bool                            notifyActorCreated(const NvBlastActor& actor) = 0;

    /**
    Notify stress solver on destroyed actor.

    Call this function when actor is destroyed (split futher) or deactivated.

    \param[in]  actor           The actor destroyed.
    */
    virtual void                            notifyActorDestroyed(const NvBlastActor& actor) = 0;

    /**
    Apply external impulse on particular actor of family. This function will find nearest actor's graph node to apply impulse on.

    \param[in]  actor           The actor to apply impulse on.
    \param[in]  localPosition   Local position in actor's coordinates to apply impulse on.
    \param[in]  localForce      Force to apply in local actor's coordinates.
    \param[in]  mode            The mode to use when applying the force/impulse(see #ExtForceMode)

    \return true iff node was found and force applied.
    */
    virtual bool                            addForce(const NvBlastActor& actor, NvcVec3 localPosition, NvcVec3 localForce, ExtForceMode::Enum mode = ExtForceMode::FORCE) = 0;

    /**
    Apply external impulse on particular node.

    \param[in]  graphNodeIndex  The graph node index to apply impulse on. See #NvBlastSupportGraph.
    \param[in]  localForce      Force to apply in local actor's coordinates.
    \param[in]  mode            The mode to use when applying the force/impulse(see #ExtForceMode)
    */
    virtual void                            addForce(uint32_t graphNodeIndex, NvcVec3 localForce, ExtForceMode::Enum mode = ExtForceMode::FORCE) = 0;

    /**
    addForce on a known graph node that also records WHERE the force is applied.

    The overload above only needs the resultant and discards the point, which is
    all the bond solve requires. Chunk crushing needs the point: a per-chunk
    stress tensor is a sum over (branch vector x force), so a contact whose
    application point is dropped contributes nothing to the chunk's own stress
    state and an impact would raise the stress of every chunk around the one it
    actually hit.

    Prefer this for external CONTACT forces. Body forces (gravity, centrifugal)
    should keep using the plain overload -- they are not surface tractions and
    do not belong in the sum.

    \param[in] graphNodeIndex   Node to load.
    \param[in] localPosition    Application point, actor-local, same frame as the node centroids.
    \param[in] localForce       Force (or acceleration, per mode), actor-local.
    \param[in] mode             Whether localForce is a force or an acceleration.
    */
    virtual void                            addForceAt(uint32_t graphNodeIndex, NvcVec3 localPosition, NvcVec3 localForce, ExtForceMode::Enum mode = ExtForceMode::FORCE) = 0;

    /**
    Apply external gravity on particular actor of family. This function applies gravity on every node withing actor, so it makes sense only for static actors.

    \param[in]  actor           The actor to apply gravitational acceleration on.
    \param[in]  localGravity    Gravity to apply in local actor's coordinates. ExtForceMode::ACCELERATION is used.

    \return true iff acceleration was applied on at least one node.
    */
    virtual bool                            addGravity(const NvBlastActor& actor, NvcVec3 localGravity) = 0;

    /**
    Apply centrifugal acceleration produced by actor's angular movement.

    \param[in]  actor                   The actor to apply impulse on.
    \param[in]  localCenterMass         Actor's local center of mass.
    \param[in]  localAngularVelocity    Local angular velocity of an actor.

    \return true iff force was applied on at least one node.
    */
    virtual bool                            addCentrifugalAcceleration(const NvBlastActor& actor, NvcVec3 localCenterMass, NvcVec3 localAngularVelocity) = 0;

    /**
    Update stress solver.

    Actual performance of stress calculation happens there. Call it after all relevant forces were applied, usually every frame.
    */
    virtual void                            update() = 0;

    /**
    Get overstressed/broken bonds count.

    This count is updated after every update() call. Number of overstressed bond directly hints if any bond fracture is recommended by stress solver.

    \return the overstressed bonds count.
    */
    virtual uint32_t                        getOverstressedBondCount() const = 0;

    /**
    Number of connected components ("islands") in the solver graph after the last update().
    Static nodes are treated as cut points, so structures sharing only a static/world node
    are counted as separate islands. Foundation for island-aware (per-component) solving.

    \return the island count.
    */
    virtual uint32_t                        getIslandCount() const = 0;

    /**
    Enable or disable island-aware solving: each disconnected component ("island") is solved
    independently. With a single island this is identical to the default whole-graph solve;
    with multiple islands the result matches within solver tolerance. Default: disabled.

    \param[in]  enabled     true to solve per-island, false for the whole-graph solve.
    */
    virtual void                            setIslandAware(bool enabled) = 0;

    /**
    \return whether island-aware solving is currently enabled.
    */
    virtual bool                            getIslandAware() const = 0;

    /**
    Enable or disable skipping of settled islands (requires island-aware solving). An island whose
    velocity inputs are unchanged since its last solve and that already converged is not re-solved (the
    solve would be a no-op); its impulses/stresses are kept. Any new input (contact, wake) re-solves it
    the same frame, and topology changes rebuild the baseline first. Paused, never evicted. Default: off.

    \param[in]  enabled     true to skip settled islands.
    */
    virtual void                            setSkipSettled(bool enabled) = 0;
    /// See ExtStressGpuSolveParams::skipStableUnconverged. GPU path only; the
    /// CPU island skip still requires convergence (documented divergence).
    virtual void                            setSkipStableUnconverged(bool enabled) = 0;

    /**
    \return whether settled-island skipping is currently enabled.
    */
    virtual bool                            getSkipSettled() const = 0;

    /**
    \return the number of settled islands skipped during the last update().
    */
    virtual uint32_t                        getIslandsSkipped() const = 0;

    /**
    \return the number of islands the last update partitioned the graph into for the per-island solve
    (0 unless island-aware solving ran with >1 island). islandsSkipped is always <= this.
    */
    virtual uint32_t                        getIslandsTotal() const = 0;

    /**
    Enable the optional CUDA CGNR backend. Graph topology and warm-start state
    remain resident on the GPU; the CPU implementation remains the fallback.

    \return true when the requested backend is available and selected.
    */
    virtual bool                            setGpuAccelerated(bool enabled) = 0;
    virtual void                            setGpuCudaContext(void* cudaContext) = 0;
    virtual void                            setGpuMinimumBondCount(uint32_t bondCount) = 0;
    virtual bool                            getGpuAccelerated() const = 0;
    virtual float                           getGpuSolveMilliseconds() const = 0;

    /// Host wall time inside the GPU solve, split into work and waiting.
    /// Only the first is reclaimable by faster host code.
    /// Host walls around the GPU solve: the pre-solve initialize, the graph
    /// solve call itself, and the post-solve error walk.
    virtual float                           getInitializeMilliseconds() const = 0;
    virtual float                           getGraphSolveMilliseconds() const = 0;
    virtual float                           getCalcErrorMilliseconds() const = 0;
    /// Per-bond host copy of solved impulses out of the GPU buffer.
    /// Host walks bracketing the GPU solve inside GraphProcessor::solve.
    virtual float                           getHostWalkInMilliseconds() const = 0;
    virtual float                           getHostResetMilliseconds() const = 0;
    virtual float                           getHostBondStressMilliseconds() const = 0;
    virtual float                           getHostNodeStressMilliseconds() const = 0;
    virtual float                           getGpuImpulseCopyMilliseconds() const = 0;
    /// Solver bond groups whose stress was provably unchanged and skipped.
    /// MUST be non-zero when the skip is enabled, or the A/B is measuring
    /// nothing and cannot tell "no win" from "never ran".
    virtual uint64_t                        getBondStressGroupsSkipped() const = 0;
    /// Verify-mode audit of the parallel bond-stress walk: group comparisons
    /// performed, and orderings that disagreed. Mismatches must be zero; zero
    /// CHECKS means the audit never ran and is inconclusive, not a pass.
    virtual uint64_t                        getBondStressParallelChecks() const = 0;
    virtual uint64_t                        getBondStressParallelMismatches() const = 0;
    virtual uint32_t                        getGpuImpulseCopyCount() const = 0;
    virtual float                           getGpuHostWorkMilliseconds() const = 0;
    virtual float                           getGpuHostBlockedMilliseconds() const = 0;
    virtual uint64_t                        getGpuHostToDeviceBytes() const = 0;
    virtual uint64_t                        getGpuDeviceToHostBytes() const = 0;

    /**
    Generate fracture commands for particular actor.

    Calling this function if getOverstressedBondCount() == 0 or actor has no bond doesn't make sense, bondFractureCount will be '0'.
    Filled fracture commands buffer can be passed directly to NvBlastActorApplyFracture.

    IMPORTANT: NvBlastFractureBuffers::bondFractures will point to internal stress solver memory which will be valid till next call
    of any of generateFractureCommands() functions or stress solver release() call.

    \param[in]  actor                   The actor to fill fracture commands for.
    \param[in]  commands                Pointer to command buffer to fill.
    */
    virtual void                            generateFractureCommands(const NvBlastActor& actor, NvBlastFractureBuffers& commands) = 0;

    /**
    Generate fracture commands for every actor in family.

    Actors and commands buffer must be passed in order to be filled. It's recommended for bufferSize to be the count of actor with more then one bond in family.

    Calling this function if getOverstressedBondCount() == 0 or actor has no bond doesn't make sense, '0' will be returned.

    IMPORTANT: NvBlastFractureBuffers::bondFractures will point to internal stress solver memory which will be valid till next call
    of any of generateFractureCommands() functions or stress solver release() call.

    \param[out] buffer          A user-supplied array of NvBlastActor pointers to fill.
    \param[out] commandsBuffer  A user-supplied array of NvBlastFractureBuffers to fill.
    \param[in]  bufferSize      The number of elements available to write into buffer.

    \return the number of actors and command buffers written to the buffer.
    */
    virtual uint32_t                        generateFractureCommandsPerActor(const NvBlastActor** actorBuffer, NvBlastFractureBuffers* commandsBuffer, uint32_t bufferSize) = 0;

    /**
    Reset stress solver.

    Stress solver uses warm start internally, calling this function will flush all previous data calculated and also zeros frame count.
    This function is to be used for debug purposes.
    */
    virtual void                            reset() = 0;

    /**
    Get stress solver linear error.

    \return the total linear error of stress calculation.
    */
    virtual float                           getStressErrorLinear() const = 0;

    /**
    Get stress solver angular error.

    \return the total angular error of stress calculation.
    */
    virtual float                           getStressErrorAngular() const = 0;

    /**
    Whether or not the solver converged to a solution within the desired error.

    \return true iff the solver converged.
    */
    virtual bool                            converged() const = 0;

    /**
    Get stress solver total frames count (update() calls) since it was created (or reset).

    \return the frames count.
    */
    virtual uint32_t                        getFrameCount() const = 0;

    /**
    Get stress solver bonds count, after graph reduction was applied.

    \return the bonds count.
    */
    virtual uint32_t                        getBondCount() const = 0;

    /**
    Get stress solver excess force related to broken bonds for the given actor.
    This is intended to be called after damage is applied to bonds and actors are split, but before the next call to 'update()'.
    Force is intended to be applied to the center of mass, torque due to linear forces that happen away from the COM are converted
    to torque as part of this function.

    \return true if data was gathered, false otherwise.
    */
    virtual bool                            getExcessForces(uint32_t actorIndex, const NvcVec3& com, NvcVec3& force, NvcVec3& torque) = 0;

    /**
    Read back the per-bond stress state from the last update(), indexed by ASSET bond index
    (the same indexing as NvBlastAssetGetBonds), so callers can relate stress to authored
    joint geometry.

    Compression and tension are mutually exclusive (they measure stress along the bond normal
    in opposite directions), so at most one of them is non-zero for a given bond. Shear is
    independent and can co-exist with either. All values are pressures, directly comparable to
    the ExtStressSolverSettings elastic/fatal limits — dividing by the matching limit gives the
    utilisation of that joint, and its reciprocal is the safety factor.

    Bonds that are broken, were reduced away by graph reduction, or are otherwise absent from
    the solver graph are written as 0.

    Any of the output pointers may be null to skip that component.

    \param[out] compression     Array of at least `capacity` floats, or null.
    \param[out] tension         Array of at least `capacity` floats, or null.
    \param[out] shear           Array of at least `capacity` floats, or null.
    \param[in]  capacity        Number of entries the output arrays can hold.

    \return the number of entries written (min of capacity and the asset's bond count).
    */
    virtual uint32_t                        getBondStresses(float* compression, float* tension, float* shear, uint32_t capacity) const = 0;


    /**
    Debug Render Mode
    */
    enum DebugRenderMode
    {
        STRESS_PCT_MAX = 0,         //!<    render the maximum of the compression, tension, and shear stress percentages
        STRESS_PCT_COMPRESSION = 1, //!<    render the compression stress percentage
        STRESS_PCT_TENSION = 2,     //!<    render the tension stress percentage
        STRESS_PCT_SHEAR = 3,       //!<    render the shear stress percentage
    };

    /**
    Used to store a single line and colour for debug rendering.
    */
    struct DebugLine
    {
        DebugLine(const NvcVec3& p0, const NvcVec3& p1, const uint32_t& c)
            : pos0(p0), color0(c), pos1(p1), color1(c) {}

        NvcVec3 pos0;
        uint32_t        color0;
        NvcVec3 pos1;
        uint32_t        color1;
    };

    /**
    Debug Buffer
    */
    struct DebugBuffer
    {
        const DebugLine* lines;
        uint32_t         lineCount;
    };

    /**
    Fill debug render for passed array of support graph nodes.

    NOTE: Returned DebugBuffer points into internal memory which is valid till next fillDebugRender() call.

    \param[in]  nodes           Node indices of support graph to debug render for.
    \param[in]  nodeCount       Node indices count.
    \param[in]  mode            Debug render mode.
    \param[in]  scale           Scale to be applied on impulses.

    \return debug buffer with array of lines
    */
    virtual const DebugBuffer               fillDebugRender(const uint32_t* nodes, uint32_t nodeCount, DebugRenderMode mode, float scale = 1.0f) = 0;
};

} // namespace Blast
} // namespace Nv


#endif // ifndef NVBLASTEXTSTRESSSOLVER_H
