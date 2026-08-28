#include "ext_stress_bridge.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#include "NvBlast.h"
#include "NvBlastGlobals.h"
#include "NvBlastSupportGraph.h"
#include "NvBlastExtStressSolver.h"

namespace
{

using namespace Nv::Blast;

struct ExtStressSolverHandleImpl
{
    ExtStressSolver* solver{nullptr};
    NvBlastAsset* asset{nullptr};
    NvBlastFamily* family{nullptr};

    void* assetMem{nullptr};
    void* assetScratch{nullptr};
    void* familyMem{nullptr};
    void* actorScratch{nullptr};

    std::vector<uint32_t> inputToGraph;
    std::vector<uint32_t> graphNodeIndices;
    std::vector<uint32_t> graphToInput;

    struct ActorEntry
    {
        NvBlastActor* actor{nullptr};
        uint32_t actorIndex{UINT32_MAX};
        std::vector<uint32_t> graphNodes;
        std::vector<uint32_t> inputNodes;
    };

    std::vector<ActorEntry> actors;
    // O(1) input-node -> owning-actor lookup. inputNodeToActorSlot[n] is the slot
    // in `actors` that owns input node n (UINT32_MAX = unowned). Rebuilt lazily by
    // ensureActorIndex() when `actorIndexDirty` is set after an actor-table change
    // (rebuildActorTable / apply_fracture_commands / reset). Replaces the former
    // O(total nodes) linear scan that ran once per injected force.
    std::vector<uint32_t> inputNodeToActorSlot;
    // O(1) actorIndex -> slot lookup, rebuilt alongside inputNodeToActorSlot.
    // Gravity injection asks for one actor per body every tick, so a linear
    // scan here is quadratic in the body count of a collapsing structure --
    // which is exactly the case that has to stay real time.
    std::vector<uint32_t> actorIndexToSlot;
    bool actorIndexDirty{true};
    std::vector<uint8_t> splitScratch;
    std::vector<NvBlastActor*> splitActors;
    std::vector<NvBlastBondFractureData> fractureScratch;

    // Chunk fracture commands produced by the last generate*() call, keyed by
    // actorIndex. They are addressed by ASSET CHUNK INDEX, which is an internal
    // detail of how this bridge lays out its single-level asset, so they are
    // not handed to the caller -- apply_fracture_commands re-attaches them.
    // Copied rather than pointed at: the solver's own buffer is only valid
    // until its next generate*() call.
    std::vector<std::pair<uint32_t, std::vector<NvBlastChunkFractureData>>> pendingChunkFractures;
    std::vector<uint32_t> crushScratch;
    std::vector<float> nodeScratch;
    std::vector<NvBlastChunkFractureData> chunkFractureScratch;

    const std::vector<NvBlastChunkFractureData>* findPendingChunkFractures(uint32_t actorIndex) const
    {
        for (const auto& entry : pendingChunkFractures)
        {
            if (entry.first == actorIndex)
            {
                return &entry.second;
            }
        }
        return nullptr;
    }
};

uint32_t mapGraphNodeToInput(const ExtStressSolverHandleImpl& handle, uint32_t graphIndex)
{
    if (graphIndex < handle.graphToInput.size())
    {
        return handle.graphToInput[graphIndex];
    }
    return UINT32_MAX;
}

uint32_t mapInputNodeToGraph(const ExtStressSolverHandleImpl& handle, uint32_t inputIndex)
{
    if (inputIndex < handle.inputToGraph.size())
    {
        return handle.inputToGraph[inputIndex];
    }
    return UINT32_MAX;
}

inline StressVec3 toStressVec3(const NvcVec3& value)
{
    StressVec3 result;
    result.x = value.x;
    result.y = value.y;
    result.z = value.z;
    return result;
}

inline NvcVec3 toNvcVec3(const StressVec3& value)
{
    return NvcVec3{value.x, value.y, value.z};
}

void releaseHandle(ExtStressSolverHandleImpl* handle)
{
    if (!handle)
    {
        return;
    }

    handle->actors.clear();
    handle->splitScratch.clear();
    handle->splitActors.clear();
    handle->fractureScratch.clear();

    if (handle->solver)
    {
        handle->solver->release();
        handle->solver = nullptr;
    }

    if (handle->actorScratch)
    {
        NVBLAST_FREE(handle->actorScratch);
        handle->actorScratch = nullptr;
    }

    if (handle->familyMem)
    {
        NVBLAST_FREE(handle->familyMem);
        handle->familyMem = nullptr;
        handle->family = nullptr;
    }

    if (handle->assetScratch)
    {
        NVBLAST_FREE(handle->assetScratch);
        handle->assetScratch = nullptr;
    }

    if (handle->assetMem)
    {
        NVBLAST_FREE(handle->assetMem);
        handle->assetMem = nullptr;
        handle->asset = nullptr;
    }

    delete handle;
}

void stressLog(int type, const char* msg, const char* file, int line)
{
    NV_UNUSED(type);
    NV_UNUSED(file);
    NV_UNUSED(line);
#ifdef DEBUG
    if (msg)
    {
        std::fprintf(stderr, "[Blast][ExtStress] %s (%s:%d)\n", msg, file ? file : "", line);
    }
#else
    NV_UNUSED(msg);
#endif
}

const NvBlastLog kLogFn = stressLog;

inline ExtForceMode::Enum toForceMode(uint32_t mode)
{
    switch (mode)
    {
    case ExtForceMode::ACCELERATION:
        return ExtForceMode::ACCELERATION;
    case ExtForceMode::FORCE:
    default:
        return ExtForceMode::FORCE;
    }
}

inline ExtStressSolverSettings toSettings(const ExtStressSolverSettingsDesc* settingsDesc)
{
    ExtStressSolverSettings settings;
    if (!settingsDesc)
    {
        return settings;
    }

    settings.maxSolverIterationsPerFrame = settingsDesc->max_solver_iterations_per_frame;
    settings.graphReductionLevel = settingsDesc->graph_reduction_level;
    return settings;
}

inline ExtStressMaterial toMaterial(const ExtStressMaterialDesc& desc)
{
    ExtStressMaterial material;
    material.compressionElasticLimit = desc.compression_elastic_limit;
    material.compressionFatalLimit = desc.compression_fatal_limit;
    material.tensionElasticLimit = desc.tension_elastic_limit;
    material.tensionFatalLimit = desc.tension_fatal_limit;
    material.shearElasticLimit = desc.shear_elastic_limit;
    material.shearFatalLimit = desc.shear_fatal_limit;
    material.crush.capPressure = desc.crush_cap_pressure;
    material.crush.cohesion = desc.crush_cohesion;
    material.crush.frictionSlope = desc.crush_friction_slope;
    material.crush.crushEnergy = desc.crush_energy;
    material.crush.crushViscosity = desc.crush_viscosity;
    material.crush.strainRateExponent = desc.crush_strain_rate_exponent;
    material.crush.referenceStrainRate = desc.crush_reference_strain_rate;
    material.crush.debrisMassFraction = desc.crush_debris_mass_fraction;
    material.crush.debrisFragmentCount = desc.crush_debris_fragment_count;
    return material;
}

extern "C" uint32_t ext_stress_sizeof_actor()
{
    return static_cast<uint32_t>(sizeof(ExtStressSolverHandleImpl::ActorEntry));
}

extern "C" uint32_t ext_stress_sizeof_actor_buffer()
{
    return static_cast<uint32_t>(sizeof(ExtStressActor));
}

extern "C" uint32_t ext_stress_sizeof_ext_split_event()
{
    return static_cast<uint32_t>(sizeof(ExtStressSplitEvent));
}

void rebuildActorTable(ExtStressSolverHandleImpl& handle)
{
    handle.actors.clear();

    const uint32_t actorCount = NvBlastFamilyGetActorCount(handle.family, kLogFn);
    handle.actors.reserve(actorCount);

    std::vector<NvBlastActor*> actorBuffer(actorCount);
    NvBlastFamilyGetActors(actorBuffer.data(), actorCount, handle.family, kLogFn);

    for (uint32_t i = 0; i < actorCount; ++i)
    {
        NvBlastActor* llActor = actorBuffer[i];
        if (!llActor)
        {
            continue;
        }

        ExtStressSolverHandleImpl::ActorEntry entry;
        entry.actor = llActor;
        entry.actorIndex = NvBlastActorGetIndex(llActor, kLogFn);

        const uint32_t graphNodeCount = NvBlastActorGetGraphNodeCount(llActor, kLogFn);
        entry.graphNodes.resize(graphNodeCount);
        if (graphNodeCount > 0)
        {
            NvBlastActorGetGraphNodeIndices(entry.graphNodes.data(), graphNodeCount, llActor, kLogFn);
        }

        entry.inputNodes.resize(graphNodeCount);
        for (uint32_t n = 0; n < graphNodeCount; ++n)
        {
            const uint32_t graphNode = entry.graphNodes[n];
            entry.inputNodes[n] = mapGraphNodeToInput(handle, graphNode);
        }

        handle.actors.push_back(std::move(entry));
    }
    handle.actorIndexDirty = true;
}

void ensureActorIndex(ExtStressSolverHandleImpl& handle);

ExtStressSolverHandleImpl::ActorEntry* findActorByIndex(ExtStressSolverHandleImpl& handle, uint32_t actorIndex)
{
    ensureActorIndex(handle);
    if (actorIndex < handle.actorIndexToSlot.size())
    {
        const uint32_t slot = handle.actorIndexToSlot[actorIndex];
        if (slot < handle.actors.size())
        {
            return &handle.actors[slot];
        }
    }
    return nullptr;
}

/// E6: A/B for the apply-path bookkeeping rewrite (default ON, =0 restores
/// the erase-per-split / rebuild-per-command path). One binary, two arms.
///
/// Three superlinear terms lived here, all bookkeeping, all invisible at one
/// split and multiplicative at four hundred: a full O(total nodes) index
/// rebuild once per fracture command (every erase/push re-dirtied it), a
/// linear actor scan per generated actor, and an O(bodies) vector erase per
/// split. Same values, same order — only the cost changes.
static bool applyIndexIncremental()
{
    static const bool enabled = [] {
        const char* raw = std::getenv("BLAST_APPLY_INDEX_INCREMENTAL");
        return raw == nullptr || std::string(raw) != "0";
    }();
    return enabled;
}

const ExtStressSolverHandleImpl::ActorEntry* findActorByPointer(const ExtStressSolverHandleImpl& handle, const NvBlastActor* actor)
{
    for (const auto& entry : handle.actors)
    {
        if (entry.actor == actor)
        {
            return &entry;
        }
    }
    return nullptr;
}

// Rebuild the input-node -> actor-slot table when stale. One O(total graph
// nodes) pass, amortised across a whole frame's force injections: the actor
// table only changes on fracture/reset, never during injection, so all the
// per-force lookups in a frame share a single rebuild. This replaces the prior
// O(total nodes) linear scan that ran once *per* injected force.
void ensureActorIndex(ExtStressSolverHandleImpl& handle)
{
    if (!handle.actorIndexDirty)
    {
        return;
    }

    handle.inputNodeToActorSlot.assign(handle.inputToGraph.size(), UINT32_MAX);
    uint32_t maxActorIndex = 0;
    for (const auto& entry : handle.actors)
    {
        maxActorIndex = std::max(maxActorIndex, entry.actorIndex);
    }
    handle.actorIndexToSlot.assign(
        handle.actors.empty() ? 0 : static_cast<size_t>(maxActorIndex) + 1, UINT32_MAX);
    for (uint32_t slot = 0; slot < handle.actors.size(); ++slot)
    {
        const uint32_t actorIndex = handle.actors[slot].actorIndex;
        if (actorIndex < handle.actorIndexToSlot.size()
            && handle.actorIndexToSlot[actorIndex] == UINT32_MAX)
        {
            handle.actorIndexToSlot[actorIndex] = slot;
        }
        for (uint32_t inputNode : handle.actors[slot].inputNodes)
        {
            // First writer wins, matching the prior loop's first-match semantics
            // (each input node is owned by exactly one actor, so this is moot in
            // practice but keeps behaviour identical defensively).
            if (inputNode < handle.inputNodeToActorSlot.size()
                && handle.inputNodeToActorSlot[inputNode] == UINT32_MAX)
            {
                handle.inputNodeToActorSlot[inputNode] = slot;
            }
        }
    }
    handle.actorIndexDirty = false;
}

ExtStressSolverHandleImpl::ActorEntry* findActorOwningInputNode(ExtStressSolverHandleImpl& handle, uint32_t inputIndex)
{
    ensureActorIndex(handle);
    if (inputIndex < handle.inputNodeToActorSlot.size())
    {
        const uint32_t slot = handle.inputNodeToActorSlot[inputIndex];
        if (slot < handle.actors.size())
        {
            return &handle.actors[slot];
        }
    }
    return nullptr;
}

} // namespace

extern "C" ExtStressSolverHandle*
ext_stress_solver_create(const ExtStressNodeDesc* nodes,
                        uint32_t node_count,
                        const ExtStressBondDesc* bonds,
                        uint32_t bond_count,
                        const ExtStressMaterialDesc* materials,
                        uint32_t material_count,
                        const ExtStressSolverSettingsDesc* settingsDesc)
{
    if (!nodes || node_count == 0U || !bonds || bond_count == 0U)
    {
        return nullptr;
    }
    // Every bond must reference a material inside the effective table (one
    // default entry when no table is supplied). An out-of-range index is an
    // authoring error, reported loudly rather than clamped into silence.
    const uint32_t effectiveMaterialCount =
        (materials && material_count > 0U) ? material_count : 1U;
    for (uint32_t i = 0; i < bond_count; ++i)
    {
        if (bonds[i].material >= effectiveMaterialCount)
        {
            kLogFn(
                NvBlastMessage::Error,
                "ext_stress_solver_create: bond material index out of range",
                __FILE__,
                __LINE__);
            return nullptr;
        }
    }

    ExtStressSolverHandleImpl* handle = new (std::nothrow) ExtStressSolverHandleImpl();
    if (!handle)
    {
        return nullptr;
    }

    std::vector<NvBlastChunkDesc> chunkDescs;
    chunkDescs.resize(node_count + 1U);

    NvBlastChunkDesc& rootChunk = chunkDescs[0];
    rootChunk.centroid[0] = 0.0f;
    rootChunk.centroid[1] = 0.0f;
    rootChunk.centroid[2] = 0.0f;
    rootChunk.volume = std::max(1.0f, static_cast<float>(node_count));
    rootChunk.parentChunkDescIndex = UINT32_MAX;
    rootChunk.flags = NvBlastChunkDesc::NoFlags;
    rootChunk.userData = 0U;

    for (uint32_t i = 0; i < node_count; ++i)
    {
        NvBlastChunkDesc& desc = chunkDescs[i + 1];
        desc.centroid[0] = nodes[i].centroid.x;
        desc.centroid[1] = nodes[i].centroid.y;
        desc.centroid[2] = nodes[i].centroid.z;
        desc.volume = nodes[i].volume > 0.0f ? nodes[i].volume : std::max(nodes[i].mass, 1.0f);
        desc.parentChunkDescIndex = 0;
        desc.flags = NvBlastChunkDesc::SupportFlag;
        desc.userData = i;
    }

    std::vector<NvBlastBondDesc> bondDescs;
    bondDescs.resize(bond_count);
    for (uint32_t i = 0; i < bond_count; ++i)
    {
        NvBlastBondDesc& desc = bondDescs[i];
        desc.chunkIndices[0] = bonds[i].node0 + 1U;
        desc.chunkIndices[1] = bonds[i].node1 + 1U;

        desc.bond.centroid[0] = bonds[i].centroid.x;
        desc.bond.centroid[1] = bonds[i].centroid.y;
        desc.bond.centroid[2] = bonds[i].centroid.z;

        desc.bond.normal[0] = bonds[i].normal.x;
        desc.bond.normal[1] = bonds[i].normal.y;
        desc.bond.normal[2] = bonds[i].normal.z;

        desc.bond.area = bonds[i].area > 0.0f ? bonds[i].area : 1.0f;
        desc.bond.userData = i;
    }

    NvBlastAssetDesc assetDesc;
    assetDesc.chunkCount = static_cast<uint32_t>(chunkDescs.size());
    assetDesc.chunkDescs = chunkDescs.data();
    assetDesc.bondCount = static_cast<uint32_t>(bondDescs.size());
    assetDesc.bondDescs = bondDescs.data();

    const size_t scratchSize = NvBlastGetRequiredScratchForCreateAsset(&assetDesc, kLogFn);
    handle->assetScratch = NVBLAST_ALLOC(scratchSize);
    if (!handle->assetScratch)
    {
        releaseHandle(handle);
        return nullptr;
    }

    const size_t assetMemSize = NvBlastGetAssetMemorySize(&assetDesc, kLogFn);
    handle->assetMem = NVBLAST_ALLOC(assetMemSize);
    if (!handle->assetMem)
    {
        releaseHandle(handle);
        return nullptr;
    }

    handle->asset = NvBlastCreateAsset(handle->assetMem, &assetDesc, handle->assetScratch, kLogFn);
    if (!handle->asset)
    {
        releaseHandle(handle);
        return nullptr;
    }

    const size_t familyMemSize = NvBlastAssetGetFamilyMemorySize(handle->asset, kLogFn);
    handle->familyMem = NVBLAST_ALLOC(familyMemSize);
    if (!handle->familyMem)
    {
        releaseHandle(handle);
        return nullptr;
    }

    handle->family = NvBlastAssetCreateFamily(handle->familyMem, handle->asset, kLogFn);
    if (!handle->family)
    {
        releaseHandle(handle);
        return nullptr;
    }

    NvBlastActorDesc actorDesc{};
    // Bond health is remaining contact area — pure geometry. Strength is the
    // per-bond material's limits; ductility is that material's elastic->fatal
    // band. Uniform health=1 once made every joint equally fragile regardless
    // of authored area; do not reintroduce it.
    std::vector<float> initialBondHealths(bond_count);
    for (uint32_t i = 0; i < bond_count; ++i)
    {
        initialBondHealths[i] = bonds[i].area > 0.0f ? bonds[i].area : 1.0f;
    }
    actorDesc.initialBondHealths = initialBondHealths.data();
    actorDesc.uniformInitialBondHealth = 0.0f;
    actorDesc.uniformInitialLowerSupportChunkHealth = 1.0f;

    const size_t actorScratchSize = NvBlastFamilyGetRequiredScratchForCreateFirstActor(handle->family, kLogFn);
    handle->actorScratch = NVBLAST_ALLOC(actorScratchSize);
    if (!handle->actorScratch)
    {
        releaseHandle(handle);
        return nullptr;
    }

    NvBlastActor* createdActor = NvBlastFamilyCreateFirstActor(handle->family, &actorDesc, handle->actorScratch, kLogFn);
    if (!createdActor)
    {
        releaseHandle(handle);
        return nullptr;
    }

    ExtStressSolverSettings settings = toSettings(settingsDesc);
    handle->solver = ExtStressSolver::create(*handle->family, settings);
    if (!handle->solver)
    {
        releaseHandle(handle);
        return nullptr;
    }

    if (materials && material_count > 0U)
    {
        std::vector<ExtStressMaterial> table(material_count);
        for (uint32_t i = 0; i < material_count; ++i)
        {
            table[i] = toMaterial(materials[i]);
        }
        handle->solver->setMaterials(table.data(), material_count);
    }
    {
        std::vector<uint32_t> bondMaterials(bond_count);
        for (uint32_t i = 0; i < bond_count; ++i)
        {
            bondMaterials[i] = bonds[i].material;
        }
        handle->solver->setBondMaterials(bondMaterials.data(), bond_count);
    }

    const NvBlastSupportGraph supportGraph = NvBlastAssetGetSupportGraph(handle->asset, kLogFn);
    handle->inputToGraph.assign(node_count, UINT32_MAX);
    handle->graphNodeIndices.resize(supportGraph.nodeCount);
    handle->graphToInput.assign(supportGraph.nodeCount, UINT32_MAX);
    for (uint32_t graphIndex = 0; graphIndex < supportGraph.nodeCount; ++graphIndex)
    {
        handle->graphNodeIndices[graphIndex] = graphIndex;
        const uint32_t chunkIndex = supportGraph.chunkIndices[graphIndex];
        if (chunkIndex > 0 && chunkIndex <= node_count)
        {
            const uint32_t inputIndex = chunkIndex - 1U;
            handle->inputToGraph[inputIndex] = graphIndex;
            handle->graphToInput[graphIndex] = inputIndex;
            const ExtStressNodeDesc& nodeDesc = nodes[inputIndex];
            handle->solver->setNodeInfo(graphIndex,
                                        nodeDesc.mass,
                                        nodeDesc.volume > 0.0f ? nodeDesc.volume : std::max(nodeDesc.mass, 1.0f),
                                        toNvcVec3(nodeDesc.centroid));
        }
    }

    rebuildActorTable(*handle);
    for (auto& entry : handle->actors)
    {
        if (entry.actor)
        {
            handle->solver->notifyActorCreated(*entry.actor);
        }
    }

    return reinterpret_cast<ExtStressSolverHandle*>(handle);
}

extern "C" void
ext_stress_solver_destroy(ExtStressSolverHandle* handlePtr)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle)
    {
        return;
    }

    for (auto& entry : handle->actors)
    {
        if (entry.actor)
        {
            handle->solver->notifyActorDestroyed(*entry.actor);
        }
    }
    releaseHandle(handle);
}

extern "C" void
ext_stress_solver_set_settings(ExtStressSolverHandle* handlePtr, const ExtStressSolverSettingsDesc* settingsDesc)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver)
    {
        return;
    }

    ExtStressSolverSettings settings = toSettings(settingsDesc);
    handle->solver->setSettings(settings);
}

extern "C" uint32_t
ext_stress_solver_graph_node_count(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return handle ? static_cast<uint32_t>(handle->graphNodeIndices.size()) : 0U;
}

extern "C" uint32_t
ext_stress_solver_bond_count(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getBondCount() : 0U;
}

extern "C" void
ext_stress_solver_reset(ExtStressSolverHandle* handlePtr)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver)
    {
        return;
    }
    handle->solver->reset();
    // Defensive: the actor table is unchanged by a solver reset, but re-arm the
    // index so any lookup after a reset can't observe a stale slot mapping.
    handle->actorIndexDirty = true;
}

namespace
{
// Shared force-application path used by both the single-force and batched
// entry points so their behaviour can never diverge. Bond-graph nodes keep
// stable graph indices after actor splits, so inject those in O(1) rather than
// asking addForce(actor, position) to scan every graph node in the actor for
// every contact. Subsupport chunks have no graph index and retain the actor
// position lookup as their fallback.
inline void applyForceToInputNode(ExtStressSolverHandleImpl& handle,
                                  uint32_t node_index,
                                  const NvcVec3& pos,
                                  const NvcVec3& force,
                                  ExtForceMode::Enum mode)
{
    if (node_index >= handle.inputToGraph.size())
    {
        return;
    }

    const uint32_t graphIndex = handle.inputToGraph[node_index];

    if (graphIndex != UINT32_MAX)
    {
        // Position-aware: these are contact forces, and a per-chunk stress
        // tensor is built from where they act, not just their resultant.
        handle.solver->addForceAt(graphIndex, pos, force, mode);
        return;
    }

    if (auto* entry = findActorOwningInputNode(handle, node_index))
    {
        if (entry->actor)
        {
            handle.solver->addForce(*entry->actor, pos, force, mode);
            return;
        }
    }
}
} // namespace

extern "C" void
ext_stress_solver_add_force(ExtStressSolverHandle* handlePtr,
                           uint32_t node_index,
                           const StressVec3* local_position,
                           const StressVec3* local_force,
                           uint32_t mode)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver)
    {
        return;
    }

    const NvcVec3 force = local_force ? toNvcVec3(*local_force) : NvcVec3{0.0f, 0.0f, 0.0f};
    const NvcVec3 pos = local_position ? toNvcVec3(*local_position) : NvcVec3{0.0f, 0.0f, 0.0f};
    applyForceToInputNode(*handle, node_index, pos, force, toForceMode(mode));
}

// Batched external-force injection. Applies `count` forces in a single FFI
// crossing instead of one crossing per force, mirroring
// ext_stress_solver_add_force for every entry. The parallel input arrays are:
//   node_indices[i]              -> input node index for force i
//   local_positions[3*i + 0..2]  -> body-local application point (x, y, z)
//   local_forces[3*i + 0..2]     -> body-local force vector (x, y, z)
// `mode` is shared by every force (the contact-injection caller always uses
// Force mode). A null position or force array is treated as all-zero. Returns
// the number of entries processed.
extern "C" uint32_t
ext_stress_solver_add_all_forces(ExtStressSolverHandle* handlePtr,
                                 const uint32_t* node_indices,
                                 const float* local_positions,
                                 const float* local_forces,
                                 uint32_t count,
                                 uint32_t mode)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver || !node_indices || count == 0U)
    {
        return 0U;
    }

    const ExtForceMode::Enum forceMode = toForceMode(mode);
    for (uint32_t i = 0; i < count; ++i)
    {
        NvcVec3 pos{0.0f, 0.0f, 0.0f};
        NvcVec3 force{0.0f, 0.0f, 0.0f};
        if (local_positions)
        {
            const float* p = local_positions + static_cast<size_t>(i) * 3U;
            pos = NvcVec3{p[0], p[1], p[2]};
        }
        if (local_forces)
        {
            const float* f = local_forces + static_cast<size_t>(i) * 3U;
            force = NvcVec3{f[0], f[1], f[2]};
        }
        applyForceToInputNode(*handle, node_indices[i], pos, force, forceMode);
    }
    return count;
}

extern "C" void
ext_stress_solver_add_gravity(ExtStressSolverHandle* handlePtr, const StressVec3* local_gravity)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver)
    {
        return;
    }

    const NvcVec3 gravity = local_gravity ? toNvcVec3(*local_gravity) : NvcVec3{0.0f, 0.0f, 0.0f};
    for (auto& entry : handle->actors)
    {
        if (entry.actor)
        {
            handle->solver->addGravity(*entry.actor, gravity);
        }
    }
}

extern "C" uint8_t
ext_stress_solver_add_actor_gravity(ExtStressSolverHandle* handlePtr,
                                    uint32_t actor_index,
                                    const StressVec3* local_gravity)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver)
    {
        return 0U;
    }

    auto* entry = findActorByIndex(*handle, actor_index);
    if (!entry || !entry->actor)
    {
        return 0U;
    }

    const NvcVec3 gravity = local_gravity ? toNvcVec3(*local_gravity) : NvcVec3{0.0f, 0.0f, 0.0f};
    handle->solver->addGravity(*entry->actor, gravity);
    return 1U;
}

extern "C" uint8_t
ext_stress_solver_add_centrifugal_acceleration(ExtStressSolverHandle* handlePtr,
                                               uint32_t actor_index,
                                               const StressVec3* local_center_mass,
                                               const StressVec3* local_angular_velocity)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver)
    {
        return 0U;
    }

    auto* entry = findActorByIndex(*handle, actor_index);
    if (!entry || !entry->actor)
    {
        return 0U;
    }

    const NvcVec3 centerMass = local_center_mass ? toNvcVec3(*local_center_mass) : NvcVec3{0.0f, 0.0f, 0.0f};
    const NvcVec3 angularVelocity =
        local_angular_velocity ? toNvcVec3(*local_angular_velocity) : NvcVec3{0.0f, 0.0f, 0.0f};
    return handle->solver->addCentrifugalAcceleration(*entry->actor, centerMass, angularVelocity) ? 1U : 0U;
}

extern "C" uint32_t
ext_stress_solver_add_all_actor_gravity(ExtStressSolverHandle* handlePtr,
                                        float world_gravity_x,
                                        float world_gravity_y,
                                        float world_gravity_z,
                                        const float* actor_rotations,
                                        uint32_t rotation_count)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver)
    {
        return 0U;
    }

    const float gx = world_gravity_x;
    const float gy = world_gravity_y;
    const float gz = world_gravity_z;

    uint32_t applied = 0U;
    for (auto& entry : handle->actors)
    {
        if (!entry.actor)
        {
            continue;
        }

        NvcVec3 localGravity{gx, gy, gz};

        // Rotate world gravity into the actor's body-local frame using its
        // rigid-body orientation. This mirrors the per-actor math previously
        // performed in JS (destructible-core.ts) so behaviour is unchanged,
        // it just happens here in a single FFI crossing for all actors.
        if (actor_rotations && entry.actorIndex < rotation_count)
        {
            const float* q = actor_rotations + static_cast<size_t>(entry.actorIndex) * 4U;
            const float qx = q[0];
            const float qy = q[1];
            const float qz = q[2];
            const float qw = q[3];

            localGravity.x = qw * qw * gx + 2.0f * qy * qw * gz - 2.0f * qz * qw * gy + qx * qx * gx
                + 2.0f * qy * qx * gy + 2.0f * qz * qx * gz - qz * qz * gx - qy * qy * gx;
            localGravity.y = 2.0f * qx * qy * gx + qy * qy * gy + 2.0f * qz * qy * gz + 2.0f * qw * qz * gx
                - qz * qz * gy + qw * qw * gy - 2.0f * qx * qw * gz - qx * qx * gy;
            localGravity.z = 2.0f * qx * qz * gx + 2.0f * qy * qz * gy + qz * qz * gz - 2.0f * qw * qy * gx
                - qy * qy * gz + 2.0f * qw * qx * gy - qx * qx * gz + qw * qw * gz;
        }

        handle->solver->addGravity(*entry.actor, localGravity);
        ++applied;
    }

    return applied;
}

extern "C" void
ext_stress_solver_update(ExtStressSolverHandle* handlePtr)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver)
    {
        return;
    }
    handle->solver->update();
}

extern "C" uint32_t
ext_stress_solver_overstressed_bond_count(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getOverstressedBondCount() : 0U;
}

extern "C" uint8_t
ext_stress_solver_set_materials(ExtStressSolverHandle* handlePtr,
                                const ExtStressMaterialDesc* materials,
                                uint32_t material_count)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver || !materials || material_count == 0U)
    {
        return 0;
    }
    std::vector<ExtStressMaterial> table(material_count);
    for (uint32_t i = 0; i < material_count; ++i)
    {
        table[i] = toMaterial(materials[i]);
    }
    handle->solver->setMaterials(table.data(), material_count);
    return 1;
}

extern "C" uint32_t
ext_stress_solver_get_bond_utilisations(const ExtStressSolverHandle* handlePtr,
                                        float* out_utilisation,
                                        uint32_t capacity)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver)
    {
        return 0U;
    }
    return handle->solver->getBondUtilisations(out_utilisation, capacity);
}

/* --- chunk crushing -------------------------------------------------------
   The solver speaks GRAPH node indices; every C entry point speaks INPUT node
   indices (the order nodes were supplied to create). These translate. */

extern "C" uint8_t
ext_stress_solver_set_node_materials(ExtStressSolverHandle* handlePtr,
                                     const uint32_t* material_indices,
                                     uint32_t node_count)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver)
    {
        return 0;
    }

    const uint32_t graphCount = static_cast<uint32_t>(handle->graphNodeIndices.size());
    std::vector<uint32_t> graphMaterials(graphCount, 0U);
    if (material_indices)
    {
        const uint32_t count = std::min<uint32_t>(node_count, static_cast<uint32_t>(handle->inputToGraph.size()));
        for (uint32_t input = 0; input < count; ++input)
        {
            const uint32_t graphIndex = handle->inputToGraph[input];
            if (graphIndex < graphCount)
            {
                graphMaterials[graphIndex] = material_indices[input];
            }
        }
    }
    handle->solver->setNodeMaterials(graphMaterials.data(), graphCount);
    return 1;
}

extern "C" uint8_t
ext_stress_solver_set_node_strain_rates(ExtStressSolverHandle* handlePtr,
                                        const float* strain_rates,
                                        uint32_t node_count,
                                        float delta_time)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver)
    {
        return 0;
    }

    const uint32_t graphCount = static_cast<uint32_t>(handle->graphNodeIndices.size());
    handle->nodeScratch.assign(graphCount, 0.0f);
    if (strain_rates)
    {
        const uint32_t count = std::min<uint32_t>(node_count, static_cast<uint32_t>(handle->inputToGraph.size()));
        for (uint32_t input = 0; input < count; ++input)
        {
            const uint32_t graphIndex = handle->inputToGraph[input];
            if (graphIndex < graphCount)
            {
                handle->nodeScratch[graphIndex] = strain_rates[input];
            }
        }
    }
    handle->solver->setNodeStrainRates(handle->nodeScratch.data(), graphCount, delta_time);
    return 1;
}

extern "C" uint32_t
ext_stress_solver_get_node_crush_damage(const ExtStressSolverHandle* handlePtr,
                                        float* out_damage,
                                        uint32_t capacity)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver || !out_damage)
    {
        return 0U;
    }

    const uint32_t graphCount = static_cast<uint32_t>(handle->graphNodeIndices.size());
    std::vector<float> graphDamage(graphCount, 0.0f);
    handle->solver->getNodeCrushDamage(graphDamage.data(), graphCount);

    const uint32_t count = std::min<uint32_t>(capacity, static_cast<uint32_t>(handle->inputToGraph.size()));
    for (uint32_t input = 0; input < count; ++input)
    {
        const uint32_t graphIndex = handle->inputToGraph[input];
        out_damage[input] = graphIndex < graphCount ? graphDamage[graphIndex] : 0.0f;
    }
    return count;
}

extern "C" uint32_t
ext_stress_solver_get_node_stress_invariants(const ExtStressSolverHandle* handlePtr,
                                             float* out_pressure,
                                             float* out_deviator,
                                             uint32_t capacity)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver || (!out_pressure && !out_deviator))
    {
        return 0U;
    }

    const uint32_t graphCount = static_cast<uint32_t>(handle->graphNodeIndices.size());
    std::vector<float> pressure(graphCount, 0.0f);
    std::vector<float> deviator(graphCount, 0.0f);
    handle->solver->getNodeStressInvariants(pressure.data(), deviator.data(), graphCount);

    const uint32_t count = std::min<uint32_t>(capacity, static_cast<uint32_t>(handle->inputToGraph.size()));
    for (uint32_t input = 0; input < count; ++input)
    {
        const uint32_t graphIndex = handle->inputToGraph[input];
        const bool valid = graphIndex < graphCount;
        if (out_pressure) out_pressure[input] = valid ? pressure[graphIndex] : 0.0f;
        if (out_deviator) out_deviator[input] = valid ? deviator[graphIndex] : 0.0f;
    }
    return count;
}

extern "C" uint32_t
ext_stress_solver_get_node_crush_utilisation(const ExtStressSolverHandle* handlePtr,
                                             float* out_utilisation,
                                             uint32_t capacity)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver || !out_utilisation)
    {
        return 0U;
    }

    const uint32_t graphCount = static_cast<uint32_t>(handle->graphNodeIndices.size());
    std::vector<float> graphUtilisation(graphCount, 0.0f);
    handle->solver->getNodeCrushUtilisation(graphUtilisation.data(), graphCount);

    const uint32_t count = std::min<uint32_t>(capacity, static_cast<uint32_t>(handle->inputToGraph.size()));
    for (uint32_t input = 0; input < count; ++input)
    {
        const uint32_t graphIndex = handle->inputToGraph[input];
        out_utilisation[input] = graphIndex < graphCount ? graphUtilisation[graphIndex] : 0.0f;
    }
    return count;
}

extern "C" uint32_t
ext_stress_solver_get_crushed_nodes(ExtStressSolverHandle* handlePtr,
                                    uint32_t* out_node_indices,
                                    uint32_t capacity)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver || !out_node_indices || capacity == 0U)
    {
        return 0U;
    }

    handle->crushScratch.resize(capacity);
    const uint32_t drained = handle->solver->getCrushedNodes(handle->crushScratch.data(), capacity);

    uint32_t written = 0;
    for (uint32_t i = 0; i < drained; ++i)
    {
        const uint32_t input = mapGraphNodeToInput(*handle, handle->crushScratch[i]);
        if (input != UINT32_MAX)
        {
            out_node_indices[written++] = input;
        }
    }
    return written;
}

extern "C" uint8_t
ext_stress_solver_retire_crushed_node(ExtStressSolverHandle* handlePtr, uint32_t node_index)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver)
    {
        return 0U;
    }

    ensureActorIndex(*handle);
    if (node_index >= handle->inputNodeToActorSlot.size())
    {
        return 0U;
    }
    const uint32_t slot = handle->inputNodeToActorSlot[node_index];
    if (slot >= handle->actors.size())
    {
        return 0U;
    }

    auto& entry = handle->actors[slot];
    if (!entry.actor || entry.inputNodes.size() != 1U || entry.inputNodes[0] != node_index)
    {
        // Still load-bearing for other chunks; retiring it would delete them too.
        return 0U;
    }

    handle->solver->notifyActorDestroyed(*entry.actor);
    NvBlastActorDeactivate(entry.actor, kLogFn);
    handle->actors.erase(handle->actors.begin() + static_cast<std::ptrdiff_t>(slot));
    handle->actorIndexDirty = true;
    return 1U;
}

extern "C" uint8_t
ext_stress_solver_is_crush_enabled(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver && handle->solver->isCrushEnabled()) ? 1U : 0U;
}

extern "C" uint32_t ext_stress_abi_version(void)
{
    return EXT_STRESS_ABI_VERSION;
}

extern "C" uint32_t ext_stress_sizeof_material_desc(void)
{
    return static_cast<uint32_t>(sizeof(ExtStressMaterialDesc));
}

extern "C" uint32_t
ext_stress_solver_get_bond_stresses(const ExtStressSolverHandle* handlePtr,
                                    float* out_compression,
                                    float* out_tension,
                                    float* out_shear,
                                    uint32_t capacity)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver)
    {
        return 0U;
    }
    return handle->solver->getBondStresses(
        out_compression, out_tension, out_shear, capacity);
}

extern "C" uint32_t
ext_stress_solver_get_bond_healths(const ExtStressSolverHandle* handlePtr,
                                   float* out_health,
                                   uint32_t capacity)
{
    // Live bond health, indexed by ASSET bond index -- the same indexing as
    // NvBlastAssetGetBonds and as ext_stress_solver_get_bond_stresses.
    //
    // This exists because the fracture command stream is a *damage* stream, not
    // a break stream: generateStressDamage issues a command every tick a bond
    // is overstressed while its health is still above zero, and the command's
    // `health` field is the damage applied, not what remains. A caller that
    // treats each command as a break overcounts badly -- measured 1067
    // "breaks" against a 546-bond tower. Health crossing zero is the break.
    auto* handle = const_cast<ExtStressSolverHandleImpl*>(
        reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr));
    if (!handle || !handle->family || !out_health || capacity == 0U)
    {
        return 0U;
    }

    // The health array is per-family, so any live actor exposes the whole of
    // it. Actors are searched rather than assuming slot 0 is alive: slots are
    // vacated as actors are destroyed.
    const float* healths = nullptr;
    for (const auto& entry : handle->actors)
    {
        if (entry.actor)
        {
            healths = NvBlastActorGetBondHealths(entry.actor, logLL);
            if (healths)
            {
                break;
            }
        }
    }
    if (!healths)
    {
        return 0U;
    }

    const uint32_t bondCount = handle->asset ? NvBlastAssetGetBondCount(handle->asset, logLL) : 0U;
    const uint32_t n = bondCount < capacity ? bondCount : capacity;
    for (uint32_t i = 0; i < n; ++i)
    {
        out_health[i] = healths[i];
    }
    return n;
}

extern "C" uint32_t
ext_stress_solver_fill_debug_render(const ExtStressSolverHandle* handlePtr,
                                    uint32_t mode,
                                    float scale,
                                    ExtStressDebugLine* out_lines,
                                    uint32_t max_lines)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver || !out_lines || max_lines == 0U)
    {
        return 0U;
    }

    const ExtStressSolver::DebugBuffer buffer = handle->solver->fillDebugRender(
        handle->graphNodeIndices.data(),
        static_cast<uint32_t>(handle->graphNodeIndices.size()),
        static_cast<ExtStressSolver::DebugRenderMode>(mode),
        scale);

    const uint32_t count = std::min(buffer.lineCount, max_lines);
    for (uint32_t i = 0; i < count; ++i)
    {
        const ExtStressSolver::DebugLine& line = buffer.lines[i];
        out_lines[i].p0 = toStressVec3(line.pos0);
        out_lines[i].p1 = toStressVec3(line.pos1);
        out_lines[i].color0 = line.color0;
        out_lines[i].color1 = line.color1;
    }

    return count;
}

namespace
{

} // namespace

extern "C" uint8_t
ext_stress_solver_generate_fracture_commands(const ExtStressSolverHandle* handlePtr,
                                             ExtStressFractureCommands* out_commands,
                                             ExtStressBondFracture* bond_buffer,
                                             uint32_t max_bonds)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver || !out_commands || !bond_buffer || max_bonds == 0U)
    {
        if (out_commands)
        {
            out_commands->bondFractures = nullptr;
            out_commands->bondFractureCount = 0U;
            out_commands->actorIndex = UINT32_MAX;
        }
        return 0U;
    }

    const uint32_t totalActors = static_cast<uint32_t>(handle->actors.size());
    if (totalActors == 0U)
    {
        out_commands->bondFractures = nullptr;
        out_commands->bondFractureCount = 0U;
        out_commands->actorIndex = UINT32_MAX;
        return 1U;
    }

    std::vector<NvBlastFractureBuffers> buffers(totalActors);
    std::vector<const NvBlastActor*> llActors(totalActors, nullptr);
    const uint32_t generated = handle->solver->generateFractureCommandsPerActor(llActors.data(), buffers.data(), totalActors);

    if (generated == 0U)
    {
        out_commands->bondFractures = nullptr;
        out_commands->bondFractureCount = 0U;
        out_commands->actorIndex = UINT32_MAX;
        return 1U;
    }

    uint32_t best = 0U;
    for (uint32_t i = 1; i < generated; ++i)
    {
        if (buffers[i].bondFractureCount > buffers[best].bondFractureCount)
        {
            best = i;
        }
    }

    const NvBlastFractureBuffers& commands = buffers[best];
    const auto* entry = findActorByPointer(*handle, llActors[best]);
    const uint32_t actorIndex = entry ? entry->actorIndex : UINT32_MAX;

    const uint32_t available = commands.bondFractureCount;
    const uint32_t toCopy = std::min(available, max_bonds);
    if (toCopy == 0U)
    {
        out_commands->bondFractures = nullptr;
        out_commands->bondFractureCount = 0U;
        out_commands->actorIndex = actorIndex;
        return 1U;
    }

    const NvBlastBondFractureData* src = commands.bondFractures;
    if (!src)
    {
        out_commands->bondFractures = nullptr;
        out_commands->bondFractureCount = 0U;
        out_commands->actorIndex = actorIndex;
        return 1U;
    }

    for (uint32_t i = 0; i < toCopy; ++i)
    {
        const NvBlastBondFractureData& fracture = src[i];
        ExtStressBondFracture converted{};
        converted.userdata = fracture.userdata;
        converted.nodeIndex0 = mapGraphNodeToInput(*handle, fracture.nodeIndex0);
        converted.nodeIndex1 = mapGraphNodeToInput(*handle, fracture.nodeIndex1);
        converted.health = fracture.health;
        bond_buffer[i] = converted;
    }

    out_commands->bondFractures = bond_buffer;
    out_commands->bondFractureCount = toCopy;
    out_commands->actorIndex = actorIndex;
    return available <= max_bonds ? 1U : 2U; // 2 indicates truncation
}

extern "C" uint32_t
ext_stress_solver_actor_count(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return handle ? static_cast<uint32_t>(handle->actors.size()) : 0U;
}

extern "C" uint8_t
ext_stress_solver_get_excess_forces(const ExtStressSolverHandle* handlePtr,
                                    uint32_t actor_index,
                                    const StressVec3* center_of_mass,
                                    StressVec3* out_force,
                                    StressVec3* out_torque)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver || !out_force || !out_torque)
    {
        return 0U;
    }

    const NvcVec3 com = center_of_mass ? toNvcVec3(*center_of_mass) : NvcVec3{0.0f, 0.0f, 0.0f};
    NvcVec3 force{};
    NvcVec3 torque{};
    if (handle->solver->getExcessForces(actor_index, com, force, torque))
    {
        *out_force = toStressVec3(force);
        *out_torque = toStressVec3(torque);
        return 1U;
    }

    return 0U;
}

extern "C" float
ext_stress_solver_get_linear_error(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getStressErrorLinear() : 0.0f;
}

extern "C" float
ext_stress_solver_get_angular_error(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getStressErrorAngular() : 0.0f;
}

extern "C" uint8_t
ext_stress_solver_converged(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver && handle->solver->converged()) ? 1U : 0U;
}

extern "C" uint32_t
ext_stress_solver_island_count(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getIslandCount() : 0U;
}

extern "C" void
ext_stress_solver_set_island_aware(ExtStressSolverHandle* handlePtr, uint8_t enabled)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (handle && handle->solver)
    {
        handle->solver->setIslandAware(enabled != 0);
    }
}

extern "C" uint8_t
ext_stress_solver_get_island_aware(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver && handle->solver->getIslandAware()) ? 1U : 0U;
}

extern "C" void
ext_stress_solver_set_skip_settled(ExtStressSolverHandle* handlePtr, uint8_t enabled)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (handle && handle->solver)
    {
        handle->solver->setSkipSettled(enabled != 0);
    }
}

extern "C" void
ext_stress_solver_set_skip_stable_unconverged(ExtStressSolverHandle* handlePtr, uint8_t enabled)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (handle && handle->solver)
    {
        handle->solver->setSkipStableUnconverged(enabled != 0);
    }
}

extern "C" uint8_t
ext_stress_solver_get_skip_settled(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver && handle->solver->getSkipSettled()) ? 1U : 0U;
}

extern "C" uint32_t
ext_stress_solver_islands_skipped(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getIslandsSkipped() : 0U;
}

extern "C" uint32_t
ext_stress_solver_islands_total(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getIslandsTotal() : 0U;
}

extern "C" uint8_t
ext_stress_solver_set_gpu_accelerated(ExtStressSolverHandle* handlePtr, uint8_t enabled)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver
            && handle->solver->setGpuAccelerated(enabled != 0))
        ? 1U
        : 0U;
}

extern "C" void
ext_stress_solver_set_gpu_cuda_context(ExtStressSolverHandle* handlePtr, void* cudaContext)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (handle && handle->solver)
    {
        handle->solver->setGpuCudaContext(cudaContext);
    }
}

extern "C" void
ext_stress_solver_set_gpu_minimum_bond_count(
    ExtStressSolverHandle* handlePtr,
    uint32_t bondCount)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (handle && handle->solver)
    {
        handle->solver->setGpuMinimumBondCount(bondCount);
    }
}

extern "C" uint8_t
ext_stress_solver_get_gpu_accelerated(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver && handle->solver->getGpuAccelerated()) ? 1U : 0U;
}

extern "C" float
ext_stress_solver_gpu_solve_milliseconds(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getGpuSolveMilliseconds() : 0.0f;
}

extern "C" float
ext_stress_solver_impulse_copy_milliseconds(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getGpuImpulseCopyMilliseconds() : 0.0f;
}

extern "C" uint64_t
ext_stress_solver_bond_stress_groups_skipped(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getBondStressGroupsSkipped() : 0U;
}

extern "C" uint64_t
ext_stress_solver_bond_stress_parallel_checks(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getBondStressParallelChecks() : 0U;
}

extern "C" uint64_t
ext_stress_solver_bond_stress_parallel_mismatches(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getBondStressParallelMismatches() : 0U;
}

extern "C" float
ext_stress_solver_host_walk_in_milliseconds(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getHostWalkInMilliseconds() : 0.0f;
}

extern "C" float
ext_stress_solver_host_reset_milliseconds(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getHostResetMilliseconds() : 0.0f;
}

extern "C" float
ext_stress_solver_host_bond_stress_milliseconds(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getHostBondStressMilliseconds() : 0.0f;
}

extern "C" float
ext_stress_solver_host_node_stress_milliseconds(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getHostNodeStressMilliseconds() : 0.0f;
}

extern "C" float
ext_stress_solver_graph_solve_milliseconds(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getGraphSolveMilliseconds() : 0.0f;
}

extern "C" float
ext_stress_solver_initialize_milliseconds(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getInitializeMilliseconds() : 0.0f;
}

extern "C" float
ext_stress_solver_calc_error_milliseconds(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getCalcErrorMilliseconds() : 0.0f;
}

extern "C" float
ext_stress_solver_gpu_host_work_milliseconds(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getGpuHostWorkMilliseconds() : 0.0f;
}

extern "C" float
ext_stress_solver_gpu_host_blocked_milliseconds(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getGpuHostBlockedMilliseconds() : 0.0f;
}

extern "C" uint64_t
ext_stress_solver_gpu_host_to_device_bytes(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getGpuHostToDeviceBytes() : 0U;
}

extern "C" uint64_t
ext_stress_solver_gpu_device_to_host_bytes(const ExtStressSolverHandle* handlePtr)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    return (handle && handle->solver) ? handle->solver->getGpuDeviceToHostBytes() : 0U;
}

extern "C" uint32_t
ext_stress_sizeof_ext_node_desc()
{
    return static_cast<uint32_t>(sizeof(ExtStressNodeDesc));
}

extern "C" uint32_t
ext_stress_sizeof_ext_bond_desc()
{
    return static_cast<uint32_t>(sizeof(ExtStressBondDesc));
}

extern "C" uint32_t
ext_stress_sizeof_ext_settings()
{
    return static_cast<uint32_t>(sizeof(ExtStressSolverSettingsDesc));
}

extern "C" uint32_t
ext_stress_sizeof_ext_debug_line()
{
    return static_cast<uint32_t>(sizeof(ExtStressDebugLine));
}

extern "C" uint32_t
ext_stress_sizeof_ext_bond_fracture()
{
    return static_cast<uint32_t>(sizeof(ExtStressBondFracture));
}

extern "C" uint32_t
ext_stress_sizeof_ext_fracture_commands()
{
    return static_cast<uint32_t>(sizeof(ExtStressFractureCommands));
}

extern "C" uint8_t ext_stress_solver_collect_actors(const ExtStressSolverHandle* handlePtr,
                                                     ExtStressActor* actor_buffer,
                                                     uint32_t actor_capacity,
                                                     uint32_t* nodes_buffer,
                                                     uint32_t nodes_capacity,
                                                     uint32_t* out_actor_count,
                                                     uint32_t* out_node_count)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !actor_buffer || actor_capacity == 0U)
    {
        if (out_actor_count)
        {
            *out_actor_count = 0;
        }
        if (out_node_count)
        {
            *out_node_count = 0;
        }
        return 0U;
    }

    const uint32_t totalActors = static_cast<uint32_t>(handle->actors.size());
    uint32_t copiedActors = 0;
    uint32_t copiedNodes = 0;

    for (uint32_t i = 0; i < totalActors && copiedActors < actor_capacity; ++i)
    {
        const auto& entry = handle->actors[i];
        ExtStressActor actor{};
        actor.actorIndex = entry.actorIndex;
        actor.nodeCount = static_cast<uint32_t>(entry.inputNodes.size());
        actor.nodes = nullptr;

        if (nodes_buffer && copiedNodes + actor.nodeCount <= nodes_capacity)
        {
            std::memcpy(nodes_buffer + copiedNodes,
                        entry.inputNodes.data(),
                        actor.nodeCount * sizeof(uint32_t));
            actor.nodes = nodes_buffer + copiedNodes;
            copiedNodes += actor.nodeCount;
        }

        actor_buffer[copiedActors++] = actor;
    }

    if (out_actor_count)
    {
        *out_actor_count = copiedActors;
    }
    if (out_node_count)
    {
        *out_node_count = copiedNodes;
    }

    const bool fullyCopiedActors = copiedActors == totalActors;
    const bool fullyCopiedNodes = (!nodes_buffer || copiedNodes <= nodes_capacity);
    return (fullyCopiedActors && fullyCopiedNodes) ? 1U : 2U;
}

extern "C" uint8_t ext_stress_solver_generate_fracture_commands_per_actor(const ExtStressSolverHandle* handlePtr,
                                                                           ExtStressFractureCommands* command_buffer,
                                                                           uint32_t command_capacity,
                                                                           ExtStressBondFracture* bond_buffer,
                                                                           uint32_t bond_capacity,
                                                                           uint32_t* out_command_count,
                                                                           uint32_t* out_bond_count)
{
    const auto* handle = reinterpret_cast<const ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver || !command_buffer || command_capacity == 0U || !bond_buffer || bond_capacity == 0U)
    {
        if (out_command_count)
        {
            *out_command_count = 0;
        }
        if (out_bond_count)
        {
            *out_bond_count = 0;
        }
        return 0U;
    }

    const uint32_t totalActors = static_cast<uint32_t>(handle->actors.size());
    uint32_t commandCount = 0;
    uint32_t bondOffset = 0;

    std::vector<NvBlastFractureBuffers> buffers(totalActors);
    std::vector<const NvBlastActor*> llActors(totalActors);

    for (uint32_t i = 0; i < totalActors; ++i)
    {
        const auto& entry = handle->actors[i];
        llActors[i] = entry.actor;
        buffers[i] = NvBlastFractureBuffers{};
    }

    // The solver fills the first `generated` entries and also overwrites llActors[]
    // so that buffers[i] corresponds to llActors[i].
    const uint32_t generated = handle->solver->generateFractureCommandsPerActor(llActors.data(), buffers.data(), totalActors);

    // Non-const: the chunk fracture cache is rebuilt from this generate pass.
    auto* mutableHandle = const_cast<ExtStressSolverHandleImpl*>(handle);
    mutableHandle->pendingChunkFractures.clear();

    // E6: one O(bodies) map build per CALL replaces one O(bodies) linear scan
    // per generated actor. generate mutates no actor entries, so pointers
    // stay valid for the whole loop.
    std::unordered_map<const NvBlastActor*, const ExtStressSolverHandleImpl::ActorEntry*> actorsByPointer;
    if (applyIndexIncremental())
    {
        actorsByPointer.reserve(handle->actors.size());
        for (const auto& actorEntry : handle->actors)
        {
            actorsByPointer.emplace(actorEntry.actor, &actorEntry);
        }
    }

    for (uint32_t i = 0; i < generated && commandCount < command_capacity; ++i)
    {
        const NvBlastFractureBuffers& buffer = buffers[i];
        const ExtStressSolverHandleImpl::ActorEntry* entry;
        if (applyIndexIncremental())
        {
            const auto found = actorsByPointer.find(llActors[i]);
            entry = found == actorsByPointer.end() ? nullptr : found->second;
        }
        else
        {
            entry = findActorByPointer(*handle, llActors[i]);
        }
        const uint32_t actorIndex = entry ? entry->actorIndex : UINT32_MAX;

        const uint32_t bondCount = buffer.bondFractureCount;
        const uint32_t chunkCount = buffer.chunkFractureCount;
        // A crush-only actor has no bond fractures but is still real work.
        if (bondCount == 0 && chunkCount == 0)
        {
            continue;
        }

        if (chunkCount > 0 && buffer.chunkFractures)
        {
            std::vector<NvBlastChunkFractureData> chunks(
                buffer.chunkFractures, buffer.chunkFractures + chunkCount);
            mutableHandle->pendingChunkFractures.emplace_back(actorIndex, std::move(chunks));
        }

        if (bondOffset + bondCount > bond_capacity)
        {
            if (out_command_count)
            {
                *out_command_count = commandCount;
            }
            if (out_bond_count)
            {
                *out_bond_count = bondOffset;
            }
            return 2U;
        }

        for (uint32_t b = 0; b < bondCount; ++b)
        {
            const NvBlastBondFractureData& src = buffer.bondFractures[b];
            ExtStressBondFracture dst{};
            dst.userdata = src.userdata;
            dst.nodeIndex0 = mapGraphNodeToInput(*handle, src.nodeIndex0);
            dst.nodeIndex1 = mapGraphNodeToInput(*handle, src.nodeIndex1);
            dst.health = src.health;
            bond_buffer[bondOffset + b] = dst;
        }

        ExtStressFractureCommands cmd{};
        cmd.actorIndex = actorIndex;
        cmd.bondFractures = bondCount > 0 ? bond_buffer + bondOffset : nullptr;
        cmd.bondFractureCount = bondCount;
        cmd.chunkFractureCount = chunkCount;

        command_buffer[commandCount++] = cmd;
        bondOffset += bondCount;
    }

    if (out_command_count)
    {
        *out_command_count = commandCount;
    }
    if (out_bond_count)
    {
        *out_bond_count = bondOffset;
    }

    return (commandCount == generated) ? 1U : 2U;
}

extern "C" uint8_t ext_stress_solver_apply_fracture_commands(ExtStressSolverHandle* handlePtr,
                                                              const ExtStressFractureCommands* command_buffer,
                                                              uint32_t command_count,
                                                              ExtStressSplitEvent* events_buffer,
                                                              uint32_t event_capacity,
                                                              ExtStressActor* child_buffer,
                                                              uint32_t child_capacity,
                                                              uint32_t* out_event_count,
                                                              uint32_t* out_child_count,
                                                              uint32_t* nodes_buffer,
                                                              uint32_t nodes_capacity,
                                                              uint32_t* out_node_count)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver || !command_buffer || command_count == 0U)
    {
        if (out_event_count) { *out_event_count = 0; }
        if (out_child_count) { *out_child_count = 0; }
        if (out_node_count) { *out_node_count = 0; }
        return 0U;
    }

    // This call retires parent actor entries and pushes child entries below.
    // E6 (incremental): build the index ONCE here, keep it correct in place
    // at each mutation, and compact + dirty once at the end — instead of a
    // full O(total nodes) rebuild on the first lookup after every mutation,
    // which made a 400-split tick quadratic.
    if (applyIndexIncremental())
    {
        handle->actorIndexDirty = true;
        ensureActorIndex(*handle);
    }
    else
    {
        handle->actorIndexDirty = true;
    }
    bool anyTombstone = false;

    uint32_t storedEvents = 0;
    uint32_t storedChildren = 0;
    uint32_t storedNodes = 0;
    bool truncated = false;

    for (uint32_t commandIndex = 0; commandIndex < command_count; ++commandIndex)
    {
        const ExtStressFractureCommands& command = command_buffer[commandIndex];
        const ExtStressBondFracture* fractures = command.bondFractures;
        const uint32_t fractureCount = (fractures != nullptr) ? command.bondFractureCount : 0U;
        const uint32_t actorIndex = command.actorIndex;

        const std::vector<NvBlastChunkFractureData>* chunkFractures =
            command.chunkFractureCount > 0U ? handle->findPendingChunkFractures(actorIndex) : nullptr;

        if (fractureCount == 0U && chunkFractures == nullptr)
        {
            continue;
        }

        auto* actorEntry = findActorByIndex(*handle, actorIndex);
        if (!actorEntry || !actorEntry->actor)
        {
            continue;
        }

        const size_t entryIndex = static_cast<size_t>(actorEntry - handle->actors.data());

        handle->fractureScratch.resize(fractureCount);
        for (uint32_t i = 0; i < fractureCount; ++i)
        {
            const ExtStressBondFracture& src = fractures[i];
            NvBlastBondFractureData dst{};
            dst.userdata = src.userdata;
            dst.nodeIndex0 = mapInputNodeToGraph(*handle, src.nodeIndex0);
            dst.nodeIndex1 = mapInputNodeToGraph(*handle, src.nodeIndex1);
            dst.health = src.health;
            handle->fractureScratch[i] = dst;
        }

        NvBlastFractureBuffers buffers{};
        buffers.bondFractureCount = fractureCount;
        buffers.bondFractures = fractureCount > 0U ? handle->fractureScratch.data() : nullptr;

        // Applying a chunk fracture zeroes every bond incident on that chunk
        // and drops its node out of the island graph, which is exactly the
        // structural meaning of "this chunk is gone". The caller still has to
        // remove the corresponding body/shape from its own physics scene --
        // Blast keeps a health-exhausted leaf chunk alive as an inert actor.
        if (chunkFractures != nullptr)
        {
            handle->chunkFractureScratch = *chunkFractures;
            buffers.chunkFractureCount = static_cast<uint32_t>(handle->chunkFractureScratch.size());
            buffers.chunkFractures = handle->chunkFractureScratch.data();
        }

        NvBlastActorApplyFracture(nullptr, actorEntry->actor, &buffers, kLogFn, nullptr);

        if (!NvBlastActorIsSplitRequired(actorEntry->actor, kLogFn))
        {
            continue;
        }

        const size_t scratchSize = NvBlastActorGetRequiredScratchForSplit(actorEntry->actor, kLogFn);
        handle->splitScratch.resize(scratchSize);
        const uint32_t maxChildren = NvBlastActorGetMaxActorCountForSplit(actorEntry->actor, kLogFn);
        handle->splitActors.resize(maxChildren);

        NvBlastActorSplitEvent splitEvent{};
        splitEvent.deletedActor = actorEntry->actor;
        splitEvent.newActors = handle->splitActors.data();

        const uint32_t created = NvBlastActorSplit(&splitEvent,
                                                   actorEntry->actor,
                                                   maxChildren,
                                                   handle->splitScratch.data(),
                                                   kLogFn,
                                                   nullptr);

        if (created == 0U)
        {
            continue;
        }

        handle->solver->notifyActorDestroyed(*actorEntry->actor);

        if (entryIndex < handle->actors.size())
        {
            if (applyIndexIncremental())
            {
                // E6: tombstone instead of erase — no slot shifts, so the
                // index stays valid without a rebuild. The parent's
                // actorIndex mapping is cleared NOW, before its children
                // register (a child may legitimately reuse the family
                // index). Node lists freed here; nothing reads a tombstone.
                // Survivor order is preserved by the stable compaction at
                // the end, so the final vector is byte-for-byte the sequence
                // the erase path produced.
                ExtStressSolverHandleImpl::ActorEntry& dead = handle->actors[entryIndex];
                if (dead.actorIndex < handle->actorIndexToSlot.size())
                {
                    handle->actorIndexToSlot[dead.actorIndex] = UINT32_MAX;
                }
                dead.actor = nullptr;
                dead.actorIndex = UINT32_MAX;
                dead.graphNodes.clear();
                dead.inputNodes.clear();
                anyTombstone = true;
            }
            else
            {
                handle->actors.erase(handle->actors.begin() + static_cast<std::ptrdiff_t>(entryIndex));
                // Erasing shifts every later entry left. The actorIndex -> slot
                // table must be invalidated HERE, not just on push_back: a second
                // fracture command in this same call would otherwise resolve its
                // actor through stale slots and read a DIFFERENT actor's node
                // list -- which is exactly how promotions ended up pairing chunks
                // from buildings seventy metres apart. The old linear scan was
                // immune because it read the live vector; the indexed lookup is
                // only correct if every mutation marks it dirty.
                handle->actorIndexDirty = true;
            }
        }

        ExtStressSplitEvent* evt = nullptr;
        if (events_buffer && storedEvents < event_capacity)
        {
            evt = &events_buffer[storedEvents];
            evt->parentActorIndex = actorIndex;
            evt->childCount = 0U;
            evt->children = nullptr;
        }
        else
        {
            truncated = true;
        }

        for (uint32_t i = 0; i < created; ++i)
        {
            NvBlastActor* child = splitEvent.newActors[i];
            if (!child)
            {
                continue;
            }

            handle->solver->notifyActorCreated(*child);

            ExtStressSolverHandleImpl::ActorEntry entry;
            entry.actor = child;
            entry.actorIndex = NvBlastActorGetIndex(child, kLogFn);

            const uint32_t graphNodeCount = NvBlastActorGetGraphNodeCount(child, kLogFn);
            entry.graphNodes.resize(graphNodeCount);
            if (graphNodeCount > 0)
            {
                NvBlastActorGetGraphNodeIndices(entry.graphNodes.data(), graphNodeCount, child, kLogFn);
            }
            entry.inputNodes.resize(graphNodeCount);
            for (uint32_t n = 0; n < graphNodeCount; ++n)
            {
                entry.inputNodes[n] = mapGraphNodeToInput(*handle, entry.graphNodes[n]);
            }

            bool childStored = false;
            if (child_buffer && storedChildren < child_capacity)
            {
                ExtStressActor& childOut = child_buffer[storedChildren];
                childOut.actorIndex = entry.actorIndex;
                childOut.nodeCount = static_cast<uint32_t>(entry.inputNodes.size());
                childOut.nodes = nullptr;

                if (nodes_buffer && storedNodes + childOut.nodeCount <= nodes_capacity)
                {
                    std::memcpy(nodes_buffer + storedNodes,
                                entry.inputNodes.data(),
                                childOut.nodeCount * sizeof(uint32_t));
                    childOut.nodes = nodes_buffer + storedNodes;
                    storedNodes += childOut.nodeCount;
                }
                else if (childOut.nodeCount > 0)
                {
                    truncated = true;
                }

                if (evt)
                {
                    if (!evt->children)
                    {
                        evt->children = &child_buffer[storedChildren];
                    }
                    evt->childCount += 1U;
                }

                ++storedChildren;
                childStored = true;
            }
            else
            {
                truncated = true;
            }

            if (applyIndexIncremental())
            {
                // E6: register the child's mappings in place of the rebuild.
                const uint32_t slot = static_cast<uint32_t>(handle->actors.size());
                if (entry.actorIndex != UINT32_MAX)
                {
                    if (entry.actorIndex >= handle->actorIndexToSlot.size())
                    {
                        handle->actorIndexToSlot.resize(entry.actorIndex + 1, UINT32_MAX);
                    }
                    handle->actorIndexToSlot[entry.actorIndex] = slot;
                }
                for (uint32_t inputNode : entry.inputNodes)
                {
                    if (inputNode < handle->inputNodeToActorSlot.size())
                    {
                        handle->inputNodeToActorSlot[inputNode] = slot;
                    }
                }
                handle->actors.push_back(std::move(entry));
            }
            else
            {
                handle->actors.push_back(std::move(entry));
                handle->actorIndexDirty = true;
            }
        }

        if (evt)
        {
            ++storedEvents;
        }
    }

    if (applyIndexIncremental() && anyTombstone)
    {
        // E6: stable compaction — one O(bodies) pass per CALL, in place of an
        // O(bodies) memmove per split. Relative survivor order and the
        // appended-children tail are exactly what erase-in-place produced.
        // Slots change here, so the index goes dirty ONCE; the next lookup
        // (first force injection of the next tick) rebuilds it, which is the
        // same steady-state the old path had after any apply.
        std::size_t write = 0;
        for (std::size_t read = 0; read < handle->actors.size(); ++read)
        {
            if (handle->actors[read].actor != nullptr)
            {
                if (write != read)
                {
                    handle->actors[write] = std::move(handle->actors[read]);
                }
                ++write;
            }
        }
        handle->actors.resize(write);
        handle->actorIndexDirty = true;
    }

    if (out_event_count)
    {
        *out_event_count = storedEvents;
    }
    if (out_child_count)
    {
        *out_child_count = storedChildren;
    }
    if (out_node_count)
    {
        *out_node_count = storedNodes;
    }

    return truncated ? 2U : 1U;
}

