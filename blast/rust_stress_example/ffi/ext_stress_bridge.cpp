#include "ext_stress_bridge.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
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
    bool actorIndexDirty{true};
    std::vector<uint8_t> splitScratch;
    std::vector<NvBlastActor*> splitActors;
    std::vector<NvBlastBondFractureData> fractureScratch;

    // Static splash adjacency (CSR) for solver-side splash expansion
    // (ext_stress_solver_add_splash_hits): hit node n's neighbours are entries
    // [splashStarts[n], splashStarts[n+1]) of splashNeighbors / splashWeights /
    // splashNeighborPos. Weights stay double and each neighbour force component is
    // computed as float(double(f32 force) * double weight) — the exact rounding
    // sequence of the JS expansion loop this replaces, so both paths are
    // byte-identical. Neighbour application points are the authored node offsets,
    // pre-rounded to f32 by the caller (same single rounding as the JS path).
    std::vector<uint32_t> splashStarts;
    std::vector<uint32_t> splashNeighbors;
    std::vector<double> splashWeights;
    std::vector<float> splashNeighborPos;
    // 1 = node excluded from splash. Mirrors the JS `chunk.active` filter, which can go
    // false (node destroyed) before the actor split that isolates the node is applied.
    std::vector<uint8_t> splashNodeInactive;
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
    settings.compressionElasticLimit = settingsDesc->compression_elastic_limit;
    settings.compressionFatalLimit = settingsDesc->compression_fatal_limit;
    settings.tensionElasticLimit = settingsDesc->tension_elastic_limit;
    settings.tensionFatalLimit = settingsDesc->tension_fatal_limit;
    settings.shearElasticLimit = settingsDesc->shear_elastic_limit;
    settings.shearFatalLimit = settingsDesc->shear_fatal_limit;
    return settings;
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

ExtStressSolverHandleImpl::ActorEntry* findActorByIndex(ExtStressSolverHandleImpl& handle, uint32_t actorIndex)
{
    for (auto& entry : handle.actors)
    {
        if (entry.actorIndex == actorIndex)
        {
            return &entry;
        }
    }
    return nullptr;
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
    for (uint32_t slot = 0; slot < handle.actors.size(); ++slot)
    {
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
                        const ExtStressSolverSettingsDesc* settingsDesc)
{
    if (!nodes || node_count == 0U || !bonds || bond_count == 0U)
    {
        return nullptr;
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
    actorDesc.uniformInitialBondHealth = 1.0f;
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
// entry points so their behaviour can never diverge. Mirrors the original
// ext_stress_solver_add_force logic exactly: prefer the actor that owns the
// input node (so the force lands in the right family after splits); otherwise
// fall back to the graph-node index.
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

    if (auto* entry = findActorOwningInputNode(handle, node_index))
    {
        if (entry->actor)
        {
            // The actor overload of addForce exists to translate an arbitrary position
            // into its nearest graph node with an O(actor nodes) scan — but every caller
            // here applies the force AT a node's own centroid, so the nearest node IS
            // `node_index`. When the input→graph mapping is 1:1 (graphReductionLevel 0 —
            // the runtime's pinned setting) we can skip the scan and accumulate on the
            // node directly. Two behaviours of the scan path are preserved exactly:
            // single-node actors drop the force (the stress solve ignores them), and a
            // merged/reduced graph (mapping not 1:1) falls back to the scan, where the
            // nearest merged node is not guaranteed to be this node's group.
            if (handle.graphNodeIndices.size() == handle.inputToGraph.size()
                && graphIndex != UINT32_MAX)
            {
                if (NvBlastActorGetGraphNodeCount(entry->actor, kLogFn) > 1)
                {
                    handle.solver->addForce(graphIndex, force, mode);
                }
                return;
            }
            handle.solver->addForce(*entry->actor, pos, force, mode);
            return;
        }
    }

    if (graphIndex != UINT32_MAX)
    {
        handle.solver->addForce(graphIndex, force, mode);
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

// ── Solver-side splash expansion ─────────────────────────────────────────────
// The JS contact-injection pipeline used to expand each contact hit into its
// precomputed splash neighbours in JS (one buffered force per neighbour), then
// submit the whole batch. That expansion loop was the costliest part of contact
// injection at scale (~3.4 ms mean during impacts on bench:large). These entry
// points move the expansion across the FFI boundary: the caller ships the static
// adjacency once, then per frame submits only the resolved HITS; the solver
// expands hit → neighbours internally in the same (contact, neighbour) order and
// with the same rounding sequence as the JS loop, so accumulated per-node forces
// are byte-identical.

extern "C" uint32_t
ext_stress_solver_set_splash_adjacency(ExtStressSolverHandle* handlePtr,
                                       const uint32_t* starts,
                                       const uint32_t* neighbors,
                                       const double* weights,
                                       const float* neighbor_positions,
                                       uint32_t node_count,
                                       uint32_t entry_count)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !starts || node_count == 0U)
    {
        return 0U;
    }
    if (entry_count > 0U && (!neighbors || !weights || !neighbor_positions))
    {
        return 0U;
    }
    handle->splashStarts.assign(starts, starts + static_cast<size_t>(node_count) + 1U);
    handle->splashNeighbors.assign(neighbors, neighbors + entry_count);
    handle->splashWeights.assign(weights, weights + entry_count);
    handle->splashNeighborPos.assign(neighbor_positions,
                                     neighbor_positions + static_cast<size_t>(entry_count) * 3U);
    handle->splashNodeInactive.assign(node_count, 0U);
    return entry_count;
}

extern "C" void
ext_stress_solver_set_splash_node_inactive(ExtStressSolverHandle* handlePtr, uint32_t node_index)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle)
    {
        return;
    }
    if (node_index < handle->splashNodeInactive.size())
    {
        handle->splashNodeInactive[node_index] = 1U;
    }
}

extern "C" uint32_t
ext_stress_solver_add_splash_hits(ExtStressSolverHandle* handlePtr,
                                  const uint32_t* hit_nodes,
                                  const float* hit_positions,
                                  const float* hit_forces,
                                  uint32_t count,
                                  uint32_t mode)
{
    auto* handle = reinterpret_cast<ExtStressSolverHandleImpl*>(handlePtr);
    if (!handle || !handle->solver || !hit_nodes || !hit_positions || !hit_forces || count == 0U)
    {
        return 0U;
    }
    if (handle->splashStarts.size() < 2U)
    {
        return 0U; // adjacency was never shipped — caller should use the JS path
    }

    const ExtForceMode::Enum forceMode = toForceMode(mode);
    const uint32_t nodeCount = static_cast<uint32_t>(handle->splashStarts.size() - 1U);
    uint32_t applied = 0U;

    for (uint32_t i = 0; i < count; ++i)
    {
        const uint32_t hit = hit_nodes[i];
        const float* hp = hit_positions + static_cast<size_t>(i) * 3U;
        const float* hf = hit_forces + static_cast<size_t>(i) * 3U;
        const NvcVec3 hitPos{hp[0], hp[1], hp[2]};
        const NvcVec3 hitForce{hf[0], hf[1], hf[2]};

        // Hit node first at full strength, then its splash neighbours — the same
        // per-contact order as the JS loop, so accumulated per-node forces match.
        applyForceToInputNode(*handle, hit, hitPos, hitForce, forceMode);
        ++applied;

        if (hit >= nodeCount)
        {
            continue;
        }
        // Same-body filter: the JS loop compares chunk bodyHandles; actors map 1:1 to
        // bodies, so owning-actor identity is the same predicate on the solver side.
        const auto* hitEntry = findActorOwningInputNode(*handle, hit);
        if (!hitEntry)
        {
            continue;
        }
        const uint32_t aStart = handle->splashStarts[hit];
        const uint32_t aEnd = handle->splashStarts[hit + 1U];
        for (uint32_t k = aStart; k < aEnd; ++k)
        {
            const uint32_t j = handle->splashNeighbors[k];
            if (j < handle->splashNodeInactive.size() && handle->splashNodeInactive[j] != 0U)
            {
                continue;
            }
            if (findActorOwningInputNode(*handle, j) != hitEntry)
            {
                continue;
            }
            const double w = handle->splashWeights[k];
            // float(double(f32) * double) — the exact rounding the JS expansion performs
            // (f32 stash force read back as f64, multiplied by the f64 weight, stored f32).
            const NvcVec3 f{static_cast<float>(hf[0] * w),
                            static_cast<float>(hf[1] * w),
                            static_cast<float>(hf[2] * w)};
            const float* np = handle->splashNeighborPos.data() + static_cast<size_t>(k) * 3U;
            const NvcVec3 pos{np[0], np[1], np[2]};
            // Shared primitive (same one the JS fallback's add_all_forces uses), so the
            // two splash paths can never diverge — including its direct-graph-node fast
            // path and single-node-actor drop.
            applyForceToInputNode(*handle, j, pos, f, forceMode);
            ++applied;
        }
    }
    return applied;
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

    for (uint32_t i = 0; i < generated && commandCount < command_capacity; ++i)
    {
        const NvBlastFractureBuffers& buffer = buffers[i];
        const auto* entry = findActorByPointer(*handle, llActors[i]);
        const uint32_t actorIndex = entry ? entry->actorIndex : UINT32_MAX;

        const uint32_t bondCount = buffer.bondFractureCount;
        if (bondCount == 0)
        {
            continue;
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
        cmd.bondFractures = bond_buffer + bondOffset;
        cmd.bondFractureCount = bondCount;

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

    // This call erases the parent actor entry and pushes child entries below, so
    // the input-node -> actor index must be rebuilt before the next lookup.
    handle->actorIndexDirty = true;

    uint32_t storedEvents = 0;
    uint32_t storedChildren = 0;
    uint32_t storedNodes = 0;
    bool truncated = false;

    for (uint32_t commandIndex = 0; commandIndex < command_count; ++commandIndex)
    {
        const ExtStressFractureCommands& command = command_buffer[commandIndex];
        const ExtStressBondFracture* fractures = command.bondFractures;
        const uint32_t fractureCount = command.bondFractureCount;
        const uint32_t actorIndex = command.actorIndex;

        if (!fractures || fractureCount == 0U)
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
        buffers.bondFractures = handle->fractureScratch.data();

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
            handle->actors.erase(handle->actors.begin() + static_cast<std::ptrdiff_t>(entryIndex));
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

            handle->actors.push_back(std::move(entry));
        }

        if (evt)
        {
            ++storedEvents;
        }
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

