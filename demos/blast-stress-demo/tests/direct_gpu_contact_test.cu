// A real sliding impact: verify the decoded contact wrench against momentum
// change, then consume the contact view directly in another CUDA stream and
// submit its loads to the stress solver sharing PhysX's context.
#include "../physx_scene.h"
#include "NvBlastExtStressPhysXDirectGpu.h"
#include "NvBlastExtStressGpu.h"
#include <extensions/PxCudaHelpersExt.h>
#include <cuda_runtime.h>
#include <PxContact.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace physx;
using namespace Nv::Blast;
using physx::Ext::PxCudaHelpersExt;
namespace Nv { namespace Blast {
bool launchDirectGpuContactDecode(const PxGpuContactPair*, const PxU32*, PxU32,
    ExtStressPhysXDirectGpuContact*, ExtStressPhysXDirectGpuContactStatus*, PxU32, CUstream);
}}
namespace {
void require(bool b, const char* message) { if (!b) throw std::runtime_error(message); }
void check(cudaError_t r) { require(r == cudaSuccess, cudaGetErrorString(r)); }
// Equal and opposite forces can carry zero net impulse and a nonzero couple.
// A single averaged contact would erase this load even if its force sum were exact.
void checkCouple(PxCudaContextManager& manager)
{
    PxScopedCudaLock lock(manager);
    PxContactPatch patches[2]{};
    PxContact points[2]{};
    PxFrictionPatch friction[2]{};
    float forces[2] = {1,1};
    for (unsigned i = 0; i < 2; ++i)
    {
        const float sign = i == 0 ? 1.0f : -1.0f;
        patches[i].normal = PxVec3(0,sign,0);
        patches[i].nbContacts = 1; patches[i].startContactIndex = i;
        points[i].contact = PxVec3(sign,0,0);
        friction[i].anchorCount = 1;
        friction[i].anchorPositions[0] = PxVec3(0,0,sign);
        friction[i].anchorImpulses[0] = PxVec3(sign,0,0);
    }
    PxContactPatch* dp; PxContact* dc; PxFrictionPatch* df; float* dj;
    PxGpuContactPair* pair; unsigned* count;
    ExtStressPhysXDirectGpuContact* output;
    ExtStressPhysXDirectGpuContactStatus* status;
    check(cudaMalloc(&dp,sizeof(patches))); check(cudaMalloc(&dc,sizeof(points)));
    check(cudaMalloc(&df,sizeof(friction))); check(cudaMalloc(&dj,sizeof(forces)));
    check(cudaMalloc(&pair,sizeof(*pair))); check(cudaMalloc(&count,sizeof(*count)));
    check(cudaMalloc(&output,4*sizeof(*output))); check(cudaMalloc(&status,sizeof(*status)));
    check(cudaMemcpy(dp,patches,sizeof(patches),cudaMemcpyHostToDevice));
    check(cudaMemcpy(dc,points,sizeof(points),cudaMemcpyHostToDevice));
    check(cudaMemcpy(df,friction,sizeof(friction),cudaMemcpyHostToDevice));
    check(cudaMemcpy(dj,forces,sizeof(forces),cudaMemcpyHostToDevice));
    PxGpuContactPair header = PxGpuContactPair();
    header.contactPatches = reinterpret_cast<PxU8*>(dp);
    header.contactPoints = reinterpret_cast<PxU8*>(dc);
    header.frictionPatches = reinterpret_cast<PxU8*>(df);
    header.contactForces = dj; header.nbContacts = 2; header.nbPatches = 2;
    const unsigned one = 1;
    check(cudaMemcpy(pair,&header,sizeof(header),cudaMemcpyHostToDevice));
    check(cudaMemcpy(count,&one,sizeof(one),cudaMemcpyHostToDevice));
    check(cudaMemset(status,0,sizeof(*status)));
    require(launchDirectGpuContactDecode(pair,count,1,output,status,4,nullptr), "couple decode failed");
    check(cudaStreamSynchronize(nullptr));
    ExtStressPhysXDirectGpuContact result[4]; ExtStressPhysXDirectGpuContactStatus resultStatus{};
    check(cudaMemcpy(result,output,sizeof(result),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&resultStatus,status,sizeof(resultStatus),cudaMemcpyDeviceToHost));
    require(resultStatus.count == 4 && !resultStatus.overflow, "couple contacts were lost");
    PxVec3 linear(0.0f), angular(0.0f);
    for (const auto& c : result) { linear += c.impulseOnActor0; angular += c.worldPosition.cross(c.impulseOnActor0); }
    require(linear.magnitude() < 1e-6f && (angular-PxVec3(0,2,2)).magnitude() < 1e-6f,
        "contact decoder erased the couple");
    check(cudaFree(dp)); check(cudaFree(dc)); check(cudaFree(df)); check(cudaFree(dj));
    check(cudaFree(pair)); check(cudaFree(count)); check(cudaFree(output)); check(cudaFree(status));
}

void step(PxScene& scene) { scene.simulate(1.0f/60.0f); require(scene.fetchResults(true), "fetch failed"); }
__global__ void contactLoads(ExtStressPhysXDirectGpuContactView view,
    PxRigidActor* body, ExtStressGpuImpulse* loads)
{
    // A tiny validation fixture uses one thread to make summation reproducible.
    // Production mapping needs shape/node ownership and an island load kernel.
    loads[0] = {}; loads[1] = {};
    if (view.status->overflow || view.status->count > view.capacity) { return; }
    PxVec3 impulse(0.0f);
    for (unsigned i = 0; i < view.status->count; ++i)
    {
        const auto& c = view.contacts[i];
        if (c.actor0 == body) impulse += c.impulseOnActor0;
        else if (c.actor1 == body) impulse -= c.impulseOnActor0;
    }
    loads[1].linear = {impulse.x, impulse.y, impulse.z};
}
}
int main()
{
    try
    {
        blast_demo::SceneCapacity capacity;
        capacity.maxBodies = 64; capacity.maxShapes = 64; capacity.maxContactPairs = 4096;
        blast_demo::PhysXScene context(blast_demo::PhysicsMode::Gpu, true, capacity, nullptr, true);
        auto& scene = context.scene();
        auto& cuda = *context.cudaContextManager();
        checkCouple(cuda);
        auto* body = context.physics().createRigidDynamic(PxTransform(PxVec3(0, 0.7f, 0)));
        auto* shape = context.physics().createShape(PxBoxGeometry(0.5f), context.material(), true);
        require(body && shape && body->attachShape(*shape), "body creation failed");
        shape->release();
        body->setMass(2.0f); body->setMassSpaceInertiaTensor(PxVec3(1.0f));
        body->setLinearDamping(0); body->setAngularDamping(0);
        body->setLinearVelocity(PxVec3(3,-2,0));
        scene.addActor(*body);
        auto* indices = PxCudaHelpersExt::allocDeviceBuffer<PxRigidDynamicGPUIndex>(cuda, 1);
        auto* velocity = PxCudaHelpersExt::allocDeviceBuffer<PxVec3>(cuda, 1);
        auto* drain = ExtStressPhysXDirectGpuContactDrain::create(scene, 4096);
        require(drain && drain->available(), "device contact drain unavailable");
        step(scene); // initializes DirectGPUAPI and the body's GPU index
        auto index = body->getGPUIndex();
        PxCudaHelpersExt::copyHToD(cuda, indices, &index, 1);
        auto readVelocity = [&]() {
            require(scene.getDirectGPUAPI().getRigidDynamicData(velocity, indices,
                PxRigidDynamicGPUAPIReadType::eLINEAR_VELOCITY, 1), "velocity read failed");
            PxVec3 v; PxCudaHelpersExt::copyDToH(cuda, &v, velocity, 1); return v;
        };
        std::vector<ExtStressPhysXDirectGpuContact> records(4096);
        bool sawNormal = false, sawFriction = false, overflowTested = false;
        float worstError = 0.0f;
        ExtStressGpuNode nodes[2] = {{{0,0,0},0,0}, {{0,1,0},1,1}};
        ExtStressGpuBond bond{}; bond.node0=0; bond.node1=1; bond.centroid[1]=0.5f;
        auto* solver = ExtStressGpuSolver::create(nodes, 2, &bond, 1, nullptr, 0, cuda.getContext());
        require(solver, "shared-context solver create failed");
        auto* inputs = PxCudaHelpersExt::allocDeviceBuffer<ExtStressGpuImpulse>(cuda, 2);
        cudaStream_t stream{}; cudaEvent_t ready{};
        {
            PxScopedCudaLock lock(cuda);
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        }
        unsigned impactSolves = 0;
        for (unsigned tick = 0; tick < 30; ++tick)
        {
            const PxVec3 before = readVelocity();
            step(scene);
            const PxVec3 after = readVelocity();
            const unsigned count = drain->copyContacts(records.data(), records.size());
            require(drain->lastCopyComplete(), "contact copy failed or overflowed");
            PxVec3 total(0.0f);
            for (unsigned i = 0; i < count; ++i)
            {
                const auto& c = records[i];
                require(c.worldPosition.isFinite() && c.impulseOnActor0.isFinite(), "nonfinite contact");
                const PxVec3 impulse = c.actor0 == body ? c.impulseOnActor0
                    : c.actor1 == body ? -c.impulseOnActor0 : PxVec3(0.0f);
                total += impulse;
                sawNormal |= impulse.y > 0.01f;
                sawFriction |= impulse.x < -0.01f;
            }
            const PxVec3 expected = (after - before - PxVec3(0,-9.81f/60.0f,0)) * 2.0f;
            const float error = (total - expected).magnitude();
            worstError = fmaxf(worstError, error);
            require(error < 0.025f, "contact impulses disagree with physical momentum change");
            if (!count) continue;
            if (count > 1 && !overflowTested)
            {
                require(drain->copyContacts(records.data(), 1) == 0 && !drain->lastCopyComplete(),
                    "truncated contacts reported as complete");
                overflowTested = true;
            }
            ExtStressPhysXDirectGpuContactView view;
            require(drain->copyContactsDevice(view, 4096), "device contact view failed");
            {
                PxScopedCudaLock lock(cuda);
                check(cudaStreamWaitEvent(stream, reinterpret_cast<cudaEvent_t>(view.readyEvent), 0));
                contactLoads<<<1,1,0,stream>>>(view, body, inputs);
                check(cudaGetLastError()); check(cudaEventRecord(ready, stream));
            }
            ExtStressGpuSolveParams params; params.warmStart = false; params.maxIterations = 32;
            require(solver->solveDevice(inputs, 2, params, ready), "contact-to-stress GPU chain failed");
            require(solver->telemetry().hostToDeviceBytes == 0, "pipeline uploaded host loads");
            // Read outputs only for validation after the GPU chain completes.
            ExtStressGpuImpulse loadValues[2], impulse;
            PxCudaHelpersExt::copyDToH(cuda, loadValues, inputs, 2);
            const PxVec3 deviceTotal(loadValues[1].linear.x,loadValues[1].linear.y,loadValues[1].linear.z);
            require((deviceTotal-total).magnitude() < 1e-4f, "device consumer missed contacts");
            require(solver->readbackImpulses(&impulse, 1), "stress output readback failed");
            require(std::isfinite(impulse.linear.y) && std::abs(impulse.linear.y) > 1e-4f,
                "contact chain produced no stress");
            ++impactSolves;
        }
        require(sawNormal && sawFriction && overflowTested && impactSolves > 0,
            "fixture did not exercise normal impact, friction, overflow and stress");
        std::printf("contact-to-stress GPU pipeline passed: %u solves, momentum error <= %g kg m/s\n", impactSolves, worstError);
        // Last tick held live contacts. Once all dynamic actors leave, PhysX
        // takes an empty-copy branch that used to leave its old pair count.
        scene.removeActor(*body);
        step(scene);
        require(drain->copyContacts(records.data(), records.size()) == 0
            && drain->lastCopyComplete(), "empty frame reused stale GPU contacts");
        solver->release(); drain->release();
        { PxScopedCudaLock lock(cuda); check(cudaEventDestroy(ready)); check(cudaStreamDestroy(stream)); }
        PxCudaHelpersExt::freeDeviceBuffer(cuda, inputs);
        PxCudaHelpersExt::freeDeviceBuffer(cuda, velocity);
        PxCudaHelpersExt::freeDeviceBuffer(cuda, indices);
        body->release();
        require(context.healthy() && context.errors().warningCount()==0, "PhysX reported a warning");
        return 0;
    }
    catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
