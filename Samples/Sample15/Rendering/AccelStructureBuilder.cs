using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.AccelStructures;
using ILGPU.OptiX.Interop;
using ILGPU.OptiX.Pipeline;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using MeshRange = ILGPU.OptiX.Pipeline.OptixMeshRange;
using System;

namespace Sample15
{
    /// <summary>
    /// Builds the scene's acceleration structure using OptixAccelBuilder - a single GAS
    /// with one build input per mesh (see SceneData.MeshRanges), returned directly as
    /// the launch traversable. Every scene in this sample is pure triangle geometry
    /// (custom primitives/volume grid were removed in favor of procedural meshes), so no
    /// IAS or second GAS is needed.
    /// </summary>
    public sealed class AccelStructureBuilder : IDisposable
    {
        readonly CudaAccelerator accelerator;
        readonly OptixDeviceContext deviceContext;
        readonly SceneGpuBuffers buffers;

        BuiltAccelStructure trianglesGas;

        // Total hitgroup records this build told OptiX to expect (NumSbtRecords *
        // RAY_TYPE_COUNT, summed across every AddTriangleMesh() call) - set by Build().
        // SampleRenderer asserts this against the SBT's own actual HitgroupRecordCount
        // after building both; see Sample13's identical field for the bug class this
        // guards against.
        public uint TotalHitgroupRecordsUsed { get; private set; }

        public AccelStructureBuilder(CudaAccelerator accelerator, OptixDeviceContext deviceContext, SceneGpuBuffers buffers)
        {
            this.accelerator = accelerator;
            this.deviceContext = deviceContext;
            this.buffers = buffers;
        }

        // Builds the triangles GAS and returns its traversable handle directly.
        public IntPtr Build(SceneData scene, MeshRange[] triangleMeshRanges)
        {
            bool hasTriangles = scene.Vertices.Length > 0 && scene.Indices.Length > 0;
            if (!hasTriangles)
                return IntPtr.Zero;

            // Every scene here is fully static (no per-frame GAS refit) and this GAS is
            // built once per scene switch, then traced by every ray of every frame for as
            // long as that scene stays active - exactly the case OptiX's own guidance
            // recommends PREFER_FAST_TRACE for (pay more at the one-time build, trace
            // faster every ray afterward). AllowCompaction() is the companion flag - also
            // a one-time cost that shrinks the GAS's actual VRAM footprint for the rest of
            // that scene's lifetime. No AllowUpdate() needed since nothing refits this GAS.
            var builder = new OptixAccelBuilder()
                .WithDeviceContext(deviceContext)
                .WithAccelerator(accelerator)
                .PreferFastTrace()
                .AllowCompaction();

            // One build input per mesh (see SceneData.MeshRanges) instead of one merged
            // build input for the whole scene. Every mesh shares the same device
            // vertex/material-id buffers (only ever offset, never duplicated) and the
            // same global/absolute Materials[] palette.
            uint numSbtRecords = (uint)scene.Materials.Length;
            uint totalHitgroupRecords = 0;
            foreach (var range in triangleMeshRanges)
            {
                builder.AddTriangleMesh(buffers.Vertices, buffers.Indices,
                    buffers.TriangleMaterialIds, numSbtRecords, range.IndexStart, range.IndexCount);
                totalHitgroupRecords += numSbtRecords * Payloads.RAY_TYPE_COUNT;
            }
            TotalHitgroupRecordsUsed = totalHitgroupRecords;

            trianglesGas = builder.Build();
            return trianglesGas.TraversableHandle;
        }

        public void DisposeBuffers()
        {
            trianglesGas?.Dispose();
            trianglesGas = null;
        }

        public void Dispose() => DisposeBuffers();
    }
}
