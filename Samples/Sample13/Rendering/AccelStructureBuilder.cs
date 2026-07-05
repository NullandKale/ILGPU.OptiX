using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.AccelStructures;
using ILGPU.OptiX.Interop;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using MeshRange = ILGPU.OptiX.Pipeline.OptixMeshRange;
using System;
using System.Collections.Generic;

namespace Sample13
{
    /// <summary>
    /// Builds the scene's acceleration structures using OptixAccelBuilder.
    /// Triangle and custom-primitive geometry are built as separate GAS (per OptiX SDK constraints),
    /// then combined via an IAS. Per-frame updates for animated custom primitives use OptixAccelBuilder.Refit().
    /// </summary>
    public sealed class AccelStructureBuilder : IDisposable
    {
        readonly CudaAccelerator accelerator;
        readonly OptixDeviceContext deviceContext;
        readonly SceneGpuBuffers buffers;

        BuiltAccelStructure trianglesGas;
        BuiltAccelStructure customPrimitivesGas;
        BuiltAccelStructure ias;

        // Total hitgroup records this build told OptiX to expect, across every
        // AddTriangleMesh()/AddCustomPrimitives() call (NumSbtRecords * RAY_TYPE_COUNT,
        // summed) - set by Build(). SampleRenderer asserts this against the SBT's own
        // actual HitgroupRecordCount after building both; a mismatch means the accel
        // structure and the SBT no longer agree on the hitgroup record layout, which is
        // exactly the bug class that made every triangle/primitive silently resolve to
        // hitgroup record 0 regardless of its actual material (see this file's own fix
        // history in docs/API_BUILDER_PLAN.md's 2026-07-04 notes).
        public uint TotalHitgroupRecordsUsed { get; private set; }

        // Kept alive across frames (unlike triangleBuilder/iasBuilder, which really are
        // one-shot) so RefitCustomPrimitives() can call Refit() against the exact same
        // configured build inputs (same buffer pointers) this was built with - Refit()
        // requires that same topology, only the AABB buffer *contents* may have changed.
        // Holds no GPU resources itself (builders are ephemeral per the Phase 1 disposal
        // contract - only the MemoryBuffer references it points at, already owned by
        // SceneGpuBuffers), so keeping it alive is harmless.
        OptixAccelBuilder customPrimitivesBuilder;

        public AccelStructureBuilder(CudaAccelerator accelerator, OptixDeviceContext deviceContext, SceneGpuBuffers buffers)
        {
            this.accelerator = accelerator;
            this.deviceContext = deviceContext;
            this.buffers = buffers;
        }

        // Builds triangles + custom-primitives GAS, then combines them via IAS
        public unsafe IntPtr Build(SceneData scene, MeshRange[] triangleMeshRanges)
        {
            bool hasTriangles = scene.Vertices.Length > 0 && scene.Indices.Length > 0;
            bool hasCustomPrimitives = HasCustomPrimitives(scene);

            if (!hasTriangles && !hasCustomPrimitives)
                throw new InvalidOperationException("Scene must have at least triangles or custom primitives.");

            // Every triangle mesh / custom-primitive kind addresses the same
            // Materials.Length-sized local hitgroup-record block (see SbtBuilder.Build's
            // FillMaterialRecords) - NumSbtRecords must match that exactly, or OptiX's
            // per-build-input SBT-index-offset math and SbtBuilder's own record layout
            // silently disagree and every triangle/primitive resolves to hitgroup record 0
            // regardless of its actual assigned material (this was happening unconditionally
            // before this fix, for any scene with more than one material).
            uint numSbtRecords = (uint)scene.Materials.Length;

            uint totalHitgroupRecords = 0;

            // Build separate GAS for triangles and custom primitives
            var triangleBuilder = new OptixAccelBuilder()
                .WithDeviceContext(deviceContext)
                .WithAccelerator(accelerator);

            if (hasTriangles)
            {
                foreach (var range in triangleMeshRanges)
                {
                    triangleBuilder.AddTriangleMesh(buffers.Vertices, buffers.Indices,
                        buffers.TriangleMaterialIds, numSbtRecords, range.IndexStart, range.IndexCount);
                    totalHitgroupRecords += numSbtRecords * Payloads.RAY_TYPE_COUNT;
                }
            }

            customPrimitivesBuilder = new OptixAccelBuilder()
                .WithDeviceContext(deviceContext)
                .WithAccelerator(accelerator);

            if (hasCustomPrimitives)
            {
                if (scene.Spheres.Length > 0) { customPrimitivesBuilder.AddCustomPrimitives(buffers.SphereAabbs, buffers.SphereMaterialIds, numSbtRecords); totalHitgroupRecords += numSbtRecords * Payloads.RAY_TYPE_COUNT; }
                if (scene.Boxes.Length > 0) { customPrimitivesBuilder.AddCustomPrimitives(buffers.BoxAabbs, buffers.BoxMaterialIds, numSbtRecords); totalHitgroupRecords += numSbtRecords * Payloads.RAY_TYPE_COUNT; }
                if (scene.CylindersY.Length > 0) { customPrimitivesBuilder.AddCustomPrimitives(buffers.CylinderYAabbs, buffers.CylinderYMaterialIds, numSbtRecords); totalHitgroupRecords += numSbtRecords * Payloads.RAY_TYPE_COUNT; }
                if (scene.Disks.Length > 0) { customPrimitivesBuilder.AddCustomPrimitives(buffers.DiskAabbs, buffers.DiskMaterialIds, numSbtRecords); totalHitgroupRecords += numSbtRecords * Payloads.RAY_TYPE_COUNT; }
                if (scene.XYRects.Length > 0) { customPrimitivesBuilder.AddCustomPrimitives(buffers.XYRectAabbs, buffers.XYRectMaterialIds, numSbtRecords); totalHitgroupRecords += numSbtRecords * Payloads.RAY_TYPE_COUNT; }
                if (scene.XZRects.Length > 0) { customPrimitivesBuilder.AddCustomPrimitives(buffers.XZRectAabbs, buffers.XZRectMaterialIds, numSbtRecords); totalHitgroupRecords += numSbtRecords * Payloads.RAY_TYPE_COUNT; }
                if (scene.YZRects.Length > 0) { customPrimitivesBuilder.AddCustomPrimitives(buffers.YZRectAabbs, buffers.YZRectMaterialIds, numSbtRecords); totalHitgroupRecords += numSbtRecords * Payloads.RAY_TYPE_COUNT; }
                // Volume grid: NumSbtRecords=1, no materialIds - its record's own custom
                // data is never read (ShadeVolumeGrid looks up materials directly via
                // LaunchParams.Materials/VoxelMaterialIds instead), matching
                // SbtBuilder.Build's hasVolumeGrid special case.
                if (scene.VoxelMaterialIds.Length > 0) { customPrimitivesBuilder.AddCustomPrimitives(buffers.VolumeGridAabb); totalHitgroupRecords += 1u * Payloads.RAY_TYPE_COUNT; }

                if (scene.HasAnimatedGeometry) customPrimitivesBuilder.AllowUpdate();
            }

            TotalHitgroupRecordsUsed = totalHitgroupRecords;

            // Build both structures separately
            if (hasTriangles) trianglesGas = triangleBuilder.Build();
            if (hasCustomPrimitives) customPrimitivesGas = customPrimitivesBuilder.Build();

            // Combine via IAS if both present
            if (hasTriangles && hasCustomPrimitives)
            {
                int triangleMeshCount = triangleMeshRanges.Length;
                uint customSbtOffset = (uint)(triangleMeshCount * scene.Materials.Length) * Payloads.RAY_TYPE_COUNT;

                var iasBuilder = new OptixAccelBuilder()
                    .WithDeviceContext(deviceContext)
                    .WithAccelerator(accelerator);

                ias = iasBuilder.BuildInstanceAccelFromHandles(
                    new[] { trianglesGas.TraversableHandle, customPrimitivesGas.TraversableHandle },
                    new[] { 0u, customSbtOffset });

                return ias.TraversableHandle;
            }

            // Return whichever structure exists
            return hasTriangles ? trianglesGas.TraversableHandle : customPrimitivesGas.TraversableHandle;
        }

        // Per-frame refit for animated custom primitives - the caller must have already
        // re-uploaded fresh contents into the SAME device buffers (see
        // SceneGpuBuffers.UpdateAnimatedSpheres) before calling this. Refits in place via
        // the same customPrimitivesBuilder instance used at Build() time, so OptiX sees
        // the exact build-input topology/buffer pointers it originally built - only the
        // AABB contents actually changed.
        public void RefitCustomPrimitives(SceneData scene)
        {
            if (customPrimitivesGas == null || customPrimitivesBuilder == null || !scene.HasAnimatedGeometry)
                return;

            customPrimitivesBuilder.Refit(customPrimitivesGas);
        }

        public void DisposeBuffers()
        {
            ias?.Dispose();
            customPrimitivesGas?.Dispose();
            trianglesGas?.Dispose();
        }

        public void Dispose() => DisposeBuffers();

        static bool HasCustomPrimitives(SceneData scene) =>
            scene.Spheres.Length > 0 || scene.Boxes.Length > 0 ||
            scene.CylindersY.Length > 0 || scene.Disks.Length > 0 ||
            scene.XYRects.Length > 0 || scene.XZRects.Length > 0 ||
            scene.YZRects.Length > 0 || scene.VoxelMaterialIds.Length > 0;
    }
}
