using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.AccelStructures;
using ILGPU.OptiX.Interop;
using ILGPU.OptiX.Pipeline;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using MeshRange = ILGPU.OptiX.Pipeline.OptixMeshRange;
using System;
using System.Collections.Generic;

namespace Sample13
{
    /// <summary>
    /// Builds the scene's acceleration structures using OptixAccelBuilder. Triangle
    /// geometry and custom-primitive geometry cannot be combined as multiple build
    /// inputs within a single GAS (confirmed against the OptiX SDK's own
    /// optixSimpleMotionBlur sample - it builds one GAS per build-input type and
    /// combines them with an IAS instead). So triangles and custom primitives each get
    /// their own GAS here (all 7 custom-primitive kinds plus the volume grid share ONE
    /// GAS as multiple build inputs of the same type), and an IAS with one instance per
    /// GAS ties them together - always, even when only one of the two GAS exists, for a
    /// uniform interface (matches this sample's original hand-rolled behavior); each
    /// OptixInstance.SbtOffset picks up where the previous instance's hitgroup records
    /// left off, matching SbtBuilder's record layout.
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
        // structure and the SBT no longer agree on the hitgroup record layout, and
        // every triangle/primitive would silently resolve to hitgroup record 0
        // regardless of its actual material.
        public uint TotalHitgroupRecordsUsed { get; private set; }

        // Kept alive across frames so RefitCustomPrimitives() can Refit() against the
        // exact same configured build inputs (same buffer pointers) used at Build()
        // time. Holds no GPU resources itself - only the MemoryBuffer references it
        // points at, already owned by SceneGpuBuffers.
        OptixAccelBuilder customPrimitivesBuilder;

        public AccelStructureBuilder(CudaAccelerator accelerator, OptixDeviceContext deviceContext, SceneGpuBuffers buffers)
        {
            this.accelerator = accelerator;
            this.deviceContext = deviceContext;
            this.buffers = buffers;
        }

        // Builds both GASes plus the IAS and returns the IAS traversable handle.
        public unsafe IntPtr Build(SceneData scene, MeshRange[] triangleMeshRanges)
        {
            bool hasTriangles = scene.Vertices.Length > 0 && scene.Indices.Length > 0;
            bool hasCustomPrimitives = HasCustomPrimitives(scene);

            if (!hasTriangles && !hasCustomPrimitives)
                throw new InvalidOperationException("Scene must have at least triangles or custom primitives.");

            // Every triangle mesh / custom-primitive kind addresses the same
            // Materials.Length-sized local hitgroup-record block (see SbtBuilder's own
            // FillMaterialRecords) - NumSbtRecords must match that exactly.
            uint numSbtRecords = (uint)scene.Materials.Length;

            uint totalHitgroupRecords = 0;

            var triangleBuilder = new OptixAccelBuilder()
                .WithDeviceContext(deviceContext)
                .WithAccelerator(accelerator);

            if (hasTriangles)
            {
                // One build input per mesh (see SceneData.MeshRanges) instead of one
                // merged build input for the whole scene - every mesh shares the same
                // device vertex/material-id buffers (only ever offset, never
                // duplicated) and the same global/absolute Materials[] palette.
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
                // SbtBuilder's hasVolumeGrid special case. A null SbtIndexOffsetBuffer
                // is valid per optix_types.h ("May be NULL" - every entry is then sbt
                // index 0), so no dedicated device buffer is needed.
                if (scene.VoxelMaterialIds.Length > 0) { customPrimitivesBuilder.AddCustomPrimitives(buffers.VolumeGridAabb); totalHitgroupRecords += 1u * Payloads.RAY_TYPE_COUNT; }

                if (scene.HasAnimatedGeometry) customPrimitivesBuilder.AllowUpdate();
            }

            TotalHitgroupRecordsUsed = totalHitgroupRecords;

            if (hasTriangles) trianglesGas = triangleBuilder.Build();
            if (hasCustomPrimitives) customPrimitivesGas = customPrimitivesBuilder.Build();

            // Always combine via IAS, even with only one GAS present, for a uniform
            // interface (matches this sample's original hand-rolled behavior).
            int triangleMeshCount = hasTriangles ? triangleMeshRanges.Length : 0;

            var handles = new List<IntPtr>();
            var offsets = new List<uint>();
            if (hasTriangles)
            {
                handles.Add(trianglesGas.TraversableHandle);
                offsets.Add(0u);
            }
            if (hasCustomPrimitives)
            {
                // Triangle records (if present) occupy [0, triangleMeshCount *
                // materials.Length*RAY_TYPE_COUNT) in SbtBuilder's flat array
                // (RAY_TYPE_COUNT=2, radiance+shadow per material); the
                // custom-primitives instance's records start right after (or at 0, if
                // there's no triangle instance).
                uint customSbtOffset = hasTriangles ? (uint)(triangleMeshCount * scene.Materials.Length) * Payloads.RAY_TYPE_COUNT : 0u;
                handles.Add(customPrimitivesGas.TraversableHandle);
                offsets.Add(customSbtOffset);
            }

            var iasBuilder = new OptixAccelBuilder()
                .WithDeviceContext(deviceContext)
                .WithAccelerator(accelerator);

            ias = iasBuilder.BuildInstanceAccelFromHandles(handles.ToArray(), offsets.ToArray());
            return ias.TraversableHandle;
        }

        // Per-frame refit path for scenes with animated geometry (bobbing spheres) - the
        // caller must have re-uploaded fresh contents into the SAME device buffers first
        // (see SceneGpuBuffers.UpdateAnimatedSpheres). Refits customPrimitivesGas in
        // place via the same customPrimitivesBuilder instance used at Build() time, so
        // OptiX sees the exact build-input topology/buffer pointers it originally built -
        // only the AABB contents actually changed. OptiX's update operation always
        // returns the same traversable handle it was given at the original BUILD, so the
        // IAS instance pointing at this GAS's handle never needs touching.
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
            scene.Spheres.Length > 0 || scene.Boxes.Length > 0
            || scene.CylindersY.Length > 0 || scene.Disks.Length > 0
            || scene.XYRects.Length > 0 || scene.XZRects.Length > 0 || scene.YZRects.Length > 0
            || scene.VoxelMaterialIds.Length > 0;
    }
}
