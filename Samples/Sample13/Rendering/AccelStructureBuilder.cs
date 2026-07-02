using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.Interop;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;

namespace Sample13
{
    /// <summary>
    /// Builds the scene's acceleration structures. Triangle geometry and
    /// custom-primitive geometry cannot be combined as multiple build inputs within a
    /// single GAS (confirmed against the OptiX SDK's own optixSimpleMotionBlur sample -
    /// it builds one GAS per build-input type and combines them with an IAS instead).
    /// So triangles and custom primitives each get their own GAS here (all 7
    /// custom-primitive kinds share ONE GAS as multiple build inputs of the same type),
    /// and an IAS with one instance per GAS ties them together; each
    /// OptixInstance.SbtOffset picks up where the previous instance's hitgroup records
    /// left off, matching SbtBuilder's record layout.
    /// </summary>
    public sealed class AccelStructureBuilder : IDisposable
    {
        readonly CudaAccelerator accelerator;
        readonly OptixDeviceContext deviceContext;
        readonly SceneGpuBuffers buffers;

        MemoryBuffer1D<byte, Stride1D.Dense> trianglesGasBuffer;
        MemoryBuffer1D<byte, Stride1D.Dense> customPrimitivesGasBuffer;
        MemoryBuffer1D<byte, Stride1D.Dense> iasBuffer;
        MemoryBuffer1D<OptixInstance, Stride1D.Dense> instancesBuffer;

        // Persistent scratch temp buffer for per-frame OPTIX_BUILD_OPERATION_UPDATE
        // calls against the custom-primitives GAS (only allocated/used when the active
        // scene has SceneData.HasAnimatedGeometry) - sized once from
        // OptixAccelBufferSizes.TempUpdateSizeInBytes at the scene's initial BUILD, then
        // reused every frame (update temp-buffer requirements don't grow beyond what a
        // fixed topology needs, since only AABB/vertex contents change, never primitive
        // counts).
        MemoryBuffer1D<byte, Stride1D.Dense> customPrimitivesUpdateTempBuffer;

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
            bool hasCustomPrimitives = scene.Spheres.Length > 0 || scene.Boxes.Length > 0
                || scene.CylindersY.Length > 0 || scene.Disks.Length > 0
                || scene.XYRects.Length > 0 || scene.XZRects.Length > 0 || scene.YZRects.Length > 0
                || scene.VoxelMaterialIds.Length > 0;

            IntPtr trianglesGasHandle = hasTriangles ? BuildTrianglesGas(scene, triangleMeshRanges) : IntPtr.Zero;
            IntPtr customPrimitivesGasHandle = hasCustomPrimitives ? BuildOrUpdateCustomPrimitivesGas(scene, OptixBuildOperation.OPTIX_BUILD_OPERATION_BUILD) : IntPtr.Zero;

            return BuildInstanceAccel(scene, triangleMeshRanges, trianglesGasHandle, hasTriangles, customPrimitivesGasHandle, hasCustomPrimitives);
        }

        // Per-frame refit path for scenes with animated geometry (bobbing spheres) -
        // the caller must have re-uploaded fresh contents into the SAME device buffers
        // first (see SceneGpuBuffers.UpdateAnimatedSpheres).
        public void RefitCustomPrimitives(SceneData scene) =>
            BuildOrUpdateCustomPrimitivesGas(scene, OptixBuildOperation.OPTIX_BUILD_OPERATION_UPDATE);

        public void DisposeBuffers()
        {
            customPrimitivesUpdateTempBuffer?.Dispose();
            customPrimitivesUpdateTempBuffer = null;

            instancesBuffer?.Dispose(); instancesBuffer = null;
            iasBuffer?.Dispose(); iasBuffer = null;
            customPrimitivesGasBuffer?.Dispose(); customPrimitivesGasBuffer = null;
            trianglesGasBuffer?.Dispose(); trianglesGasBuffer = null;
        }

        public void Dispose() => DisposeBuffers();

        // One build input per mesh (see SceneData.MeshRanges) instead of one merged
        // build input for the whole scene - mirrors BuildOrUpdateCustomPrimitivesGas's
        // multiple-build-inputs-in-one-GAS pattern below. Every mesh shares the same
        // device vertex/material-id buffers (only ever offset, never duplicated) and
        // the same global/absolute Materials[] palette (no per-mesh SBT-index
        // remapping - see SbtLayout.GetTriangleMeshRanges's comment), so vertexBuffers/
        // sharedFlags below are declared once and reused by every build input, unlike
        // BuildOrUpdateCustomPrimitivesGas's per-kind AABB buffers which genuinely
        // differ per build input.
        unsafe IntPtr BuildTrianglesGas(SceneData scene, MeshRange[] ranges)
        {
            var vertexBuffers = stackalloc IntPtr[1];
            vertexBuffers[0] = buffers.Vertices.NativePtr;
            var sharedFlags = stackalloc uint[scene.Materials.Length];

            var buildInputs = new OptixBuildInput[ranges.Length];
            for (int i = 0; i < ranges.Length; i++)
            {
                MeshRange range = ranges[i];
                OptixBuildInput input = new OptixBuildInput()
                {
                    Type = OptixBuildInputType.OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
                };

                input.TriangleArray.VertexFormat = OptixVertexFormat.OPTIX_VERTEX_FORMAT_FLOAT3;
                input.TriangleArray.VertexStrideInBytes = (uint)sizeof(Vec3);
                input.TriangleArray.NumVerticies = (uint)scene.Vertices.Length;
                input.TriangleArray.VertexBuffers = new IntPtr(vertexBuffers);

                input.TriangleArray.IndexFormat = OptixIndicesFormat.OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                input.TriangleArray.IndexStrideInBytes = (uint)sizeof(Vec3i);
                input.TriangleArray.NumIndexTriplets = (uint)range.IndexCount;
                input.TriangleArray.IndexBuffer = IntPtr.Add(buffers.Indices.NativePtr, range.IndexStart * sizeof(Vec3i));

                input.TriangleArray.Flags = sharedFlags;
                input.TriangleArray.NumSbtRecords = (uint)scene.Materials.Length;
                input.TriangleArray.SbtIndexOffsetBuffer = IntPtr.Add(buffers.TriangleMaterialIds.NativePtr, range.IndexStart * sizeof(uint));
                input.TriangleArray.SbtIndexOffsetSizeInBytes = sizeof(uint);
                input.TriangleArray.SbtIndexOffsetStrideInBytes = 0;

                buildInputs[i] = input;
            }

            OptixAccelBuildOptions accelOptions = new OptixAccelBuildOptions()
            {
                BuildFlags = OptixBuildFlags.OPTIX_BUILD_FLAG_NONE,
                Operation = OptixBuildOperation.OPTIX_BUILD_OPERATION_BUILD
            };
            accelOptions.MotionOptions.NumKeys = 1;

            OptixAccelBufferSizes bufferSizes = deviceContext.AccelComputeMemoryUsage(accelOptions, buildInputs);

            using MemoryBuffer1D<byte, Stride1D.Dense> tempBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.TempSizeInBytes);
            trianglesGasBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.OutputSizeInBytes);

            return deviceContext.AccelBuild(accelerator.DefaultStream, accelOptions, buildInputs, tempBuffer, trianglesGasBuffer, ReadOnlySpan<OptixAccelEmitDesc>.Empty);
        }

        // Builds ONE GAS from multiple OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES build
        // inputs (one per active primitive kind, in canonical HitKind order) - this is
        // supported (same build-input TYPE repeated, like multiple triangle meshes
        // already sharing one GAS), unlike mixing triangles with custom primitives.
        // Every stackalloc'd AABB-buffer-pointer array and the shared flags array must
        // stay alive for this whole method body (stackalloc's lifetime is the enclosing
        // method's stack frame), so this can't be split into per-kind helper methods
        // the way BuildTrianglesGas is.
        //
        // Also doubles as the per-frame refit path for scenes with
        // SceneData.HasAnimatedGeometry (bobbing spheres): called again every frame
        // with operation=OPTIX_BUILD_OPERATION_UPDATE after the caller has re-uploaded
        // fresh contents into the SAME sphere/AABB device buffers (their addresses
        // never change between frames, only their contents) - OptiX's update operation
        // requires the exact same build-input topology/buffer pointers as the original
        // build, just refreshed data, and always returns the same traversable handle it
        // was given at the original BUILD (see docs/SAMPLE13_PLAN.md's AS-update design
        // note), so the IAS instance pointing at this GAS's handle never needs touching.
        unsafe IntPtr BuildOrUpdateCustomPrimitivesGas(SceneData scene, OptixBuildOperation operation)
        {
            var buildInputsList = new List<OptixBuildInput>();

            // Every custom-primitive build input can share this one all-zero
            // (OPTIX_GEOMETRY_FLAG_NONE) flags array - none of our materials need
            // per-geometry build flags, and "size must match numSbtRecords" is
            // satisfied for every kind since NumSbtRecords is always scene.Materials.Length.
            var sharedFlags = stackalloc uint[scene.Materials.Length];

            unsafe OptixBuildInput MakeCustomPrimitiveInput(IntPtr* aabbBufferSlot, IntPtr aabbBufferPtr, uint numPrimitives, IntPtr sbtIndexOffsetBuffer)
            {
                aabbBufferSlot[0] = aabbBufferPtr;
                var input = new OptixBuildInput { Type = OptixBuildInputType.OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES };
                input.CustomPrimitiveArray.AabbBuffers = new IntPtr(aabbBufferSlot);
                input.CustomPrimitiveArray.NumPrimitives = numPrimitives;
                input.CustomPrimitiveArray.Stride = 0; // tightly packed, sizeof(OptixAabb)
                input.CustomPrimitiveArray.Flags = sharedFlags;
                input.CustomPrimitiveArray.NumSbtRecords = (uint)scene.Materials.Length;
                input.CustomPrimitiveArray.SbtIndexOffsetBuffer = sbtIndexOffsetBuffer;
                input.CustomPrimitiveArray.SbtIndexOffsetSizeInBytes = sizeof(uint);
                input.CustomPrimitiveArray.SbtIndexOffsetStrideInBytes = 0;
                return input;
            }

            var sphereAabbSlot = stackalloc IntPtr[1];
            if (scene.Spheres.Length > 0)
                buildInputsList.Add(MakeCustomPrimitiveInput(sphereAabbSlot, buffers.SphereAabbs.NativePtr, (uint)scene.Spheres.Length, buffers.SphereMaterialIds.NativePtr));

            var boxAabbSlot = stackalloc IntPtr[1];
            if (scene.Boxes.Length > 0)
                buildInputsList.Add(MakeCustomPrimitiveInput(boxAabbSlot, buffers.BoxAabbs.NativePtr, (uint)scene.Boxes.Length, buffers.BoxMaterialIds.NativePtr));

            var cylinderYAabbSlot = stackalloc IntPtr[1];
            if (scene.CylindersY.Length > 0)
                buildInputsList.Add(MakeCustomPrimitiveInput(cylinderYAabbSlot, buffers.CylinderYAabbs.NativePtr, (uint)scene.CylindersY.Length, buffers.CylinderYMaterialIds.NativePtr));

            var diskAabbSlot = stackalloc IntPtr[1];
            if (scene.Disks.Length > 0)
                buildInputsList.Add(MakeCustomPrimitiveInput(diskAabbSlot, buffers.DiskAabbs.NativePtr, (uint)scene.Disks.Length, buffers.DiskMaterialIds.NativePtr));

            var xyRectAabbSlot = stackalloc IntPtr[1];
            if (scene.XYRects.Length > 0)
                buildInputsList.Add(MakeCustomPrimitiveInput(xyRectAabbSlot, buffers.XYRectAabbs.NativePtr, (uint)scene.XYRects.Length, buffers.XYRectMaterialIds.NativePtr));

            var xzRectAabbSlot = stackalloc IntPtr[1];
            if (scene.XZRects.Length > 0)
                buildInputsList.Add(MakeCustomPrimitiveInput(xzRectAabbSlot, buffers.XZRectAabbs.NativePtr, (uint)scene.XZRects.Length, buffers.XZRectMaterialIds.NativePtr));

            var yzRectAabbSlot = stackalloc IntPtr[1];
            if (scene.YZRects.Length > 0)
                buildInputsList.Add(MakeCustomPrimitiveInput(yzRectAabbSlot, buffers.YZRectAabbs.NativePtr, (uint)scene.YZRects.Length, buffers.YZRectMaterialIds.NativePtr));

            // Volume grid: NumPrimitives=1 (one AABB for the whole grid), NumSbtRecords=1
            // (its record's custom data is unused - ShadeVolumeGrid looks up materials
            // directly via LaunchParams.Materials/VoxelMaterialIds instead, since a
            // single primitive can't otherwise carry a per-voxel material through the
            // SbtIndexOffsetBuffer mechanism - see ShadingHelpers.ShadeVolumeGrid). A
            // null SbtIndexOffsetBuffer is valid per optix_types.h ("May be NULL" -
            // every entry is then sbt index 0), so no dedicated device buffer is needed.
            var volumeGridAabbSlot = stackalloc IntPtr[1];
            if (scene.VoxelMaterialIds.Length > 0)
            {
                volumeGridAabbSlot[0] = buffers.VolumeGridAabb.NativePtr;
                var volumeGridInput = new OptixBuildInput { Type = OptixBuildInputType.OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES };
                volumeGridInput.CustomPrimitiveArray.AabbBuffers = new IntPtr(volumeGridAabbSlot);
                volumeGridInput.CustomPrimitiveArray.NumPrimitives = 1;
                volumeGridInput.CustomPrimitiveArray.Stride = 0;
                volumeGridInput.CustomPrimitiveArray.Flags = sharedFlags;
                volumeGridInput.CustomPrimitiveArray.NumSbtRecords = 1;
                volumeGridInput.CustomPrimitiveArray.SbtIndexOffsetBuffer = IntPtr.Zero;
                buildInputsList.Add(volumeGridInput);
            }

            OptixAccelBuildOptions accelOptions = new OptixAccelBuildOptions()
            {
                BuildFlags = scene.HasAnimatedGeometry ? OptixBuildFlags.OPTIX_BUILD_FLAG_ALLOW_UPDATE : OptixBuildFlags.OPTIX_BUILD_FLAG_NONE,
                Operation = operation
            };
            accelOptions.MotionOptions.NumKeys = 1;

            OptixBuildInput[] buildInputs = buildInputsList.ToArray();

            if (operation == OptixBuildOperation.OPTIX_BUILD_OPERATION_UPDATE)
            {
                // Same output buffer as the original BUILD (required by OptiX for
                // update), and the persistent update-sized temp buffer allocated back
                // when that BUILD ran.
                return deviceContext.AccelBuild(accelerator.DefaultStream, accelOptions, buildInputs, customPrimitivesUpdateTempBuffer, customPrimitivesGasBuffer, ReadOnlySpan<OptixAccelEmitDesc>.Empty);
            }

            OptixAccelBufferSizes bufferSizes = deviceContext.AccelComputeMemoryUsage(accelOptions, buildInputs);

            using MemoryBuffer1D<byte, Stride1D.Dense> tempBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.TempSizeInBytes);
            customPrimitivesGasBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.OutputSizeInBytes);

            if (scene.HasAnimatedGeometry)
            {
                customPrimitivesUpdateTempBuffer?.Dispose();
                customPrimitivesUpdateTempBuffer = accelerator.Allocate1D<byte>((long)Math.Max(1UL, bufferSizes.TempUpdateSizeInBytes));
            }

            return deviceContext.AccelBuild(accelerator.DefaultStream, accelOptions, buildInputs, tempBuffer, customPrimitivesGasBuffer, ReadOnlySpan<OptixAccelEmitDesc>.Empty);
        }

        unsafe IntPtr BuildInstanceAccel(SceneData scene, MeshRange[] triangleMeshRanges, IntPtr trianglesGasHandle, bool hasTriangles, IntPtr customPrimitivesGasHandle, bool hasCustomPrimitives)
        {
            // Triangle records (if present) occupy [0, triangleMeshCount *
            // materials.Length*RAY_TYPE_COUNT) in SbtBuilder's flat array
            // (RAY_TYPE_COUNT=2, radiance+shadow per material - matches
            // Payloads.RAY_TYPE_COUNT; triangleMeshCount is >1 whenever the
            // triangles GAS has multiple per-mesh build inputs, see
            // SbtLayout.GetTriangleMeshRanges/BuildTrianglesGas); the custom-primitives
            // instance's records start right after (or at 0, if there's no triangle
            // instance - e.g. the volume-grid-only test scene). All triangle build
            // inputs still live in the ONE triangles GAS, so only one IAS instance is
            // needed for them here, regardless of how many mesh build inputs it has.
            var instances = new List<OptixInstance>();
            if (hasTriangles)
                instances.Add(MakeInstance(trianglesGasHandle, sbtOffset: 0));
            if (hasCustomPrimitives)
            {
                int triangleMeshCount = hasTriangles ? triangleMeshRanges.Length : 0;
                uint customPrimitivesSbtOffset = hasTriangles ? (uint)(triangleMeshCount * scene.Materials.Length * 2) : 0;
                instances.Add(MakeInstance(customPrimitivesGasHandle, customPrimitivesSbtOffset));
            }

            instancesBuffer = accelerator.Allocate1D(instances.ToArray());

            OptixBuildInput instanceInput = new OptixBuildInput()
            {
                Type = OptixBuildInputType.OPTIX_BUILD_INPUT_TYPE_INSTANCES,
            };
            instanceInput.InstanceArray.Instances = instancesBuffer.NativePtr;
            instanceInput.InstanceArray.NumInstances = (uint)instances.Count;
            instanceInput.InstanceArray.InstanceStride = 0;

            OptixAccelBuildOptions accelOptions = new OptixAccelBuildOptions()
            {
                BuildFlags = OptixBuildFlags.OPTIX_BUILD_FLAG_NONE,
                Operation = OptixBuildOperation.OPTIX_BUILD_OPERATION_BUILD
            };
            accelOptions.MotionOptions.NumKeys = 1;

            OptixBuildInput[] buildInputs = { instanceInput };
            OptixAccelBufferSizes bufferSizes = deviceContext.AccelComputeMemoryUsage(accelOptions, buildInputs);

            using MemoryBuffer1D<byte, Stride1D.Dense> tempBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.TempSizeInBytes);
            iasBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.OutputSizeInBytes);

            return deviceContext.AccelBuild(accelerator.DefaultStream, accelOptions, buildInputs, tempBuffer, iasBuffer, ReadOnlySpan<OptixAccelEmitDesc>.Empty);
        }

        static unsafe OptixInstance MakeInstance(IntPtr gasHandle, uint sbtOffset)
        {
            OptixInstance instance = default;
            for (int i = 0; i < 12; i++)
                instance.Transform[i] = OptixInstance.IdentityTransform[i];
            instance.InstanceId = 0;
            instance.SbtOffset = sbtOffset;
            instance.VisibilityMask = 0xFF;
            instance.Flags = 0; // OPTIX_INSTANCE_FLAG_NONE
            instance.TraversableHandle = unchecked((ulong)gasHandle.ToInt64());
            return instance;
        }
    }
}
