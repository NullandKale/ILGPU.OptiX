// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixAccelBuilder.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Util;
using System;
using System.Collections.Generic;
using static ILGPU.Stride1D;

namespace ILGPU.OptiX.AccelStructures
{
    /// <summary>
    /// Manages the full acceleration structure build pipeline.
    /// Supports GAS (geometry via triangles/custom-primitives) or IAS (instances).
    /// For complex multi-GAS scenarios (e.g., separate triangle + custom-primitive GAS combined via IAS),
    /// use BuildSeparateGeometry() to get individual structures, then BuildInstanceAccel() to combine them.
    /// </summary>
    public sealed class OptixAccelBuilder : DisposeBase
    {
        private class TriangleMesh
        {
            public MemoryBuffer Vertices { get; set; }
            public MemoryBuffer Indices { get; set; }
            public MemoryBuffer MaterialIds { get; set; }
            public uint NumSbtRecords { get; set; }
            public long IndexStart { get; set; }
            public long IndexCount { get; set; }
        }

        private class CustomPrimitive
        {
            public MemoryBuffer AabbBuffer { get; set; }
            public MemoryBuffer MaterialIds { get; set; }
            public uint NumSbtRecords { get; set; }
        }

        private OptixDeviceContext deviceContext;
        private CudaAccelerator accelerator;
        private readonly List<TriangleMesh> triangles = new List<TriangleMesh>();
        private readonly List<CustomPrimitive> customPrimitives = new List<CustomPrimitive>();
        private (MemoryBuffer buffer, uint count)? instances;
        private bool allowCompaction;
        private bool preferFastTrace;
        private bool preferFastBuild;
        private bool allowUpdate;

        /// <summary>
        /// Sets the device context for acceleration structure building.
        /// </summary>
        public OptixAccelBuilder WithDeviceContext(OptixDeviceContext ctx)
        {
            if (ctx == null)
                throw new ArgumentNullException(nameof(ctx), "Device context must not be null.");

            deviceContext = ctx;
            return this;
        }

        /// <summary>
        /// Sets the accelerator for GPU memory operations.
        /// </summary>
        public OptixAccelBuilder WithAccelerator(CudaAccelerator accel)
        {
            if (accel == null)
                throw new ArgumentNullException(nameof(accel), "Accelerator must not be null.");

            accelerator = accel;
            return this;
        }

        /// <summary>
        /// Adds a triangle mesh, or a sub-range of one. When <paramref name="materialIds"/>
        /// is supplied, <paramref name="numSbtRecords"/> must be at least one greater than
        /// the largest material ID referenced by that buffer, since it tells OptiX how
        /// many hitgroup records are associated with this build input.
        /// </summary>
        /// <param name="vertices">
        /// The mesh's full vertex buffer. Unlike <paramref name="indices"/>, this is never
        /// sub-ranged - every mesh sharing one big vertex buffer passes the same buffer
        /// here, only offset by which indices reference which vertices.
        /// </param>
        /// <param name="indices">
        /// The (possibly shared) index buffer, one <c>Vec3i</c> (3 uints) per triangle.
        /// </param>
        /// <param name="indexStart">
        /// Triangle (not raw index) offset into <paramref name="indices"/> where this
        /// mesh's own triangles begin - 0 means "from the start of the buffer". Lets
        /// multiple meshes share one big index buffer as separate build inputs, each
        /// covering only its own sub-range, instead of every build input covering the
        /// whole shared buffer.
        /// </param>
        /// <param name="indexCount">
        /// Number of triangles this mesh occupies starting at <paramref name="indexStart"/>,
        /// or -1 (default) to mean "the rest of the buffer from indexStart" - matches the
        /// old always-whole-buffer behavior when indexStart is also left at 0.
        /// </param>
        public OptixAccelBuilder AddTriangleMesh(MemoryBuffer vertices, MemoryBuffer indices,
            MemoryBuffer materialIds = null, uint numSbtRecords = 1,
            long indexStart = 0, long indexCount = -1)
        {
            if (vertices == null || vertices.Length == 0)
                throw new ArgumentException("Vertices buffer must not be null or empty.", nameof(vertices));
            if (indices == null || indices.Length == 0)
                throw new ArgumentException("Indices buffer must not be null or empty.", nameof(indices));
            if (indexStart < 0)
                throw new ArgumentOutOfRangeException(nameof(indexStart), "Index start must be >= 0.");

            long resolvedIndexCount = indexCount < 0 ? indices.Length - indexStart : indexCount;
            if (resolvedIndexCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(indexCount), "Index count must be > 0.");

            triangles.Add(new TriangleMesh
            {
                Vertices = vertices,
                Indices = indices,
                MaterialIds = materialIds,
                NumSbtRecords = numSbtRecords,
                IndexStart = indexStart,
                IndexCount = resolvedIndexCount
            });
            return this;
        }

        /// <summary>
        /// Adds a kind of custom primitives (e.g. all spheres, or all boxes) as one build
        /// input within the resulting GAS. Call multiple times to combine several kinds
        /// (each its own build input) into a single GAS - matches OptiX's own constraint
        /// that a single build input's primitives must all be the same intersection
        /// program, while multiple build inputs of that type can still share one GAS.
        /// </summary>
        /// <param name="aabbBuffer">The AABB buffer for this primitive kind.</param>
        /// <param name="materialIds">
        /// Per-primitive material/SBT index buffer for this kind, or null if every
        /// primitive of this kind uses hitgroup record 0 (OptiX treats a null
        /// SbtIndexOffsetBuffer as "every entry is index 0").
        /// </param>
        /// <param name="numSbtRecords">
        /// Number of hitgroup records this kind's primitives can address - must be at
        /// least one greater than the largest ID referenced by <paramref name="materialIds"/>.
        /// </param>
        public OptixAccelBuilder AddCustomPrimitives(MemoryBuffer aabbBuffer,
            MemoryBuffer materialIds = null, uint numSbtRecords = 1)
        {
            if (aabbBuffer == null || aabbBuffer.Length == 0)
                throw new ArgumentException("AABB buffer must not be null or empty.", nameof(aabbBuffer));

            customPrimitives.Add(new CustomPrimitive
            {
                AabbBuffer = aabbBuffer,
                MaterialIds = materialIds,
                NumSbtRecords = numSbtRecords
            });
            return this;
        }

        /// <summary>
        /// Adds instances.
        /// </summary>
        public OptixAccelBuilder AddInstances(MemoryBuffer instanceBuffer, uint numInstances)
        {
            if (instanceBuffer == null || instanceBuffer.Length == 0)
                throw new ArgumentException("Instance buffer must not be null or empty.", nameof(instanceBuffer));
            if (numInstances == 0)
                throw new ArgumentException("Number of instances must be > 0.", nameof(numInstances));

            instances = (instanceBuffer, numInstances);
            return this;
        }

        /// <summary>
        /// Enables acceleration structure compaction.
        /// </summary>
        public OptixAccelBuilder AllowCompaction()
        {
            allowCompaction = true;
            return this;
        }

        /// <summary>
        /// Prefers fast tracing over fast building.
        /// </summary>
        public OptixAccelBuilder PreferFastTrace()
        {
            preferFastTrace = true;
            return this;
        }

        /// <summary>
        /// Prefers fast building over fast tracing.
        /// </summary>
        public OptixAccelBuilder PreferFastBuild()
        {
            preferFastBuild = true;
            return this;
        }

        /// <summary>
        /// Allows updating the acceleration structure.
        /// </summary>
        public OptixAccelBuilder AllowUpdate()
        {
            allowUpdate = true;
            return this;
        }

        /// <summary>
        /// Builds the acceleration structure and returns the result with all GPU buffers owned.
        /// </summary>
        public unsafe BuiltAccelStructure Build(CudaStream stream = null)
        {
            if (deviceContext == null)
                throw new InvalidOperationException("Device context must be set via WithDeviceContext().");
            if (accelerator == null)
                throw new InvalidOperationException("Accelerator must be set via WithAccelerator().");

            bool hasTriangles = triangles.Count > 0;
            bool hasCustomPrimitives = customPrimitives.Count > 0;
            bool hasInstances = instances.HasValue;

            if (!hasTriangles && !hasCustomPrimitives && !hasInstances)
                throw new InvalidOperationException(
                    "At least one of AddTriangleMesh(), AddCustomPrimitives(), or AddInstances() must be called.");

            // Cannot mix GAS and IAS types
            int gasCount = (hasTriangles ? 1 : 0) + (hasCustomPrimitives ? 1 : 0);
            if (gasCount > 0 && hasInstances)
                throw new InvalidOperationException(
                    "Cannot mix geometry (triangles/custom-primitives) with instances; choose one type.");

            MemoryBuffer1D<byte, Dense> tempBuffer = null;
            MemoryBuffer1D<byte, Dense> outputBuffer = null;
            MemoryBuffer1D<byte, Dense> compactedBuffer = null;
            MemoryBuffer1D<ulong, Dense> compactSizeBuffer = null;

            try
            {
                // Build the list of OptixBuildInput structures
                var buildInputList = new List<OptixBuildInput>();

                // Add triangle meshes
                if (hasTriangles)
                {
                    // One pointer slot per mesh - each build input's VertexBuffers must
                    // keep pointing at its own slot for the lifetime of this method, since
                    // the native AccelBuild call below reads them only after this whole
                    // loop finishes. A single reused slot would leave every build input
                    // pointing at whichever mesh was added last.
                    var vertexBufferPtrs = stackalloc IntPtr[triangles.Count];

                    for (int meshIndex = 0; meshIndex < triangles.Count; meshIndex++)
                    {
                        var mesh = triangles[meshIndex];
                        vertexBufferPtrs[meshIndex] = mesh.Vertices.NativePtr;

                        var input = new OptixBuildInput
                        {
                            Type = OptixBuildInputType.OPTIX_BUILD_INPUT_TYPE_TRIANGLES
                        };

                        input.TriangleArray.VertexFormat = OptixVertexFormat.OPTIX_VERTEX_FORMAT_FLOAT3;
                        input.TriangleArray.VertexStrideInBytes = sizeof(float) * 3;  // Vec3 = 3 floats
                        input.TriangleArray.NumVertices = (uint)mesh.Vertices.Length;
                        input.TriangleArray.VertexBuffers = new IntPtr(vertexBufferPtrs + meshIndex);

                        // Indices/materialIds are byte-offset by IndexStart (a triangle
                        // count) so multiple meshes can share one big index/materialIds
                        // buffer as separate build inputs, each covering only its own
                        // sub-range - IntPtr.Add with a 0 offset is a no-op, so this is
                        // exactly the old whole-buffer behavior when IndexStart is 0.
                        const int vec3iSizeInBytes = sizeof(uint) * 3;
                        input.TriangleArray.IndexFormat = OptixIndicesFormat.OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                        input.TriangleArray.IndexStrideInBytes = vec3iSizeInBytes;
                        input.TriangleArray.NumIndexTriplets = (uint)mesh.IndexCount;
                        input.TriangleArray.IndexBuffer = IntPtr.Add(
                            mesh.Indices.NativePtr, (int)(mesh.IndexStart * vec3iSizeInBytes));

                        // OptiX requires one flag per SBT record, not one flag per
                        // build input - a 1-element array here was fine while
                        // NumSbtRecords was always 1, but reading past it once a mesh
                        // has multiple materials returns garbage flag bits and OptiX
                        // rejects the build with OPTIX_ERROR_INVALID_VALUE.
                        var meshFlags = stackalloc uint[(int)mesh.NumSbtRecords];
                        for (int flagIndex = 0; flagIndex < mesh.NumSbtRecords; flagIndex++)
                            meshFlags[flagIndex] = 0; // OPTIX_GEOMETRY_FLAG_NONE

                        input.TriangleArray.Flags = meshFlags;
                        input.TriangleArray.NumSbtRecords = mesh.NumSbtRecords;
                        input.TriangleArray.SbtIndexOffsetBuffer = mesh.MaterialIds == null
                            ? IntPtr.Zero
                            : IntPtr.Add(mesh.MaterialIds.NativePtr, (int)(mesh.IndexStart * sizeof(uint)));
                        input.TriangleArray.SbtIndexOffsetSizeInBytes = sizeof(uint);
                        input.TriangleArray.SbtIndexOffsetStrideInBytes = 0;

                        buildInputList.Add(input);
                    }
                }

                // Add custom primitives
                if (hasCustomPrimitives)
                {
                    // One pointer slot per primitive set - see the triangle loop above
                    // for why a single reused slot is unsafe here.
                    var aabbPtrs = stackalloc IntPtr[customPrimitives.Count];

                    for (int primIndex = 0; primIndex < customPrimitives.Count; primIndex++)
                    {
                        var prim = customPrimitives[primIndex];
                        aabbPtrs[primIndex] = prim.AabbBuffer.NativePtr;

                        var input = new OptixBuildInput
                        {
                            Type = OptixBuildInputType.OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES
                        };

                        // One flag per SBT record (see the identical comment in the
                        // triangle loop above) - a shared 1-element array is only safe
                        // while every kind's NumSbtRecords is 1.
                        var kindFlags = stackalloc uint[(int)prim.NumSbtRecords];
                        for (int flagIndex = 0; flagIndex < prim.NumSbtRecords; flagIndex++)
                            kindFlags[flagIndex] = 0; // OPTIX_GEOMETRY_FLAG_NONE

                        input.CustomPrimitiveArray.AabbBuffers = new IntPtr(aabbPtrs + primIndex);
                        input.CustomPrimitiveArray.NumPrimitives = (uint)prim.AabbBuffer.Length;
                        input.CustomPrimitiveArray.Stride = 0;
                        input.CustomPrimitiveArray.Flags = kindFlags;
                        input.CustomPrimitiveArray.NumSbtRecords = prim.NumSbtRecords;
                        input.CustomPrimitiveArray.SbtIndexOffsetBuffer = prim.MaterialIds?.NativePtr ?? IntPtr.Zero;
                        input.CustomPrimitiveArray.SbtIndexOffsetSizeInBytes = sizeof(uint);
                        input.CustomPrimitiveArray.SbtIndexOffsetStrideInBytes = 0;

                        buildInputList.Add(input);
                    }
                }

                // Add instances
                if (hasInstances)
                {
                    var (instanceBuffer, numInstances) = instances.Value;
                    var input = new OptixBuildInput
                    {
                        Type = OptixBuildInputType.OPTIX_BUILD_INPUT_TYPE_INSTANCES
                    };

                    input.InstanceArray.Instances = instanceBuffer.NativePtr;
                    input.InstanceArray.NumInstances = numInstances;

                    buildInputList.Add(input);
                }

                // Set up build options
                var buildOptions = new OptixAccelBuildOptions
                {
                    BuildFlags = OptixBuildFlags.OPTIX_BUILD_FLAG_NONE,
                    Operation = OptixBuildOperation.OPTIX_BUILD_OPERATION_BUILD
                };

                if (allowCompaction)
                    buildOptions.BuildFlags |= OptixBuildFlags.OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
                if (preferFastTrace)
                    buildOptions.BuildFlags |= OptixBuildFlags.OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
                if (preferFastBuild)
                    buildOptions.BuildFlags |= OptixBuildFlags.OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
                if (allowUpdate)
                    buildOptions.BuildFlags |= OptixBuildFlags.OPTIX_BUILD_FLAG_ALLOW_UPDATE;

                buildOptions.MotionOptions.NumKeys = 1;

                // Compute memory requirements
                var bufferSizes = deviceContext.AccelComputeMemoryUsage(buildOptions, buildInputList.ToArray());

                // Allocate temp and output buffers
                tempBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.TempSizeInBytes);
                outputBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.OutputSizeInBytes);

                // Prepare emitted properties for compaction size if needed
                OptixAccelEmitDesc[] emittedProps = System.Array.Empty<OptixAccelEmitDesc>();

                if (allowCompaction)
                {
                    // Allocate a device buffer to hold the emitted compaction size
                    compactSizeBuffer = accelerator.Allocate1D<ulong>(1);
                    emittedProps = new[]
                    {
                        new OptixAccelEmitDesc
                        {
                            Type = OptixAccelPropertyType.OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
                            Result = compactSizeBuffer.NativePtr
                        }
                    };
                }

                var buildStream = stream ?? accelerator.DefaultStream;

                // Build acceleration structure
                IntPtr asHandle = deviceContext.AccelBuild(
                    buildStream,
                    buildOptions,
                    buildInputList.ToArray(),
                    tempBuffer.View,
                    outputBuffer.View,
                    emittedProps);

                // Dispose temp buffer after build
                tempBuffer?.Dispose();
                tempBuffer = null;

                if (allowCompaction)
                {
                    // The emitted compacted size is only known once the build above has
                    // actually finished on the device - reading it back requires a sync
                    // point even if the caller supplied their own stream for otherwise-
                    // async use (there's no way to know the compacted size without
                    // waiting for it).
                    accelerator.Synchronize();

                    var compactedSizeHost = new ulong[1];
                    compactSizeBuffer.CopyToCPU(compactedSizeHost);
                    compactSizeBuffer.Dispose();
                    compactSizeBuffer = null;

                    compactedBuffer = accelerator.Allocate1D<byte>((long)Math.Max(1UL, compactedSizeHost[0]));
                    IntPtr compactedHandle = deviceContext.AccelCompact(buildStream, asHandle, compactedBuffer.View);

                    // AccelCompact's copy runs asynchronously on buildStream, same as
                    // AccelBuild - must finish before outputBuffer (the uncompacted
                    // source it's still reading from) is disposed below.
                    accelerator.Synchronize();

                    outputBuffer.Dispose();
                    outputBuffer = null;

                    var compactedResult = new BuiltAccelStructure(compactedHandle, compactedBuffer);
                    compactedBuffer = null; // Result now owns it
                    return compactedResult;
                }

                // Synchronize if stream is null (synchronous build)
                if (stream == null)
                {
                    accelerator.Synchronize();
                }

                // Return result owning the output buffer
                var result = new BuiltAccelStructure(asHandle, outputBuffer);
                outputBuffer = null; // Result now owns it
                return result;
            }
            catch (Exception)
            {
                // Clean up on exception
                tempBuffer?.Dispose();
                outputBuffer?.Dispose();
                compactedBuffer?.Dispose();
                compactSizeBuffer?.Dispose();
                throw;
            }
        }

        /// <summary>
        /// For complex multi-GAS scenarios: builds triangles and custom-primitives as separate GAS structures.
        /// Returns (trianglesGas, customPrimitivesGas) - one or both may be null if not present.
        /// User is responsible for disposing returned structures.
        /// </summary>
        public (BuiltAccelStructure trianglesGas, BuiltAccelStructure customPrimitivesGas) BuildSeparateGeometry(CudaStream stream = null)
        {
            if (deviceContext == null)
                throw new InvalidOperationException("Device context must be set via WithDeviceContext().");
            if (accelerator == null)
                throw new InvalidOperationException("Accelerator must be set via WithAccelerator().");

            BuiltAccelStructure trianglesResult = null;
            BuiltAccelStructure customResult = null;

            try
            {
                // Build triangles separately if present
                if (triangles.Count > 0)
                {
                    var triangleBuilder = new OptixAccelBuilder()
                        .WithDeviceContext(deviceContext)
                        .WithAccelerator(accelerator);

                    foreach (var mesh in triangles)
                        triangleBuilder.AddTriangleMesh(mesh.Vertices, mesh.Indices, mesh.MaterialIds, mesh.NumSbtRecords, mesh.IndexStart, mesh.IndexCount);

                    if (allowUpdate) triangleBuilder.AllowUpdate();
                    if (preferFastTrace) triangleBuilder.PreferFastTrace();
                    if (preferFastBuild) triangleBuilder.PreferFastBuild();
                    if (allowCompaction) triangleBuilder.AllowCompaction();

                    trianglesResult = triangleBuilder.Build(stream);
                }

                // Build custom primitives separately if present
                if (customPrimitives.Count > 0)
                {
                    var customBuilder = new OptixAccelBuilder()
                        .WithDeviceContext(deviceContext)
                        .WithAccelerator(accelerator);

                    foreach (var prim in customPrimitives)
                        customBuilder.AddCustomPrimitives(prim.AabbBuffer, prim.MaterialIds, prim.NumSbtRecords);

                    if (allowUpdate) customBuilder.AllowUpdate();
                    if (preferFastTrace) customBuilder.PreferFastTrace();
                    if (preferFastBuild) customBuilder.PreferFastBuild();
                    if (allowCompaction) customBuilder.AllowCompaction();

                    customResult = customBuilder.Build(stream);
                }

                return (trianglesResult, customResult);
            }
            catch
            {
                trianglesResult?.Dispose();
                customResult?.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Builds an IAS (Instance Acceleration Structure) from multiple GAS handles.
        /// Used after BuildSeparateGeometry() to combine multiple GAS into a single traversable.
        /// </summary>
        /// <param name="gasHandles">The traversable handle of each GAS to instance.</param>
        /// <param name="sbtOffsets">Each instance's hitgroup SBT offset.</param>
        /// <param name="stream">The CUDA stream, or null to build synchronously on the accelerator's default stream.</param>
        /// <param name="transforms">
        /// Each instance's row-major 3x4 object-to-world transform (12 floats), or null
        /// for every instance to use the identity transform.
        /// </param>
        /// <param name="visibilityMasks">
        /// Each instance's visibility mask, or null for every instance to use 255 (visible to all rays).
        /// </param>
        /// <param name="instanceFlags">
        /// Each instance's <c>OptixInstanceFlags</c> bitmask, or null for every instance to use none.
        /// </param>
        public unsafe BuiltAccelStructure BuildInstanceAccelFromHandles(
            IntPtr[] gasHandles,
            uint[] sbtOffsets,
            CudaStream stream = null,
            float[][] transforms = null,
            byte[] visibilityMasks = null,
            uint[] instanceFlags = null)
        {
            if (deviceContext == null)
                throw new InvalidOperationException("Device context must be set via WithDeviceContext().");
            if (accelerator == null)
                throw new InvalidOperationException("Accelerator must be set via WithAccelerator().");
            if (gasHandles == null || gasHandles.Length == 0)
                throw new ArgumentException("At least one GAS handle must be provided.", nameof(gasHandles));
            if (sbtOffsets == null || sbtOffsets.Length != gasHandles.Length)
                throw new ArgumentException("Must provide exactly one SBT offset per GAS handle.", nameof(sbtOffsets));
            if (transforms != null && transforms.Length != gasHandles.Length)
                throw new ArgumentException("Must provide exactly one transform per GAS handle.", nameof(transforms));
            if (visibilityMasks != null && visibilityMasks.Length != gasHandles.Length)
                throw new ArgumentException("Must provide exactly one visibility mask per GAS handle.", nameof(visibilityMasks));
            if (instanceFlags != null && instanceFlags.Length != gasHandles.Length)
                throw new ArgumentException("Must provide exactly one flags value per GAS handle.", nameof(instanceFlags));

            MemoryBuffer1D<byte, Dense> iasBuffer = null;
            MemoryBuffer1D<OptixInstance, Dense> instancesBuffer = null;

            try
            {
                // Create instances for each GAS
                var instances = new OptixInstance[gasHandles.Length];
                for (int i = 0; i < gasHandles.Length; i++)
                {
                    instances[i] = new OptixInstance
                    {
                        TraversableHandle = (ulong)gasHandles[i].ToInt64(),
                        InstanceId = (uint)i,
                        VisibilityMask = visibilityMasks?[i] ?? 255,
                        SbtOffset = sbtOffsets[i],
                        Flags = instanceFlags?[i] ?? 0
                    };

                    if (transforms == null)
                    {
                        for (int j = 0; j < 12; j++)
                            instances[i].Transform[j] = (j % 4 == j / 4) ? 1.0f : 0.0f; // Identity matrix
                    }
                    else
                    {
                        var transform = transforms[i];
                        if (transform == null || transform.Length != 12)
                            throw new ArgumentException(
                                $"Transform at index {i} must have exactly 12 elements (row-major 3x4).",
                                nameof(transforms));
                        for (int j = 0; j < 12; j++)
                            instances[i].Transform[j] = transform[j];
                    }
                }

                instancesBuffer = accelerator.Allocate1D(instances);

                // Build IAS
                var instanceInput = new OptixBuildInput
                {
                    Type = OptixBuildInputType.OPTIX_BUILD_INPUT_TYPE_INSTANCES
                };
                instanceInput.InstanceArray.Instances = instancesBuffer.NativePtr;
                instanceInput.InstanceArray.NumInstances = (uint)instances.Length;
                instanceInput.InstanceArray.InstanceStride = 0;

                var accelOptions = new OptixAccelBuildOptions
                {
                    BuildFlags = OptixBuildFlags.OPTIX_BUILD_FLAG_NONE,
                    Operation = OptixBuildOperation.OPTIX_BUILD_OPERATION_BUILD
                };

                OptixAccelBufferSizes bufferSizes = deviceContext.AccelComputeMemoryUsage(accelOptions, new[] { instanceInput });

                using var tempBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.TempSizeInBytes);
                iasBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.OutputSizeInBytes);

                var iasHandle = deviceContext.AccelBuild(
                    stream ?? accelerator.DefaultStream,
                    accelOptions,
                    new[] { instanceInput },
                    tempBuffer,
                    iasBuffer,
                    ReadOnlySpan<OptixAccelEmitDesc>.Empty);

                if (stream == null)
                    accelerator.Synchronize();

                var result = new BuiltAccelStructure(iasHandle, iasBuffer);
                iasBuffer = null; // Result now owns it
                return result;
            }
            catch
            {
                iasBuffer?.Dispose();
                throw;
            }
            finally
            {
                // instancesBuffer is only needed for the build itself (the returned
                // BuiltAccelStructure owns iasBuffer, not this) - always released here,
                // on both the success and exception paths, so it isn't also disposed
                // above in the catch block.
                instancesBuffer?.Dispose();
            }
        }

        /// <summary>
        /// Refits (OPTIX_BUILD_OPERATION_UPDATE) an existing structure in place, reusing
        /// its output buffer and a persistent update-temp buffer owned by it. The caller
        /// must have re-uploaded fresh contents into the SAME device buffers this builder
        /// was configured with (via AddTriangleMesh/AddCustomPrimitives on THIS instance) -
        /// OptiX's update operation requires the exact same build-input topology/buffer
        /// pointers as the original BUILD, just refreshed data. Requires AllowUpdate() to
        /// have been called on this builder (and on whatever builder originally produced
        /// <paramref name="existing"/> via Build() - the ALLOW_UPDATE flag must have been
        /// set at that original build for OptiX to permit refitting it now).
        /// </summary>
        public unsafe void Refit(BuiltAccelStructure existing, CudaStream stream = null)
        {
            if (deviceContext == null)
                throw new InvalidOperationException("Device context must be set via WithDeviceContext().");
            if (accelerator == null)
                throw new InvalidOperationException("Accelerator must be set via WithAccelerator().");
            if (existing == null)
                throw new ArgumentNullException(nameof(existing));
            if (!allowUpdate)
                throw new InvalidOperationException(
                    "Refit() requires AllowUpdate() to have been called on this builder.");

            bool hasTriangles = triangles.Count > 0;
            bool hasCustomPrimitives = customPrimitives.Count > 0;

            if (!hasTriangles && !hasCustomPrimitives)
                throw new InvalidOperationException(
                    "At least one of AddTriangleMesh() or AddCustomPrimitives() must be called before Refit().");
            if (instances.HasValue)
                throw new InvalidOperationException("Refit() does not support instance (IAS) build inputs.");

            // Deliberately duplicated (not shared with Build()) rather than factored into
            // a helper: the stackalloc'd pointer arrays below must stay alive on THIS
            // stack frame for as long as buildInputList's IntPtr fields point into them -
            // returning them from a helper method would let the stack frame unwind while
            // AccelComputeMemoryUsage/AccelBuild still read those pointers, reproducing
            // the exact dangling-pointer bug documented in "GPU Testing Results" item 3.
            var buildInputList = new List<OptixBuildInput>();

            if (hasTriangles)
            {
                var vertexBufferPtrs = stackalloc IntPtr[triangles.Count];

                for (int meshIndex = 0; meshIndex < triangles.Count; meshIndex++)
                {
                    var mesh = triangles[meshIndex];
                    vertexBufferPtrs[meshIndex] = mesh.Vertices.NativePtr;

                    var input = new OptixBuildInput { Type = OptixBuildInputType.OPTIX_BUILD_INPUT_TYPE_TRIANGLES };

                    input.TriangleArray.VertexFormat = OptixVertexFormat.OPTIX_VERTEX_FORMAT_FLOAT3;
                    input.TriangleArray.VertexStrideInBytes = sizeof(float) * 3;
                    input.TriangleArray.NumVertices = (uint)mesh.Vertices.Length;
                    input.TriangleArray.VertexBuffers = new IntPtr(vertexBufferPtrs + meshIndex);

                    const int vec3iSizeInBytes = sizeof(uint) * 3;
                    input.TriangleArray.IndexFormat = OptixIndicesFormat.OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                    input.TriangleArray.IndexStrideInBytes = vec3iSizeInBytes;
                    input.TriangleArray.NumIndexTriplets = (uint)mesh.IndexCount;
                    input.TriangleArray.IndexBuffer = IntPtr.Add(
                        mesh.Indices.NativePtr, (int)(mesh.IndexStart * vec3iSizeInBytes));

                    var meshFlags = stackalloc uint[(int)mesh.NumSbtRecords];
                    for (int flagIndex = 0; flagIndex < mesh.NumSbtRecords; flagIndex++)
                        meshFlags[flagIndex] = 0;

                    input.TriangleArray.Flags = meshFlags;
                    input.TriangleArray.NumSbtRecords = mesh.NumSbtRecords;
                    input.TriangleArray.SbtIndexOffsetBuffer = mesh.MaterialIds == null
                        ? IntPtr.Zero
                        : IntPtr.Add(mesh.MaterialIds.NativePtr, (int)(mesh.IndexStart * sizeof(uint)));
                    input.TriangleArray.SbtIndexOffsetSizeInBytes = sizeof(uint);
                    input.TriangleArray.SbtIndexOffsetStrideInBytes = 0;

                    buildInputList.Add(input);
                }
            }

            if (hasCustomPrimitives)
            {
                var aabbPtrs = stackalloc IntPtr[customPrimitives.Count];

                for (int primIndex = 0; primIndex < customPrimitives.Count; primIndex++)
                {
                    var prim = customPrimitives[primIndex];
                    aabbPtrs[primIndex] = prim.AabbBuffer.NativePtr;

                    var input = new OptixBuildInput { Type = OptixBuildInputType.OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES };

                    var kindFlags = stackalloc uint[(int)prim.NumSbtRecords];
                    for (int flagIndex = 0; flagIndex < prim.NumSbtRecords; flagIndex++)
                        kindFlags[flagIndex] = 0;

                    input.CustomPrimitiveArray.AabbBuffers = new IntPtr(aabbPtrs + primIndex);
                    input.CustomPrimitiveArray.NumPrimitives = (uint)prim.AabbBuffer.Length;
                    input.CustomPrimitiveArray.Stride = 0;
                    input.CustomPrimitiveArray.Flags = kindFlags;
                    input.CustomPrimitiveArray.NumSbtRecords = prim.NumSbtRecords;
                    input.CustomPrimitiveArray.SbtIndexOffsetBuffer = prim.MaterialIds?.NativePtr ?? IntPtr.Zero;
                    input.CustomPrimitiveArray.SbtIndexOffsetSizeInBytes = sizeof(uint);
                    input.CustomPrimitiveArray.SbtIndexOffsetStrideInBytes = 0;

                    buildInputList.Add(input);
                }
            }

            var buildOptions = new OptixAccelBuildOptions
            {
                BuildFlags = OptixBuildFlags.OPTIX_BUILD_FLAG_ALLOW_UPDATE,
                Operation = OptixBuildOperation.OPTIX_BUILD_OPERATION_UPDATE
            };
            if (preferFastTrace)
                buildOptions.BuildFlags |= OptixBuildFlags.OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
            if (preferFastBuild)
                buildOptions.BuildFlags |= OptixBuildFlags.OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
            buildOptions.MotionOptions.NumKeys = 1;

            var bufferSizes = deviceContext.AccelComputeMemoryUsage(buildOptions, buildInputList.ToArray());

            // Reuse the persistent update-temp buffer if it's already large enough
            // (sized once, reused every subsequent frame - update-temp requirements
            // don't grow beyond what a fixed topology needs, since only buffer contents
            // change between refits, never primitive counts).
            if (existing.UpdateTempBuffer == null ||
                existing.UpdateTempBuffer.LengthInBytes < (long)bufferSizes.TempUpdateSizeInBytes)
            {
                existing.UpdateTempBuffer?.Dispose();
                existing.UpdateTempBuffer = accelerator.Allocate1D<byte>(
                    (long)Math.Max(1UL, bufferSizes.TempUpdateSizeInBytes));
            }

            var buildStream = stream ?? accelerator.DefaultStream;
            IntPtr handle = deviceContext.AccelBuild(
                buildStream,
                buildOptions,
                buildInputList.ToArray(),
                existing.UpdateTempBuffer.View,
                existing.GasBuffer.View,
                ReadOnlySpan<OptixAccelEmitDesc>.Empty);

            if (stream == null)
                accelerator.Synchronize();

            // OptiX guarantees OPTIX_BUILD_OPERATION_UPDATE returns the same handle it
            // was given at the original BUILD, but assign it back defensively rather than
            // assume that's true in every driver version.
            existing.SetTraversableHandle(handle);
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
        }
    }

    /// <summary>
    /// Result of acceleration structure building, owning all GPU buffers.
    /// </summary>
    public sealed class BuiltAccelStructure : IDisposable
    {
        private bool disposed;

        public IntPtr TraversableHandle { get; private set; }
        public MemoryBuffer1D<byte, Dense> GasBuffer { get; }

        /// <summary>
        /// Persistent scratch buffer for OptixAccelBuilder.Refit() calls against this
        /// structure - null until the first Refit(), then reused across every subsequent
        /// one (see Refit()'s own doc comment).
        /// </summary>
        internal MemoryBuffer1D<byte, Dense> UpdateTempBuffer { get; set; }

        internal BuiltAccelStructure(IntPtr handle, MemoryBuffer1D<byte, Dense> gasBuffer)
        {
            TraversableHandle = handle;
            GasBuffer = gasBuffer;
        }

        internal void SetTraversableHandle(IntPtr handle) => TraversableHandle = handle;

        public void Dispose()
        {
            if (!disposed)
            {
                disposed = true;
                UpdateTempBuffer?.Dispose();
                GasBuffer?.Dispose();
                GC.SuppressFinalize(this);
            }
        }
    }
}
