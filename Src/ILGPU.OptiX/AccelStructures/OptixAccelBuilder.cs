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
using ILGPU.OptiX.Pipeline;
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
            public BuiltOpacityMicromapArray OpacityMicromapArray { get; set; }
        }

        private class CustomPrimitive
        {
            public MemoryBuffer AabbBuffer { get; set; }
            public MemoryBuffer MaterialIds { get; set; }
            public uint NumSbtRecords { get; set; }
        }

        private class Curve
        {
            public OptixPrimitiveType CurveType { get; set; }
            public MemoryBuffer Vertices { get; set; }
            public MemoryBuffer Widths { get; set; }
            public MemoryBuffer Indices { get; set; }
            public OptixCurveEndcapFlags EndcapFlags { get; set; }
        }

        private OptixDeviceContext deviceContext;
        private CudaAccelerator accelerator;
        private readonly List<TriangleMesh> triangles = new List<TriangleMesh>();
        private readonly List<CustomPrimitive> customPrimitives = new List<CustomPrimitive>();
        private readonly List<Curve> curves = new List<Curve>();
        private (MemoryBuffer buffer, uint count)? instances;
        private bool allowCompaction;
        private bool preferFastTrace;
        private bool preferFastBuild;
        private bool allowUpdate;
        private bool emitAabbs;

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
        /// <param name="opacityMicromapArray">
        /// An opacity micromap array (<see cref="OptixOpacityMicromapBuilder.BuildUniformStateArray"/>) to attach
        /// to this mesh in LINEAR indexing mode (triangle[i] uses the array's i-th
        /// micromap) - lets the driver skip any-hit invocations on triangles marked
        /// fully transparent/opaque. Must have exactly one entry per triangle in this
        /// mesh (<paramref name="indexCount"/>) if supplied. Null (default) attaches no
        /// opacity micromap - ordinary opaque triangle behavior.
        /// </param>
        public OptixAccelBuilder AddTriangleMesh(MemoryBuffer vertices, MemoryBuffer indices,
            MemoryBuffer materialIds = null, uint numSbtRecords = 1,
            long indexStart = 0, long indexCount = -1,
            BuiltOpacityMicromapArray opacityMicromapArray = null)
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
                IndexCount = resolvedIndexCount,
                OpacityMicromapArray = opacityMicromapArray
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
        /// Adds a kind of curve ("strand") geometry as one build input within the
        /// resulting GAS - call multiple times to
        /// combine several curve kinds, same rule as <see cref="AddCustomPrimitives"/>.
        /// Curves use OptiX's own built-in intersection program (there is no
        /// user-suppliable one) - see <see cref="Pipeline.RayTracingPipelineBuilder{TLaunchParams}.HitGroupWithBuiltinIS"/>
        /// (via <c>RayTypeBuilder.HitGroupWithBuiltinIS</c>) to wire the matching
        /// hitgroup. Unlike triangles/custom-primitives, a curves build input always
        /// has exactly one SBT record (OptiX's own limitation - "each curves build
        /// input has only one SBT record"), so combine multiple materials via
        /// multiple <see cref="AddCurves"/> calls (one build input per kind/material),
        /// not a per-primitive material-ID buffer.
        /// </summary>
        /// <param name="curveType">
        /// The curve basis/degree (must be one of the Round*/Flat* types, not
        /// <see cref="OptixPrimitiveType.Custom"/>, <see cref="OptixPrimitiveType.Triangle"/>,
        /// or <see cref="OptixPrimitiveType.Sphere"/>).
        /// </param>
        /// <param name="vertices">Curve control-point positions (one per vertex, shared across all segments).</param>
        /// <param name="widths">Per-vertex curve radius - same length as <paramref name="vertices"/>.</param>
        /// <param name="indices">
        /// One entry per curve segment: the index into <paramref name="vertices"/>/<paramref name="widths"/>
        /// of that segment's first of (degree+1) consecutive control points.
        /// </param>
        /// <param name="endcapFlags">Curve end cap flags - default has no end caps for quadratic/cubic curves.</param>
        public OptixAccelBuilder AddCurves(
            OptixPrimitiveType curveType,
            MemoryBuffer vertices,
            MemoryBuffer widths,
            MemoryBuffer indices,
            OptixCurveEndcapFlags endcapFlags = OptixCurveEndcapFlags.Default)
        {
            if (curveType == OptixPrimitiveType.Custom || curveType == OptixPrimitiveType.Triangle)
                throw new ArgumentException(
                    "curveType must be one of the Round*/Flat* curve types.", nameof(curveType));
            if (vertices == null || vertices.Length == 0)
                throw new ArgumentException("Vertices buffer must not be null or empty.", nameof(vertices));
            if (widths == null || widths.Length != vertices.Length)
                throw new ArgumentException("Widths buffer must have the same length as vertices.", nameof(widths));
            if (indices == null || indices.Length == 0)
                throw new ArgumentException("Indices buffer must not be null or empty.", nameof(indices));

            curves.Add(new Curve
            {
                CurveType = curveType,
                Vertices = vertices,
                Widths = widths,
                Indices = indices,
                EndcapFlags = endcapFlags
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
        /// Requests that the built acceleration structure's world-space AABB(s) be
        /// emitted and read back to the host, available afterwards via
        /// <see cref="BuiltAccelStructure.Aabbs"/>. Forces a device synchronization
        /// after the build (like AllowCompaction() already does) to read the result
        /// back immediately.
        /// </summary>
        public OptixAccelBuilder EmitAabbs()
        {
            emitAabbs = true;
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
            bool hasCurves = curves.Count > 0;
            bool hasInstances = instances.HasValue;

            if (!hasTriangles && !hasCustomPrimitives && !hasCurves && !hasInstances)
                throw new InvalidOperationException(
                    "At least one of AddTriangleMesh(), AddCustomPrimitives(), AddCurves(), or AddInstances() must be called.");

            // Cannot mix GAS and IAS types
            int gasCount = (hasTriangles ? 1 : 0) + (hasCustomPrimitives ? 1 : 0) + (hasCurves ? 1 : 0);
            if (gasCount > 0 && hasInstances)
                throw new InvalidOperationException(
                    "Cannot mix geometry (triangles/custom-primitives/curves) with instances; choose one type.");

            MemoryBuffer1D<byte, Dense> tempBuffer = null;
            MemoryBuffer1D<byte, Dense> outputBuffer = null;
            MemoryBuffer1D<byte, Dense> compactedBuffer = null;
            MemoryBuffer1D<ulong, Dense> compactSizeBuffer = null;
            MemoryBuffer1D<OptixAabb, Dense> aabbsBuffer = null;

            try
            {
                // Build the list of OptixBuildInput structures
                var buildInputList = new List<OptixBuildInput>();

                // Parallel to buildInputList (same order, same count) - one relocation
                // descriptor per build input, needed later if the caller relocates this
                // structure via BuiltAccelStructure.Relocate().
                var relocateInputList = new List<OptixRelocateInput>();

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
                            Type = OptixBuildInputType.Triangles
                        };

                        input.TriangleArray.VertexFormat = OptixVertexFormat.Float3;
                        input.TriangleArray.VertexStrideInBytes = sizeof(float) * 3;  // Vec3 = 3 floats
                        input.TriangleArray.NumVertices = (uint)mesh.Vertices.Length;
                        input.TriangleArray.VertexBuffers = new IntPtr(vertexBufferPtrs + meshIndex);

                        // Indices/materialIds are byte-offset by IndexStart (a triangle
                        // count) so multiple meshes can share one big index/materialIds
                        // buffer as separate build inputs, each covering only its own
                        // sub-range - IntPtr.Add with a 0 offset is a no-op, so this is
                        // exactly the old whole-buffer behavior when IndexStart is 0.
                        const int vec3iSizeInBytes = sizeof(uint) * 3;
                        input.TriangleArray.IndexFormat = OptixIndicesFormat.UnsignedInt3;
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

                        if (mesh.OpacityMicromapArray != null)
                        {
                            input.TriangleArray.OpacityMicromap = new OptixBuildInputOpacityMicromap
                            {
                                IndexingMode = OptixOpacityMicromapArrayIndexingMode.Linear,
                                OpacityMicromapArray = mesh.OpacityMicromapArray.ArrayDevicePointer,
                                NumMicromapUsageCounts = mesh.OpacityMicromapArray.NumUsageCounts,
                                MicromapUsageCounts = mesh.OpacityMicromapArray.UsageCountsPtr
                            };
                        }

                        buildInputList.Add(input);
                        relocateInputList.Add(new OptixRelocateInput
                        {
                            Type = OptixBuildInputType.Triangles,
                            TriangleArray = new OptixRelocateInputTriangleArray
                            {
                                NumSbtRecords = mesh.NumSbtRecords
                            }
                        });
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
                            Type = OptixBuildInputType.CustomPrimitives
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
                        // Custom-primitive inputs require no relocation data beyond the
                        // Type tag (see OptixRelocateInput's doc comment).
                        relocateInputList.Add(new OptixRelocateInput
                        {
                            Type = OptixBuildInputType.CustomPrimitives
                        });
                    }
                }

                // Add curves
                if (hasCurves)
                {
                    // One pointer slot per curve kind for each of vertex/width buffers
                    // - same "must outlive this whole method" reasoning as the
                    // triangle/custom-primitive loops above. Curves have no per-motion-
                    // key array here since this builder only builds single-key (static)
                    // curve geometry - each slot is a 1-element "array of device
                    // pointers" (vertexBuffers/widthBuffers are themselves arrays of
                    // per-motion-key pointers per optix_types.h, size 1 for no motion).
                    var vertexPtrs = stackalloc IntPtr[curves.Count];
                    var widthPtrs = stackalloc IntPtr[curves.Count];

                    for (int curveIndex = 0; curveIndex < curves.Count; curveIndex++)
                    {
                        var curve = curves[curveIndex];
                        vertexPtrs[curveIndex] = curve.Vertices.NativePtr;
                        widthPtrs[curveIndex] = curve.Widths.NativePtr;

                        var input = new OptixBuildInput
                        {
                            Type = OptixBuildInputType.Curves
                        };

                        input.CurveArray.CurveType = curve.CurveType;
                        input.CurveArray.NumPrimitives = (uint)curve.Indices.Length;
                        input.CurveArray.VertexBuffers = new IntPtr(vertexPtrs + curveIndex);
                        input.CurveArray.NumVertices = (uint)curve.Vertices.Length;
                        input.CurveArray.VertexStrideInBytes = 0;
                        input.CurveArray.WidthBuffers = new IntPtr(widthPtrs + curveIndex);
                        input.CurveArray.WidthStrideInBytes = 0;
                        input.CurveArray.NormalBuffers = IntPtr.Zero;
                        input.CurveArray.NormalStrideInBytes = 0;
                        input.CurveArray.IndexBuffer = curve.Indices.NativePtr;
                        input.CurveArray.IndexStrideInBytes = 0;
                        input.CurveArray.Flag = 0; // OPTIX_GEOMETRY_FLAG_NONE
                        input.CurveArray.PrimitiveIndexOffset = 0;
                        input.CurveArray.EndcapFlags = (uint)curve.EndcapFlags;

                        buildInputList.Add(input);
                        // Curves require no relocation data beyond the Type tag (same
                        // as custom primitives - see OptixRelocateInput's doc comment).
                        relocateInputList.Add(new OptixRelocateInput
                        {
                            Type = OptixBuildInputType.Curves
                        });
                    }
                }

                // Add instances
                if (hasInstances)
                {
                    var (instanceBuffer, numInstances) = instances.Value;
                    var input = new OptixBuildInput
                    {
                        Type = OptixBuildInputType.Instances
                    };

                    input.InstanceArray.Instances = instanceBuffer.NativePtr;
                    input.InstanceArray.NumInstances = numInstances;

                    buildInputList.Add(input);
                    relocateInputList.Add(new OptixRelocateInput
                    {
                        Type = OptixBuildInputType.Instances,
                        InstanceArray = new OptixRelocateInputInstanceArray
                        {
                            NumInstances = numInstances,
                            TraversableHandles = IntPtr.Zero
                        }
                    });
                }

                // Set up build options
                var buildOptions = new OptixAccelBuildOptions
                {
                    BuildFlags = OptixBuildFlags.None,
                    Operation = OptixBuildOperation.Build
                };

                if (allowCompaction)
                    buildOptions.BuildFlags |= OptixBuildFlags.AllowCompaction;
                if (preferFastTrace)
                    buildOptions.BuildFlags |= OptixBuildFlags.PreferFastTrace;
                if (preferFastBuild)
                    buildOptions.BuildFlags |= OptixBuildFlags.PreferFastBuild;
                if (allowUpdate)
                    buildOptions.BuildFlags |= OptixBuildFlags.AllowUpdate;

                buildOptions.MotionOptions.NumKeys = 1;

                // Compute memory requirements
                var bufferSizes = deviceContext.AccelComputeMemoryUsage(buildOptions, buildInputList.ToArray());

                // Allocate temp and output buffers
                tempBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.TempSizeInBytes);
                outputBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.OutputSizeInBytes);

                // Prepare emitted properties for compaction size / AABBs if requested
                var emittedPropsList = new List<OptixAccelEmitDesc>();

                if (allowCompaction)
                {
                    // Allocate a device buffer to hold the emitted compaction size
                    compactSizeBuffer = accelerator.Allocate1D<ulong>(1);
                    emittedPropsList.Add(new OptixAccelEmitDesc
                    {
                        Type = OptixAccelPropertyType.CompactedSize,
                        Result = compactSizeBuffer.NativePtr
                    });
                }

                if (emitAabbs)
                {
                    // One OptixAabb per motion key (always 1 here - see MotionOptions.NumKeys
                    // below; this is not yet configurable).
                    aabbsBuffer = accelerator.Allocate1D<OptixAabb>(1);
                    emittedPropsList.Add(new OptixAccelEmitDesc
                    {
                        Type = OptixAccelPropertyType.Aabbs,
                        Result = aabbsBuffer.NativePtr
                    });
                }

                var emittedProps = emittedPropsList.ToArray();

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

                // Emitted AABBs are only known once the build above has actually
                // finished on the device - reading them back requires a sync point
                // even if the caller supplied their own stream (same reasoning as the
                // compaction-size readback below).
                OptixAabb[] aabbsHost = null;
                if (emitAabbs)
                {
                    accelerator.Synchronize();
                    aabbsHost = new OptixAabb[1];
                    aabbsBuffer.CopyToCPU(aabbsHost);
                    aabbsBuffer.Dispose();
                    aabbsBuffer = null;
                }

                var relocateInputs = relocateInputList.ToArray();

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

                    var compactedResult = new BuiltAccelStructure(compactedHandle, compactedBuffer)
                    {
                        RelocateInputs = relocateInputs,
                        Aabbs = aabbsHost
                    };
                    compactedBuffer = null; // Result now owns it
                    return compactedResult;
                }

                // Synchronize if stream is null (synchronous build)
                if (stream == null)
                {
                    accelerator.Synchronize();
                }

                // Return result owning the output buffer
                var result = new BuiltAccelStructure(asHandle, outputBuffer)
                {
                    RelocateInputs = relocateInputs,
                    Aabbs = aabbsHost
                };
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
                aabbsBuffer?.Dispose();
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
                    Type = OptixBuildInputType.Instances
                };
                instanceInput.InstanceArray.Instances = instancesBuffer.NativePtr;
                instanceInput.InstanceArray.NumInstances = (uint)instances.Length;
                instanceInput.InstanceArray.InstanceStride = 0;

                var accelOptions = new OptixAccelBuildOptions
                {
                    BuildFlags = OptixBuildFlags.None,
                    Operation = OptixBuildOperation.Build
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
        /// Refits (OptixBuildOperation.Update) an existing structure in place, reusing
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

                    var input = new OptixBuildInput { Type = OptixBuildInputType.Triangles };

                    input.TriangleArray.VertexFormat = OptixVertexFormat.Float3;
                    input.TriangleArray.VertexStrideInBytes = sizeof(float) * 3;
                    input.TriangleArray.NumVertices = (uint)mesh.Vertices.Length;
                    input.TriangleArray.VertexBuffers = new IntPtr(vertexBufferPtrs + meshIndex);

                    const int vec3iSizeInBytes = sizeof(uint) * 3;
                    input.TriangleArray.IndexFormat = OptixIndicesFormat.UnsignedInt3;
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

                    var input = new OptixBuildInput { Type = OptixBuildInputType.CustomPrimitives };

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
                BuildFlags = OptixBuildFlags.AllowUpdate,
                Operation = OptixBuildOperation.Update
            };
            if (preferFastTrace)
                buildOptions.BuildFlags |= OptixBuildFlags.PreferFastTrace;
            if (preferFastBuild)
                buildOptions.BuildFlags |= OptixBuildFlags.PreferFastBuild;
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

            // OptiX guarantees OptixBuildOperation.Update returns the same handle it
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

        /// <summary>
        /// World-space AABB(s) emitted during the build, one per motion key, or null if
        /// the builder's EmitAabbs() was not called.
        /// </summary>
        [CLSCompliant(false)]
        public OptixAabb[] Aabbs { get; internal set; }

        /// <summary>
        /// One relocation descriptor per build input this structure was originally built
        /// with, in build order - used by Relocate() to populate optixAccelRelocate's
        /// relocateInputs parameter. Set by OptixAccelBuilder.Build().
        /// </summary>
        [CLSCompliant(false)]
        internal OptixRelocateInput[] RelocateInputs { get; set; }

        internal BuiltAccelStructure(IntPtr handle, MemoryBuffer1D<byte, Dense> gasBuffer)
        {
            TraversableHandle = handle;
            GasBuffer = gasBuffer;
        }

        internal void SetTraversableHandle(IntPtr handle) => TraversableHandle = handle;

        /// <summary>
        /// Relocates this acceleration structure into a freshly allocated target buffer
        /// on the given device context, returning a new <see cref="BuiltAccelStructure"/>
        /// that owns it. This structure is left unchanged (and still usable/disposable)
        /// afterwards. Throws if OptiX reports the source structure is not relocation-
        /// compatible with <paramref name="deviceContext"/> (mismatched compile/ABI
        /// options between the two - relocating within the same process/device, as in a
        /// round-trip test, is always compatible).
        /// </summary>
        /// <param name="deviceContext">The OptiX device context to relocate on.</param>
        /// <param name="accelerator">The accelerator to allocate the target buffer from.</param>
        /// <param name="stream">The CUDA stream, or null to relocate synchronously.</param>
        [CLSCompliant(false)]
        public BuiltAccelStructure Relocate(
            ILGPU.OptiX.Pipeline.OptixDeviceContext deviceContext,
            CudaAccelerator accelerator,
            CudaStream stream = null)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (accelerator == null)
                throw new ArgumentNullException(nameof(accelerator));
            if (RelocateInputs == null || RelocateInputs.Length == 0)
                throw new InvalidOperationException(
                    "This BuiltAccelStructure has no relocation info - it must have come " +
                    "from OptixAccelBuilder.Build().");

            var info = deviceContext.AccelGetRelocationInfo(TraversableHandle);
            if (!deviceContext.CheckRelocationCompatibility(info))
                throw new InvalidOperationException(
                    "This acceleration structure is not relocation-compatible with the " +
                    "given device context (mismatched device/driver/compile options).");

            MemoryBuffer1D<byte, Dense> targetBuffer = null;
            try
            {
                targetBuffer = accelerator.Allocate1D<byte>(GasBuffer.LengthInBytes);

                var relocateStream = stream ?? accelerator.DefaultStream;

                // optixAccelRelocate does NOT copy the acceleration structure's memory -
                // per optix_host.h: "This function only operates on the relocated memory
                // whose new location is specified by 'targetAccel'... The original memory
                // (source) is not required to be valid, only the OptixRelocationInfo." The
                // caller is responsible for getting the bytes into targetAccel first (a
                // device-to-device copy here, since both buffers live on the same
                // accelerator - a real cross-device/cross-process relocation would copy via
                // host memory or P2P instead). Skipping this step leaves targetBuffer as
                // uninitialized device memory, which optixAccelRelocate then reinterprets as
                // a traversable - undefined behavior that manifested as a sticky CUDA
                // "unspecified launch failure" the first time this was tried.
                GasBuffer.CopyTo(relocateStream, 0, targetBuffer.View);

                var handle = deviceContext.AccelRelocate(
                    relocateStream, info, RelocateInputs, targetBuffer.View);

                if (stream == null)
                    accelerator.Synchronize();

                var result = new BuiltAccelStructure(handle, targetBuffer)
                {
                    RelocateInputs = RelocateInputs
                };
                targetBuffer = null; // Result now owns it
                return result;
            }
            catch
            {
                // Once a CUDA context faults, further calls on it (including Dispose's
                // own cudaFree) also throw - swallow a secondary failure here so the
                // original exception (the actually useful one) isn't masked by it.
                try { targetBuffer?.Dispose(); } catch { /* see comment above */ }
                throw;
            }
        }

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
