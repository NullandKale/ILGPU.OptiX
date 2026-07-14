// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: RayTracingPipeline.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX.Interop;
using ILGPU.OptiX.Util;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Util;
using ILGPU.OptiX.DeviceApi;
using ILGPU.OptiX.Native;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Owns the whole "programs -> pipeline -> SBT -> launch" lifecycle for one
    /// launch-params type, built by <see cref="OptixRayTracer.CreatePipeline{TLaunchParams}"/>.
    /// Fixes the per-launch GPU-memory leak in <see cref="OptixLaunchExtensions"/> by
    /// construction: the launch-params buffer and the native SBT copy are allocated
    /// once and reused for every <see cref="Launch"/> call, not per call.
    /// Not thread-safe - matches every sample's existing single-threaded-per-frame
    /// render loop; do not call <see cref="Launch"/> or <see cref="SetHitRecords{TMaterial}"/>
    /// from more than one thread concurrently.
    /// </summary>
    [CLSCompliant(false)]
    public sealed class RayTracingPipeline<TLaunchParams> : DisposeBase
        where TLaunchParams : unmanaged
    {
        private readonly OptixDeviceContext deviceContext;
        private readonly OptixKernel raygenKernel;
        private readonly List<BuiltRayType<TLaunchParams>> rayTypes;
        private readonly Dictionary<string, int> rayTypeIndicesByName;

        private MemoryBuffer? raygenBuffer;
        private MemoryBuffer? missBuffer;
        private MemoryBuffer? hitgroupBuffer;
        private MemoryBuffer1D<TLaunchParams, Stride1D.Dense>? launchParamsBuffer;
        private readonly SafeHGlobal sbtHandle;
        private OptixShaderBindingTable sbt;

        /// <summary>
        /// The underlying OptiX pipeline, for power users who need something this
        /// facade does not (yet) expose.
        /// </summary>
        public OptixPipeline Pipeline { get; }

        /// <summary>
        /// The number of ray types declared on this pipeline - the value every
        /// <see cref="OptixTrace.Trace{T}"/> call's <c>sbtStride</c> argument should be.
        /// </summary>
        public int RayTypeCount => rayTypes.Count;

        /// <summary>
        /// The number of hitgroup records currently uploaded by
        /// <see cref="SetHitRecords{TMaterial}(ReadOnlySpan{TMaterial}, int)"/> - 0
        /// until it has been called at least once. Exposed so callers that build their
        /// own acceleration structure can cross-check its NumSbtRecords sum against
        /// this pipeline's SBT, the same invariant a mismatched value here silently
        /// violates at launch time otherwise.
        /// </summary>
        public uint HitgroupRecordCount => sbt.HitgroupRecordCount;

        /// <summary>
        /// How many OptixTasks were executed to compile the raygen module, if
        /// <see cref="RayTracingPipelineBuilder{TLaunchParams}.CompileRaygenWithTasks"/>
        /// was used - 0 if the raygen module was compiled the normal (blocking, single
        /// call) way.
        /// </summary>
        public int RaygenCompileTaskCount { get; }

        internal RayTracingPipeline(
            OptixDeviceContext deviceContext,
            OptixPipeline pipeline,
            OptixKernel raygenKernel,
            List<BuiltRayType<TLaunchParams>> rayTypes,
            int raygenCompileTaskCount = 0)
        {
            this.deviceContext = deviceContext;
            Pipeline = pipeline;
            this.raygenKernel = raygenKernel;
            this.rayTypes = rayTypes;
            rayTypeIndicesByName = rayTypes.ToDictionary(rt => rt.Name, rt => rt.Index);
            RaygenCompileTaskCount = raygenCompileTaskCount;

            sbtHandle = SafeHGlobal.Alloc(Marshal.SizeOf<OptixShaderBindingTable>());
            InitializeRaygenAndMissRecords();
        }

        /// <summary>
        /// Resolves a ray type's SBT index (declaration order in the pipeline
        /// builder) - the value to pass as <c>sbtOffset</c>/<c>missSbtIndex</c> to
        /// <see cref="OptixTrace.Trace{T}"/>.
        /// </summary>
        public int RayTypeIndex(string name) =>
            rayTypeIndicesByName.TryGetValue(name, out var index)
                ? index
                : throw new ArgumentException($"Unknown ray type '{name}'.", nameof(name));

        private void InitializeRaygenAndMissRecords()
        {
            var accelerator = deviceContext.Accelerator;

            var raygenBytes = OptixSbtRecords.Pack<byte>(
                new[] { raygenKernel },
                new byte[] { 0 });
            raygenBuffer = accelerator.Allocate1D(raygenBytes);

            var missKernels = rayTypes.Select(rt => rt.MissKernel).ToArray();
            var missBytes = OptixSbtRecords.Pack<byte>(missKernels, new byte[missKernels.Length]);
            missBuffer = accelerator.Allocate1D(missBytes);

            sbt = new OptixShaderBindingTable
            {
                RaygenRecord = raygenBuffer.NativePtr,
                MissRecordBase = missBuffer.NativePtr,
                MissRecordStrideInBytes = (uint)OptixSbtRecords.StrideOf<byte>(),
                MissRecordCount = (uint)missKernels.Length,
            };
            WriteSbt();
        }

        /// <summary>
        /// Packs and uploads one <see cref="SbtRecord{TMaterial}"/> per material per
        /// ray type, using the same <paramref name="materials"/> data for every ray
        /// type - the common case where every ray type's hit program reads the same
        /// per-material data. Use
        /// <see cref="SetHitRecords{TMaterial}(TMaterial[][], int)"/> instead if
        /// different ray types need different per-material data (e.g. a shadow ray
        /// type whose hit program ignores most fields and should get zeroed records).
        /// </summary>
        /// <param name="materials">One entry per material.</param>
        /// <param name="repeatCount">See the other overload.</param>
        public void SetHitRecords<TMaterial>(ReadOnlySpan<TMaterial> materials, int repeatCount = 1)
            where TMaterial : unmanaged
        {
            var materialsArray = materials.ToArray();
            var perRayType = new TMaterial[rayTypes.Count][];
            for (var r = 0; r < rayTypes.Count; r++)
                perRayType[r] = materialsArray;
            SetHitRecords(perRayType, repeatCount);
        }

        /// <summary>
        /// Packs and uploads one <see cref="SbtRecord{TMaterial}"/> per
        /// <paramref name="entries"/> element per ray type, where each entry names
        /// which of a ray type's (possibly several) declared
        /// <see cref="RayTypeBuilder{TLaunchParams}.HitGroup{TMaterial}(string, Action{TLaunchParams}, Action{TLaunchParams}?, Action{TLaunchParams}?)"/>
        /// program groups backs that record - the overload geometry with multiple
        /// intersection programs per ray type needs (Sample13/14's custom-primitive
        /// pattern: one kind per primitive type, sharing one closest-hit/any-hit pair).
        /// Layout and interleaving match the other overloads exactly - entries in
        /// order, ray types interleaved within each entry, the whole sequence repeated
        /// <paramref name="repeatCount"/> times - so
        /// <c>sbtOffset = entryIndex * RayTypeCount, sbtStride = RayTypeCount</c>
        /// addresses a given entry/ray-type combination, and the caller controls entry
        /// order to match its own GAS build-input / SbtIndexOffsetBuffer layout exactly
        /// (this method has no opinion on acceleration-structure layout - that stays
        /// entirely with <see cref="OptixAccelBuilder"/>, per every other builder in
        /// this library).
        /// </summary>
        /// <param name="entries">
        /// One entry per record-per-entry-slot (in declaration/geometry order); every
        /// <see cref="HitGroupEntry{TMaterial}.Kind"/> must match a
        /// <c>HitGroup(kind, ...)</c> declared on every ray type in this pipeline.
        /// </param>
        /// <param name="repeatCount">See the other overloads.</param>
        public void SetHitRecords<TMaterial>(ReadOnlySpan<HitGroupEntry<TMaterial>> entries, int repeatCount = 1)
            where TMaterial : unmanaged
        {
            foreach (var rayType in rayTypes)
            {
                if (rayType.HitGroupDataType != typeof(TMaterial))
                    throw new ArgumentException(
                        $"Ray type '{rayType.Name}' was declared with HitGroup<{rayType.HitGroupDataType.Name}>, " +
                        $"but SetHitRecords was called with {typeof(TMaterial).Name}. Every ray type on a pipeline " +
                        "must use the same hit-group material type.",
                        nameof(TMaterial));
            }
            if (repeatCount < 0)
                throw new ArgumentOutOfRangeException(nameof(repeatCount));

            var entryCount = entries.Length;
            var blockSize = entryCount * rayTypes.Count;
            var kernels = new OptixKernel[blockSize * repeatCount];
            var data = new TMaterial[blockSize * repeatCount];
            for (var b = 0; b < repeatCount; b++)
            {
                for (var e = 0; e < entryCount; e++)
                {
                    var entry = entries[e];
                    for (var r = 0; r < rayTypes.Count; r++)
                    {
                        if (!rayTypes[r].HitGroupKernelsByKind.TryGetValue(entry.Kind, out var kernel))
                        {
                            throw new ArgumentException(
                                $"Ray type '{rayTypes[r].Name}' has no HitGroup(\"{entry.Kind}\", ...) configured, " +
                                $"but an entry with Kind=\"{entry.Kind}\" was passed to SetHitRecords.",
                                nameof(entries));
                        }

                        var i = (b * blockSize) + (e * rayTypes.Count) + r;
                        kernels[i] = kernel;
                        data[i] = entry.Data;
                    }
                }
            }

            UploadHitRecords(kernels, data);
        }

        /// <summary>
        /// Packs and uploads one <see cref="SbtRecord{TMaterial}"/> per material per
        /// ray type - <c>materialsByRayType[0].Length * RayTypeCount</c> records total,
        /// interleaved as [mat0-rayType0, mat0-rayType1, ..., mat1-rayType0, ...] so
        /// <c>sbtOffset = rayTypeIndex, sbtStride = RayTypeCount</c> (the values
        /// <see cref="OptixTrace.Trace{T}"/> defaults to / <see cref="RayTypeIndex"/>
        /// returns) address the right record. Every ray type's
        /// <see cref="RayTypeBuilder{TLaunchParams}.HitGroup{TMaterial}"/> must have
        /// been declared with this same <typeparamref name="TMaterial"/>.
        /// Synchronizes the accelerator before freeing the previous hitgroup buffer,
        /// so an in-flight launch reading the old table is never invalidated out from
        /// under it.
        /// </summary>
        /// <param name="materialsByRayType">
        /// One array per ray type (in declaration order, length must equal
        /// <see cref="RayTypeCount"/>), each with one entry per material (all arrays
        /// the same length).
        /// </param>
        /// <param name="repeatCount">
        /// Repeats the entire [mat0-rayType0, mat0-rayType1, ...] block this many
        /// times (identical data each time), for scenes built from multiple triangle
        /// GAS build inputs that each need their own copy of the same material table -
        /// OptiX sums <c>NumSbtRecords</c> across build inputs within a GAS to compute
        /// each build input's base SBT-GAS-index, so every build input's own
        /// SbtIndexOffsetBuffer values stay local/0-based against its own repeated
        /// block. 1 (no repetition) is correct for every scene with a single triangle
        /// build input.
        /// </param>
        public void SetHitRecords<TMaterial>(TMaterial[][] materialsByRayType, int repeatCount = 1)
            where TMaterial : unmanaged
        {
            if (materialsByRayType == null)
                throw new ArgumentNullException(nameof(materialsByRayType));
            if (materialsByRayType.Length != rayTypes.Count)
                throw new ArgumentException(
                    $"Expected one materials array per ray type ({rayTypes.Count}), got {materialsByRayType.Length}.",
                    nameof(materialsByRayType));
            if (repeatCount < 0)
                throw new ArgumentOutOfRangeException(nameof(repeatCount));

            foreach (var rayType in rayTypes)
            {
                if (rayType.HitGroupDataType != typeof(TMaterial))
                    throw new ArgumentException(
                        $"Ray type '{rayType.Name}' was declared with HitGroup<{rayType.HitGroupDataType.Name}>, " +
                        $"but SetHitRecords was called with {typeof(TMaterial).Name}. Every ray type on a pipeline " +
                        "must use the same hit-group material type.",
                        nameof(TMaterial));
            }

            var materialCount = materialsByRayType[0].Length;
            for (var r = 1; r < materialsByRayType.Length; r++)
            {
                if (materialsByRayType[r].Length != materialCount)
                    throw new ArgumentException(
                        "Every ray type's materials array must have the same length.",
                        nameof(materialsByRayType));
            }

            var blockSize = materialCount * rayTypes.Count;
            var kernels = new OptixKernel[blockSize * repeatCount];
            var data = new TMaterial[blockSize * repeatCount];
            for (var b = 0; b < repeatCount; b++)
            {
                for (var m = 0; m < materialCount; m++)
                {
                    for (var r = 0; r < rayTypes.Count; r++)
                    {
                        if (!rayTypes[r].HitGroupKernelsByKind.TryGetValue(HitGroupKind.Default, out var kernel))
                        {
                            throw new InvalidOperationException(
                                $"Ray type '{rayTypes[r].Name}' declares multiple named hit groups; use the " +
                                "SetHitRecords overload that takes HitGroupEntry<TMaterial> entries instead.");
                        }

                        var i = (b * blockSize) + (m * rayTypes.Count) + r;
                        kernels[i] = kernel;
                        data[i] = materialsByRayType[r][m];
                    }
                }
            }

            UploadHitRecords(kernels, data);
        }

        private void UploadHitRecords<TMaterial>(OptixKernel[] kernels, TMaterial[] data)
            where TMaterial : unmanaged
        {
            var packedBytes = OptixSbtRecords.Pack<TMaterial>(kernels, data);
            var newBuffer = deviceContext.Accelerator.Allocate1D(packedBytes);

            // Freeing GPU memory the device may still be reading from an in-flight
            // launch is undefined behavior - synchronize first. This runs once per
            // scene/material change, not per frame, so the cost is negligible.
            deviceContext.Accelerator.Synchronize();
            hitgroupBuffer?.Dispose();
            hitgroupBuffer = newBuffer;

            sbt.HitgroupRecordBase = hitgroupBuffer.NativePtr;
            sbt.HitgroupRecordStrideInBytes = (uint)OptixSbtRecords.StrideOf<TMaterial>();
            sbt.HitgroupRecordCount = (uint)kernels.Length;
            WriteSbt();
        }

        private void WriteSbt() =>
            Marshal.StructureToPtr(sbt, sbtHandle.NativePtr, false);

        /// <summary>
        /// Launches the pipeline on <paramref name="stream"/>. Safe to call once per
        /// frame forever - the launch-params buffer is allocated once (on first call)
        /// and reused, unlike <see cref="OptixLaunchExtensions.OptixLaunch"/>'s
        /// per-call allocate/free.
        /// </summary>
        public void Launch(
            CudaStream stream,
            in TLaunchParams launchParams,
            int width,
            int height,
            int depth = 1)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (hitgroupBuffer == null)
                throw new InvalidOperationException("SetHitRecords() must be called at least once before Launch().");

            if (launchParamsBuffer == null)
                launchParamsBuffer = deviceContext.Accelerator.Allocate1D<TLaunchParams>(1);

            var localLaunchParams = launchParams;
            launchParamsBuffer.View.CopyFromCPU(stream, ref localLaunchParams, 1);

            OptixException.ThrowIfFailed(
                OptixAPI.Current.Launch(
                    Pipeline.PipelinePtr,
                    stream.StreamPtr,
                    launchParamsBuffer.NativePtr,
                    (uint)Marshal.SizeOf<TLaunchParams>(),
                    sbtHandle.NativePtr,
                    (uint)width,
                    (uint)height,
                    (uint)depth));
        }

        /// <summary>
        /// Launches the pipeline on the accelerator's default stream.
        /// </summary>
        public void Launch(in TLaunchParams launchParams, int width, int height, int depth = 1) =>
            Launch((CudaStream)deviceContext.Accelerator.DefaultStream, launchParams, width, height, depth);

        /// <summary cref="DisposeBase.Dispose(bool)"/>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                Pipeline.Dispose();
                raygenKernel.Dispose();
                foreach (var rayType in rayTypes)
                {
                    rayType.MissKernel.Dispose();
                    foreach (var hitGroupKernel in rayType.HitGroupKernelsByKind.Values)
                        hitGroupKernel.Dispose();
                }
                raygenBuffer?.Dispose();
                missBuffer?.Dispose();
                hitgroupBuffer?.Dispose();
                launchParamsBuffer?.Dispose();
                sbtHandle.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
