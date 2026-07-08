// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixAPI.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX.Interop;
using ILGPU.OptiX.Pipeline;
using ILGPU.OptiX.Util;
using ILGPU.Util;
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

// disable: max_line_length
#pragma warning disable CA1707 // Identifiers should not contain underscores

namespace ILGPU.OptiX
{
    /// <summary>
    /// Wrapper for the OptiX library functions.
    /// </summary>
    public sealed partial class OptixAPI : DisposeBase
    {
        #region Static

        public const int OPTIX_SBT_RECORD_ALIGNMENT = 16;
        public const int OPTIX_SBT_RECORD_HEADER_SIZE = 32;

        private const int DEFAULT_LOG_SIZE = 2048;

        /// <summary>
        /// The current instance of the API.
        /// </summary>
        public static OptixAPI Current { get; } = new OptixAPI();

        #endregion

        #region Instance

        /// <summary>
        /// The OptiX DLL handle.
        /// </summary>
        private IntPtr hmodule = IntPtr.Zero;

        /// <summary>
        /// The OptiX function table.
        /// </summary>
        private OptixFunctionTable functionTable;

        #endregion

        #region Delegate cache

        // Each native entry point below is resolved into a delegate once and reused,
        // instead of calling Marshal.GetDelegateForFunctionPointer on every single
        // invocation (delegate creation is not free, and several of these - notably
        // the denoiser calls - run once per rendered frame in a real render loop, not
        // just at setup time). Cleared by ClearDelegateCache (called from Uninit())
        // since the underlying function pointers in functionTable are only valid
        // between a matching Init()/Uninit() pair - a stale cached delegate surviving
        // past Uninit() would call into a pointer the just-freed nvoptix.dll no longer
        // backs.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T GetOrCreateDelegate<T>(ref T? cache, IntPtr functionPointer)
            where T : Delegate =>
            cache ??= Marshal.GetDelegateForFunctionPointer<T>(functionPointer);

        // On return, OptiX's log-size out-parameter reports the full length of the log
        // message, INCLUDING when that message was truncated to fit the buffer - it is
        // not clamped to the buffer capacity the caller supplied on input. Reading
        // reportedLength bytes straight out of a fixed-size buffer would read past the
        // end of that buffer whenever the driver's message is longer than
        // DEFAULT_LOG_SIZE, so this always clamps to the buffer's actual capacity.
        private static unsafe string ReadLog(byte* logPtr, ulong reportedLength, int bufferCapacity)
        {
            int length = (int)Math.Min(reportedLength, (ulong)bufferCapacity);
            return Encoding.UTF8.GetString(logPtr, length);
        }

        private DeviceContextCreate? cachedDeviceContextCreate;
        private DeviceContextDestroy? cachedDeviceContextDestroy;
        private ModuleCreate? cachedModuleCreate;
        private ModuleDestroy? cachedModuleDestroy;
        private ProgramGroupCreate? cachedProgramGroupCreate;
        private ProgramGroupDestroy? cachedProgramGroupDestroy;
        private ProgramGroupGetStackSize? cachedProgramGroupGetStackSize;
        private PipelineCreate? cachedPipelineCreate;
        private PipelineDestroy? cachedPipelineDestroy;
        private PipelineSetStackSize? cachedPipelineSetStackSize;
        private SbtRecordPackHeader? cachedSbtRecordPackHeader;
        private Launch? cachedLaunch;
        private AccelComputeMemoryUsage? cachedAccelComputeMemoryUsage;
        private AccelBuild? cachedAccelBuild;
        private AccelCompact? cachedAccelCompact;
        private DenoiserCreate? cachedDenoiserCreate;
        private DenoiserDestroy? cachedDenoiserDestroy;
        private DenoiserComputeMemoryResources? cachedDenoiserComputeMemoryResources;
        private DenoiserSetup? cachedDenoiserSetup;
        private DenoiserInvoke? cachedDenoiserInvoke;
        private DenoiserComputeIntensity? cachedDenoiserComputeIntensity;
        private DenoiserComputeAverageColor? cachedDenoiserComputeAverageColor;

        private void ClearDelegateCache()
        {
            cachedDeviceContextCreate = null;
            cachedDeviceContextDestroy = null;
            cachedModuleCreate = null;
            cachedModuleDestroy = null;
            cachedProgramGroupCreate = null;
            cachedProgramGroupDestroy = null;
            cachedProgramGroupGetStackSize = null;
            cachedPipelineCreate = null;
            cachedPipelineDestroy = null;
            cachedPipelineSetStackSize = null;
            cachedSbtRecordPackHeader = null;
            cachedLaunch = null;
            cachedAccelComputeMemoryUsage = null;
            cachedAccelBuild = null;
            cachedAccelCompact = null;
            cachedDenoiserCreate = null;
            cachedDenoiserDestroy = null;
            cachedDenoiserComputeMemoryResources = null;
            cachedDenoiserSetup = null;
            cachedDenoiserInvoke = null;
            cachedDenoiserComputeIntensity = null;
            cachedDenoiserComputeAverageColor = null;
        }

        #endregion

        #region Methods

        /// <summary>
        /// Creates a new OptiX device context.
        /// </summary>
        /// <param name="cudaContext">The CUDA context.</param>
        /// <param name="options">The OptiX device context options.</param>
        /// <param name="deviceContext">Filled in with the new device context.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult DeviceContextCreate(
            IntPtr cudaContext,
            OptixDeviceContextOptions options,
            out IntPtr deviceContext)
        {
            var func = GetOrCreateDelegate(ref cachedDeviceContextCreate, functionTable.OptixDeviceContextCreate);
            return func(cudaContext, options, out deviceContext);
        }

        /// <summary>
        /// Destroys the OptiX device context.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <returns>The OptiX result.</returns>
        public OptixResult DeviceContextDestroy(IntPtr deviceContext)
        {
            var func = GetOrCreateDelegate(ref cachedDeviceContextDestroy, functionTable.OptixDeviceContextDestroy);
            return func(deviceContext);
        }

        /// <summary>
        /// Creates a new OptiX module.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="moduleCompileOptions">The module compile options.</param>
        /// <param name="pipelineCompileOptions">The pipeline compile options.</param>
        /// <param name="ptxString">The module PTX code.</param>
        /// <param name="module">Filled in with the new module.</param>
        /// <param name="logString">Filled in with the log string.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public unsafe OptixResult ModuleCreate(
            IntPtr deviceContext,
            OptixModuleCompileOptions moduleCompileOptions,
            OptixPipelineCompileOptions pipelineCompileOptions,
            string ptxString,
            out IntPtr module,
            out string logString)
        {
            var func = GetOrCreateDelegate(ref cachedModuleCreate, functionTable.OptixModuleCreate);

            using var moduleCompileOptionsPtr = SafeHGlobal.AllocFrom(moduleCompileOptions);
            using var pipelineCompileOptionsPtr = SafeHGlobal.AllocFrom(pipelineCompileOptions);

            var ptxStringBytes = Encoding.UTF8.GetBytes(ptxString);
            var logBytes = new byte[DEFAULT_LOG_SIZE];
            fixed (byte* ptxStringPtr = ptxStringBytes)
            fixed (byte* logPtr = logBytes)
            {
                ulong logLength = (ulong)logBytes.Length;
                var result = func(
                    deviceContext,
                    moduleCompileOptionsPtr,
                    pipelineCompileOptionsPtr,
                    new IntPtr(ptxStringPtr),
                    (ulong)ptxStringBytes.Length,
                    new IntPtr(logPtr),
                    ref logLength,
                    out module
                );
                if (result != OptixResult.OPTIX_SUCCESS)
                {
                    logString = ReadLog(logPtr, logLength, logBytes.Length);
                }
                else
                {
                    logString = string.Empty;
                }
                return result;
            }
        }

        /// <summary>
        /// Destroys the OptiX module.
        /// </summary>
        /// <param name="module">The OptiX module.</param>
        /// <returns>The OptiX result.</returns>
        public OptixResult ModuleDestroy(IntPtr module)
        {
            var func = GetOrCreateDelegate(ref cachedModuleDestroy, functionTable.OptixModuleDestroy);
            return func(module);
        }

        /// <summary>
        /// Creates a new OptiX program group.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="programDescriptions">The program group descriptions.</param>
        /// <param name="programGroupOptions">The program group options.</param>
        /// <param name="programGroup">Filled in with the new program group.</param>
        /// <param name="logString">Filled in with the log string.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public unsafe OptixResult ProgramGroupCreate(
            IntPtr deviceContext,
            ReadOnlySpan<OptixProgramGroupDesc> programDescriptions,
            OptixProgramGroupOptions programGroupOptions,
            out IntPtr programGroup,
            out string logString)
        {
            var func = GetOrCreateDelegate(ref cachedProgramGroupCreate, functionTable.OptixProgramGroupCreate);

            using var programGroupOptionsPtr = SafeHGlobal.AllocFrom(programGroupOptions);
            using var programDescriptionsPtr = SafeHGlobal.AllocFrom(programDescriptions);

            var logBytes = new byte[DEFAULT_LOG_SIZE];
            fixed (byte* logPtr = logBytes)
            {
                ulong logLength = (ulong)logBytes.Length;
                var result = func(
                    deviceContext,
                    programDescriptionsPtr,
                    (uint)programDescriptions.Length,
                    programGroupOptionsPtr,
                    new IntPtr(logPtr),
                    ref logLength,
                    out programGroup
                );
                if (result != OptixResult.OPTIX_SUCCESS)
                {
                    logString = ReadLog(logPtr, logLength, logBytes.Length);
                }
                else
                {
                    logString = string.Empty;
                }
                return result;
            }
        }

        /// <summary>
        /// Destroys the OptiX program group.
        /// </summary>
        /// <param name="programGroup">The OptiX program group.</param>
        /// <returns>The OptiX result.</returns>
        public OptixResult ProgramGroupDestroy(IntPtr programGroup)
        {
            var func = GetOrCreateDelegate(ref cachedProgramGroupDestroy, functionTable.OptixProgramGroupDestroy);
            return func(programGroup);
        }

        /// <summary>
        /// Queries a program group's stack size requirements, feeding
        /// <see cref="OptixStackSizeUtil.ComputeStackSizes"/> so pipeline stack sizes
        /// can be computed instead of guessed.
        /// </summary>
        /// <param name="programGroup">The OptiX program group.</param>
        /// <param name="stackSizes">The program group's stack size requirements.</param>
        /// <param name="pipeline">
        /// An optional pipeline to additionally account for external function calls;
        /// pass <see cref="IntPtr.Zero"/> if not applicable.
        /// </param>
        [CLSCompliant(false)]
        public OptixResult ProgramGroupGetStackSize(
            IntPtr programGroup,
            out OptixStackSizes stackSizes,
            IntPtr pipeline)
        {
            var func = GetOrCreateDelegate(ref cachedProgramGroupGetStackSize, functionTable.OptixProgramGroupGetStackSize);
            return func(programGroup, out stackSizes, pipeline);
        }

        /// <summary>
        /// Creates a new OptiX pipeline.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="pipelineCompileOptions">The pipeline compile options.</param>
        /// <param name="pipelineLinkOptions">The pipeline link options.</param>
        /// <param name="programGroups">The program groups.</param>
        /// <param name="numProgramGroups">The number of program groups.</param>
        /// <param name="pipeline">Filled in with the new pipeline.</param>
        /// <param name="logString">Filled in with the log string.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public unsafe OptixResult PipelineCreate(
            IntPtr deviceContext,
            OptixPipelineCompileOptions pipelineCompileOptions,
            OptixPipelineLinkOptions pipelineLinkOptions,
            IntPtr programGroups,
            uint numProgramGroups,
            out IntPtr pipeline,
            out string logString)
        {
            var func = GetOrCreateDelegate(ref cachedPipelineCreate, functionTable.OptixPipelineCreate);

            using var pipelineCompileOptionsPtr = SafeHGlobal.AllocFrom(pipelineCompileOptions);
            using var pipelineLinkOptionsPtr = SafeHGlobal.AllocFrom(pipelineLinkOptions);

            var logBytes = new byte[DEFAULT_LOG_SIZE];
            fixed (byte* logPtr = logBytes)
            {
                ulong logLength = (ulong)logBytes.Length;
                var result = func(
                    deviceContext,
                    pipelineCompileOptionsPtr,
                    pipelineLinkOptionsPtr,
                    programGroups,
                    numProgramGroups,
                    new IntPtr(logPtr),
                    ref logLength,
                    out pipeline
                );
                if (result != OptixResult.OPTIX_SUCCESS)
                {
                    logString = ReadLog(logPtr, logLength, logBytes.Length);
                }
                else
                {
                    logString = string.Empty;
                }
                return result;
            }
        }

        /// <summary>
        /// Destroys the OptiX program group.
        /// </summary>
        /// <param name="pipeline">The OptiX pipeline.</param>
        /// <returns>The OptiX result.</returns>
        public OptixResult PipelineDestroy(IntPtr pipeline)
        {
            var func = GetOrCreateDelegate(ref cachedPipelineDestroy, functionTable.OptixPipelineDestroy);
            return func(pipeline);
        }

        /// <summary>
        /// Configures the pipeline stack size.
        /// </summary>
        /// <param name="pipeline">The OptiX pipeline.</param>
        /// <param name="directCallableStackSizeFromTraversal">
        /// The direct stack size requirement for direct callables invoked from IS or AH.
        /// </param>
        /// <param name="directCallableStackSizeFromState">
        /// The direct stack size requirement for direct callables invoked from RG, MS,
        /// or CH.</param>
        /// <param name="continuationStackSize">
        /// The continuation stack requirement.
        /// </param>
        /// <param name="maxTraversableGraphDepth">
        /// The maximum depth of a traversable graph passed to trace.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult PipelineSetStackSize(
            IntPtr pipeline,
            uint directCallableStackSizeFromTraversal,
            uint directCallableStackSizeFromState,
            uint continuationStackSize,
            uint maxTraversableGraphDepth)
        {
            var func = GetOrCreateDelegate(ref cachedPipelineSetStackSize, functionTable.OptixPipelineSetStackSize);
            return func(
                pipeline,
                directCallableStackSizeFromTraversal,
                directCallableStackSizeFromState,
                continuationStackSize,
                maxTraversableGraphDepth);
        }

        /// <summary>
        /// Configures the pipeline stack size.
        /// </summary>
        /// <param name="programGroup">The OptiX program group.</param>
        /// <param name="sbtRecordHeaderHostPointer">
        /// Filled in with the result SBT record header.
        /// </param>
        /// <returns>The OptiX result.</returns>
        public OptixResult SbtRecordPackHeader(
            IntPtr programGroup,
            IntPtr sbtRecordHeaderHostPointer)
        {
            var func = GetOrCreateDelegate(ref cachedSbtRecordPackHeader, functionTable.OptixSbtRecordPackHeader);
            return func(
                programGroup,
                sbtRecordHeaderHostPointer);
        }

        /// <summary>
        /// Launches the OptiX pipeline.
        /// </summary>
        /// <param name="pipeline">The OptiX pipeline.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="pipelineParams">The pipeline parameters.</param>
        /// <param name="pipelineParamsSize">The pipeline parameters size.</param>
        /// <param name="sbt">The shader binding table.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="depth">The depth.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult Launch(
            IntPtr pipeline,
            IntPtr stream,
            IntPtr pipelineParams,
            uint pipelineParamsSize,
            IntPtr sbt,
            uint width,
            uint height,
            uint depth)
        {
            var func = GetOrCreateDelegate(ref cachedLaunch, functionTable.OptixLaunch);
            return func(
                pipeline,
                stream,
                pipelineParams,
                pipelineParamsSize,
                sbt,
                width,
                height,
                depth);
        }

        /// <summary>
        /// Calculates acceleration structure size.
        /// </summary>
        /// <param name="context">The OptiX device context.</param>
        /// <param name="accelOptions">The acceleration structure build options.</param>
        /// <param name="buildInputs">The build inputs.</param>
        /// <param name="numBuildInputs">The build inputs count.</param>
        /// <param name="bufferSizes">The acceleration structure size output.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult AccelComputeMemoryUsage(
        IntPtr context,
        IntPtr accelOptions,
        IntPtr buildInputs,
        uint numBuildInputs,
        IntPtr bufferSizes)
        {
            var func = GetOrCreateDelegate(ref cachedAccelComputeMemoryUsage, functionTable.OptixAccelComputeMemoryUsage);

            return func(
                context,
                accelOptions,
                buildInputs,
                numBuildInputs,
                bufferSizes);
        }

        /// <summary>
        /// Builds acceleration structure.
        /// </summary>
        /// <param name="context">The OptiX device context.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="accelOptions">The acceleration structure build options.</param>
        /// <param name="buildInputs">The build inputs.</param>
        /// <param name="numBuildInputs">The build inputs count.</param>
        /// <param name="tempBuffer">The temp build buffer, after this call the temp buffer is filled with garbage.</param>
        /// <param name="tempBufferSizeInBytes">The temp buffer size.</param>
        /// <param name="outputBuffer">The build output buffer.</param>
        /// <param name="outputBufferSizeInBytes">The output buffer size.</param>
        /// <param name="outputHandle">The OptixTraversableHandle pointer.</param>
        /// <param name="emittedProperties">The acceleration structure emitted properties.</param>
        /// <param name="numEmittedProperties">Emitted properties count.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult AccelBuild(
        IntPtr context,
        IntPtr stream,
        IntPtr accelOptions,
        IntPtr buildInputs,
        uint numBuildInputs,
        IntPtr tempBuffer,
        ulong tempBufferSizeInBytes,
        IntPtr outputBuffer,
        ulong outputBufferSizeInBytes,
        IntPtr outputHandle,
        IntPtr emittedProperties,
        uint numEmittedProperties)
        {
            var func = GetOrCreateDelegate(ref cachedAccelBuild, functionTable.OptixAccelBuild);

            return func(
                context,
                stream,
                accelOptions,
                buildInputs,
                numBuildInputs,
                tempBuffer,
                tempBufferSizeInBytes,
                outputBuffer,
                outputBufferSizeInBytes,
                outputHandle,
                emittedProperties,
                numEmittedProperties);
        }

        /// <summary>
        /// Compacts a previously built acceleration structure into a smaller output
        /// buffer. The input handle must have been built with
        /// OPTIX_BUILD_FLAG_ALLOW_COMPACTION and an OPTIX_PROPERTY_TYPE_COMPACTED_SIZE
        /// emitted property (see AccelBuild's emittedProperties parameter).
        /// </summary>
        /// <param name="context">The OptiX device context.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="inputHandle">The traversable handle of the uncompacted acceleration structure.</param>
        /// <param name="outputBuffer">The compacted output buffer.</param>
        /// <param name="outputBufferSizeInBytes">The compacted output buffer size (the compacted size read back from the emitted property).</param>
        /// <param name="outputHandle">The OptixTraversableHandle pointer, filled in with the new (compacted) handle.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult AccelCompact(
        IntPtr context,
        IntPtr stream,
        ulong inputHandle,
        IntPtr outputBuffer,
        ulong outputBufferSizeInBytes,
        IntPtr outputHandle)
        {
            var func = GetOrCreateDelegate(ref cachedAccelCompact, functionTable.OptixAccelCompact);

            return func(
                context,
                stream,
                inputHandle,
                outputBuffer,
                outputBufferSizeInBytes,
                outputHandle);
        }

        /// <summary>
        /// Creates a new OptiX denoiser.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="modelKind">The denoiser model kind.</param>
        /// <param name="options">The denoiser options.</param>
        /// <param name="denoiser">Filled in with the new denoiser.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult DenoiserCreate(
            IntPtr deviceContext,
            OptixDenoiserModelKind modelKind,
            OptixDenoiserOptions options,
            out IntPtr denoiser)
        {
            var func = GetOrCreateDelegate(ref cachedDenoiserCreate, functionTable.OptixDenoiserCreate);
            using var optionsPtr = SafeHGlobal.AllocFrom(options);
            return func(deviceContext, modelKind, optionsPtr, out denoiser);
        }

        /// <summary>
        /// Destroys the OptiX denoiser.
        /// </summary>
        /// <param name="denoiser">The OptiX denoiser.</param>
        /// <returns>The OptiX result.</returns>
        public OptixResult DenoiserDestroy(IntPtr denoiser)
        {
            var func = GetOrCreateDelegate(ref cachedDenoiserDestroy, functionTable.OptixDenoiserDestroy);
            return func(denoiser);
        }

        /// <summary>
        /// Computes the memory resources required by the denoiser.
        /// </summary>
        /// <param name="denoiser">The OptiX denoiser.</param>
        /// <param name="maximumInputWidth">The maximum input image width.</param>
        /// <param name="maximumInputHeight">The maximum input image height.</param>
        /// <param name="sizes">Filled in with the required memory sizes.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public unsafe OptixResult DenoiserComputeMemoryResources(
            IntPtr denoiser,
            uint maximumInputWidth,
            uint maximumInputHeight,
            out OptixDenoiserSizes sizes)
        {
            var func = GetOrCreateDelegate(ref cachedDenoiserComputeMemoryResources, functionTable.OptixDenoiserComputeMemoryResources);
            var sizesPtr = stackalloc OptixDenoiserSizes[1];
            var result = func(denoiser, maximumInputWidth, maximumInputHeight, new IntPtr(sizesPtr));
            sizes = sizesPtr[0];
            return result;
        }

        /// <summary>
        /// Sets up the denoiser for a given input resolution.
        /// </summary>
        /// <param name="denoiser">The OptiX denoiser.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="inputWidth">The input image width.</param>
        /// <param name="inputHeight">The input image height.</param>
        /// <param name="state">The denoiser state buffer.</param>
        /// <param name="stateSizeInBytes">The denoiser state buffer size.</param>
        /// <param name="scratch">The denoiser scratch buffer.</param>
        /// <param name="scratchSizeInBytes">The denoiser scratch buffer size.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult DenoiserSetup(
            IntPtr denoiser,
            IntPtr stream,
            uint inputWidth,
            uint inputHeight,
            IntPtr state,
            ulong stateSizeInBytes,
            IntPtr scratch,
            ulong scratchSizeInBytes)
        {
            var func = GetOrCreateDelegate(ref cachedDenoiserSetup, functionTable.OptixDenoiserSetup);
            return func(denoiser, stream, inputWidth, inputHeight, state, stateSizeInBytes, scratch, scratchSizeInBytes);
        }

        /// <summary>
        /// Invokes the denoiser.
        /// </summary>
        /// <param name="denoiser">The OptiX denoiser.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="parameters">The denoiser parameters.</param>
        /// <param name="denoiserState">The denoiser state buffer.</param>
        /// <param name="denoiserStateSizeInBytes">The denoiser state buffer size.</param>
        /// <param name="guideLayer">The denoiser guide layer.</param>
        /// <param name="layers">The denoiser input/output layers.</param>
        /// <param name="inputOffsetX">The input tile X offset.</param>
        /// <param name="inputOffsetY">The input tile Y offset.</param>
        /// <param name="scratch">The denoiser scratch buffer.</param>
        /// <param name="scratchSizeInBytes">The denoiser scratch buffer size.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult DenoiserInvoke(
            IntPtr denoiser,
            IntPtr stream,
            OptixDenoiserParams parameters,
            IntPtr denoiserState,
            ulong denoiserStateSizeInBytes,
            OptixDenoiserGuideLayer guideLayer,
            ReadOnlySpan<OptixDenoiserLayer> layers,
            uint inputOffsetX,
            uint inputOffsetY,
            IntPtr scratch,
            ulong scratchSizeInBytes)
        {
            var func = GetOrCreateDelegate(ref cachedDenoiserInvoke, functionTable.OptixDenoiserInvoke);
            using var parametersPtr = SafeHGlobal.AllocFrom(parameters);
            using var guideLayerPtr = SafeHGlobal.AllocFrom(guideLayer);
            using var layersPtr = SafeHGlobal.AllocFrom(layers);
            return func(
                denoiser,
                stream,
                parametersPtr,
                denoiserState,
                denoiserStateSizeInBytes,
                guideLayerPtr,
                layersPtr,
                (uint)layers.Length,
                inputOffsetX,
                inputOffsetY,
                scratch,
                scratchSizeInBytes);
        }

        /// <summary>
        /// Invokes the denoiser using already-marshaled native buffers for
        /// <paramref name="parameters"/>/<paramref name="guideLayer"/>/
        /// <paramref name="layers"/>, instead of the struct-taking overload's
        /// per-call <see cref="SafeHGlobal.AllocFrom{T}(T)"/> marshaling. For
        /// callers invoked every frame (e.g. a real-time denoiser loop) that already
        /// own persistent native buffers sized once and overwritten per call.
        /// </summary>
        /// <param name="denoiser">The OptiX denoiser.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="parameters">Native buffer holding a marshaled <see cref="OptixDenoiserParams"/>.</param>
        /// <param name="denoiserState">The denoiser state buffer.</param>
        /// <param name="denoiserStateSizeInBytes">The denoiser state buffer size.</param>
        /// <param name="guideLayer">Native buffer holding a marshaled <see cref="OptixDenoiserGuideLayer"/>.</param>
        /// <param name="layers">Native buffer holding <paramref name="numLayers"/> marshaled <see cref="OptixDenoiserLayer"/> entries.</param>
        /// <param name="numLayers">The number of entries in <paramref name="layers"/>.</param>
        /// <param name="inputOffsetX">The input tile X offset.</param>
        /// <param name="inputOffsetY">The input tile Y offset.</param>
        /// <param name="scratch">The denoiser scratch buffer.</param>
        /// <param name="scratchSizeInBytes">The denoiser scratch buffer size.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult DenoiserInvoke(
            IntPtr denoiser,
            IntPtr stream,
            IntPtr parameters,
            IntPtr denoiserState,
            ulong denoiserStateSizeInBytes,
            IntPtr guideLayer,
            IntPtr layers,
            uint numLayers,
            uint inputOffsetX,
            uint inputOffsetY,
            IntPtr scratch,
            ulong scratchSizeInBytes)
        {
            var func = GetOrCreateDelegate(ref cachedDenoiserInvoke, functionTable.OptixDenoiserInvoke);
            return func(
                denoiser,
                stream,
                parameters,
                denoiserState,
                denoiserStateSizeInBytes,
                guideLayer,
                layers,
                numLayers,
                inputOffsetX,
                inputOffsetY,
                scratch,
                scratchSizeInBytes);
        }

        /// <summary>
        /// Computes the average log intensity of an input image, for use as
        /// OptixDenoiserParams.HdrIntensity.
        /// </summary>
        /// <param name="denoiser">The OptiX denoiser.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="outputIntensity">The output device buffer (a single float).</param>
        /// <param name="scratch">The denoiser scratch buffer.</param>
        /// <param name="scratchSizeInBytes">The denoiser scratch buffer size.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult DenoiserComputeIntensity(
            IntPtr denoiser,
            IntPtr stream,
            OptixImage2D inputImage,
            IntPtr outputIntensity,
            IntPtr scratch,
            ulong scratchSizeInBytes)
        {
            var func = GetOrCreateDelegate(ref cachedDenoiserComputeIntensity, functionTable.OptixDenoiserComputeIntensity);
            using var inputImagePtr = SafeHGlobal.AllocFrom(inputImage);
            return func(denoiser, stream, inputImagePtr, outputIntensity, scratch, scratchSizeInBytes);
        }

        /// <summary>
        /// Computes the average log intensity using an already-marshaled native
        /// buffer for <paramref name="inputImage"/>, instead of the struct-taking
        /// overload's per-call <see cref="SafeHGlobal.AllocFrom{T}(T)"/> marshaling.
        /// For callers invoked every frame that already own a persistent native
        /// buffer sized once and overwritten per call.
        /// </summary>
        /// <param name="denoiser">The OptiX denoiser.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="inputImage">Native buffer holding a marshaled <see cref="OptixImage2D"/>.</param>
        /// <param name="outputIntensity">The output device buffer (a single float).</param>
        /// <param name="scratch">The denoiser scratch buffer.</param>
        /// <param name="scratchSizeInBytes">The denoiser scratch buffer size.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult DenoiserComputeIntensity(
            IntPtr denoiser,
            IntPtr stream,
            IntPtr inputImage,
            IntPtr outputIntensity,
            IntPtr scratch,
            ulong scratchSizeInBytes)
        {
            var func = GetOrCreateDelegate(ref cachedDenoiserComputeIntensity, functionTable.OptixDenoiserComputeIntensity);
            return func(denoiser, stream, inputImage, outputIntensity, scratch, scratchSizeInBytes);
        }

        /// <summary>
        /// Computes the average log color of an input image, for use as
        /// OptixDenoiserParams.HdrAverageColor.
        /// </summary>
        /// <param name="denoiser">The OptiX denoiser.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="outputAverageColor">The output device buffer (three floats).</param>
        /// <param name="scratch">The denoiser scratch buffer.</param>
        /// <param name="scratchSizeInBytes">The denoiser scratch buffer size.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult DenoiserComputeAverageColor(
            IntPtr denoiser,
            IntPtr stream,
            OptixImage2D inputImage,
            IntPtr outputAverageColor,
            IntPtr scratch,
            ulong scratchSizeInBytes)
        {
            var func = GetOrCreateDelegate(ref cachedDenoiserComputeAverageColor, functionTable.OptixDenoiserComputeAverageColor);
            using var inputImagePtr = SafeHGlobal.AllocFrom(inputImage);
            return func(denoiser, stream, inputImagePtr, outputAverageColor, scratch, scratchSizeInBytes);
        }

        #endregion

        #region IDisposable

        /// <summary cref="DisposeBase.Dispose(bool)"/>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                Uninit();
            }
            base.Dispose(disposing);
        }

        #endregion
    }
}

#pragma warning restore CA1707 // Identifiers should not contain underscores
