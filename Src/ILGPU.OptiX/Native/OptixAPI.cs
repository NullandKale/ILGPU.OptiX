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
using ILGPU.OptiX.Denoising;
using ILGPU.OptiX.CooperativeVectors;
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

// disable: max_line_length
#pragma warning disable CA1707 // Identifiers should not contain underscores

namespace ILGPU.OptiX.Native
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
        private ModuleCreateWithTasks? cachedModuleCreateWithTasks;
        private ModuleGetCompilationState? cachedModuleGetCompilationState;
        private TaskExecute? cachedTaskExecute;
        private BuiltinISModuleGet? cachedBuiltinISModuleGet;
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
        private AccelGetRelocationInfo? cachedAccelGetRelocationInfo;
        private CheckRelocationCompatibility? cachedCheckRelocationCompatibility;
        private AccelRelocate? cachedAccelRelocate;
        private AccelEmitProperty? cachedAccelEmitProperty;
        private ConvertPointerToTraversableHandle? cachedConvertPointerToTraversableHandle;
        private OpacityMicromapArrayComputeMemoryUsage? cachedOpacityMicromapArrayComputeMemoryUsage;
        private OpacityMicromapArrayBuild? cachedOpacityMicromapArrayBuild;
        private DenoiserCreate? cachedDenoiserCreate;
        private DenoiserDestroy? cachedDenoiserDestroy;
        private DenoiserComputeMemoryResources? cachedDenoiserComputeMemoryResources;
        private DenoiserSetup? cachedDenoiserSetup;
        private DenoiserInvoke? cachedDenoiserInvoke;
        private DenoiserComputeIntensity? cachedDenoiserComputeIntensity;
        private DenoiserComputeAverageColor? cachedDenoiserComputeAverageColor;
        private CoopVecMatrixConvert? cachedCoopVecMatrixConvert;
        private CoopVecMatrixComputeSize? cachedCoopVecMatrixComputeSize;

        private void ClearDelegateCache()
        {
            cachedDeviceContextCreate = null;
            cachedDeviceContextDestroy = null;
            cachedModuleCreate = null;
            cachedModuleDestroy = null;
            cachedModuleCreateWithTasks = null;
            cachedModuleGetCompilationState = null;
            cachedTaskExecute = null;
            cachedBuiltinISModuleGet = null;
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
            cachedAccelGetRelocationInfo = null;
            cachedCheckRelocationCompatibility = null;
            cachedAccelRelocate = null;
            cachedAccelEmitProperty = null;
            cachedConvertPointerToTraversableHandle = null;
            cachedOpacityMicromapArrayComputeMemoryUsage = null;
            cachedOpacityMicromapArrayBuild = null;
            cachedDenoiserCreate = null;
            cachedDenoiserDestroy = null;
            cachedDenoiserComputeMemoryResources = null;
            cachedDenoiserSetup = null;
            cachedDenoiserInvoke = null;
            cachedDenoiserComputeIntensity = null;
            cachedDenoiserComputeAverageColor = null;
            cachedCoopVecMatrixConvert = null;
            cachedCoopVecMatrixComputeSize = null;
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
        /// Creates a new OptiX module using task-based (parallelizable) compilation -
        /// returns immediately with a first OptixTask; the caller must drive that task
        /// graph via <see cref="TaskExecute"/> until <see cref="ModuleGetCompilationState"/>
        /// reports Completed or Failed before the module is usable.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="moduleCompileOptions">The module compile options.</param>
        /// <param name="pipelineCompileOptions">The pipeline compile options.</param>
        /// <param name="ptxString">The module PTX code.</param>
        /// <param name="module">Filled in with the new module.</param>
        /// <param name="firstTask">Filled in with the first task to execute.</param>
        /// <param name="logString">Filled in with the log string.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public unsafe OptixResult ModuleCreateWithTasks(
            IntPtr deviceContext,
            OptixModuleCompileOptions moduleCompileOptions,
            OptixPipelineCompileOptions pipelineCompileOptions,
            string ptxString,
            out IntPtr module,
            out IntPtr firstTask,
            out string logString)
        {
            var func = GetOrCreateDelegate(ref cachedModuleCreateWithTasks, functionTable.OptixModuleCreateWithTasks);

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
                    out module,
                    out firstTask
                );
                logString = result != OptixResult.OPTIX_SUCCESS
                    ? ReadLog(logPtr, logLength, logBytes.Length)
                    : string.Empty;
                return result;
            }
        }

        /// <summary>
        /// Queries the compilation state of a module created via
        /// <see cref="ModuleCreateWithTasks"/>.
        /// </summary>
        /// <param name="module">The OptiX module.</param>
        /// <param name="state">Filled in with the current compilation state.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult ModuleGetCompilationState(IntPtr module, out OptixModuleCompileState state)
        {
            var func = GetOrCreateDelegate(ref cachedModuleGetCompilationState, functionTable.OptixModuleGetCompilationState);
            return func(module, out state);
        }

        /// <summary>
        /// Executes a single OptixTask from a task graph started by
        /// <see cref="ModuleCreateWithTasks"/>. May produce more tasks (written into
        /// <paramref name="additionalTasks"/>, up to its length) that must themselves be
        /// executed before compilation is complete - see optix_host.h's doc comment on
        /// optixTaskExecute: "Each task can be executed in parallel and in any order."
        /// </summary>
        /// <param name="task">The task to execute.</param>
        /// <param name="additionalTasks">Buffer to receive any additional tasks produced.</param>
        /// <param name="maxNumAdditionalTasks">The capacity of <paramref name="additionalTasks"/>.</param>
        /// <param name="numAdditionalTasksCreated">Filled in with how many additional tasks were produced.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult TaskExecute(
            IntPtr task,
            IntPtr[] additionalTasks,
            uint maxNumAdditionalTasks,
            out uint numAdditionalTasksCreated)
        {
            var func = GetOrCreateDelegate(ref cachedTaskExecute, functionTable.OptixTaskExecute);
            return func(task, additionalTasks, maxNumAdditionalTasks, out numAdditionalTasksCreated);
        }

        /// <summary>
        /// Retrieves OptiX's own built-in intersection program module for a non-custom
        /// primitive type (curves, spheres) - there is no user-supplied intersection
        /// program for these types.
        /// </summary>
        /// <param name="context">The OptiX device context.</param>
        /// <param name="moduleCompileOptions">The module compile options.</param>
        /// <param name="pipelineCompileOptions">The pipeline compile options.</param>
        /// <param name="builtinISOptions">The built-in primitive type + curve/motion options.</param>
        /// <param name="builtinModule">Filled in with the built-in module.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult BuiltinISModuleGet(
            IntPtr context,
            OptixModuleCompileOptions moduleCompileOptions,
            OptixPipelineCompileOptions pipelineCompileOptions,
            OptixBuiltinISOptions builtinISOptions,
            out IntPtr builtinModule)
        {
            var func = GetOrCreateDelegate(ref cachedBuiltinISModuleGet, functionTable.OptixBuiltinISModuleGet);

            using var moduleCompileOptionsPtr = SafeHGlobal.AllocFrom(moduleCompileOptions);
            using var pipelineCompileOptionsPtr = SafeHGlobal.AllocFrom(pipelineCompileOptions);
            using var builtinISOptionsPtr = SafeHGlobal.AllocFrom(builtinISOptions);

            return func(
                context,
                moduleCompileOptionsPtr,
                pipelineCompileOptionsPtr,
                builtinISOptionsPtr,
                out builtinModule);
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
        /// AllowCompaction and an CompactedSize
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
        /// Retrieves relocation info for a previously built acceleration structure, to be
        /// passed to CheckRelocationCompatibility/AccelRelocate.
        /// </summary>
        /// <param name="context">The OptiX device context.</param>
        /// <param name="handle">The traversable handle of the built acceleration structure.</param>
        /// <param name="info">Pointer to an OptixRelocationInfo to fill in.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult AccelGetRelocationInfo(
            IntPtr context,
            ulong handle,
            IntPtr info)
        {
            var func = GetOrCreateDelegate(ref cachedAccelGetRelocationInfo, functionTable.OptixAccelGetRelocationInfo);
            return func(context, handle, info);
        }

        /// <summary>
        /// Checks whether an acceleration structure described by relocation info can be
        /// relocated on this device context.
        /// </summary>
        /// <param name="context">The OptiX device context.</param>
        /// <param name="info">Pointer to the OptixRelocationInfo from AccelGetRelocationInfo.</param>
        /// <param name="compatible">Filled in with non-zero if compatible.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult CheckRelocationCompatibility(
            IntPtr context,
            IntPtr info,
            out int compatible)
        {
            var func = GetOrCreateDelegate(ref cachedCheckRelocationCompatibility, functionTable.OptixCheckRelocationCompatibility);
            return func(context, info, out compatible);
        }

        /// <summary>
        /// Relocates a previously built acceleration structure into a new target buffer.
        /// </summary>
        /// <param name="context">The OptiX device context.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="info">Pointer to the OptixRelocationInfo from AccelGetRelocationInfo.</param>
        /// <param name="relocateInputs">Pointer to an array of OptixRelocateInput, one per original build input.</param>
        /// <param name="numRelocateInputs">Number of relocate inputs.</param>
        /// <param name="targetAccel">The target device buffer to relocate into.</param>
        /// <param name="targetAccelSizeInBytes">The target buffer size (must match the source's size).</param>
        /// <param name="targetHandle">The OptixTraversableHandle pointer, filled in with the relocated handle.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult AccelRelocate(
            IntPtr context,
            IntPtr stream,
            IntPtr info,
            IntPtr relocateInputs,
            ulong numRelocateInputs,
            IntPtr targetAccel,
            ulong targetAccelSizeInBytes,
            IntPtr targetHandle)
        {
            var func = GetOrCreateDelegate(ref cachedAccelRelocate, functionTable.OptixAccelRelocate);
            return func(
                context,
                stream,
                info,
                relocateInputs,
                numRelocateInputs,
                targetAccel,
                targetAccelSizeInBytes,
                targetHandle);
        }

        /// <summary>
        /// Emits a single post-build property (e.g. Aabbs) for a previously built
        /// acceleration structure, outside of the emittedProperties list passed to
        /// AccelBuild itself.
        /// </summary>
        /// <param name="context">The OptiX device context.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="handle">The traversable handle of the built acceleration structure.</param>
        /// <param name="emittedProperty">Pointer to a single OptixAccelEmitDesc.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult AccelEmitProperty(
            IntPtr context,
            IntPtr stream,
            ulong handle,
            IntPtr emittedProperty)
        {
            var func = GetOrCreateDelegate(ref cachedAccelEmitProperty, functionTable.OptixAccelEmitProperty);
            return func(context, stream, handle, emittedProperty);
        }

        /// <summary>
        /// Converts a raw device pointer to a static/motion transform struct into a
        /// traversable handle, for use as an OptixInstance's or another transform
        /// node's child traversable.
        /// </summary>
        /// <param name="onDevice">The OptiX device context.</param>
        /// <param name="pointer">Device pointer to the transform struct (must be OPTIX_TRANSFORM_BYTE_ALIGNMENT-aligned).</param>
        /// <param name="traversableType">The kind of transform struct at <paramref name="pointer"/>.</param>
        /// <param name="traversableHandle">Filled in with the resulting traversable handle.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult ConvertPointerToTraversableHandle(
            IntPtr onDevice,
            ulong pointer,
            AccelStructures.OptixTraversableType traversableType,
            out ulong traversableHandle)
        {
            var func = GetOrCreateDelegate(ref cachedConvertPointerToTraversableHandle, functionTable.OptixConvertPointerToTraversableHandle);
            return func(onDevice, pointer, traversableType, out traversableHandle);
        }

        /// <summary>
        /// Computes memory requirements for building an opacity micromap array.
        /// </summary>
        /// <param name="context">The OptiX device context.</param>
        /// <param name="buildInput">The opacity micromap array build input.</param>
        /// <param name="bufferSizes">Filled in with the required buffer sizes.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult OpacityMicromapArrayComputeMemoryUsage(
            IntPtr context,
            AccelStructures.OptixOpacityMicromapArrayBuildInput buildInput,
            out AccelStructures.OptixMicromapBufferSizes bufferSizes)
        {
            var func = GetOrCreateDelegate(ref cachedOpacityMicromapArrayComputeMemoryUsage, functionTable.OptixOpacityMicromapArrayComputeMemoryUsage);
            using var buildInputPtr = SafeHGlobal.AllocFrom(buildInput);
            return func(context, buildInputPtr, out bufferSizes);
        }

        /// <summary>
        /// Builds an opacity micromap array.
        /// </summary>
        /// <param name="context">The OptiX device context.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="buildInput">The opacity micromap array build input.</param>
        /// <param name="buffers">The output/temp buffers.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult OpacityMicromapArrayBuild(
            IntPtr context,
            IntPtr stream,
            AccelStructures.OptixOpacityMicromapArrayBuildInput buildInput,
            AccelStructures.OptixMicromapBuffers buffers)
        {
            var func = GetOrCreateDelegate(ref cachedOpacityMicromapArrayBuild, functionTable.OptixOpacityMicromapArrayBuild);
            using var buildInputPtr = SafeHGlobal.AllocFrom(buildInput);
            using var buffersPtr = SafeHGlobal.AllocFrom(buffers);
            return func(context, stream, buildInputPtr, buffersPtr);
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

        /// <summary>
        /// Computes the required size, in bytes, of a cooperative-vector matrix with
        /// the given shape/type/layout. Results are
        /// rounded up to a multiple of 64 bytes by the driver.
        /// </summary>
        /// <param name="context">The OptiX device context.</param>
        /// <param name="N">Matrix row count.</param>
        /// <param name="K">Matrix column count.</param>
        /// <param name="elementType">The matrix element type.</param>
        /// <param name="layout">The matrix layout.</param>
        /// <param name="rowColumnStrideInBytes">Ignored for the two "optimal" layouts.</param>
        /// <param name="sizeInBytes">Filled in with the required size in bytes.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult CoopVecMatrixComputeSize(
            IntPtr context,
            uint N,
            uint K,
            CooperativeVectors.OptixCoopVecElemType elementType,
            CooperativeVectors.OptixCoopVecMatrixLayout layout,
            ulong rowColumnStrideInBytes,
            out ulong sizeInBytes)
        {
            var func = GetOrCreateDelegate(ref cachedCoopVecMatrixComputeSize, functionTable.OptixCoopVecMatrixComputeSize);
            return func(context, N, K, elementType, layout, rowColumnStrideInBytes, out sizeInBytes);
        }

        /// <summary>
        /// Converts cooperative-vector matrices from one layout/element type to another
        /// - typically row-major (as uploaded by the
        /// host) into InferencingOptimal (as consumed by <c>optixCoopVecMatMul</c>).
        /// Prefer <see cref="CooperativeVectors.OptixCoopVecMatrixBuilder.ConvertMatrix"/>
        /// over calling this directly.
        /// </summary>
        /// <param name="context">The OptiX device context.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="numNetworks">Number of networks to convert (matrices per network described by the descriptions below).</param>
        /// <param name="inputNetworkDescription">Pointer to a marshaled OptixNetworkDescription describing the input topology.</param>
        /// <param name="inputNetworks">Device pointer to the input matrix data.</param>
        /// <param name="inputNetworkStrideInBytes">Stride between input networks; ignored if numNetworks is 1.</param>
        /// <param name="outputNetworkDescription">Pointer to a marshaled OptixNetworkDescription describing the output topology.</param>
        /// <param name="outputNetworks">Device pointer to the output matrix data.</param>
        /// <param name="outputNetworkStrideInBytes">Stride between output networks; ignored if numNetworks is 1.</param>
        /// <returns>The OptiX result.</returns>
        [CLSCompliant(false)]
        public OptixResult CoopVecMatrixConvert(
            IntPtr context,
            IntPtr stream,
            uint numNetworks,
            IntPtr inputNetworkDescription,
            ulong inputNetworks,
            ulong inputNetworkStrideInBytes,
            IntPtr outputNetworkDescription,
            ulong outputNetworks,
            ulong outputNetworkStrideInBytes)
        {
            var func = GetOrCreateDelegate(ref cachedCoopVecMatrixConvert, functionTable.OptixCoopVecMatrixConvert);
            return func(
                context,
                stream,
                numNetworks,
                inputNetworkDescription,
                inputNetworks,
                inputNetworkStrideInBytes,
                outputNetworkDescription,
                outputNetworks,
                outputNetworkStrideInBytes);
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
