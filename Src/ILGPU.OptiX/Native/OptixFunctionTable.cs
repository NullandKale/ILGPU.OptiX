// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixFunctionTable.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX.Pipeline;
using ILGPU.OptiX.Denoising;
using ILGPU.OptiX.CooperativeVectors;
using System;

#pragma warning disable CS0649 // Field is never assigned to

namespace ILGPU.OptiX.Native
{
    // Mirrors OptixFunctionTable in optix_function_table.h (OptiX SDK 9.0.0, ABI 105 -
    // this is the exact version the nvoptix.dll bundled with the installed driver
    // ships, confirmed via its file version; see OptixAPI.Init.cs for why this can
    // differ from the newest SDK you have headers for). The driver fills this struct
    // positionally via optixQueryFunctionTable, so every field must be present in this
    // exact order even if unused by this binding.
    internal struct OptixFunctionTable
    {
        // Error handling
        public IntPtr OptixGetErrorName;
        public IntPtr OptixGetErrorString;

        // Device context
        public IntPtr OptixDeviceContextCreate;
        public IntPtr OptixDeviceContextDestroy;
        public IntPtr OptixDeviceContextGetProperty;
        public IntPtr OptixDeviceContextSetLogCallback;
        public IntPtr OptixDeviceContextSetCacheEnabled;
        public IntPtr OptixDeviceContextSetCacheLocation;
        public IntPtr OptixDeviceContextSetCacheDatabaseSizes;
        public IntPtr OptixDeviceContextGetCacheEnabled;
        public IntPtr OptixDeviceContextGetCacheLocation;
        public IntPtr OptixDeviceContextGetCacheDatabaseSizes;

        // Modules
        public IntPtr OptixModuleCreate;
        public IntPtr OptixModuleCreateWithTasks;
        public IntPtr OptixModuleGetCompilationState;
        public IntPtr OptixModuleDestroy;
        public IntPtr OptixBuiltinISModuleGet;

        // Tasks
        public IntPtr OptixTaskExecute;

        // Program groups
        public IntPtr OptixProgramGroupCreate;
        public IntPtr OptixProgramGroupDestroy;
        public IntPtr OptixProgramGroupGetStackSize;

        // Pipeline
        public IntPtr OptixPipelineCreate;
        public IntPtr OptixPipelineDestroy;
        public IntPtr OptixPipelineSetStackSize;

        // Acceleration structures
        public IntPtr OptixAccelComputeMemoryUsage;
        public IntPtr OptixAccelBuild;
        public IntPtr OptixAccelGetRelocationInfo;
        public IntPtr OptixCheckRelocationCompatibility;
        public IntPtr OptixAccelRelocate;
        public IntPtr OptixAccelCompact;
        public IntPtr OptixAccelEmitProperty;
        public IntPtr OptixConvertPointerToTraversableHandle;
        public IntPtr OptixOpacityMicromapArrayComputeMemoryUsage;
        public IntPtr OptixOpacityMicromapArrayBuild;
        public IntPtr OptixOpacityMicromapArrayGetRelocationInfo;
        public IntPtr OptixOpacityMicromapArrayRelocate;
        public IntPtr OptixDisplacementMicromapArrayComputeMemoryUsage;
        public IntPtr OptixDisplacementMicromapArrayBuild;
        public IntPtr OptixClusterAccelComputeMemoryUsage;
        public IntPtr OptixClusterAccelBuild;

        // Launch
        public IntPtr OptixSbtRecordPackHeader;
        public IntPtr OptixLaunch;

        // Cooperative Vector
        public IntPtr OptixCoopVecMatrixConvert;
        public IntPtr OptixCoopVecMatrixComputeSize;

        // Denoiser
        public IntPtr OptixDenoiserCreate;
        public IntPtr OptixDenoiserDestroy;
        public IntPtr OptixDenoiserComputeMemoryResources;
        public IntPtr OptixDenoiserSetup;
        public IntPtr OptixDenoiserInvoke;
        public IntPtr OptixDenoiserComputeIntensity;
        public IntPtr OptixDenoiserComputeAverageColor;
        public IntPtr OptixDenoiserCreateWithUserModel;
    }

    internal delegate OptixResult DeviceContextCreate(
        IntPtr cudaContext,
        OptixDeviceContextOptions options,
        out IntPtr deviceContext);

    internal delegate OptixResult DeviceContextDestroy(IntPtr deviceContext);

    internal delegate OptixResult ModuleCreate(
        IntPtr deviceContext,
        IntPtr moduleCompileOptions,
        IntPtr pipelineCompileOptions,
        IntPtr ptxString,
        ulong ptxStringSize,
        IntPtr logString,
        ref ulong logStringSize,
        out IntPtr module);

    internal delegate OptixResult ModuleDestroy(IntPtr module);

    internal delegate OptixResult BuiltinISModuleGet(
        IntPtr context,
        IntPtr moduleCompileOptions,
        IntPtr pipelineCompileOptions,
        IntPtr builtinISOptions,
        out IntPtr builtinModule);

    internal delegate OptixResult ModuleCreateWithTasks(
        IntPtr deviceContext,
        IntPtr moduleCompileOptions,
        IntPtr pipelineCompileOptions,
        IntPtr ptxString,
        ulong ptxStringSize,
        IntPtr logString,
        ref ulong logStringSize,
        out IntPtr module,
        out IntPtr firstTask);

    internal delegate OptixResult ModuleGetCompilationState(
        IntPtr module,
        out Pipeline.OptixModuleCompileState state);

    internal delegate OptixResult TaskExecute(
        IntPtr task,
        [System.Runtime.InteropServices.Out] IntPtr[] additionalTasks,
        uint maxNumAdditionalTasks,
        out uint numAdditionalTasksCreated);

    internal delegate OptixResult ProgramGroupCreate(
        IntPtr deviceContext,
        IntPtr programDescriptions,
        uint numProgramGroups,
        IntPtr programGroupOptions,
        IntPtr logString,
        ref ulong logStringSize,
        out IntPtr programGroups);

    internal delegate OptixResult ProgramGroupDestroy(IntPtr programGroup);

    internal delegate OptixResult ProgramGroupGetStackSize(
        IntPtr programGroup,
        out OptixStackSizes stackSizes,
        IntPtr pipeline);

    internal delegate OptixResult PipelineCreate(
        IntPtr deviceContext,
        IntPtr pipelineCompileOptions,
        IntPtr pipelineLinkOptions,
        IntPtr programGroups,
        uint numProgramGroups,
        IntPtr logString,
        ref ulong logStringSize,
        out IntPtr pipeline);
    internal delegate OptixResult PipelineDestroy(IntPtr pipeline);

    internal delegate OptixResult PipelineSetStackSize(
        IntPtr pipeline,
        uint directCallableStackSizeFromTraversal,
        uint directCallableStackSizeFromState,
        uint continuationStackSize,
        uint maxTraversableGraphDepth);

    internal delegate OptixResult SbtRecordPackHeader(
        IntPtr programGroup,
        IntPtr sbtRecordHeaderHostPtr);

    internal delegate OptixResult Launch(
        IntPtr pipeline,
        IntPtr stream,
        IntPtr pipelineParams,
        uint pipelineParamsSize,
        IntPtr sbt,
        uint width,
        uint height,
        uint depth);

    internal delegate OptixResult AccelComputeMemoryUsage(
        IntPtr context,
        IntPtr accelOptions,
        IntPtr buildInputs,
        uint numBuildInputs,
        IntPtr bufferSizes);

    internal delegate OptixResult AccelBuild(
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
        uint numEmittedProperties);

    internal delegate OptixResult AccelCompact(
        IntPtr context,
        IntPtr stream,
        ulong inputHandle,
        IntPtr outputBuffer,
        ulong outputBufferSizeInBytes,
        IntPtr outputHandle);

    internal delegate OptixResult AccelGetRelocationInfo(
        IntPtr context,
        ulong handle,
        IntPtr info);

    internal delegate OptixResult CheckRelocationCompatibility(
        IntPtr context,
        IntPtr info,
        out int compatible);

    internal delegate OptixResult AccelRelocate(
        IntPtr context,
        IntPtr stream,
        IntPtr info,
        IntPtr relocateInputs,
        ulong numRelocateInputs,
        IntPtr targetAccel,
        ulong targetAccelSizeInBytes,
        IntPtr targetHandle);

    internal delegate OptixResult AccelEmitProperty(
        IntPtr context,
        IntPtr stream,
        ulong handle,
        IntPtr emittedProperty);

    internal delegate OptixResult ConvertPointerToTraversableHandle(
        IntPtr onDevice,
        ulong pointer,
        AccelStructures.OptixTraversableType traversableType,
        out ulong traversableHandle);

    internal delegate OptixResult OpacityMicromapArrayComputeMemoryUsage(
        IntPtr context,
        IntPtr buildInput,
        out AccelStructures.OptixMicromapBufferSizes bufferSizes);

    internal delegate OptixResult OpacityMicromapArrayBuild(
        IntPtr context,
        IntPtr stream,
        IntPtr buildInput,
        IntPtr buffers);

    internal delegate OptixResult DenoiserCreate(
        IntPtr context,
        OptixDenoiserModelKind modelKind,
        IntPtr options,
        out IntPtr denoiser);

    internal delegate OptixResult DenoiserDestroy(IntPtr denoiser);

    internal delegate OptixResult DenoiserComputeMemoryResources(
        IntPtr denoiser,
        uint maximumInputWidth,
        uint maximumInputHeight,
        IntPtr returnSizes);

    internal delegate OptixResult DenoiserSetup(
        IntPtr denoiser,
        IntPtr stream,
        uint inputWidth,
        uint inputHeight,
        IntPtr state,
        ulong stateSizeInBytes,
        IntPtr scratch,
        ulong scratchSizeInBytes);

    internal delegate OptixResult DenoiserInvoke(
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
        ulong scratchSizeInBytes);

    internal delegate OptixResult DenoiserComputeIntensity(
        IntPtr denoiser,
        IntPtr stream,
        IntPtr inputImage,
        IntPtr outputIntensity,
        IntPtr scratch,
        ulong scratchSizeInBytes);

    internal delegate OptixResult DenoiserComputeAverageColor(
        IntPtr denoiser,
        IntPtr stream,
        IntPtr inputImage,
        IntPtr outputAverageColor,
        IntPtr scratch,
        ulong scratchSizeInBytes);

    internal delegate OptixResult CoopVecMatrixConvert(
        IntPtr context,
        IntPtr stream,
        uint numNetworks,
        IntPtr inputNetworkDescription,
        ulong inputNetworks,
        ulong inputNetworkStrideInBytes,
        IntPtr outputNetworkDescription,
        ulong outputNetworks,
        ulong outputNetworkStrideInBytes);

    internal delegate OptixResult CoopVecMatrixComputeSize(
        IntPtr context,
        uint N,
        uint K,
        CooperativeVectors.OptixCoopVecElemType elementType,
        CooperativeVectors.OptixCoopVecMatrixLayout layout,
        ulong rowColumnStrideInBytes,
        out ulong sizeInBytes);

    // Returns a static const char* owned by the driver - marshal manually via
    // Marshal.PtrToStringAnsi, never free it.
    internal delegate IntPtr GetErrorName(OptixResult result);

    internal delegate IntPtr GetErrorString(OptixResult result);

    internal delegate OptixResult DeviceContextGetProperty(
        IntPtr context,
        Pipeline.OptixDeviceProperty property,
        IntPtr value,
        UIntPtr sizeInBytes);

    internal delegate OptixResult DeviceContextSetLogCallback(
        IntPtr context,
        IntPtr callbackFunction,
        IntPtr callbackData,
        uint callbackLevel);

    internal delegate OptixResult DeviceContextSetCacheEnabled(
        IntPtr context,
        int enabled);

    internal delegate OptixResult DeviceContextGetCacheEnabled(
        IntPtr context,
        out int enabled);

    internal delegate OptixResult DeviceContextSetCacheLocation(
        IntPtr context,
        IntPtr location);

    internal delegate OptixResult DeviceContextGetCacheLocation(
        IntPtr context,
        IntPtr location,
        UIntPtr locationSize);

    internal delegate OptixResult DeviceContextSetCacheDatabaseSizes(
        IntPtr context,
        UIntPtr lowWaterMark,
        UIntPtr highWaterMark);

    internal delegate OptixResult DeviceContextGetCacheDatabaseSizes(
        IntPtr context,
        out UIntPtr lowWaterMark,
        out UIntPtr highWaterMark);

    internal delegate OptixResult OpacityMicromapArrayGetRelocationInfo(
        IntPtr context,
        ulong opacityMicromapArray,
        IntPtr info);

    internal delegate OptixResult OpacityMicromapArrayRelocate(
        IntPtr context,
        IntPtr stream,
        IntPtr info,
        ulong targetOpacityMicromapArray,
        UIntPtr targetOpacityMicromapArraySizeInBytes);

    internal delegate OptixResult DenoiserCreateWithUserModel(
        IntPtr context,
        IntPtr userData,
        UIntPtr userDataSizeInBytes,
        out IntPtr denoiser);

    internal delegate OptixResult ClusterAccelComputeMemoryUsage(
        IntPtr context,
        AccelStructures.OptixClusterAccelBuildMode buildMode,
        IntPtr buildInput,
        out AccelStructures.OptixAccelBufferSizes bufferSizes);

    internal delegate OptixResult ClusterAccelBuild(
        IntPtr context,
        IntPtr stream,
        IntPtr buildModeDesc,
        IntPtr buildInput,
        ulong argsArray,
        ulong argsCount,
        uint argsStrideInBytes);
}

#pragma warning restore CS0649 // Field is never assigned to
