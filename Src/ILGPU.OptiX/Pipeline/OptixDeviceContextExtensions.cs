// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixDeviceContextExtensions.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX.Interop;
using ILGPU.OptiX.Util;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.OptiX.Native;
using ILGPU.OptiX.AccelStructures;
using ILGPU.OptiX.Denoising;
using System;
using System.Runtime.InteropServices;

// disable: max_line_length

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Extension functions for OptixDeviceContext.
    /// </summary>
    public static class OptixDeviceContextExtensions
    {
        /// <summary>
        /// When set, every kernel module compiled via <see cref="CreateModule"/> has
        /// its generated PTX written to a file in <see cref="System.IO.Path.GetTempPath"/>
        /// named "ilgpu-optix-debug-{kernelPrefix}-{entryFunctionName}.ptx". Off by
        /// default - intended for diagnosing PTX generation issues, not for routine use.
        /// </summary>
        public static bool DumpGeneratedPtx { get; set; }

        /// <summary>
        /// Creates a new OptiX raygen kernel.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="raygenKernel">The raygen kernel.</param>
        /// <param name="moduleCompileOptions">The module compile options.</param>
        /// <param name="pipelineCompileOptions">The pipeline compile options.</param>
        /// <returns>The raygen kernel.</returns>
        [CLSCompliant(false)]
        public static OptixKernel CreateRaygenKernel<TLaunchParams>(
            this OptixDeviceContext deviceContext,
            Action<TLaunchParams> raygenKernel,
            OptixModuleCompileOptions moduleCompileOptions,
            OptixPipelineCompileOptions pipelineCompileOptions)
            where TLaunchParams : unmanaged
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            using var module = deviceContext.CreateModule(
                raygenKernel,
                OptixKernel.RAYGEN_PREFIX,
                moduleCompileOptions,
                pipelineCompileOptions,
                out var raygenEntryFunctionName);
            using var name = SafeHGlobal.FromString(raygenEntryFunctionName);
            using var programGroup = deviceContext.CreateProgramGroup(
                new OptixProgramGroupDesc()
                {
                    Kind = OptixProgramGroupKind.Raygen,
                    Raygen =
                        new OptixProgramGroupSingleModule()
                        {
                            Module = module.ModulePtr,
                            EntryFunctionName = name
                        }
                });
            return new OptixKernel(
                new[] { module.Transfer() },
                programGroup.Transfer());
        }

        /// <summary>
        /// Creates a new OptiX raygen kernel, compiling its module via OptiX's
        /// task-based (parallelizable) compilation path (<see cref="CreateModuleWithTasks"/>)
        /// instead of the single blocking call <see cref="CreateRaygenKernel{TLaunchParams}(OptixDeviceContext, Action{TLaunchParams}, OptixModuleCompileOptions, OptixPipelineCompileOptions)"/>
        /// uses. Functionally identical result - same module, same program group - only
        /// the compilation strategy differs.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="raygenKernel">The raygen kernel.</param>
        /// <param name="moduleCompileOptions">The module compile options.</param>
        /// <param name="pipelineCompileOptions">The pipeline compile options.</param>
        /// <param name="taskCount">Filled in with how many OptixTasks were executed to compile this module.</param>
        /// <returns>The raygen kernel.</returns>
        [CLSCompliant(false)]
        public static OptixKernel CreateRaygenKernelWithTasks<TLaunchParams>(
            this OptixDeviceContext deviceContext,
            Action<TLaunchParams> raygenKernel,
            OptixModuleCompileOptions moduleCompileOptions,
            OptixPipelineCompileOptions pipelineCompileOptions,
            out int taskCount)
            where TLaunchParams : unmanaged
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            var ptxAssembly = deviceContext.GeneratePTX(
                raygenKernel, OptixKernel.RAYGEN_PREFIX, out var raygenEntryFunctionName);

            if (DumpGeneratedPtx)
            {
                System.IO.File.WriteAllText(
                    System.IO.Path.Combine(
                        System.IO.Path.GetTempPath(),
                        $"ilgpu-optix-debug-{OptixKernel.RAYGEN_PREFIX}-{raygenEntryFunctionName}.ptx"),
                    ptxAssembly);
            }

            using var module = deviceContext.CreateModuleWithTasks(
                moduleCompileOptions, pipelineCompileOptions, ptxAssembly, out taskCount);
            using var name = SafeHGlobal.FromString(raygenEntryFunctionName);
            using var programGroup = deviceContext.CreateProgramGroup(
                new OptixProgramGroupDesc()
                {
                    Kind = OptixProgramGroupKind.Raygen,
                    Raygen =
                        new OptixProgramGroupSingleModule()
                        {
                            Module = module.ModulePtr,
                            EntryFunctionName = name
                        }
                });
            return new OptixKernel(
                new[] { module.Transfer() },
                programGroup.Transfer());
        }

        /// <summary>
        /// Creates a new OptiX miss kernel.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="missKernel">The miss kernel.</param>
        /// <param name="moduleCompileOptions">The module compile options.</param>
        /// <param name="pipelineCompileOptions">The pipeline compile options.</param>
        /// <returns>The miss kernel.</returns>
        [CLSCompliant(false)]
        public static OptixKernel CreateMissKernel<TLaunchParams>(
            this OptixDeviceContext deviceContext,
            Action<TLaunchParams> missKernel,
            OptixModuleCompileOptions moduleCompileOptions,
            OptixPipelineCompileOptions pipelineCompileOptions)
            where TLaunchParams : unmanaged
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            using var module = deviceContext.CreateModule(
                missKernel,
                OptixKernel.MISS_PREFIX,
                moduleCompileOptions,
                pipelineCompileOptions,
                out var missEntryFunctionName);
            using var name = SafeHGlobal.FromString(missEntryFunctionName);

            using var programGroup = deviceContext.CreateProgramGroup(
                new OptixProgramGroupDesc()
                {
                    Kind = OptixProgramGroupKind.Miss,
                    Miss =
                        new OptixProgramGroupSingleModule()
                        {
                            Module = module.ModulePtr,
                            EntryFunctionName = name
                        }
                });
            return new OptixKernel(
                new[] { module.Transfer() },
                programGroup.Transfer());
        }

        /// <summary>
        /// Creates a new OptiX exception kernel - runs when the launch hits an
        /// exception enabled via
        /// <see cref="OptixPipelineCompileOptionsBuilder.WithExceptionFlags"/> (or a
        /// user exception thrown via <c>OptixThrowException</c>). Read the exception
        /// code/details inside via <c>OptixGetExceptionInfo</c>.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="exceptionKernel">The exception kernel.</param>
        /// <param name="moduleCompileOptions">The module compile options.</param>
        /// <param name="pipelineCompileOptions">The pipeline compile options.</param>
        /// <returns>The exception kernel.</returns>
        [CLSCompliant(false)]
        public static OptixKernel CreateExceptionKernel<TLaunchParams>(
            this OptixDeviceContext deviceContext,
            Action<TLaunchParams> exceptionKernel,
            OptixModuleCompileOptions moduleCompileOptions,
            OptixPipelineCompileOptions pipelineCompileOptions)
            where TLaunchParams : unmanaged
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            using var module = deviceContext.CreateModule(
                exceptionKernel,
                OptixKernel.EXCEPTION_PREFIX,
                moduleCompileOptions,
                pipelineCompileOptions,
                out var exceptionEntryFunctionName);
            using var name = SafeHGlobal.FromString(exceptionEntryFunctionName);

            using var programGroup = deviceContext.CreateProgramGroup(
                new OptixProgramGroupDesc()
                {
                    Kind = OptixProgramGroupKind.Exception,
                    Exception =
                        new OptixProgramGroupSingleModule()
                        {
                            Module = module.ModulePtr,
                            EntryFunctionName = name
                        }
                });
            return new OptixKernel(
                new[] { module.Transfer() },
                programGroup.Transfer());
        }

        /// <summary>
        /// Creates a new OptiX direct-callable (DC) kernel, invokable from device
        /// code via <c>OptixCallables.DirectCall(sbtIndex)</c> after registering the
        /// kernel's program group in the SBT callables table
        /// (<see cref="OptixSbtBuilder.SetCallableRecords"/>). The callable receives
        /// the launch params like any other program; see
        /// <c>OptixCallables</c> for the current zero-argument calling convention.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="callableKernel">The direct callable kernel.</param>
        /// <param name="moduleCompileOptions">The module compile options.</param>
        /// <param name="pipelineCompileOptions">The pipeline compile options.</param>
        /// <returns>The callable kernel.</returns>
        [CLSCompliant(false)]
        public static OptixKernel CreateDirectCallableKernel<TLaunchParams>(
            this OptixDeviceContext deviceContext,
            Action<TLaunchParams> callableKernel,
            OptixModuleCompileOptions moduleCompileOptions,
            OptixPipelineCompileOptions pipelineCompileOptions)
            where TLaunchParams : unmanaged
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            var ptxAssembly = deviceContext.GenerateCallablePTX(
                callableKernel,
                OptixKernel.DIRECT_CALLABLE_PREFIX,
                out var entryFunctionName);

            if (DumpGeneratedPtx)
            {
                System.IO.File.WriteAllText(
                    System.IO.Path.Combine(
                        System.IO.Path.GetTempPath(),
                        $"ilgpu-optix-debug-{OptixKernel.DIRECT_CALLABLE_PREFIX}-{entryFunctionName}.ptx"),
                    ptxAssembly);
            }

            using var module = deviceContext.CreateModule(
                moduleCompileOptions, pipelineCompileOptions, ptxAssembly);
            using var name = SafeHGlobal.FromString(entryFunctionName);

            using var programGroup = deviceContext.CreateProgramGroup(
                new OptixProgramGroupDesc()
                {
                    Kind = OptixProgramGroupKind.Callables,
                    Callables =
                        new OptixProgramGroupCallables()
                        {
                            ModuleDC = module.ModulePtr,
                            EntryFunctionNameDC = name
                        }
                });
            return new OptixKernel(
                new[] { module.Transfer() },
                programGroup.Transfer());
        }

        /// <summary>
        /// Creates a new OptiX continuation-callable (CC) kernel, invokable from
        /// device code via <c>OptixCallables.ContinuationCall(sbtIndex)</c> after
        /// registering the kernel's program group in the SBT callables table
        /// (<see cref="OptixSbtBuilder.SetCallableRecords"/>). Continuation callables
        /// consume continuation stack - account for them in the pipeline stack sizes.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="callableKernel">The continuation callable kernel.</param>
        /// <param name="moduleCompileOptions">The module compile options.</param>
        /// <param name="pipelineCompileOptions">The pipeline compile options.</param>
        /// <returns>The callable kernel.</returns>
        [CLSCompliant(false)]
        public static OptixKernel CreateContinuationCallableKernel<TLaunchParams>(
            this OptixDeviceContext deviceContext,
            Action<TLaunchParams> callableKernel,
            OptixModuleCompileOptions moduleCompileOptions,
            OptixPipelineCompileOptions pipelineCompileOptions)
            where TLaunchParams : unmanaged
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            var ptxAssembly = deviceContext.GenerateCallablePTX(
                callableKernel,
                OptixKernel.CONTINUATION_CALLABLE_PREFIX,
                out var entryFunctionName);

            if (DumpGeneratedPtx)
            {
                System.IO.File.WriteAllText(
                    System.IO.Path.Combine(
                        System.IO.Path.GetTempPath(),
                        $"ilgpu-optix-debug-{OptixKernel.CONTINUATION_CALLABLE_PREFIX}-{entryFunctionName}.ptx"),
                    ptxAssembly);
            }

            using var module = deviceContext.CreateModule(
                moduleCompileOptions, pipelineCompileOptions, ptxAssembly);
            using var name = SafeHGlobal.FromString(entryFunctionName);

            using var programGroup = deviceContext.CreateProgramGroup(
                new OptixProgramGroupDesc()
                {
                    Kind = OptixProgramGroupKind.Callables,
                    Callables =
                        new OptixProgramGroupCallables()
                        {
                            ModuleCC = module.ModulePtr,
                            entryFunctionNameCC = name
                        }
                });
            return new OptixKernel(
                new[] { module.Transfer() },
                programGroup.Transfer());
        }

        /// <summary>
        /// Creates a new OptiX hitgroup kernel.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="closestHitKernel">The closest hit kernel.</param>
        /// <param name="anyHitKernel">The any hit kernel.</param>
        /// <param name="intersectionKernel">The intersection kernel.</param>
        /// <param name="moduleCompileOptions">The module compile options.</param>
        /// <param name="pipelineCompileOptions">The pipeline compile options.</param>
        /// <returns>The hitgroup kernel.</returns>
        [CLSCompliant(false)]
        public static OptixKernel CreateHitgroupKernel<TLaunchParams>(
            this OptixDeviceContext deviceContext,
            Action<TLaunchParams> closestHitKernel,
            Action<TLaunchParams> anyHitKernel,
            Action<TLaunchParams> intersectionKernel,
            OptixModuleCompileOptions moduleCompileOptions,
            OptixPipelineCompileOptions pipelineCompileOptions)
            where TLaunchParams : unmanaged
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            using var closestHitModule =
                deviceContext.CreateModule(
                    closestHitKernel,
                    OptixKernel.CLOSESTHIT_PREFIX,
                    moduleCompileOptions,
                    pipelineCompileOptions,
                    out var closestHitEntryFunctionName);
            using var chName = SafeHGlobal.FromString(closestHitEntryFunctionName);

            using var anyHitModule =
                deviceContext.CreateModule(
                   anyHitKernel,
                    OptixKernel.ANYHIT_PREFIX,
                    moduleCompileOptions,
                    pipelineCompileOptions,
                    out var anyHitEntryFunctionName);
            using var ahName = SafeHGlobal.FromString(anyHitEntryFunctionName);

            using var intersectionModule =
                deviceContext.CreateModule(
                    intersectionKernel,
                    OptixKernel.INTERSECTION_PREFIX,
                    moduleCompileOptions,
                    pipelineCompileOptions,
                    out var intersectionEntryFunctionName);
            using var isName = SafeHGlobal.FromString(intersectionEntryFunctionName);

            using var programGroup = deviceContext.CreateProgramGroup(
                new OptixProgramGroupDesc()
                {
                    Kind = OptixProgramGroupKind.Hitgroup,
                    Hitgroup =
                        new OptixProgramGroupHitgroup()
                        {
                            ModuleCH = closestHitModule.ModulePtr,
                            EntryFunctionNameCH = chName,
                            ModuleAH = anyHitModule.ModulePtr,
                            EntryFunctionNameAH = ahName,
                            ModuleIS = intersectionModule.ModulePtr,
                            EntryFunctionNameIS = isName
                        }
                });

            return new OptixKernel(
                new[]
                {
                    closestHitModule.Transfer(),
                    anyHitModule.Transfer(),
                    intersectionModule.Transfer()
                },
                programGroup.Transfer());
        }

        /// <summary>
        /// Retrieves OptiX's own built-in intersection program module for a non-custom
        /// primitive type (curves, spheres). There is
        /// no user-supplied intersection program possible for these types, unlike
        /// custom primitives.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="primitiveType">The built-in primitive type (must not be <see cref="OptixPrimitiveType.Custom"/> or <see cref="OptixPrimitiveType.Triangle"/>).</param>
        /// <param name="moduleCompileOptions">The module compile options.</param>
        /// <param name="pipelineCompileOptions">The pipeline compile options.</param>
        /// <param name="curveEndcapFlags">Curve end cap flags - ignored for non-curve types.</param>
        /// <param name="usesMotionBlur">Whether vertex motion blur is used for this primitive.</param>
        [CLSCompliant(false)]
        public static OptixModule GetBuiltinISModule(
            this OptixDeviceContext deviceContext,
            OptixPrimitiveType primitiveType,
            OptixModuleCompileOptions moduleCompileOptions,
            OptixPipelineCompileOptions pipelineCompileOptions,
            OptixCurveEndcapFlags curveEndcapFlags = OptixCurveEndcapFlags.Default,
            bool usesMotionBlur = false)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            var result = OptixAPI.Current.BuiltinISModuleGet(
                deviceContext.DeviceContextPtr,
                moduleCompileOptions,
                pipelineCompileOptions,
                new OptixBuiltinISOptions
                {
                    BuiltinISModuleType = primitiveType,
                    UsesMotionBlur = usesMotionBlur ? 1 : 0,
                    CurveEndcapFlags = (uint)curveEndcapFlags,
                },
                out var module);
            OptixException.ThrowIfFailed(result);
            return new OptixModule(module);
        }

        /// <summary>
        /// Creates a new OptiX hitgroup kernel whose intersection program is OptiX's
        /// own built-in module for <paramref name="primitiveType"/>
        /// instead of a compiled kernel - required
        /// for curve/sphere geometry, which has no user-suppliable intersection
        /// program.
        /// </summary>
        [CLSCompliant(false)]
        public static OptixKernel CreateHitgroupKernelWithBuiltinIS<TLaunchParams>(
            this OptixDeviceContext deviceContext,
            Action<TLaunchParams> closestHitKernel,
            Action<TLaunchParams> anyHitKernel,
            OptixPrimitiveType primitiveType,
            OptixModuleCompileOptions moduleCompileOptions,
            OptixPipelineCompileOptions pipelineCompileOptions,
            OptixCurveEndcapFlags curveEndcapFlags = OptixCurveEndcapFlags.Default)
            where TLaunchParams : unmanaged
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            using var closestHitModule =
                deviceContext.CreateModule(
                    closestHitKernel,
                    OptixKernel.CLOSESTHIT_PREFIX,
                    moduleCompileOptions,
                    pipelineCompileOptions,
                    out var closestHitEntryFunctionName);
            using var chName = SafeHGlobal.FromString(closestHitEntryFunctionName);

            using var anyHitModule =
                deviceContext.CreateModule(
                   anyHitKernel,
                    OptixKernel.ANYHIT_PREFIX,
                    moduleCompileOptions,
                    pipelineCompileOptions,
                    out var anyHitEntryFunctionName);
            using var ahName = SafeHGlobal.FromString(anyHitEntryFunctionName);

            using var builtinModule = deviceContext.GetBuiltinISModule(
                primitiveType, moduleCompileOptions, pipelineCompileOptions, curveEndcapFlags);

            using var programGroup = deviceContext.CreateProgramGroup(
                new OptixProgramGroupDesc()
                {
                    Kind = OptixProgramGroupKind.Hitgroup,
                    Hitgroup =
                        new OptixProgramGroupHitgroup()
                        {
                            ModuleCH = closestHitModule.ModulePtr,
                            EntryFunctionNameCH = chName,
                            ModuleAH = anyHitModule.ModulePtr,
                            EntryFunctionNameAH = ahName,
                            ModuleIS = builtinModule.ModulePtr,
                            EntryFunctionNameIS = IntPtr.Zero
                        }
                });

            return new OptixKernel(
                new[]
                {
                    closestHitModule.Transfer(),
                    anyHitModule.Transfer(),
                    builtinModule.Transfer()
                },
                programGroup.Transfer());
        }

        /// <summary>
        /// Creates a new OptiX raygen kernel using compile options stored in the device context.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="raygenKernel">The raygen kernel.</param>
        /// <returns>The raygen kernel.</returns>
        [CLSCompliant(false)]
        public static OptixKernel CreateRaygenKernel<TLaunchParams>(
            this OptixDeviceContext deviceContext,
            Action<TLaunchParams> raygenKernel)
            where TLaunchParams : unmanaged
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (deviceContext.ModuleCompileOptions == null)
                throw new InvalidOperationException("Module compile options not configured in device context. Call WithModuleCompileOptions() first.");
            if (deviceContext.PipelineCompileOptions == null)
                throw new InvalidOperationException("Pipeline compile options not configured in device context. Call WithPipelineCompileOptions() first.");

            return deviceContext.CreateRaygenKernel(
                raygenKernel,
                deviceContext.ModuleCompileOptions.Value,
                deviceContext.PipelineCompileOptions.Value);
        }

        /// <summary>
        /// Creates a new OptiX miss kernel using compile options stored in the device context.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="missKernel">The miss kernel.</param>
        /// <returns>The miss kernel.</returns>
        [CLSCompliant(false)]
        public static OptixKernel CreateMissKernel<TLaunchParams>(
            this OptixDeviceContext deviceContext,
            Action<TLaunchParams> missKernel)
            where TLaunchParams : unmanaged
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (deviceContext.ModuleCompileOptions == null)
                throw new InvalidOperationException("Module compile options not configured in device context. Call WithModuleCompileOptions() first.");
            if (deviceContext.PipelineCompileOptions == null)
                throw new InvalidOperationException("Pipeline compile options not configured in device context. Call WithPipelineCompileOptions() first.");

            return deviceContext.CreateMissKernel(
                missKernel,
                deviceContext.ModuleCompileOptions.Value,
                deviceContext.PipelineCompileOptions.Value);
        }

        /// <summary>
        /// Creates a new OptiX hitgroup kernel using compile options stored in the device context.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="closestHitKernel">The closest hit kernel.</param>
        /// <param name="anyHitKernel">The any hit kernel.</param>
        /// <param name="intersectionKernel">The intersection kernel.</param>
        /// <returns>The hitgroup kernel.</returns>
        [CLSCompliant(false)]
        public static OptixKernel CreateHitgroupKernel<TLaunchParams>(
            this OptixDeviceContext deviceContext,
            Action<TLaunchParams> closestHitKernel,
            Action<TLaunchParams> anyHitKernel,
            Action<TLaunchParams> intersectionKernel)
            where TLaunchParams : unmanaged
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (deviceContext.ModuleCompileOptions == null)
                throw new InvalidOperationException("Module compile options not configured in device context. Call WithModuleCompileOptions() first.");
            if (deviceContext.PipelineCompileOptions == null)
                throw new InvalidOperationException("Pipeline compile options not configured in device context. Call WithPipelineCompileOptions() first.");

            return deviceContext.CreateHitgroupKernel(
                closestHitKernel,
                anyHitKernel,
                intersectionKernel,
                deviceContext.ModuleCompileOptions.Value,
                deviceContext.PipelineCompileOptions.Value);
        }

        /// <summary>
        /// Creates a new OptiX module.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="kernel">The kernel.</param>
        /// <param name="kernelPrefix">The prefix to the kernel name.</param>
        /// <param name="moduleCompileOptions">The module compile options.</param>
        /// <param name="pipelineCompileOptions">The pipeline compile options.</param>
        /// <param name="entryFunctionName">Filled in with the function name.</param>
        /// <returns>The module.</returns>
        private static OptixModule CreateModule<TLaunchParams>(
            this OptixDeviceContext deviceContext,
            Action<TLaunchParams> kernel,
            string kernelPrefix,
            OptixModuleCompileOptions moduleCompileOptions,
            OptixPipelineCompileOptions pipelineCompileOptions,
            out string? entryFunctionName)
            where TLaunchParams : unmanaged
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            if (kernel?.Method == null)
            {
                entryFunctionName = null;
                return OptixModule.CreateEmpty();
            }

            // Compile the action into PTX
            var ptxAssembly =
                deviceContext.GeneratePTX(
                    kernel,
                    kernelPrefix,
                    out entryFunctionName);

            if (DumpGeneratedPtx)
            {
                System.IO.File.WriteAllText(
                    System.IO.Path.Combine(
                        System.IO.Path.GetTempPath(),
                        $"ilgpu-optix-debug-{kernelPrefix}-{entryFunctionName}.ptx"),
                    ptxAssembly);
            }

            return deviceContext.CreateModule(
                moduleCompileOptions,
                pipelineCompileOptions,
                ptxAssembly);
        }

        /// <summary>
        /// Creates a new OptiX module.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="moduleCompileOptions">The module compile options.</param>
        /// <param name="pipelineCompileOptions">The pipeline compile options.</param>
        /// <param name="ptxString">The module PTX code.</param>
        /// <param name="functionName">The entry function name.</param>
        /// <returns>The module.</returns>
        [CLSCompliant(false)]
        public static OptixModule CreateModule(
            this OptixDeviceContext deviceContext,
            OptixModuleCompileOptions moduleCompileOptions,
            OptixPipelineCompileOptions pipelineCompileOptions,
            string ptxString)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            var result = OptixAPI.Current.ModuleCreate(
                deviceContext.DeviceContextPtr,
                moduleCompileOptions,
                pipelineCompileOptions,
                ptxString,
                out var module,
                out var logString);
            OptixException.ThrowIfFailed(result, logString);
            return new OptixModule(module);
        }

        // Maximum additional OptixTasks OptixAPI.TaskExecute may hand back per call -
        // OptiX's own doc comment on optixTaskExecute doesn't specify a recommended
        // size, so this follows the pattern of a small fixed buffer (matches other
        // fixed-capacity out-array idioms already used elsewhere in this codebase).
        private const int MaxAdditionalTasksPerExecute = 32;

        /// <summary>
        /// Creates a new OptiX module using task-based (parallelizable) compilation -
        /// drives the whole task graph to completion internally (fanning it out across
        /// the .NET thread pool via nested <see cref="System.Threading.Tasks.Task.Run(Action)"/>
        /// calls) and returns only once the module has finished compiling, same
        /// synchronous contract as <see cref="CreateModule(OptixDeviceContext, OptixModuleCompileOptions, OptixPipelineCompileOptions, string)"/> -
        /// no <c>OptixTask</c>/<c>IntPtr</c> handles are ever exposed to the caller.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="moduleCompileOptions">The module compile options.</param>
        /// <param name="pipelineCompileOptions">The pipeline compile options.</param>
        /// <param name="ptxString">The module PTX code.</param>
        /// <param name="taskCount">Filled in with how many OptixTasks were executed.</param>
        /// <returns>The module.</returns>
        [CLSCompliant(false)]
        public static OptixModule CreateModuleWithTasks(
            this OptixDeviceContext deviceContext,
            OptixModuleCompileOptions moduleCompileOptions,
            OptixPipelineCompileOptions pipelineCompileOptions,
            string ptxString,
            out int taskCount)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            var result = OptixAPI.Current.ModuleCreateWithTasks(
                deviceContext.DeviceContextPtr,
                moduleCompileOptions,
                pipelineCompileOptions,
                ptxString,
                out var module,
                out var firstTask,
                out var logString);
            OptixException.ThrowIfFailed(result, logString);

            var executedCount = new int[1];
            ExecuteTaskGraph(firstTask, executedCount);
            taskCount = executedCount[0];

            OptixException.ThrowIfFailed(
                OptixAPI.Current.ModuleGetCompilationState(module, out var state));
            if (state != OptixModuleCompileState.Completed)
            {
                OptixAPI.Current.ModuleDestroy(module);
                throw new OptixException(
                    $"Task-based module compilation did not complete successfully " +
                    $"(state={state}).{Environment.NewLine}{logString}");
            }

            return new OptixModule(module);
        }

        // Fans a task graph out across the .NET thread pool: each OptixTask may spawn
        // more tasks when executed (optixTaskExecute's additionalTasks out-param),
        // executed as child Task.Run calls awaited via Task.WaitAll - matches
        // optix_host.h's own guidance on optixTaskExecute ("Each task can be executed
        // in parallel and in any order"), letting the thread pool bound actual
        // parallelism instead of a hand-rolled worker-thread count.
        private static void ExecuteTaskGraph(IntPtr rootTask, int[] executedCount)
        {
            void Recurse(IntPtr task)
            {
                var additionalTasks = new IntPtr[MaxAdditionalTasksPerExecute];
                OptixException.ThrowIfFailed(
                    OptixAPI.Current.TaskExecute(
                        task, additionalTasks, MaxAdditionalTasksPerExecute, out var numCreated));
                System.Threading.Interlocked.Increment(ref executedCount[0]);

                if (numCreated == 0)
                    return;

                // optixTaskExecute's contract caps numCreated at the maxNumAdditionalTasks
                // we passed in, but clamp defensively anyway - an unexpectedly large task
                // graph should surface as the ModuleCompileState.Completed check failing
                // with a clear message below, not an out-of-bounds array read here.
                int count = System.Math.Min((int)numCreated, MaxAdditionalTasksPerExecute);
                var children = new System.Threading.Tasks.Task[count];
                for (int i = 0; i < count; i++)
                {
                    var childTask = additionalTasks[i];
                    children[i] = System.Threading.Tasks.Task.Run(() => Recurse(childTask));
                }
                System.Threading.Tasks.Task.WaitAll(children);
            }
            Recurse(rootTask);
        }

        /// <summary>
        /// Creates a new OptiX program group.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="programDescriptions">The program group descriptions.</param>
        /// <returns>The program group.</returns>
        [CLSCompliant(false)]
        public static OptixProgramGroup CreateProgramGroup(
            this OptixDeviceContext deviceContext,
            params OptixProgramGroupDesc[] programDescriptions)
        {
            return deviceContext.CreateProgramGroup(default, programDescriptions);
        }

        /// <summary>
        /// Creates a new OptiX program group.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="programGroupOptions">The program group options.</param>
        /// <param name="programDescriptions">The program group descriptions.</param>
        /// <returns>The program group.</returns>
        [CLSCompliant(false)]
        public static OptixProgramGroup CreateProgramGroup(
            this OptixDeviceContext deviceContext,
            OptixProgramGroupOptions programGroupOptions,
            params OptixProgramGroupDesc[] programDescriptions)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            var result =
                OptixAPI.Current.ProgramGroupCreate(
                    deviceContext.DeviceContextPtr,
                    programDescriptions,
                    programGroupOptions,
                    out var programGroup,
                    out var logString);
            OptixException.ThrowIfFailed(result, logString);
            return new OptixProgramGroup(programGroup);
        }

        /// <summary>
        /// Creates a new OptiX pipeline.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="pipelineCompileOptions">The pipeline compile options.</param>
        /// <param name="pipelineLinkOptions">The pipeline link options.</param>
        /// <param name="programGroups">The program groups.</param>
        /// <returns>The pipeline.</returns>
        [CLSCompliant(false)]
        public static OptixPipeline CreatePipeline(
            this OptixDeviceContext deviceContext,
            OptixPipelineCompileOptions pipelineCompileOptions,
            OptixPipelineLinkOptions pipelineLinkOptions,
            params OptixProgramGroup[] programGroups)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (programGroups == null)
                throw new ArgumentNullException(nameof(programGroups));

            using var programGroupsPtr = SafeHGlobal.Alloc<IntPtr>(programGroups.Length);
            IntPtr nextPtr = programGroupsPtr;

            foreach (var programGroup in programGroups)
            {
                Marshal.WriteIntPtr(nextPtr, programGroup.ProgramGroupPtr);
                nextPtr += Marshal.SizeOf<IntPtr>();
            }

            var result = OptixAPI.Current.PipelineCreate(
                deviceContext.DeviceContextPtr,
                pipelineCompileOptions,
                pipelineLinkOptions,
                programGroupsPtr,
                (uint)programGroups.Length,
                out var pipeline,
                out var logString);
            OptixException.ThrowIfFailed(result, logString);
            return new OptixPipeline(pipeline);
        }

        /// <summary>
        /// Calculates accelleration structure size
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="accelOptions">The acceleration structure build options.</param>
        /// <param name="buildInputs">The build inputs.</param>
        /// <returns>The acceleration structure size output.</returns>
        [CLSCompliant(false)]
        public unsafe static OptixAccelBufferSizes AccelComputeMemoryUsage(
            this OptixDeviceContext deviceContext,
            OptixAccelBuildOptions accelOptions,
            params OptixBuildInput[] buildInputs)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (buildInputs == null)
                throw new ArgumentNullException(nameof(buildInputs));

            using var accelBuildOptions = SafeHGlobal.AllocFrom(accelOptions);
            using var accelBuildInputs = SafeHGlobal.AllocFrom<OptixBuildInput>(buildInputs);

            var bufferSizes = stackalloc OptixAccelBufferSizes[1];
            OptixException.ThrowIfFailed(
                OptixAPI.Current.AccelComputeMemoryUsage(
                    deviceContext.DeviceContextPtr,
                    accelBuildOptions,
                    accelBuildInputs,
                    (uint)buildInputs.Length,
                    new IntPtr(bufferSizes)));
            return bufferSizes[0];
        }

        /// <summary>
        /// Builds Acceleration Structure
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="stream">The current cuda stream.</param>
        /// <param name="accelOptions">The acceleration structure build options.</param>
        /// <param name="buildInputs">The build inputs.</param>
        /// <param name="tempBuffer">The temp build buffer, after this call the temp buffer is filled with garbage.</param>
        /// <param name="outputBuffer">The build output buffer.</param>
        /// <returns>The output device pointer</returns>
        [CLSCompliant(false)]
        public unsafe static IntPtr AccelBuild(
            this OptixDeviceContext deviceContext,
            AcceleratorStream stream,
            OptixAccelBuildOptions accelOptions,
            ReadOnlySpan<OptixBuildInput> buildInputs,
            ArrayView1D<byte, Stride1D.Dense> tempBuffer,
            ArrayView1D<byte, Stride1D.Dense> outputBuffer,
            ReadOnlySpan<OptixAccelEmitDesc> emittedProperties)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (stream is not CudaStream cudaStream)
                throw new ArgumentOutOfRangeException(nameof(stream));

            using var accelBuildOptions = SafeHGlobal.AllocFrom(accelOptions);
            using var accelBuildInputs = SafeHGlobal.AllocFrom(buildInputs);

            // SafeHGlobal.AllocFrom always Marshal.AllocHGlobal's a real (non-null)
            // block, even for a zero-length span - OptiX's validator rejects a non-null
            // emittedProperties pointer when numEmittedProperties is 0 ("emittedProperties
            // is non-null but numEmittedProperties is 0"), so an empty span must pass a
            // true null pointer instead of an empty-but-allocated one.
            using var emittedPropertiesInputs = emittedProperties.IsEmpty
                ? null
                : SafeHGlobal.AllocFrom(emittedProperties);
            IntPtr emittedPropertiesPtr = emittedProperties.IsEmpty ? IntPtr.Zero : emittedPropertiesInputs!;

            var asHandle = stackalloc IntPtr[1];
            OptixException.ThrowIfFailed(
                OptixAPI.Current.AccelBuild(
                    deviceContext.DeviceContextPtr,
                    cudaStream.StreamPtr,
                    accelBuildOptions,
                    accelBuildInputs,
                    (uint)buildInputs.Length,
                    tempBuffer.BaseView.LoadEffectiveAddressAsPtr(),
                    (ulong)tempBuffer.LengthInBytes,
                    outputBuffer.BaseView.LoadEffectiveAddressAsPtr(),
                    (ulong)outputBuffer.LengthInBytes,
                    new IntPtr(asHandle),
                    emittedPropertiesPtr,
                    (uint)emittedProperties.Length));
            return asHandle[0];
        }

        /// <summary>
        /// Compacts an acceleration structure into a smaller output buffer, returning
        /// the new (compacted) traversable handle. The input handle must come from an
        /// AccelBuild call whose accelOptions included AllowCompaction and whose
        /// emittedProperties included a CompactedSize entry - outputBuffer must be at
        /// least that emitted size (read back from device memory by the caller after
        /// the build's stream has been synchronized).
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="stream">The current cuda stream.</param>
        /// <param name="inputHandle">The uncompacted traversable handle from AccelBuild.</param>
        /// <param name="outputBuffer">The compacted output buffer.</param>
        /// <returns>The new (compacted) traversable handle.</returns>
        [CLSCompliant(false)]
        public unsafe static IntPtr AccelCompact(
            this OptixDeviceContext deviceContext,
            AcceleratorStream stream,
            IntPtr inputHandle,
            ArrayView1D<byte, Stride1D.Dense> outputBuffer)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (stream is not CudaStream cudaStream)
                throw new ArgumentOutOfRangeException(nameof(stream));

            var asHandle = stackalloc IntPtr[1];
            OptixException.ThrowIfFailed(
                OptixAPI.Current.AccelCompact(
                    deviceContext.DeviceContextPtr,
                    cudaStream.StreamPtr,
                    unchecked((ulong)inputHandle.ToInt64()),
                    outputBuffer.BaseView.LoadEffectiveAddressAsPtr(),
                    (ulong)outputBuffer.LengthInBytes,
                    new IntPtr(asHandle)));
            return asHandle[0];
        }

        /// <summary>
        /// Retrieves relocation info for an acceleration structure - opaque data
        /// describing which devices/drivers it can be relocated to/between.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="handle">The traversable handle of the built acceleration structure.</param>
        [CLSCompliant(false)]
        public unsafe static OptixRelocationInfo AccelGetRelocationInfo(
            this OptixDeviceContext deviceContext,
            IntPtr handle)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            var info = stackalloc OptixRelocationInfo[1];
            OptixException.ThrowIfFailed(
                OptixAPI.Current.AccelGetRelocationInfo(
                    deviceContext.DeviceContextPtr,
                    unchecked((ulong)handle.ToInt64()),
                    new IntPtr(info)));
            return info[0];
        }

        /// <summary>
        /// Checks whether an acceleration structure described by <paramref name="info"/>
        /// (from <see cref="AccelGetRelocationInfo"/>) can be relocated on this device
        /// context - i.e. built on a matching device/driver combination.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="info">Relocation info from <see cref="AccelGetRelocationInfo"/>.</param>
        [CLSCompliant(false)]
        public unsafe static bool CheckRelocationCompatibility(
            this OptixDeviceContext deviceContext,
            OptixRelocationInfo info)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            using var infoPtr = SafeHGlobal.AllocFrom(info);
            OptixException.ThrowIfFailed(
                OptixAPI.Current.CheckRelocationCompatibility(
                    deviceContext.DeviceContextPtr,
                    infoPtr,
                    out int compatible));
            return compatible != 0;
        }

        /// <summary>
        /// Relocates an acceleration structure into <paramref name="targetAccel"/>,
        /// returning the relocated traversable handle. <paramref name="relocateInputs"/>
        /// must have exactly one entry per build input the source acceleration
        /// structure was built with, in the same order.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="stream">The current CUDA stream.</param>
        /// <param name="info">Relocation info from <see cref="AccelGetRelocationInfo"/>.</param>
        /// <param name="relocateInputs">One entry per original build input.</param>
        /// <param name="targetAccel">
        /// The target device buffer to relocate into - must be exactly as large as the
        /// source acceleration structure's own output buffer.
        /// </param>
        [CLSCompliant(false)]
        public unsafe static IntPtr AccelRelocate(
            this OptixDeviceContext deviceContext,
            AcceleratorStream stream,
            OptixRelocationInfo info,
            ReadOnlySpan<OptixRelocateInput> relocateInputs,
            ArrayView1D<byte, Stride1D.Dense> targetAccel)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (stream is not CudaStream cudaStream)
                throw new ArgumentOutOfRangeException(nameof(stream));
            if (relocateInputs.IsEmpty)
                throw new ArgumentException("At least one relocate input is required.", nameof(relocateInputs));

            using var infoPtr = SafeHGlobal.AllocFrom(info);
            using var relocateInputsPtr = SafeHGlobal.AllocFrom(relocateInputs);

            var targetHandle = stackalloc IntPtr[1];
            OptixException.ThrowIfFailed(
                OptixAPI.Current.AccelRelocate(
                    deviceContext.DeviceContextPtr,
                    cudaStream.StreamPtr,
                    infoPtr,
                    relocateInputsPtr,
                    (ulong)relocateInputs.Length,
                    targetAccel.BaseView.LoadEffectiveAddressAsPtr(),
                    (ulong)targetAccel.LengthInBytes,
                    new IntPtr(targetHandle)));
            return targetHandle[0];
        }

        /// <summary>
        /// Emits a single post-build property (e.g. <see cref="OptixAccelPropertyType.Aabbs"/>)
        /// for an acceleration structure, outside of the emittedProperties list passed
        /// to its AccelBuild call.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="stream">The current CUDA stream.</param>
        /// <param name="handle">The traversable handle of the built acceleration structure.</param>
        /// <param name="emittedProperty">The property to emit, and where to write its result.</param>
        [CLSCompliant(false)]
        public unsafe static void AccelEmitProperty(
            this OptixDeviceContext deviceContext,
            AcceleratorStream stream,
            IntPtr handle,
            OptixAccelEmitDesc emittedProperty)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (stream is not CudaStream cudaStream)
                throw new ArgumentOutOfRangeException(nameof(stream));

            using var emittedPropertyPtr = SafeHGlobal.AllocFrom(emittedProperty);
            OptixException.ThrowIfFailed(
                OptixAPI.Current.AccelEmitProperty(
                    deviceContext.DeviceContextPtr,
                    cudaStream.StreamPtr,
                    unchecked((ulong)handle.ToInt64()),
                    emittedPropertyPtr));
        }

        /// <summary>
        /// Converts a raw device pointer to a static/motion transform struct (already
        /// uploaded by the caller) into a traversable handle, for use as an
        /// <see cref="OptixInstance.TraversableHandle"/> or another transform node's
        /// child. Prefer <see cref="OptixMotionTransformBuilder.BuildMatrixMotionTransform"/>
        /// over calling this directly - it also owns the device buffer.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="devicePointer">
        /// Device pointer to the transform struct - must be a multiple of 64 bytes
        /// (OPTIX_TRANSFORM_BYTE_ALIGNMENT).
        /// </param>
        /// <param name="traversableType">The kind of transform struct at <paramref name="devicePointer"/>.</param>
        [CLSCompliant(false)]
        public static IntPtr ConvertPointerToTraversableHandle(
            this OptixDeviceContext deviceContext,
            IntPtr devicePointer,
            OptixTraversableType traversableType)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            OptixException.ThrowIfFailed(
                OptixAPI.Current.ConvertPointerToTraversableHandle(
                    deviceContext.DeviceContextPtr,
                    unchecked((ulong)devicePointer.ToInt64()),
                    traversableType,
                    out var traversableHandle));
            return unchecked((IntPtr)(long)traversableHandle);
        }

        /// <summary>
        /// Computes memory requirements for building an opacity micromap array. Prefer
        /// <see cref="OptixOpacityMicromapBuilder.BuildUniformStateArray"/> over
        /// calling this directly.
        /// </summary>
        [CLSCompliant(false)]
        public static OptixMicromapBufferSizes OpacityMicromapArrayComputeMemoryUsage(
            this OptixDeviceContext deviceContext,
            OptixOpacityMicromapArrayBuildInput buildInput)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            OptixException.ThrowIfFailed(
                OptixAPI.Current.OpacityMicromapArrayComputeMemoryUsage(
                    deviceContext.DeviceContextPtr, buildInput, out var bufferSizes));
            return bufferSizes;
        }

        /// <summary>
        /// Builds an opacity micromap array. Prefer
        /// <see cref="OptixOpacityMicromapBuilder.BuildUniformStateArray"/> over
        /// calling this directly.
        /// </summary>
        [CLSCompliant(false)]
        public static void OpacityMicromapArrayBuild(
            this OptixDeviceContext deviceContext,
            AcceleratorStream stream,
            OptixOpacityMicromapArrayBuildInput buildInput,
            OptixMicromapBuffers buffers)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (stream is not CudaStream cudaStream)
                throw new ArgumentOutOfRangeException(nameof(stream));

            OptixException.ThrowIfFailed(
                OptixAPI.Current.OpacityMicromapArrayBuild(
                    deviceContext.DeviceContextPtr, cudaStream.StreamPtr, buildInput, buffers));
        }

        /// <summary>
        /// Creates a new OptiX denoiser.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="modelKind">The denoiser model kind.</param>
        /// <param name="options">The denoiser options.</param>
        /// <returns>The denoiser.</returns>
        [CLSCompliant(false)]
        public static OptixDenoiser CreateDenoiser(
            this OptixDeviceContext deviceContext,
            OptixDenoiserModelKind modelKind,
            OptixDenoiserOptions options)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            var result = OptixAPI.Current.DenoiserCreate(
                deviceContext.DeviceContextPtr,
                modelKind,
                options,
                out var denoiser);
            OptixException.ThrowIfFailed(result);
            return new OptixDenoiser(denoiser);
        }

        /// <summary>
        /// Creates a new OptiX denoiser from a user-supplied model blob
        /// (optixDenoiserCreateWithUserModel) instead of a built-in model kind.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="userModelData">The user model training data blob.</param>
        /// <returns>The denoiser.</returns>
        [CLSCompliant(false)]
        public static OptixDenoiser CreateDenoiserWithUserModel(
            this OptixDeviceContext deviceContext,
            byte[] userModelData)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (userModelData == null || userModelData.Length == 0)
                throw new ArgumentException(
                    "User model data must not be null or empty.", nameof(userModelData));

            using var dataPtr = SafeHGlobal.AllocFrom<byte>(userModelData);
            var result = OptixAPI.Current.DenoiserCreateWithUserModel(
                deviceContext.DeviceContextPtr,
                dataPtr,
                (ulong)userModelData.Length,
                out var denoiser);
            OptixException.ThrowIfFailed(result);
            return new OptixDenoiser(denoiser);
        }

        /// <summary>
        /// Queries a device/driver property (optixDeviceContextGetProperty) - driver
        /// limits like max trace depth, RT core version, or feature support flags
        /// (SER, cooperative vectors, cluster accels). Every property is a 4-byte
        /// unsigned value.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="property">The property to query.</param>
        [CLSCompliant(false)]
        public static uint GetProperty(
            this OptixDeviceContext deviceContext,
            OptixDeviceProperty property)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            OptixException.ThrowIfFailed(
                OptixAPI.Current.DeviceContextGetProperty(
                    deviceContext.DeviceContextPtr, property, out var value));
            return value;
        }

        /// <summary>
        /// Enables or disables the OptiX compilation disk cache
        /// (optixDeviceContextSetCacheEnabled).
        /// </summary>
        [CLSCompliant(false)]
        public static void SetCacheEnabled(
            this OptixDeviceContext deviceContext,
            bool enabled)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            OptixException.ThrowIfFailed(
                OptixAPI.Current.DeviceContextSetCacheEnabled(
                    deviceContext.DeviceContextPtr, enabled));
        }

        /// <summary>
        /// Queries whether the OptiX compilation disk cache is enabled
        /// (optixDeviceContextGetCacheEnabled).
        /// </summary>
        [CLSCompliant(false)]
        public static bool GetCacheEnabled(this OptixDeviceContext deviceContext)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            OptixException.ThrowIfFailed(
                OptixAPI.Current.DeviceContextGetCacheEnabled(
                    deviceContext.DeviceContextPtr, out var enabled));
            return enabled;
        }

        /// <summary>
        /// Sets the disk cache directory (optixDeviceContextSetCacheLocation) - the
        /// directory is created if it does not exist. Throws
        /// OPTIX_ERROR_DISK_CACHE_INVALID_PATH for an unusable path.
        /// </summary>
        [CLSCompliant(false)]
        public static void SetCacheLocation(
            this OptixDeviceContext deviceContext,
            string location)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (string.IsNullOrEmpty(location))
                throw new ArgumentException(
                    "Cache location must not be null or empty.", nameof(location));

            OptixException.ThrowIfFailed(
                OptixAPI.Current.DeviceContextSetCacheLocation(
                    deviceContext.DeviceContextPtr, location));
        }

        /// <summary>
        /// Queries the disk cache directory (optixDeviceContextGetCacheLocation).
        /// </summary>
        [CLSCompliant(false)]
        public static string GetCacheLocation(this OptixDeviceContext deviceContext)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            OptixException.ThrowIfFailed(
                OptixAPI.Current.DeviceContextGetCacheLocation(
                    deviceContext.DeviceContextPtr, out var location));
            return location ?? string.Empty;
        }

        /// <summary>
        /// Sets the disk cache garbage-collection watermarks in bytes
        /// (optixDeviceContextSetCacheDatabaseSizes). Both 0 disables garbage
        /// collection; otherwise highWaterMark must exceed lowWaterMark.
        /// </summary>
        [CLSCompliant(false)]
        public static void SetCacheDatabaseSizes(
            this OptixDeviceContext deviceContext,
            ulong lowWaterMark,
            ulong highWaterMark)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            OptixException.ThrowIfFailed(
                OptixAPI.Current.DeviceContextSetCacheDatabaseSizes(
                    deviceContext.DeviceContextPtr, lowWaterMark, highWaterMark));
        }

        /// <summary>
        /// Queries the disk cache garbage-collection watermarks in bytes
        /// (optixDeviceContextGetCacheDatabaseSizes).
        /// </summary>
        [CLSCompliant(false)]
        public static (ulong LowWaterMark, ulong HighWaterMark) GetCacheDatabaseSizes(
            this OptixDeviceContext deviceContext)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            OptixException.ThrowIfFailed(
                OptixAPI.Current.DeviceContextGetCacheDatabaseSizes(
                    deviceContext.DeviceContextPtr, out var low, out var high));
            return (low, high);
        }

        /// <summary>
        /// Computes the buffer sizes for a cluster acceleration build
        /// (optixClusterAccelComputeMemoryUsage). tempUpdateSizeInBytes is always 0.
        /// Requires driver support - check
        /// <see cref="GetProperty"/>(<see cref="OptixDeviceProperty.ClusterAccel"/>)
        /// first.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="buildMode">The build mode the later build will use.</param>
        /// <param name="buildInput">The build type and per-object limits.</param>
        [CLSCompliant(false)]
        public static OptixAccelBufferSizes ClusterAccelComputeMemoryUsage(
            this OptixDeviceContext deviceContext,
            OptixClusterAccelBuildMode buildMode,
            in OptixClusterAccelBuildInput buildInput)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            using var buildInputPtr = SafeHGlobal.AllocFrom(buildInput);
            OptixException.ThrowIfFailed(
                OptixAPI.Current.ClusterAccelComputeMemoryUsage(
                    deviceContext.DeviceContextPtr,
                    buildMode,
                    buildInputPtr,
                    out var bufferSizes));
            return bufferSizes;
        }

        /// <summary>
        /// Builds cluster objects - CLASes from triangles, cluster templates, or
        /// GASes over clusters - from device-side argument arrays
        /// (optixClusterAccelBuild). This is an indirect multi-build: the per-object
        /// <c>*Args</c> records (e.g.
        /// <see cref="OptixClusterAccelBuildInputTrianglesArgs"/>) live in device
        /// memory at <paramref name="argsArray"/>. The pipeline consuming the
        /// result must be compiled with
        /// <see cref="OptixPipelineCompileOptionsBuilder.WithAllowClusteredGeometry"/>.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="stream">The current CUDA stream.</param>
        /// <param name="buildModeDesc">Where to write output for the selected mode.</param>
        /// <param name="buildInput">The build type and per-object limits.</param>
        /// <param name="argsArray">Device pointer to the per-object args array.</param>
        /// <param name="argsCountDevicePointer">
        /// Optional device pointer to a uint object count; 0 uses the build input's
        /// maxArgCount.
        /// </param>
        /// <param name="argsStrideInBytes">Optional args stride; 0 uses the natural stride.</param>
        [CLSCompliant(false)]
        public static void ClusterAccelBuild(
            this OptixDeviceContext deviceContext,
            AcceleratorStream stream,
            in OptixClusterAccelBuildModeDesc buildModeDesc,
            in OptixClusterAccelBuildInput buildInput,
            ulong argsArray,
            ulong argsCountDevicePointer = 0,
            uint argsStrideInBytes = 0)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (stream is not CudaStream cudaStream)
                throw new ArgumentOutOfRangeException(nameof(stream));

            using var buildModeDescPtr = SafeHGlobal.AllocFrom(buildModeDesc);
            using var buildInputPtr = SafeHGlobal.AllocFrom(buildInput);
            OptixException.ThrowIfFailed(
                OptixAPI.Current.ClusterAccelBuild(
                    deviceContext.DeviceContextPtr,
                    cudaStream.StreamPtr,
                    buildModeDescPtr,
                    buildInputPtr,
                    argsArray,
                    argsCountDevicePointer,
                    argsStrideInBytes));
        }

        /// <summary>
        /// Retrieves relocation info for an opacity micromap array - the OMM
        /// counterpart of <see cref="AccelGetRelocationInfo"/>.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="opacityMicromapArray">Device pointer of the built OMM array.</param>
        [CLSCompliant(false)]
        public unsafe static OptixRelocationInfo OpacityMicromapArrayGetRelocationInfo(
            this OptixDeviceContext deviceContext,
            ulong opacityMicromapArray)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            var info = stackalloc OptixRelocationInfo[1];
            OptixException.ThrowIfFailed(
                OptixAPI.Current.OpacityMicromapArrayGetRelocationInfo(
                    deviceContext.DeviceContextPtr,
                    opacityMicromapArray,
                    new IntPtr(info)));
            return info[0];
        }

        /// <summary>
        /// Updates a relocated opacity micromap array - the OMM counterpart of
        /// <see cref="AccelRelocate"/>. Does not copy the array bytes; the caller
        /// must have copied them into <paramref name="targetArray"/> first.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="stream">The current CUDA stream.</param>
        /// <param name="info">Relocation info from <see cref="OpacityMicromapArrayGetRelocationInfo"/>.</param>
        /// <param name="targetArray">
        /// The target device buffer holding the copied OMM array bytes - must be
        /// exactly as large as the source array.
        /// </param>
        [CLSCompliant(false)]
        public static void OpacityMicromapArrayRelocate(
            this OptixDeviceContext deviceContext,
            AcceleratorStream stream,
            OptixRelocationInfo info,
            ArrayView1D<byte, Stride1D.Dense> targetArray)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (stream is not CudaStream cudaStream)
                throw new ArgumentOutOfRangeException(nameof(stream));

            using var infoPtr = SafeHGlobal.AllocFrom(info);
            OptixException.ThrowIfFailed(
                OptixAPI.Current.OpacityMicromapArrayRelocate(
                    deviceContext.DeviceContextPtr,
                    cudaStream.StreamPtr,
                    infoPtr,
                    unchecked((ulong)targetArray.BaseView.LoadEffectiveAddressAsPtr().ToInt64()),
                    (ulong)targetArray.LengthInBytes));
        }
    }
}
