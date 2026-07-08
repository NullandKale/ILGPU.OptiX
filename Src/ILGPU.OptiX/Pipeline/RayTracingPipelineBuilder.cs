// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: RayTracingPipelineBuilder.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Configures one ray type within a <see cref="RayTracingPipelineBuilder{TLaunchParams}"/> -
    /// its payload struct (which determines its register count), miss program, and
    /// hit-group program(s). A ray type's position in the builder's declaration order
    /// becomes its SBT index (<see cref="RayTracingPipeline{TLaunchParams}.RayTypeIndex"/>),
    /// matching every sample's existing "radiance = 0, shadow = 1" convention.
    /// </summary>
    public sealed class RayTypeBuilder<TLaunchParams> where TLaunchParams : unmanaged
    {
        internal string Name { get; }
        internal int PayloadCount { get; private set; }
        internal Action<TLaunchParams>? MissKernel { get; private set; }
        internal Action<TLaunchParams>? ClosestHitKernel { get; private set; }
        internal Action<TLaunchParams>? AnyHitKernel { get; private set; }
        internal Action<TLaunchParams>? IntersectionKernel { get; private set; }
        internal Type? HitGroupDataType { get; private set; }

        internal RayTypeBuilder(string name)
        {
            Name = name;
        }

        /// <summary>
        /// Declares this ray type's payload struct. <see cref="OptixTrace.Trace{T}"/>
        /// calls for this ray type must use exactly this type -
        /// <c>NumPayloadValues</c> for the pipeline is derived from it, so the
        /// count-drift bug class (pipeline option vs. Trace() call site vs. hit-program
        /// reads disagreeing on register count) cannot occur.
        /// </summary>
        public RayTypeBuilder<TLaunchParams> Payload<TPayload>() where TPayload : unmanaged
        {
            PayloadCount = OptixPayloadLayout.CountOf<TPayload>();
            return this;
        }

        /// <summary>
        /// Sets this ray type's miss program.
        /// </summary>
        public RayTypeBuilder<TLaunchParams> Miss(Action<TLaunchParams> missKernel)
        {
            MissKernel = missKernel ?? throw new ArgumentNullException(nameof(missKernel));
            return this;
        }

        /// <summary>
        /// Sets this ray type's hit-group programs and binds the per-material record
        /// data type. <see cref="RayTracingPipeline{TLaunchParams}.SetHitRecords{TMaterial}"/>
        /// packs one <see cref="SbtRecord{TMaterial}"/> per material per ray type using
        /// this type - every ray type on the same pipeline must agree on it.
        /// </summary>
        /// <param name="closestHit">The closest-hit program (required).</param>
        /// <param name="anyHit">The any-hit program, or null if unused.</param>
        /// <param name="intersection">
        /// The intersection program, or null to use OptiX's built-in triangle
        /// intersection (correct for all triangle-mesh geometry).
        /// </param>
        public RayTypeBuilder<TLaunchParams> HitGroup<TMaterial>(
            Action<TLaunchParams> closestHit,
            Action<TLaunchParams>? anyHit = null,
            Action<TLaunchParams>? intersection = null)
            where TMaterial : unmanaged
        {
            ClosestHitKernel = closestHit ?? throw new ArgumentNullException(nameof(closestHit));
            AnyHitKernel = anyHit;
            IntersectionKernel = intersection;
            HitGroupDataType = typeof(TMaterial);
            return this;
        }
    }

    /// <summary>
    /// Builds a <see cref="RayTracingPipeline{TLaunchParams}"/> from a raygen program
    /// and a set of ray types, applying every default described in
    /// docs/API_USABILITY_PLAN.md section 3.5 (launch-params variable name, triangle
    /// attribute count, single-GAS traversable graph, computed stack sizes, derived
    /// payload count) so the consumer only ever states what is actually unique to
    /// their app.
    /// </summary>
    public sealed class RayTracingPipelineBuilder<TLaunchParams> where TLaunchParams : unmanaged
    {
        private readonly List<RayTypeBuilder<TLaunchParams>> rayTypes = new List<RayTypeBuilder<TLaunchParams>>();
        private Action<TLaunchParams>? raygenKernel;
        private uint maxTraceDepth = 1;
        private OptixModuleCompileOptions moduleOptions = OptixCompilePresets.Release;
        private OptixTraversableGraphFlags traversableGraphFlags =
            OptixTraversableGraphFlags.OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        private OptixExceptionFlags exceptionFlags = OptixExceptionFlags.OPTIX_EXCEPTION_FLAG_NONE;

        /// <summary>
        /// Sets the raygen program. Required.
        /// </summary>
        public RayTracingPipelineBuilder<TLaunchParams> Raygen(Action<TLaunchParams> raygen)
        {
            raygenKernel = raygen ?? throw new ArgumentNullException(nameof(raygen));
            return this;
        }

        /// <summary>
        /// Declares a ray type. Declaration order is the ray type's SBT index -
        /// declare "radiance" before "shadow" to match every existing sample's
        /// convention.
        /// </summary>
        public RayTracingPipelineBuilder<TLaunchParams> RayType(
            string name,
            Action<RayTypeBuilder<TLaunchParams>> configure)
        {
            if (string.IsNullOrEmpty(name))
                throw new ArgumentException("Ray type name must not be null or empty.", nameof(name));
            if (configure == null)
                throw new ArgumentNullException(nameof(configure));

            var builder = new RayTypeBuilder<TLaunchParams>(name);
            configure(builder);

            if (builder.MissKernel == null)
                throw new InvalidOperationException($"Ray type '{name}' has no Miss() program configured.");
            if (builder.ClosestHitKernel == null)
                throw new InvalidOperationException($"Ray type '{name}' has no HitGroup() configured.");

            rayTypes.Add(builder);
            return this;
        }

        /// <summary>
        /// Sets the maximum <c>optixTrace()</c> recursion depth. Defaults to 1
        /// (raygen traces once; hit/miss programs do not trace further rays). Also
        /// feeds the automatically computed pipeline stack sizes (see
        /// docs/API_USABILITY_PLAN.md section 3.6) - no <c>SetStackSize(...)</c> magic
        /// numbers required.
        /// </summary>
        public RayTracingPipelineBuilder<TLaunchParams> MaxTraceDepth(uint depth)
        {
            maxTraceDepth = depth;
            return this;
        }

        /// <summary>
        /// Overrides the module compile options (default:
        /// <see cref="OptixCompilePresets.Release"/>, which also means
        /// <c>MaxRegisterCount = 0</c> - i.e. no limit - rather than the
        /// cargo-culted 50 every sample used to hard-code).
        /// </summary>
        public RayTracingPipelineBuilder<TLaunchParams> ModuleOptions(OptixModuleCompileOptions options)
        {
            moduleOptions = options;
            return this;
        }

        /// <summary>
        /// Overrides the traversable graph flags (default: single-GAS). Use
        /// <see cref="OptixTraversableGraphFlags.OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING"/>
        /// when launching against an instance-acceleration-structure traversable.
        /// </summary>
        public RayTracingPipelineBuilder<TLaunchParams> WithTraversableGraphFlags(
            OptixTraversableGraphFlags flags)
        {
            traversableGraphFlags = flags;
            return this;
        }

        /// <summary>
        /// Overrides the pipeline exception flags (default: none). Also switch on
        /// device-context validation mode for meaningful results from most flags.
        /// </summary>
        public RayTracingPipelineBuilder<TLaunchParams> WithExceptionFlags(OptixExceptionFlags flags)
        {
            exceptionFlags = flags;
            return this;
        }

        internal RayTracingPipeline<TLaunchParams> Build(OptixDeviceContext deviceContext)
        {
            if (raygenKernel == null)
                throw new InvalidOperationException("Raygen() must be configured before Build().");
            if (rayTypes.Count == 0)
                throw new InvalidOperationException("At least one RayType() must be configured before Build().");

            var maxPayloadCount = rayTypes.Max(r => r.PayloadCount);
            var pipelineOptions = new OptixPipelineCompileOptionsBuilder()
                .WithNumPayloadValues(maxPayloadCount)
                .WithNumAttributeValues(2)
                .WithTraversableGraphFlags(traversableGraphFlags)
                .WithExceptionFlags(exceptionFlags)
                .Build();

            var linkOptions = new OptixPipelineLinkOptions { MaxTraceDepth = maxTraceDepth };

            var allKernels = new List<OptixKernel>();
            OptixKernel? raygen = null;
            var builtRayTypes = new List<BuiltRayType<TLaunchParams>>();

            try
            {
                raygen = deviceContext.CreateRaygenKernel(raygenKernel, moduleOptions, pipelineOptions);
                allKernels.Add(raygen);

                for (var i = 0; i < rayTypes.Count; i++)
                {
                    var rt = rayTypes[i];
                    var missKernel = deviceContext.CreateMissKernel(rt.MissKernel!, moduleOptions, pipelineOptions);
                    allKernels.Add(missKernel);

                    var hitgroupKernel = deviceContext.CreateHitgroupKernel(
                        rt.ClosestHitKernel!,
                        rt.AnyHitKernel!,
                        rt.IntersectionKernel!,
                        moduleOptions,
                        pipelineOptions);
                    allKernels.Add(hitgroupKernel);

                    builtRayTypes.Add(new BuiltRayType<TLaunchParams>(
                        rt.Name,
                        i,
                        missKernel,
                        hitgroupKernel,
                        rt.HitGroupDataType!));
                }

                var pipeline = deviceContext.CreatePipeline(
                    pipelineOptions,
                    linkOptions,
                    allKernels.Select(k => k.ProgramGroup).ToArray());

                OptixStackSizeUtil.ComputeAndApply(pipeline, allKernels, maxTraceDepth);

                return new RayTracingPipeline<TLaunchParams>(
                    deviceContext,
                    pipeline,
                    raygen,
                    builtRayTypes);
            }
            catch
            {
                foreach (var kernel in allKernels)
                    kernel.Dispose();
                throw;
            }
        }
    }

    /// <summary>
    /// A fully compiled ray type, ready for <see cref="RayTracingPipeline{TLaunchParams}"/>
    /// to own.
    /// </summary>
    internal sealed class BuiltRayType<TLaunchParams> where TLaunchParams : unmanaged
    {
        public string Name { get; }
        public int Index { get; }
        public OptixKernel MissKernel { get; }
        public OptixKernel HitGroupKernel { get; }
        public Type HitGroupDataType { get; }

        public BuiltRayType(
            string name,
            int index,
            OptixKernel missKernel,
            OptixKernel hitGroupKernel,
            Type hitGroupDataType)
        {
            Name = name;
            Index = index;
            MissKernel = missKernel;
            HitGroupKernel = hitGroupKernel;
            HitGroupDataType = hitGroupDataType;
        }
    }
}
