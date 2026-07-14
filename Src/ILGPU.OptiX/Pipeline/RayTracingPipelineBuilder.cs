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
using ILGPU.OptiX.DeviceApi;
using ILGPU.OptiX.Native;
using ILGPU.OptiX.AccelStructures;
using ILGPU.OptiX.Util;

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
        internal OptixPayloadTypeID TypeId { get; private set; } = OptixPayloadTypeID.Default;
        internal Action<TLaunchParams>? MissKernel { get; private set; }
        internal Type? HitGroupDataType { get; private set; }

        internal List<HitGroupDeclaration<TLaunchParams>> HitGroups { get; } =
            new List<HitGroupDeclaration<TLaunchParams>>();

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
        /// Declares this ray type's payload struct AND opts it into OptiX's typed
        /// payload dispatch under
        /// <paramref name="typeId"/> - every <c>OptixTrace.Trace(...)</c> call for
        /// this ray type must use the matching nested class
        /// (<see cref="DeviceApi.OptixTrace.Typed0"/> for <see cref="OptixPayloadTypeID.Id0"/>,
        /// <see cref="DeviceApi.OptixTrace.Typed1"/> for <see cref="OptixPayloadTypeID.Id1"/>).
        /// Every ray type on the same pipeline must either all use
        /// <see cref="Payload{TPayload}()"/> (untyped) or all use this overload with a
        /// distinct, non-<see cref="OptixPayloadTypeID.Default"/> ID each -
        /// <see cref="RayTracingPipelineBuilder{TLaunchParams}.Build"/> validates this.
        /// </summary>
        public RayTypeBuilder<TLaunchParams> Payload<TPayload>(OptixPayloadTypeID typeId) where TPayload : unmanaged
        {
            if (typeId == OptixPayloadTypeID.Default)
                throw new ArgumentException(
                    "Use Payload<TPayload>() (no typeId) for untyped payload dispatch - " +
                    "OptixPayloadTypeID.Default cannot be used with typed dispatch.",
                    nameof(typeId));

            PayloadCount = OptixPayloadLayout.CountOf<TPayload>();
            TypeId = typeId;
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
        /// Sets this ray type's (sole) hit-group programs and binds the per-material
        /// record data type. Equivalent to
        /// <see cref="HitGroup{TMaterial}(string, Action{TLaunchParams}, Action{TLaunchParams}?, Action{TLaunchParams}?)"/>
        /// with <see cref="HitGroupKind.Default"/> - the common case where a ray type
        /// has exactly one hit group (every sample except Sample13/14's
        /// custom-primitive geometry).
        /// <see cref="RayTracingPipeline{TLaunchParams}.SetHitRecords{TMaterial}(ReadOnlySpan{TMaterial}, int)"/>
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
            where TMaterial : unmanaged =>
            HitGroup<TMaterial>(HitGroupKind.Default, closestHit, anyHit, intersection);

        /// <summary>
        /// Declares one more named hit group for this ray type, in addition to any
        /// already declared - each becomes its own compiled OptiX hit-group program
        /// group, addressed by <paramref name="kind"/> from
        /// <see cref="HitGroupEntry{TMaterial}"/> when packing records via
        /// <see cref="RayTracingPipeline{TLaunchParams}.SetHitRecords{TMaterial}(ReadOnlySpan{HitGroupEntry{TMaterial}}, int)"/>.
        /// Exists for geometry that needs multiple intersection programs sharing one
        /// ray type's closest-hit/any-hit logic - e.g. one custom-primitive kind per
        /// kind name, all dispatching through the same shared closest-hit function via
        /// <c>OptixGetHitKind</c> (Sample13/14's pattern). <paramref name="closestHit"/>
        /// and <paramref name="anyHit"/> may be the exact same delegate passed to a
        /// previous <see cref="HitGroup{TMaterial}(string, Action{TLaunchParams}, Action{TLaunchParams}?, Action{TLaunchParams}?)"/>
        /// call on this ray type - the library still compiles a distinct program group
        /// per call, since only <paramref name="intersection"/> needs to differ.
        /// </summary>
        /// <param name="kind">
        /// A name unique within this ray type, identifying this hit group's compiled
        /// program group for <see cref="HitGroupEntry{TMaterial}"/>.
        /// </param>
        /// <param name="closestHit">The closest-hit program (required).</param>
        /// <param name="anyHit">The any-hit program, or null if unused.</param>
        /// <param name="intersection">
        /// The intersection program, or null to use OptiX's built-in triangle
        /// intersection.
        /// </param>
        public RayTypeBuilder<TLaunchParams> HitGroup<TMaterial>(
            string kind,
            Action<TLaunchParams> closestHit,
            Action<TLaunchParams>? anyHit = null,
            Action<TLaunchParams>? intersection = null)
            where TMaterial : unmanaged
        {
            if (string.IsNullOrEmpty(kind))
                throw new ArgumentException("Hit-group kind must not be null or empty.", nameof(kind));
            if (closestHit == null)
                throw new ArgumentNullException(nameof(closestHit));
            if (HitGroups.Exists(h => h.Kind == kind))
                throw new ArgumentException(
                    $"Ray type '{Name}' already has a HitGroup(\"{kind}\", ...) declared.",
                    nameof(kind));
            if (HitGroupDataType != null && HitGroupDataType != typeof(TMaterial))
                throw new ArgumentException(
                    $"Ray type '{Name}' was already declared with HitGroup<{HitGroupDataType.Name}>; " +
                    $"every hit group on the same ray type must use the same material type, got {typeof(TMaterial).Name}.",
                    nameof(TMaterial));

            HitGroupDataType = typeof(TMaterial);
            HitGroups.Add(new HitGroupDeclaration<TLaunchParams>(kind, closestHit, anyHit, intersection));
            return this;
        }

        /// <summary>
        /// Declares this ray type's (sole) hit-group programs for curve or sphere
        /// geometry - <paramref name="primitiveType"/>
        /// selects OptiX's own built-in intersection program instead of a compiled
        /// one, which is the only option for these primitive types.
        /// </summary>
        /// <param name="closestHit">The closest-hit program (required).</param>
        /// <param name="anyHit">The any-hit program, or null if unused.</param>
        /// <param name="primitiveType">
        /// The built-in curve/sphere type (must not be <see cref="OptixPrimitiveType.Custom"/>
        /// or <see cref="OptixPrimitiveType.Triangle"/>).
        /// </param>
        /// <param name="endcapFlags">Curve end cap flags - ignored for spheres.</param>
        public RayTypeBuilder<TLaunchParams> HitGroupWithBuiltinIS<TMaterial>(
            Action<TLaunchParams> closestHit,
            Action<TLaunchParams>? anyHit,
            OptixPrimitiveType primitiveType,
            OptixCurveEndcapFlags endcapFlags = OptixCurveEndcapFlags.Default)
            where TMaterial : unmanaged =>
            HitGroupWithBuiltinIS<TMaterial>(HitGroupKind.Default, closestHit, anyHit, primitiveType, endcapFlags);

        /// <summary>
        /// Declares one more named hit group for this ray type using OptiX's built-in
        /// intersection program for curve/sphere geometry - the named-hit-group counterpart to
        /// <see cref="HitGroupWithBuiltinIS{TMaterial}(Action{TLaunchParams}, Action{TLaunchParams}?, OptixPrimitiveType, OptixCurveEndcapFlags)"/>,
        /// for combining curve/sphere geometry with triangles or custom primitives on
        /// the same ray type (e.g. a triangle ground plane plus curve-based "hair" -
        /// each geometry kind gets its own named hit group, sharing the ray type's
        /// miss program and material data type).
        /// </summary>
        /// <param name="kind">A name unique within this ray type.</param>
        /// <param name="closestHit">The closest-hit program (required).</param>
        /// <param name="anyHit">The any-hit program, or null if unused.</param>
        /// <param name="primitiveType">
        /// The built-in curve/sphere type (must not be <see cref="OptixPrimitiveType.Custom"/>
        /// or <see cref="OptixPrimitiveType.Triangle"/>).
        /// </param>
        /// <param name="endcapFlags">Curve end cap flags - ignored for spheres.</param>
        public RayTypeBuilder<TLaunchParams> HitGroupWithBuiltinIS<TMaterial>(
            string kind,
            Action<TLaunchParams> closestHit,
            Action<TLaunchParams>? anyHit,
            OptixPrimitiveType primitiveType,
            OptixCurveEndcapFlags endcapFlags = OptixCurveEndcapFlags.Default)
            where TMaterial : unmanaged
        {
            if (string.IsNullOrEmpty(kind))
                throw new ArgumentException("Hit-group kind must not be null or empty.", nameof(kind));
            if (closestHit == null)
                throw new ArgumentNullException(nameof(closestHit));
            if (primitiveType == OptixPrimitiveType.Custom || primitiveType == OptixPrimitiveType.Triangle)
                throw new ArgumentException(
                    "Built-in intersection is only for curve/sphere primitive types, not Custom or Triangle.",
                    nameof(primitiveType));
            if (HitGroups.Exists(h => h.Kind == kind))
                throw new ArgumentException(
                    $"Ray type '{Name}' already has a HitGroup(\"{kind}\", ...) declared.",
                    nameof(kind));
            if (HitGroupDataType != null && HitGroupDataType != typeof(TMaterial))
                throw new ArgumentException(
                    $"Ray type '{Name}' was already declared with HitGroup<{HitGroupDataType.Name}>; " +
                    $"every hit group on the same ray type must use the same material type, got {typeof(TMaterial).Name}.",
                    nameof(TMaterial));

            HitGroupDataType = typeof(TMaterial);
            HitGroups.Add(new HitGroupDeclaration<TLaunchParams>(
                kind, closestHit, anyHit, primitiveType, endcapFlags));
            return this;
        }
    }

    /// <summary>
    /// One named hit-group declaration collected by <see cref="RayTypeBuilder{TLaunchParams}"/>
    /// before <see cref="RayTracingPipelineBuilder{TLaunchParams}.Build"/> compiles it into an
    /// <see cref="OptixKernel"/>.
    /// </summary>
    internal readonly struct HitGroupDeclaration<TLaunchParams> where TLaunchParams : unmanaged
    {
        public string Kind { get; }
        public Action<TLaunchParams> ClosestHit { get; }
        public Action<TLaunchParams>? AnyHit { get; }
        public Action<TLaunchParams>? Intersection { get; }

        /// <summary>
        /// Non-null when this hit group's intersection program is OptiX's own
        /// built-in module for this primitive type (curves/spheres, which have no
        /// user-suppliable intersection program) -
        /// mutually exclusive with <see cref="Intersection"/> being non-null.
        /// </summary>
        public AccelStructures.OptixPrimitiveType? BuiltinPrimitiveType { get; }
        public AccelStructures.OptixCurveEndcapFlags EndcapFlags { get; }

        public HitGroupDeclaration(
            string kind,
            Action<TLaunchParams> closestHit,
            Action<TLaunchParams>? anyHit,
            Action<TLaunchParams>? intersection)
        {
            Kind = kind;
            ClosestHit = closestHit;
            AnyHit = anyHit;
            Intersection = intersection;
            BuiltinPrimitiveType = null;
            EndcapFlags = AccelStructures.OptixCurveEndcapFlags.Default;
        }

        public HitGroupDeclaration(
            string kind,
            Action<TLaunchParams> closestHit,
            Action<TLaunchParams>? anyHit,
            AccelStructures.OptixPrimitiveType builtinPrimitiveType,
            AccelStructures.OptixCurveEndcapFlags endcapFlags)
        {
            Kind = kind;
            ClosestHit = closestHit;
            AnyHit = anyHit;
            Intersection = null;
            BuiltinPrimitiveType = builtinPrimitiveType;
            EndcapFlags = endcapFlags;
        }
    }

    /// <summary>
    /// Builds a <see cref="RayTracingPipeline{TLaunchParams}"/> from a raygen program
    /// and a set of ray types, applying sensible defaults (launch-params variable
    /// name, triangle attribute count, single-GAS traversable graph, computed stack sizes, derived
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
            OptixTraversableGraphFlags.AllowSingleGas;
        private OptixExceptionFlags exceptionFlags = OptixExceptionFlags.None;
        private uint? maxTraversableGraphDepthOverride;
        private bool compileRaygenWithTasks;
        private bool usesMotionBlur;
        private OptixPrimitiveTypeFlags primitiveTypeFlags;
        private bool allowOpacityMicromaps;

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
            if (builder.HitGroups.Count == 0)
                throw new InvalidOperationException($"Ray type '{name}' has no HitGroup() configured.");

            rayTypes.Add(builder);
            return this;
        }

        /// <summary>
        /// Sets the maximum <c>optixTrace()</c> recursion depth. Defaults to 1
        /// (raygen traces once; hit/miss programs do not trace further rays). Also
        /// feeds the automatically computed pipeline stack sizes - no
        /// <c>SetStackSize(...)</c> magic numbers required.
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
        /// <see cref="OptixTraversableGraphFlags.AllowSingleLevelInstancing"/>
        /// when launching against an instance-acceleration-structure traversable.
        /// </summary>
        public RayTracingPipelineBuilder<TLaunchParams> WithTraversableGraphFlags(
            OptixTraversableGraphFlags flags)
        {
            traversableGraphFlags = flags;
            return this;
        }

        /// <summary>
        /// Overrides the computed pipeline stack sizes' traversal-depth argument
        /// (see <see cref="OptixStackSizeUtil.ComputeAndApply"/>). Defaults to 2 when
        /// <see cref="OptixTraversableGraphFlags.AllowSingleLevelInstancing"/>
        /// is set (an IAS-then-GAS traversable graph is two levels deep) and 1
        /// otherwise - this only exists for consumers whose traversable graph is
        /// deeper still (e.g. transform traversables), which this facade does not
        /// otherwise model.
        /// </summary>
        public RayTracingPipelineBuilder<TLaunchParams> MaxTraversableGraphDepth(uint depth)
        {
            maxTraversableGraphDepthOverride = depth;
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

        /// <summary>
        /// Compiles the raygen module via OptiX's task-based (parallelizable)
        /// compilation path (<see cref="OptixDeviceContextExtensions.CreateModuleWithTasks"/>)
        /// instead of the normal single blocking call - functionally identical result,
        /// only the compilation strategy differs. The executed task count is exposed
        /// afterwards via <see cref="RayTracingPipeline{TLaunchParams}.RaygenCompileTaskCount"/>.
        /// </summary>
        public RayTracingPipelineBuilder<TLaunchParams> CompileRaygenWithTasks()
        {
            compileRaygenWithTasks = true;
            return this;
        }

        /// <summary>
        /// Marks this pipeline's traversable graph as possibly containing motion
        /// transforms - required whenever the scene
        /// includes an <c>OptixMatrixMotionTransform</c>/<c>OptixSRTMotionTransform</c>
        /// (see <see cref="AccelStructures.OptixMotionTransformBuilder"/>) or a GAS
        /// built with more than one motion key.
        /// </summary>
        public RayTracingPipelineBuilder<TLaunchParams> UsesMotionBlur()
        {
            usesMotionBlur = true;
            return this;
        }

        /// <summary>
        /// Declares which non-default primitive types
        /// this pipeline's traversable graph may contain - required alongside curve
        /// geometry (<see cref="AccelStructures.OptixAccelBuilder.AddCurves"/> +
        /// <see cref="RayTypeBuilder{TLaunchParams}.HitGroupWithBuiltinIS{TMaterial}(Action{TLaunchParams}, Action{TLaunchParams}?, OptixPrimitiveType, OptixCurveEndcapFlags)"/>).
        /// See <see cref="OptixPipelineCompileOptionsBuilder.WithUsesPrimitiveTypeFlags"/>
        /// for the "replaces, not adds to, the implicit default" caveat.
        /// </summary>
        public RayTracingPipelineBuilder<TLaunchParams> UsesPrimitiveTypes(OptixPrimitiveTypeFlags flags)
        {
            primitiveTypeFlags = flags;
            return this;
        }

        /// <summary>
        /// Marks this pipeline as possibly using opacity micromaps
        /// (<see cref="AccelStructures.OptixOpacityMicromapBuilder.BuildUniformStateArray"/>
        /// + <see cref="AccelStructures.OptixAccelBuilder.AddTriangleMesh"/>'s
        /// <c>opacityMicromapArray</c> parameter).
        /// </summary>
        public RayTracingPipelineBuilder<TLaunchParams> AllowOpacityMicromaps()
        {
            allowOpacityMicromaps = true;
            return this;
        }

        internal RayTracingPipeline<TLaunchParams> Build(OptixDeviceContext deviceContext)
        {
            if (raygenKernel == null)
                throw new InvalidOperationException("Raygen() must be configured before Build().");
            if (rayTypes.Count == 0)
                throw new InvalidOperationException("At least one RayType() must be configured before Build().");

            // Typed payload dispatch: either every
            // ray type uses the untyped Payload<T>() (the default, unchanged
            // behavior) or every ray type uses the typed Payload<T>(typeId) overload
            // with a distinct ID - no mixing, since OptixPipelineCompileOptions.numPayloadValues
            // must be zero when any module declares payload types, which would silently
            // break an untyped ray type sharing the same pipeline.
            bool usesTypedPayloads = rayTypes.Any(r => r.TypeId != OptixPayloadTypeID.Default);
            if (usesTypedPayloads)
            {
                var untyped = rayTypes.Where(r => r.TypeId == OptixPayloadTypeID.Default).ToArray();
                if (untyped.Length > 0)
                    throw new InvalidOperationException(
                        $"Ray type '{untyped[0].Name}' uses the untyped Payload<T>(), but another ray type " +
                        "on this pipeline uses typed Payload<T>(typeId) - every ray type must use one or the " +
                        "other, not a mix.");
                var duplicateTypeId = rayTypes.GroupBy(r => r.TypeId).FirstOrDefault(g => g.Count() > 1);
                if (duplicateTypeId != null)
                    throw new InvalidOperationException(
                        $"Ray types '{string.Join("', '", duplicateTypeId.Select(r => r.Name))}' all declared " +
                        $"payload type {duplicateTypeId.Key} - every ray type must use a distinct ID.");
            }

            var maxPayloadCount = rayTypes.Max(r => r.PayloadCount);
            var pipelineOptions = new OptixPipelineCompileOptionsBuilder()
                // Must be zero when typed payloads are declared on the module compile
                // options below - OptiX derives each type's own value count from its
                // OptixPayloadType.NumPayloadValues instead.
                .WithNumPayloadValues(usesTypedPayloads ? 0 : maxPayloadCount)
                .WithNumAttributeValues(2)
                .WithTraversableGraphFlags(traversableGraphFlags)
                .WithExceptionFlags(exceptionFlags)
                .WithUsesMotionBlur(usesMotionBlur)
                .WithUsesPrimitiveTypeFlags(primitiveTypeFlags)
                .WithAllowOpacityMicromaps(allowOpacityMicromaps)
                .Build();

            var linkOptions = new OptixPipelineLinkOptions { MaxTraceDepth = maxTraceDepth };

            var allKernels = new List<OptixKernel>();
            OptixKernel? raygen = null;
            var builtRayTypes = new List<BuiltRayType<TLaunchParams>>();
            var raygenCompileTaskCount = 0;

            // Raygen genuinely traces every declared type (it's the only place
            // OptixTrace.TypedN is called), so its module declares the full set. Miss/
            // hitgroup modules only ever call OptixPayloadInterop.Get/SetXxx (which
            // compile down to optixGetPayload_N/optixSetPayload_N) - the driver
            // validates those calls against EVERY type a module declares (there's no
            // device-side optixSetPayloadTypes() call wrapped here to restrict a
            // module to a subset), so declaring the full set for e.g. a 1-payload-value
            // "shadow" miss module would make its (in-range-for-radiance's-3-values,
            // out-of-range-for-shadow's-1-value) index checks fail - confirmed via a
            // real GPU run: "Requested payload value 2 of payload type 1 in
            // optixSetPayload, but only 1 values are configured in the type." Each
            // ray type's miss/hitgroup modules are therefore compiled with ONLY that
            // ray type's own single OptixPayloadType declared instead.
            using var raygenPayloadHandles = usesTypedPayloads
                ? BuildPayloadTypeHandles(rayTypes)
                : null;
            var raygenModuleOptions = moduleOptions;
            if (usesTypedPayloads)
            {
                raygenModuleOptions.NumPayloadTypes = (uint)rayTypes.Count;
                raygenModuleOptions.PayloadTypes = raygenPayloadHandles!.ArrayPtr;
            }

            try
            {
                raygen = compileRaygenWithTasks
                    ? deviceContext.CreateRaygenKernelWithTasks(
                        raygenKernel, raygenModuleOptions, pipelineOptions, out raygenCompileTaskCount)
                    : deviceContext.CreateRaygenKernel(raygenKernel, raygenModuleOptions, pipelineOptions);
                allKernels.Add(raygen);

                for (var i = 0; i < rayTypes.Count; i++)
                {
                    var rt = rayTypes[i];

                    using var rtPayloadHandles = usesTypedPayloads
                        ? BuildPayloadTypeHandles(new List<RayTypeBuilder<TLaunchParams>> { rt })
                        : null;
                    var rtModuleOptions = moduleOptions;
                    if (usesTypedPayloads)
                    {
                        rtModuleOptions.NumPayloadTypes = 1;
                        rtModuleOptions.PayloadTypes = rtPayloadHandles!.ArrayPtr;
                    }

                    var missKernel = deviceContext.CreateMissKernel(rt.MissKernel!, rtModuleOptions, pipelineOptions);
                    allKernels.Add(missKernel);

                    var hitGroupKernelsByKind = new Dictionary<string, OptixKernel>();
                    foreach (var hg in rt.HitGroups)
                    {
                        var hitgroupKernel = hg.BuiltinPrimitiveType.HasValue
                            ? deviceContext.CreateHitgroupKernelWithBuiltinIS(
                                hg.ClosestHit,
                                hg.AnyHit!,
                                hg.BuiltinPrimitiveType.Value,
                                rtModuleOptions,
                                pipelineOptions,
                                hg.EndcapFlags)
                            : deviceContext.CreateHitgroupKernel(
                                hg.ClosestHit,
                                hg.AnyHit!,
                                hg.Intersection!,
                                rtModuleOptions,
                                pipelineOptions);
                        allKernels.Add(hitgroupKernel);
                        hitGroupKernelsByKind.Add(hg.Kind, hitgroupKernel);
                    }

                    builtRayTypes.Add(new BuiltRayType<TLaunchParams>(
                        rt.Name,
                        i,
                        missKernel,
                        hitGroupKernelsByKind,
                        rt.HitGroupDataType!));
                }

                var pipeline = deviceContext.CreatePipeline(
                    pipelineOptions,
                    linkOptions,
                    allKernels.Select(k => k.ProgramGroup).ToArray());

                var maxTraversableGraphDepth = maxTraversableGraphDepthOverride ??
                    (traversableGraphFlags.HasFlag(
                        OptixTraversableGraphFlags.AllowSingleLevelInstancing)
                        ? 2u
                        : 1u);
                OptixStackSizeUtil.ComputeAndApply(pipeline, allKernels, maxTraceDepth, maxTraversableGraphDepth);

                return new RayTracingPipeline<TLaunchParams>(
                    deviceContext,
                    pipeline,
                    raygen,
                    builtRayTypes,
                    raygenCompileTaskCount);
            }
            catch
            {
                foreach (var kernel in allKernels)
                    kernel.Dispose();
                throw;
            }
        }

        // Builds the native OptixPayloadType[] array (plus each entry's own
        // per-word-semantics array) that moduleOptionsForBuild.PayloadTypes points at -
        // every word gets OptixPayloadSemantics.FullAccess (read/write from the trace
        // caller and every shader stage), matching how untyped payloads already behave
        // with no compiler-enforced per-stage restriction. Declaration order matches
        // rayTypes' order, which is also each ray type's SBT index - not otherwise
        // significant to the driver (each OptixTrace.TypedN call site names its type
        // by ID, not by array position).
        private static PayloadTypeHandles BuildPayloadTypeHandles(List<RayTypeBuilder<TLaunchParams>> rayTypes)
        {
            var semanticsHandles = new List<SafeHGlobal>();
            try
            {
                var payloadTypes = new OptixPayloadType[rayTypes.Count];
                for (int i = 0; i < rayTypes.Count; i++)
                {
                    var count = rayTypes[i].PayloadCount;
                    var semantics = new uint[count];
                    for (int w = 0; w < count; w++)
                        semantics[w] = (uint)OptixPayloadSemantics.FullAccess;

                    var semanticsHandle = SafeHGlobal.AllocFrom<uint>(semantics);
                    semanticsHandles.Add(semanticsHandle);

                    payloadTypes[i] = new OptixPayloadType
                    {
                        NumPayloadValues = (uint)count,
                        PayloadSemantics = semanticsHandle,
                    };
                }

                var arrayHandle = SafeHGlobal.AllocFrom<OptixPayloadType>(payloadTypes);
                return new PayloadTypeHandles(arrayHandle, semanticsHandles);
            }
            catch
            {
                foreach (var handle in semanticsHandles)
                    handle.Dispose();
                throw;
            }
        }

        // Owns the native buffers BuildPayloadTypeHandles allocated - the
        // OptixPayloadType[] array itself, plus each entry's own per-word semantics
        // array it points into (SafeHGlobal.AllocFrom<OptixPayloadType> only copies
        // the struct's own bytes - including each PayloadSemantics IntPtr field's
        // *value* - not the memory that IntPtr points to, so that pointee memory needs
        // its own separate, equally long-lived allocation).
        private sealed class PayloadTypeHandles : IDisposable
        {
            private readonly SafeHGlobal arrayHandle;
            private readonly List<SafeHGlobal> semanticsHandles;

            public IntPtr ArrayPtr => arrayHandle;

            public PayloadTypeHandles(SafeHGlobal arrayHandle, List<SafeHGlobal> semanticsHandles)
            {
                this.arrayHandle = arrayHandle;
                this.semanticsHandles = semanticsHandles;
            }

            public void Dispose()
            {
                arrayHandle.Dispose();
                foreach (var handle in semanticsHandles)
                    handle.Dispose();
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
        public Dictionary<string, OptixKernel> HitGroupKernelsByKind { get; }
        public Type HitGroupDataType { get; }

        public BuiltRayType(
            string name,
            int index,
            OptixKernel missKernel,
            Dictionary<string, OptixKernel> hitGroupKernelsByKind,
            Type hitGroupDataType)
        {
            Name = name;
            Index = index;
            MissKernel = missKernel;
            HitGroupKernelsByKind = hitGroupKernelsByKind;
            HitGroupDataType = hitGroupDataType;
        }
    }
}
