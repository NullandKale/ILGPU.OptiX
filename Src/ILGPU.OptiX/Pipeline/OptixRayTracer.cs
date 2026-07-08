// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixRayTracer.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using ILGPU.Util;
using System;

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Entry point for the high-level ray tracing API - owns global OptiX init and the
    /// device context, and builds <see cref="RayTracingPipeline{TLaunchParams}"/>
    /// instances. The full "90 lines of OptiX-jargon ritual" every sample used to
    /// repeat (module/pipeline/link compile options, magic stack sizes, hand-measured
    /// SBT record structs, per-launch memory leaks) is behind
    /// <see cref="CreatePipeline{TLaunchParams}"/>; everything the raw API exposes
    /// stays reachable via <see cref="DeviceContext"/> and <see cref="Accelerator"/>
    /// for consumers who need something this facade does not yet cover.
    /// </summary>
    [CLSCompliant(false)]
    public sealed class OptixRayTracer : DisposeBase
    {
        /// <summary>
        /// The CUDA accelerator this ray tracer was created on.
        /// </summary>
        public CudaAccelerator Accelerator { get; }

        /// <summary>
        /// The underlying OptiX device context - the escape hatch for anything this
        /// facade does not (yet) expose (raw module/program-group/pipeline creation,
        /// acceleration structures, denoising).
        /// </summary>
        public OptixDeviceContext DeviceContext { get; }

        private OptixRayTracer(CudaAccelerator accelerator, OptixDeviceContext deviceContext)
        {
            Accelerator = accelerator;
            DeviceContext = deviceContext;
        }

        /// <summary>
        /// Creates a new ray tracer on <paramref name="accelerator"/>. Transparently
        /// initializes the OptiX library and creates the device context - no separate
        /// init call is needed or exists; disposing the returned instance balances it.
        /// </summary>
        public static OptixRayTracer Create(
            CudaAccelerator accelerator,
            OptixDeviceContextOptions options = default)
        {
            if (accelerator == null)
                throw new ArgumentNullException(nameof(accelerator));

            var deviceContext = accelerator.CreateDeviceContext(options);
            return new OptixRayTracer(accelerator, deviceContext);
        }

        /// <summary>
        /// Builds a ray tracing pipeline for <typeparamref name="TLaunchParams"/> - see
        /// <see cref="RayTracingPipelineBuilder{TLaunchParams}"/> for the raygen/ray-type
        /// configuration surface.
        /// </summary>
        public RayTracingPipeline<TLaunchParams> CreatePipeline<TLaunchParams>(
            Action<RayTracingPipelineBuilder<TLaunchParams>> configure)
            where TLaunchParams : unmanaged
        {
            if (configure == null)
                throw new ArgumentNullException(nameof(configure));

            var builder = new RayTracingPipelineBuilder<TLaunchParams>();
            configure(builder);
            return builder.Build(DeviceContext);
        }

        /// <summary cref="DisposeBase.Dispose(bool)"/>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
                DeviceContext.Dispose();
            base.Dispose(disposing);
        }
    }
}
