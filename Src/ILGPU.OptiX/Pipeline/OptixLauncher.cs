// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixLauncher.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX.Interop;
using ILGPU.OptiX.Native;
using ILGPU.OptiX.Util;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Util;
using System;
using System.Runtime.InteropServices;

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Persistent launch facade for a fixed pipeline/SBT pair - the facade
    /// <see cref="OptixLaunchExtensions.OptixLaunch{TLaunchParams}(CudaAccelerator, OptixPipeline, TLaunchParams, OptixShaderBindingTable, uint, uint, uint)"/>'s
    /// own NOTE asks per-frame callers to prefer: the one-element launch-params device
    /// buffer and the marshaled native SBT copy are allocated once here and reused by
    /// every <see cref="Launch"/> call, instead of allocated and freed per call (freeing
    /// GPU memory forces an implicit device sync, so the extension path serializes the
    /// device on every launch). Does not take ownership of the pipeline or the SBT's
    /// record buffers - the caller keeps those alive for this launcher's lifetime.
    /// </summary>
    /// <typeparam name="TLaunchParams">The launch-params struct type.</typeparam>
    [CLSCompliant(false)]
    public sealed class OptixLauncher<TLaunchParams> : DisposeBase
        where TLaunchParams : unmanaged
    {
        private static readonly uint LaunchParamsSizeInBytes =
            (uint)Marshal.SizeOf<TLaunchParams>();

        private readonly OptixPipeline pipeline;
        private readonly MemoryBuffer1D<TLaunchParams, Stride1D.Dense> launchParamsBuffer;
        private readonly SafeHGlobal sbtHandle;

        /// <summary>
        /// Constructs a launcher for the given pipeline/SBT pair, allocating the
        /// persistent launch-params buffer and native SBT copy up front.
        /// </summary>
        /// <param name="accelerator">The CUDA accelerator.</param>
        /// <param name="pipeline">The pipeline to launch.</param>
        /// <param name="sbt">The shader binding table; its record buffers must outlive this launcher.</param>
        public OptixLauncher(
            CudaAccelerator accelerator,
            OptixPipeline pipeline,
            OptixShaderBindingTable sbt)
        {
            if (accelerator == null)
                throw new ArgumentNullException(nameof(accelerator));
            this.pipeline = pipeline ?? throw new ArgumentNullException(nameof(pipeline));

            launchParamsBuffer = accelerator.Allocate1D<TLaunchParams>(1);
            sbtHandle = SafeHGlobal.AllocFrom(sbt);
        }

        /// <summary>
        /// Launches the pipeline on <paramref name="stream"/>, reusing the persistent
        /// launch-params buffer and SBT copy - safe to call once (or several times) per
        /// frame forever with no per-call allocations.
        /// </summary>
        /// <param name="stream">The stream to launch on.</param>
        /// <param name="launchParams">This launch's params value.</param>
        /// <param name="width">Launch width.</param>
        /// <param name="height">Launch height.</param>
        /// <param name="depth">Launch depth.</param>
        public void Launch(
            AcceleratorStream stream,
            in TLaunchParams launchParams,
            uint width,
            uint height,
            uint depth = 1)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (stream is not CudaStream cudaStream)
                throw new ArgumentOutOfRangeException(nameof(stream));

            var localLaunchParams = launchParams;
            launchParamsBuffer.View.CopyFromCPU(stream, ref localLaunchParams, 1);

            OptixException.ThrowIfFailed(
                OptixAPI.Current.Launch(
                    pipeline.PipelinePtr,
                    cudaStream.StreamPtr,
                    launchParamsBuffer.NativePtr,
                    LaunchParamsSizeInBytes,
                    sbtHandle,
                    width,
                    height,
                    depth));
        }

        /// <summary cref="DisposeBase.Dispose(bool)"/>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                launchParamsBuffer.Dispose();
                sbtHandle.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
