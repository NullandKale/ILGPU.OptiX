// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixDeviceContext.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Backends.PTX;
using ILGPU.Runtime.Cuda;
using ILGPU.Util;
using System;

namespace ILGPU.OptiX
{
    /// <summary>
    /// Wrapper for an OptiX device context with configurable compile options.
    /// </summary>
    [CLSCompliant(false)]
    public sealed class OptixDeviceContext : DisposeBase
    {
        private OptixModuleCompileOptions? moduleCompileOptions;
        private OptixPipelineCompileOptions? pipelineCompileOptions;
        private OptixPipelineLinkOptions? pipelineLinkOptions;

        #region Properties

        /// <summary>
        /// The native OptiX device context.
        /// </summary>
        public IntPtr DeviceContextPtr { get; private set; }

        /// <summary>
        /// The Cuda accelerator.
        /// </summary>
        public CudaAccelerator Accelerator { get; }

        /// <summary>
        /// The PTX backend.
        /// </summary>
        public PTXBackend Backend =>
            Accelerator.Backend;

        /// <summary>
        /// The stored module compile options (may be null if not configured).
        /// </summary>
        public OptixModuleCompileOptions? ModuleCompileOptions
        {
            get => moduleCompileOptions;
            private set => moduleCompileOptions = value;
        }

        /// <summary>
        /// The stored pipeline compile options (may be null if not configured).
        /// </summary>
        public OptixPipelineCompileOptions? PipelineCompileOptions
        {
            get => pipelineCompileOptions;
            private set => pipelineCompileOptions = value;
        }

        /// <summary>
        /// The stored pipeline link options (may be null if not configured).
        /// </summary>
        public OptixPipelineLinkOptions? PipelineLinkOptions
        {
            get => pipelineLinkOptions;
            private set => pipelineLinkOptions = value;
        }

        #endregion

        #region Instance

        /// <summary>
        /// Constructs a new device context wrapper.
        /// </summary>
        /// <param name="accelerator">The Cuda accelerator.</param>
        /// <param name="deviceContextPtr">The OptiX device context.</param>
        public OptixDeviceContext(CudaAccelerator accelerator, IntPtr deviceContextPtr)
        {
            if (deviceContextPtr == IntPtr.Zero)
                throw new ArgumentNullException(nameof(deviceContextPtr));
            Accelerator = accelerator
                ?? throw new ArgumentNullException(nameof(accelerator));
            DeviceContextPtr = deviceContextPtr;
        }

        /// <summary>
        /// Sets the module compile options and returns this context for fluent chaining.
        /// </summary>
        public OptixDeviceContext WithModuleCompileOptions(OptixModuleCompileOptions options)
        {
            ModuleCompileOptions = options;
            return this;
        }

        /// <summary>
        /// Sets the pipeline compile options and returns this context for fluent chaining.
        /// </summary>
        public OptixDeviceContext WithPipelineCompileOptions(OptixPipelineCompileOptions options)
        {
            PipelineCompileOptions = options;
            return this;
        }

        /// <summary>
        /// Sets the pipeline link options and returns this context for fluent chaining.
        /// </summary>
        public OptixDeviceContext WithPipelineLinkOptions(OptixPipelineLinkOptions options)
        {
            PipelineLinkOptions = options;
            return this;
        }

        #endregion

        #region IDisposable

        /// <summary cref="DisposeBase.Dispose(bool)"/>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (DeviceContextPtr != IntPtr.Zero)
                {
                    OptixAPI.Current.DeviceContextDestroy(DeviceContextPtr);
                    DeviceContextPtr = IntPtr.Zero;
                }
            }
            base.Dispose(disposing);
        }

        #endregion
    }
}
