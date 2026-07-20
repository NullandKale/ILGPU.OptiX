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
using ILGPU.OptiX.Native;
using System;
using System.Runtime.InteropServices;

namespace ILGPU.OptiX.Pipeline
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

        // Keeps the native log-callback delegate alive for the lifetime of this
        // context - the driver only holds the raw function pointer, which the GC
        // cannot see (per Native/OptixLogCallback.cs's own contract). Replaced on
        // every SetLogCallback call; cleared together with the context.
        private OptixLogCallback? logCallbackKeepAlive;

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

        /// <summary>
        /// Sets or replaces this context's log callback
        /// (optixDeviceContextSetLogCallback) with a managed handler, taking care of
        /// the native-delegate lifetime. Pass null to remove the callback.
        /// </summary>
        /// <param name="callback">
        /// Receives (level, tag, message) per driver log line - level 1 = fatal,
        /// 2 = error, 3 = warning, 4 = print. Must not call any OptiX API from
        /// inside the callback (driver restriction).
        /// </param>
        /// <param name="level">
        /// Maximum level to generate messages for (0 disables all messages).
        /// </param>
        public OptixDeviceContext SetLogCallback(
            Action<uint, string, string>? callback,
            uint level = 4)
        {
            if (callback == null)
            {
                OptixException.ThrowIfFailed(
                    OptixAPI.Current.DeviceContextSetLogCallback(
                        DeviceContextPtr, IntPtr.Zero, IntPtr.Zero, 0));
                logCallbackKeepAlive = null;
                return this;
            }

            var native = CreateNativeLogCallback(callback);
            OptixException.ThrowIfFailed(
                OptixAPI.Current.DeviceContextSetLogCallback(
                    DeviceContextPtr,
                    Marshal.GetFunctionPointerForDelegate(native),
                    IntPtr.Zero,
                    level));

            // Only swap the keep-alive after the driver accepted the new pointer.
            logCallbackKeepAlive = native;
            return this;
        }

        /// <summary>
        /// Wraps a managed handler in the native <see cref="OptixLogCallback"/>
        /// signature (marshaling the tag/message strings).
        /// </summary>
        internal static OptixLogCallback CreateNativeLogCallback(
            Action<uint, string, string> callback) =>
            (level, tag, message, _) => callback(
                level,
                Marshal.PtrToStringAnsi(tag) ?? string.Empty,
                Marshal.PtrToStringAnsi(message) ?? string.Empty);

        /// <summary>
        /// Adopts an already-registered native log-callback delegate whose pointer
        /// the driver holds (used by the creation-time callback overload of
        /// <see cref="OptixContext.CreateDeviceContext(CudaAccelerator, Action{uint, string, string}, uint)"/>),
        /// keeping it alive for this context's lifetime.
        /// </summary>
        internal void AdoptLogCallback(OptixLogCallback native) =>
            logCallbackKeepAlive = native;

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
                    // Balances the Init() call made by OptixContext.CreateDeviceContext.
                    OptixAPI.Current.Uninit();
                }
            }
            base.Dispose(disposing);
        }

        #endregion
    }
}
