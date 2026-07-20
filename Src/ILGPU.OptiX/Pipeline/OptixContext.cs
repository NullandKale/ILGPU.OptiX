// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixContext.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using ILGPU.OptiX.Native;
using System;
using System.Runtime.InteropServices;

namespace ILGPU.OptiX.Pipeline
{
    public static class OptixContext
    {
        /// <summary>
        /// Creates a new OptiX device context. Transparently ref-counts global OptiX
        /// library init/uninit against the returned context's lifetime - the consumer
        /// never needs to call an explicit init or uninit method; disposing every
        /// <see cref="OptixDeviceContext"/> that was created balances it automatically.
        /// </summary>
        /// <param name="accelerator">The CUDA accelerator.</param>
        /// <param name="options">The device context options.</param>
        /// <returns>The device context.</returns>
        [CLSCompliant(false)]
        public static OptixDeviceContext CreateDeviceContext(
            this CudaAccelerator accelerator,
            OptixDeviceContextOptions options = default)
        {
            if (accelerator == null)
                throw new ArgumentNullException(nameof(accelerator));

            OptixException.ThrowIfFailed(OptixAPI.Current.Init());
            try
            {
                OptixException.ThrowIfFailed(
                    OptixAPI.Current.DeviceContextCreate(
                        accelerator.NativePtr,
                        options,
                        out var deviceContext));
                return new OptixDeviceContext(accelerator, deviceContext);
            }
            catch
            {
                OptixAPI.Current.Uninit();
                throw;
            }
        }

        /// <summary>
        /// Creates a new OptiX device context with a managed log callback installed
        /// from creation time onwards (so even context-creation messages are
        /// delivered). The callback delegate's lifetime is managed by the returned
        /// context.
        /// </summary>
        /// <param name="accelerator">The CUDA accelerator.</param>
        /// <param name="logCallback">
        /// Receives (level, tag, message) per driver log line - level 1 = fatal,
        /// 2 = error, 3 = warning, 4 = print. Must not call any OptiX API from
        /// inside the callback (driver restriction).
        /// </param>
        /// <param name="logLevel">Maximum level to generate messages for.</param>
        /// <param name="validationMode">Optional device-context validation mode.</param>
        /// <returns>The device context.</returns>
        [CLSCompliant(false)]
        public static OptixDeviceContext CreateDeviceContext(
            this CudaAccelerator accelerator,
            Action<uint, string, string> logCallback,
            uint logLevel = 4,
            OptixDeviceContextValidationMode validationMode =
                OptixDeviceContextValidationMode.Off)
        {
            if (accelerator == null)
                throw new ArgumentNullException(nameof(accelerator));
            if (logCallback == null)
                throw new ArgumentNullException(nameof(logCallback));

            var native = OptixDeviceContext.CreateNativeLogCallback(logCallback);
            var options = new OptixDeviceContextOptions
            {
                LogCallbackFunction = Marshal.GetFunctionPointerForDelegate(native),
                LogCallbackData = IntPtr.Zero,
                LogCallbackLevel = (int)logLevel,
                ValidationMode = validationMode,
            };

            var context = accelerator.CreateDeviceContext(options);
            context.AdoptLogCallback(native);
            return context;
        }
    }
}
