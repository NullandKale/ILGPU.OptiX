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
    }
}
