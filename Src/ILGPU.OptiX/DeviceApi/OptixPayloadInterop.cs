// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixPayloadInterop.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides convenience wrappers for payload register bit-casting operations.
    /// Reduces boilerplate when manually manipulating individual payload registers.
    /// </summary>
    public static class OptixPayloadInterop
    {
        /// <summary>
        /// Sets a payload register to a float value (stored bit-for-bit via FloatAsInt).
        /// </summary>
        public static void SetFloat(uint index, float value)
            => OptixPayload.Set((int)index, ILGPU.Interop.FloatAsInt(value));

        /// <summary>
        /// Reads a payload register and reinterprets it as a float (via IntAsFloat).
        /// </summary>
        public static float GetFloat(uint index)
            => ILGPU.Interop.IntAsFloat(OptixPayload.Get((int)index));

        /// <summary>
        /// Sets a payload register to a uint value directly (no bit-casting).
        /// </summary>
        public static void SetUint(uint index, uint value)
            => OptixPayload.Set((int)index, value);

        /// <summary>
        /// Reads a payload register as a uint value directly (no bit-casting).
        /// </summary>
        public static uint GetUint(uint index)
            => OptixPayload.Get((int)index);
    }
}
