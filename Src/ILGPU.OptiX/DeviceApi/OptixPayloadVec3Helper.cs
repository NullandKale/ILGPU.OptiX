// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixPayloadVec3Helper.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;
using System.Numerics;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Helpers for compact Vec3/Vector3 payload packing/unpacking.
    /// Used by complex-payload path-tracing samples for radiance, normal, albedo, etc.
    /// Reduces manual 3-register assignment boilerplate to single-line calls.
    /// </summary>
    public static class OptixPayloadVec3Helper
    {
        /// <summary>
        /// Writes a Vector3 value across three consecutive payload registers, starting at startIdx.
        /// Each component is stored as a float (bit-for-bit via FloatAsInt).
        /// </summary>
        public static void SetVec3Registers(uint startIdx, Vector3 vec)
        {
            OptixPayloadInterop.SetFloat(startIdx, vec.X);
            OptixPayloadInterop.SetFloat(startIdx + 1, vec.Y);
            OptixPayloadInterop.SetFloat(startIdx + 2, vec.Z);
        }

        /// <summary>
        /// Reads three consecutive payload registers (starting at startIdx) and reconstructs a Vector3.
        /// Each register is reinterpreted as a float (via IntAsFloat).
        /// </summary>
        public static Vector3 GetVec3Registers(uint startIdx)
        {
            return new Vector3(
                OptixPayloadInterop.GetFloat(startIdx),
                OptixPayloadInterop.GetFloat(startIdx + 1),
                OptixPayloadInterop.GetFloat(startIdx + 2));
        }
    }
}
