// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetObjectRayDirection.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System.Numerics;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetObjectRayDirection built-in function -
    /// the ray direction transformed into object space. Only valid in IS/AH programs
    /// (in CH the transform stack must be applied manually via
    /// <see cref="OptixTransforms"/>, matching the SDK's documented restriction).
    /// </summary>
    public static class OptixGetObjectRayDirection
    {
        /// <summary>
        /// The object-space ray direction, as a tuple. Prefer <see cref="AsVector3"/>
        /// for new code - <c>System.Numerics.Vector3</c> is this library's standard
        /// vector type.
        /// </summary>
        public static (float X, float Y, float Z) Value
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_object_ray_direction_x, ();",
                    out float x);
                CudaAsm.Emit(
                    "call (%0), _optix_get_object_ray_direction_y, ();",
                    out float y);
                CudaAsm.Emit(
                    "call (%0), _optix_get_object_ray_direction_z, ();",
                    out float z);
                return (x, y, z);
            }
        }

        /// <summary>
        /// The object-space ray direction, as a <see cref="Vector3"/> - this library's
        /// standard vector type. Prefer this over <see cref="Value"/> in new code.
        /// </summary>
        public static Vector3 AsVector3
        {
            get
            {
                var (x, y, z) = Value;
                return new Vector3(x, y, z);
            }
        }
    }
}
