// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetWorldRayDirection.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;

namespace ILGPU.OptiX
{
    /// <summary>
    /// Provides the functionality of the optixGetWorldRayDirection built-in function.
    /// </summary>
    public static class OptixGetWorldRayDirection
    {
        public static (float X, float Y, float Z) Value
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_world_ray_direction_x, ();",
                    out float x);
                CudaAsm.Emit(
                    "call (%0), _optix_get_world_ray_direction_y, ();",
                    out float y);
                CudaAsm.Emit(
                    "call (%0), _optix_get_world_ray_direction_z, ();",
                    out float z);
                return (x, y, z);
            }
        }
    }
}
