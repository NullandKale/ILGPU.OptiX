// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetRayFlags.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetRayFlags built-in function - the
    /// <see cref="OptixRayFlags"/> passed to the optixTrace call that spawned the
    /// current IS/AH/CH/MS invocation.
    /// </summary>
    public static class OptixGetRayFlags
    {
        public static OptixRayFlags Value
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_ray_flags, ();",
                    out uint flags);
                return (OptixRayFlags)flags;
            }
        }
    }
}
