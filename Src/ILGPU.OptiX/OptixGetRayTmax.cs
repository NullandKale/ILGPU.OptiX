// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetRayTmax.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;

namespace ILGPU.OptiX
{
    /// <summary>
    /// Provides the functionality of the optixGetRayTmax built-in function - inside a
    /// closest-hit program this is the hit distance along the ray (the current tmax at
    /// hit time, matching the reference OptiX documentation's own note that this is only
    /// meaningful during intersection/any-hit/closest-hit execution).
    /// </summary>
    public static class OptixGetRayTmax
    {
        public static float Value
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_ray_tmax, ();",
                    out float t);
                return t;
            }
        }
    }
}
