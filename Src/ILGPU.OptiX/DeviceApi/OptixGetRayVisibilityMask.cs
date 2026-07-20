// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetRayVisibilityMask.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetRayVisibilityMask built-in function -
    /// the 8-bit visibility mask passed to the optixTrace call that spawned the current
    /// IS/AH/CH/MS invocation.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetRayVisibilityMask
    {
        public static uint Value
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_ray_visibility_mask, ();",
                    out uint mask);
                return mask;
            }
        }
    }
}
