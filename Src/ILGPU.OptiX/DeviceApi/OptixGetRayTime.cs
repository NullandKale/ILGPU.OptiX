// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetRayTime.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetRayTime built-in function - the motion
    /// time value passed to optixTrace. Always 0 when the pipeline is compiled without
    /// motion blur (UsesMotionBlur = false).
    /// </summary>
    public static class OptixGetRayTime
    {
        public static float Value
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_ray_time, ();",
                    out float t);
                return t;
            }
        }
    }
}
