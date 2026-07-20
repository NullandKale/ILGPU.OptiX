// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetLaunchDimensions.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetLaunchDimensions built-in function -
    /// the width/height/depth passed to optixLaunch, queryable from any program type.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetLaunchDimensions
    {
        public static (uint X, uint Y, uint Z) Value
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_launch_dimension_x, ();",
                    out uint x);
                CudaAsm.Emit(
                    "call (%0), _optix_get_launch_dimension_y, ();",
                    out uint y);
                CudaAsm.Emit(
                    "call (%0), _optix_get_launch_dimension_z, ();",
                    out uint z);
                return (x, y, z);
            }
        }
    }
}
