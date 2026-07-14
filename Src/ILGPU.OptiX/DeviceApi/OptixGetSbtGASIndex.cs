// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetSbtGASIndex.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetSbtGASIndex built-in function -
    /// identifies which build input (geometry) within the acceleration structure was
    /// hit, when a single GAS is built from multiple build inputs.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetSbtGASIndex
    {
        public static uint Value
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_read_sbt_gas_idx, ();",
                    out uint x);
                return x;
            }
        }
    }
}
