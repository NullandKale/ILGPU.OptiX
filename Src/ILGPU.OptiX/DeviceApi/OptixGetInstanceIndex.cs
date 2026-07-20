// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetInstanceIndex.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetInstanceIndex built-in function - the
    /// zero-based index of the instance within its IAS build input (unlike
    /// <see cref="OptixGetInstanceId"/>, which is the user-supplied id), or 0 when the
    /// ray hit non-instanced (bare GAS) geometry. Valid in IS/AH/CH programs.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetInstanceIndex
    {
        public static uint Value
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_read_instance_idx, ();",
                    out uint index);
                return index;
            }
        }
    }
}
