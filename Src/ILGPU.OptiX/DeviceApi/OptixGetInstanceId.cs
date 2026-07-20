// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetInstanceId.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetInstanceId built-in function - the
    /// user-supplied OptixInstance.InstanceId of the instance the current hit belongs
    /// to, or ~0u when the ray hit non-instanced (bare GAS) geometry. Valid in
    /// IS/AH/CH programs.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetInstanceId
    {
        public static uint Value
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_read_instance_id, ();",
                    out uint id);
                return id;
            }
        }
    }
}
