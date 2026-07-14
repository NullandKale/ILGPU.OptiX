// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetPrimitiveIndex.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetPrimitiveIndex built-in function.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetPrimitiveIndex
    {
        public static uint Value
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_read_primitive_idx, ();",
                    out uint x);
                return x;
            }
        }
    }
}
