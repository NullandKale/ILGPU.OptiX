// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixUndefinedValue.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixUndefinedValue built-in function - an
    /// explicitly undefined value the compiler may freely optimize around. Useful for
    /// writing payload registers whose value is intentionally unused on some paths
    /// (cheaper than a real constant, per the SDK docs).
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixUndefinedValue
    {
        public static uint Value
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_undef_value, ();", out uint value);
                return value;
            }
        }
    }
}
