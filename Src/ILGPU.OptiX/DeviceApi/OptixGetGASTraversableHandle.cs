// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetGASTraversableHandle.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetGASTraversableHandle built-in
    /// function - the traversable handle of the GAS containing the current hit.
    /// Valid in IS/AH/CH programs.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetGASTraversableHandle
    {
        public static ulong Value
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_gas_traversable_handle, ();",
                    out ulong handle);
                return handle;
            }
        }
    }
}
