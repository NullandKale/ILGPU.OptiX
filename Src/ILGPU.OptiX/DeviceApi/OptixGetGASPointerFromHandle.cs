// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetGASPointerFromHandle.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetGASPointerFromHandle built-in
    /// function - converts a GAS traversable handle back to the device address of the
    /// acceleration structure (the inverse of optixConvertPointerToTraversableHandle).
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetGASPointerFromHandle
    {
        public static ulong Get(ulong gasHandle)
        {
            Input<ulong> _handle = gasHandle;
            Output<ulong> _pointer = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_get_gas_ptr_from_handle, (%1);",
                ref _pointer, ref _handle);
            return _pointer.Value;
        }
    }
}
