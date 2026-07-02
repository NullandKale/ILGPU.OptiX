// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetSbtDataPointer.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX
{
    /// <summary>
    /// Provides the functionality of the optixGetSbtDataPointer built-in function -
    /// returns a device pointer to the custom data region of the current hitgroup's
    /// SBT record (the bytes immediately following the record's header). Cast the
    /// result to an unmanaged struct pointer matching the layout you wrote after the
    /// header when building the record (see OptixSbt.PackRecords).
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetSbtDataPointer
    {
        public static unsafe void* Value
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_sbt_data_ptr_64, ();",
                    out ulong ptr);
                return (void*)ptr;
            }
        }
    }
}
