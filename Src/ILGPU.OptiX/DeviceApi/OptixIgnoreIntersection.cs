// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixIgnoreIntersection.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixIgnoreIntersection built-in function -
    /// callable from an any-hit program to discard the current intersection and let the
    /// ray continue past it (see internal/optix_device_impl.h's zero-argument, zero-return
    /// "_optix_ignore_intersection" pseudo-call).
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixIgnoreIntersection
    {
        public static void Ignore()
        {
            CudaAsm.Emit("call _optix_ignore_intersection, ();");
        }
    }
}
