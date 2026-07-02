// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixTerminateRay.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX
{
    /// <summary>
    /// Provides the functionality of the optixTerminateRay built-in function - callable
    /// from an any-hit program to immediately stop traversal and jump to the closest-hit
    /// (or miss) program (see internal/optix_device_impl.h's zero-argument, zero-return
    /// "_optix_terminate_ray" pseudo-call).
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixTerminateRay
    {
        public static void Terminate()
        {
            CudaAsm.Emit("call _optix_terminate_ray, ();");
        }
    }
}
