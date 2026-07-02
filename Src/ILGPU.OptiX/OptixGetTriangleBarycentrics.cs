// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetTriangleBarycentrics.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;

namespace ILGPU.OptiX
{
    /// <summary>
    /// Provides the functionality of the optixGetTriangleBarycentrics built-in
    /// function.
    /// </summary>
    public static class OptixGetTriangleBarycentrics
    {
        public static (float U, float V) Value
        {
            get
            {
                // Unlike OptixGetWorldRayDirection (three separate single-output
                // calls), the OptiX device header returns both barycentric
                // coordinates from one call - "call (%0, %1), _optix_get_triangle_
                // barycentrics, ();" - so this needs EmitRef (multiple simultaneous
                // outputs) rather than Emit (single output only).
                Output<float> u = default;
                Output<float> v = default;
                CudaAsm.EmitRef(
                    "call (%0, %1), _optix_get_triangle_barycentrics, ();",
                    ref u,
                    ref v);
                return (u.Value, v.Value);
            }
        }
    }
}
