// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetHitKind.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX
{
    /// <summary>
    /// Provides the functionality of the optixGetHitKind built-in function. Built-in
    /// triangle hits report the reserved values OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE
    /// (0xFE) / OPTIX_HIT_KIND_TRIANGLE_BACK_FACE (0xFF) (optix_types.h); any hit kind a
    /// custom-primitive intersection program reports via OptixReportIntersection must
    /// stay below that range, so checking "Value &gt;= 0xFE" is sufficient to
    /// distinguish a triangle hit from a custom-primitive hit without a separate
    /// optixIsTriangleHit-style wrapper.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetHitKind
    {
        public const uint TriangleFrontFace = 0xFE;
        public const uint TriangleBackFace = 0xFF;

        public static uint Value
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_hit_kind, ();",
                    out uint kind);
                return kind;
            }
        }
    }
}
