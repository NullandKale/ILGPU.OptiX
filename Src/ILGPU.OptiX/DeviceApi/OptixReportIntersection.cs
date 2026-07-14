// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixReportIntersection.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the 0-attribute optixReportIntersection built-in
    /// function overload (internal/optix_device_impl.h's "_optix_report_intersection_0"
    /// pseudo-call) - callable only from an intersection program. Per-primitive geometry
    /// (position/normal/UV) is recovered analytically in the closest-hit program instead
    /// of being threaded through attribute registers, so the 1-8 attribute overloads
    /// (_optix_report_intersection_1..8) aren't bound here.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixReportIntersection
    {
        public static bool Report(float hitT, uint hitKind)
        {
            Input<float> _hitT = hitT;
            Input<uint> _hitKind = hitKind;
            Output<uint> _accepted = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_report_intersection_0, (%1, %2);",
                ref _accepted,
                ref _hitT,
                ref _hitKind);
            return _accepted.Value != 0;
        }
    }
}
