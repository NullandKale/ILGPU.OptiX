// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixReportIntersection.tt/OptixReportIntersection.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------


using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixReportIntersection built-in function
    /// overloads (internal/optix_device_impl.h's "_optix_report_intersection_0..8"
    /// pseudo-calls) - callable only from an intersection program. Attribute
    /// registers a0..a7 are delivered to the AH/CH programs of the hit, readable
    /// there via <see cref="OptixGetAttribute"/>; the pipeline's NumAttributeValues
    /// must cover the highest attribute count reported. Returns true if the hit was
    /// accepted (not ignored by an any-hit program).
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixReportIntersection
    {
        /// <summary>
        /// Reports an intersection at <paramref name="hitT"/> carrying
        /// no attribute registers.
        /// </summary>
        public static bool Report(
            float hitT,
            uint hitKind)
        {
            Input<float> _hitT = hitT;
            Input<uint> _hitKind = hitKind;
            Output<uint> _accepted = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_report_intersection_0, " +
                "(%1, %2);",
                ref _accepted,
                ref _hitT,
                ref _hitKind);
            return _accepted.Value != 0;
        }

        /// <summary>
        /// Reports an intersection at <paramref name="hitT"/> carrying
        /// 1 attribute register (a0).
        /// </summary>
        public static bool Report(
            float hitT,
            uint hitKind,
            uint a0)
        {
            Input<float> _hitT = hitT;
            Input<uint> _hitKind = hitKind;
            Input<uint> _a0 = a0;
            Output<uint> _accepted = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_report_intersection_1, " +
                "(%1, %2, %3);",
                ref _accepted,
                ref _hitT,
                ref _hitKind,
                ref _a0);
            return _accepted.Value != 0;
        }

        /// <summary>
        /// Reports an intersection at <paramref name="hitT"/> carrying
        /// 2 attribute registers (a0..a1).
        /// </summary>
        public static bool Report(
            float hitT,
            uint hitKind,
            uint a0,
            uint a1)
        {
            Input<float> _hitT = hitT;
            Input<uint> _hitKind = hitKind;
            Input<uint> _a0 = a0;
            Input<uint> _a1 = a1;
            Output<uint> _accepted = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_report_intersection_2, " +
                "(%1, %2, %3, %4);",
                ref _accepted,
                ref _hitT,
                ref _hitKind,
                ref _a0,
                ref _a1);
            return _accepted.Value != 0;
        }

        /// <summary>
        /// Reports an intersection at <paramref name="hitT"/> carrying
        /// 3 attribute registers (a0..a2).
        /// </summary>
        public static bool Report(
            float hitT,
            uint hitKind,
            uint a0,
            uint a1,
            uint a2)
        {
            Input<float> _hitT = hitT;
            Input<uint> _hitKind = hitKind;
            Input<uint> _a0 = a0;
            Input<uint> _a1 = a1;
            Input<uint> _a2 = a2;
            Output<uint> _accepted = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_report_intersection_3, " +
                "(%1, %2, %3, %4, %5);",
                ref _accepted,
                ref _hitT,
                ref _hitKind,
                ref _a0,
                ref _a1,
                ref _a2);
            return _accepted.Value != 0;
        }

        /// <summary>
        /// Reports an intersection at <paramref name="hitT"/> carrying
        /// 4 attribute registers (a0..a3).
        /// </summary>
        public static bool Report(
            float hitT,
            uint hitKind,
            uint a0,
            uint a1,
            uint a2,
            uint a3)
        {
            Input<float> _hitT = hitT;
            Input<uint> _hitKind = hitKind;
            Input<uint> _a0 = a0;
            Input<uint> _a1 = a1;
            Input<uint> _a2 = a2;
            Input<uint> _a3 = a3;
            Output<uint> _accepted = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_report_intersection_4, " +
                "(%1, %2, %3, %4, %5, %6);",
                ref _accepted,
                ref _hitT,
                ref _hitKind,
                ref _a0,
                ref _a1,
                ref _a2,
                ref _a3);
            return _accepted.Value != 0;
        }

        /// <summary>
        /// Reports an intersection at <paramref name="hitT"/> carrying
        /// 5 attribute registers (a0..a4).
        /// </summary>
        public static bool Report(
            float hitT,
            uint hitKind,
            uint a0,
            uint a1,
            uint a2,
            uint a3,
            uint a4)
        {
            Input<float> _hitT = hitT;
            Input<uint> _hitKind = hitKind;
            Input<uint> _a0 = a0;
            Input<uint> _a1 = a1;
            Input<uint> _a2 = a2;
            Input<uint> _a3 = a3;
            Input<uint> _a4 = a4;
            Output<uint> _accepted = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_report_intersection_5, " +
                "(%1, %2, %3, %4, %5, %6, %7);",
                ref _accepted,
                ref _hitT,
                ref _hitKind,
                ref _a0,
                ref _a1,
                ref _a2,
                ref _a3,
                ref _a4);
            return _accepted.Value != 0;
        }

        /// <summary>
        /// Reports an intersection at <paramref name="hitT"/> carrying
        /// 6 attribute registers (a0..a5).
        /// </summary>
        public static bool Report(
            float hitT,
            uint hitKind,
            uint a0,
            uint a1,
            uint a2,
            uint a3,
            uint a4,
            uint a5)
        {
            Input<float> _hitT = hitT;
            Input<uint> _hitKind = hitKind;
            Input<uint> _a0 = a0;
            Input<uint> _a1 = a1;
            Input<uint> _a2 = a2;
            Input<uint> _a3 = a3;
            Input<uint> _a4 = a4;
            Input<uint> _a5 = a5;
            Output<uint> _accepted = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_report_intersection_6, " +
                "(%1, %2, %3, %4, %5, %6, %7, %8);",
                ref _accepted,
                ref _hitT,
                ref _hitKind,
                ref _a0,
                ref _a1,
                ref _a2,
                ref _a3,
                ref _a4,
                ref _a5);
            return _accepted.Value != 0;
        }

        /// <summary>
        /// Reports an intersection at <paramref name="hitT"/> carrying
        /// 7 attribute registers (a0..a6).
        /// </summary>
        public static bool Report(
            float hitT,
            uint hitKind,
            uint a0,
            uint a1,
            uint a2,
            uint a3,
            uint a4,
            uint a5,
            uint a6)
        {
            Input<float> _hitT = hitT;
            Input<uint> _hitKind = hitKind;
            Input<uint> _a0 = a0;
            Input<uint> _a1 = a1;
            Input<uint> _a2 = a2;
            Input<uint> _a3 = a3;
            Input<uint> _a4 = a4;
            Input<uint> _a5 = a5;
            Input<uint> _a6 = a6;
            Output<uint> _accepted = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_report_intersection_7, " +
                "(%1, %2, %3, %4, %5, %6, %7, %8, %9);",
                ref _accepted,
                ref _hitT,
                ref _hitKind,
                ref _a0,
                ref _a1,
                ref _a2,
                ref _a3,
                ref _a4,
                ref _a5,
                ref _a6);
            return _accepted.Value != 0;
        }

        /// <summary>
        /// Reports an intersection at <paramref name="hitT"/> carrying
        /// 8 attribute registers (a0..a7).
        /// </summary>
        public static bool Report(
            float hitT,
            uint hitKind,
            uint a0,
            uint a1,
            uint a2,
            uint a3,
            uint a4,
            uint a5,
            uint a6,
            uint a7)
        {
            Input<float> _hitT = hitT;
            Input<uint> _hitKind = hitKind;
            Input<uint> _a0 = a0;
            Input<uint> _a1 = a1;
            Input<uint> _a2 = a2;
            Input<uint> _a3 = a3;
            Input<uint> _a4 = a4;
            Input<uint> _a5 = a5;
            Input<uint> _a6 = a6;
            Input<uint> _a7 = a7;
            Output<uint> _accepted = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_report_intersection_8, " +
                "(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10);",
                ref _accepted,
                ref _hitT,
                ref _hitKind,
                ref _a0,
                ref _a1,
                ref _a2,
                ref _a3,
                ref _a4,
                ref _a5,
                ref _a6,
                ref _a7);
            return _accepted.Value != 0;
        }

    }
}
