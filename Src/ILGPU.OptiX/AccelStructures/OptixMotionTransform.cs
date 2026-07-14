// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixMotionTransform.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Runtime.InteropServices;

#pragma warning disable CA1051 // Do not declare visible instance fields
#pragma warning disable CA1815 // Override equals and operator equals on value types
#pragma warning disable CA1028 // Enum Storage should be Int32

namespace ILGPU.OptiX.AccelStructures
{
    /// <summary>
    /// Mirrors OptixTraversableType (optix_types.h) - identifies the kind of traversable
    /// a device pointer refers to for <see cref="Pipeline.OptixDeviceContextExtensions.ConvertPointerToTraversableHandle"/>.
    /// </summary>
    [CLSCompliant(false)]
    public enum OptixTraversableType
    {
        StaticTransform = 0x21C1,
        MatrixMotionTransform = 0x21C2,
        SrtMotionTransform = 0x21C3,
    }

    /// <summary>
    /// Mirrors OptixMatrixMotionTransform (optix_types.h) - a rigid-body motion node
    /// wrapping a static child traversable (typically
    /// a GAS) with a time-varying 3x4 object-to-world matrix, interpolated between
    /// exactly 2 motion keys. Build via <see cref="OptixMotionTransformBuilder.BuildMatrixMotionTransform"/>
    /// rather than constructing this directly - the device buffer this struct is
    /// uploaded into must be OPTIX_TRANSFORM_BYTE_ALIGNMENT (64-byte) aligned, and its
    /// traversable handle comes from <c>optixConvertPointerToTraversableHandle</c>, not
    /// <c>optixAccelBuild</c>.
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct OptixMatrixMotionTransform
    {
        /// <summary>The traversable handle (typically a GAS) transformed by this node.</summary>
        public ulong Child;

        public OptixMotionOptions MotionOptions;

        public fixed uint Pad[3];

        /// <summary>
        /// Two 3x4 row-major object-to-world matrices (12 floats each, flattened to 24)
        /// - motion key 0 at <see cref="OptixMotionOptions.TimeBegin"/>, key 1
        /// at <see cref="OptixMotionOptions.TimeEnd"/>.
        /// </summary>
        public fixed float Transform[24];
    }
}

#pragma warning restore CA1051 // Do not declare visible instance fields
#pragma warning restore CA1815 // Override equals and operator equals on value types
#pragma warning restore CA1028 // Enum Storage should be Int32
