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

    /// <summary>
    /// Mirrors OptixSRTData (optix_types.h) - one motion key of an
    /// <see cref="OptixSRTMotionTransform"/>, decomposed into scale/shear (Sx, A, B,
    /// Sy, C, Sz), pivot point (Pvx/Pvy/Pvz), rotation quaternion (Qx/Qy/Qz/Qw - must
    /// be normalized), and translation (Tx/Ty/Tz). The effective transformation is
    /// (translation * rotation * scale-with-pivot); see the optix_types.h field docs
    /// for the exact matrix.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixSRTData
    {
        public float Sx;
        public float A;
        public float B;
        public float Pvx;
        public float Sy;
        public float C;
        public float Pvy;
        public float Sz;
        public float Pvz;
        public float Qx;
        public float Qy;
        public float Qz;
        public float Qw;
        public float Tx;
        public float Ty;
        public float Tz;
    }

    /// <summary>
    /// Mirrors OptixSRTMotionTransform (optix_types.h) - a motion node wrapping a
    /// static child traversable with a time-varying SRT decomposition, interpolated
    /// between exactly 2 motion keys (quaternions are nlerped, so SRT gives smooth
    /// rotation where <see cref="OptixMatrixMotionTransform"/> would shear). The same
    /// upload rules apply: 64-byte-aligned device buffer, handle via
    /// <c>optixConvertPointerToTraversableHandle</c> with
    /// <see cref="OptixTraversableType.SrtMotionTransform"/>.
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixSRTMotionTransform
    {
        /// <summary>The traversable handle (typically a GAS) transformed by this node.</summary>
        public ulong Child;

        public OptixMotionOptions MotionOptions;

        public uint Pad0;
        public uint Pad1;
        public uint Pad2;

        /// <summary>Motion key 0 at <see cref="OptixMotionOptions.TimeBegin"/>.</summary>
        public OptixSRTData SrtData0;

        /// <summary>Motion key 1 at <see cref="OptixMotionOptions.TimeEnd"/>.</summary>
        public OptixSRTData SrtData1;
    }

    /// <summary>
    /// Mirrors OptixStaticTransform (optix_types.h) - a time-invariant transform node
    /// wrapping a child traversable. Unlike instance transforms, static transforms
    /// carry their inverse explicitly - <see cref="InvTransform"/> must be the exact
    /// inverse of <see cref="Transform"/>. The same upload rules apply:
    /// 64-byte-aligned device buffer, handle via
    /// <c>optixConvertPointerToTraversableHandle</c> with
    /// <see cref="OptixTraversableType.StaticTransform"/>.
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct OptixStaticTransform
    {
        /// <summary>The traversable handle transformed by this node.</summary>
        public ulong Child;

        /// <summary>Padding to make the transformations 16-byte aligned.</summary>
        public fixed uint Pad[2];

        /// <summary>Affine object-to-world transformation as 3x4 row-major matrix.</summary>
        public fixed float Transform[12];

        /// <summary>
        /// Affine world-to-object transformation as 3x4 row-major matrix. Must be
        /// the inverse of <see cref="Transform"/>.
        /// </summary>
        public fixed float InvTransform[12];
    }
}

#pragma warning restore CA1051 // Do not declare visible instance fields
#pragma warning restore CA1815 // Override equals and operator equals on value types
#pragma warning restore CA1028 // Enum Storage should be Int32
