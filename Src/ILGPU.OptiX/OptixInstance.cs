// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixInstance.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Runtime.InteropServices;

namespace ILGPU.OptiX
{
    /// <summary>
    /// Mirrors OptiX's OptixInstance (optix_types.h) - one entry per instance in an
    /// OptixBuildInputInstanceArray, used to combine multiple GASes (of possibly
    /// different build-input types, e.g. triangles and custom primitives, which cannot
    /// be combined as multiple build inputs within a single GAS) into one Instance
    /// Acceleration Structure. Size is 80 bytes (48 + 4*4 + 8 + 8), matching the header's
    /// own "round up to 80-byte, to ensure 16-byte alignment" comment.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    [CLSCompliant(false)]
    public unsafe struct OptixInstance
    {
        /// <summary>
        /// Affine object-to-world transformation as a 3x4 row-major matrix.
        /// </summary>
        public fixed float Transform[12];

        public uint InstanceId;

        /// <summary>
        /// SBT record offset - added to each hit's GAS-local sbt index to compute the
        /// global hitgroup record index (see docs/SAMPLE13_PLAN.md's custom-primitive
        /// GAS design).
        /// </summary>
        public uint SbtOffset;

        public uint VisibilityMask;
        public uint Flags;
        public ulong TraversableHandle;
        public fixed uint Pad[2];

        public static readonly float[] IdentityTransform =
        {
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f
        };
    }
}
