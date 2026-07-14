// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixCoopVecTypes.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Runtime.InteropServices;

#pragma warning disable CA1008 // Enums should have zero value
#pragma warning disable CA1051 // Do not declare visible instance fields
#pragma warning disable CA1815 // Override equals and operator equals on value types

namespace ILGPU.OptiX.CooperativeVectors
{
    /// <summary>
    /// Mirrors OptixCoopVecElemType (optix_types.h).
    /// Shared between the host matrix-conversion API and the device intrinsics in
    /// <see cref="DeviceApi.OptixCoopVec"/> (it is a plain integer enum, so it compiles
    /// on both sides identically - same precedent as e.g. OptixRayFlags).
    /// </summary>
    [CLSCompliant(false)]
    public enum OptixCoopVecElemType : uint
    {
        Unknown = 0x2A00,
        Float16 = 0x2A01,
        Float32 = 0x2A03,
        UInt8 = 0x2A04,
        Int8 = 0x2A05,
        UInt32 = 0x2A08,
        Int32 = 0x2A09,

        /// <summary>Only supported as the matvecmul inputInterpretation/matrixElementType.</summary>
        Float8E4M3 = 0x2A0A,

        /// <summary>Only supported as the matvecmul inputInterpretation/matrixElementType.</summary>
        Float8E5M2 = 0x2A0B,
    }

    /// <summary>Mirrors OptixCoopVecMatrixLayout (optix_types.h).</summary>
    [CLSCompliant(false)]
    public enum OptixCoopVecMatrixLayout : uint
    {
        RowMajor = 0x2A40,
        ColumnMajor = 0x2A41,
        InferencingOptimal = 0x2A42,
        TrainingOptimal = 0x2A43,
    }

    /// <summary>
    /// Mirrors OptixCoopVecMatrixDescription (optix_types.h) - describes one matrix's
    /// shape/type/layout/location within a larger buffer (offsetInBytes lets several
    /// matrices - e.g. every layer of an MLP - be packed tightly into one allocation).
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixCoopVecMatrixDescription
    {
        public uint N;
        public uint K;
        public uint OffsetInBytes;
        public OptixCoopVecElemType ElementType;
        public OptixCoopVecMatrixLayout Layout;

        /// <summary>Ignored for the two "optimal" layouts.</summary>
        public uint RowColumnStrideInBytes;

        public uint SizeInBytes;
    }

    /// <summary>
    /// Native mirror of OptixNetworkDescription (optix_types.h) - a device-side pointer
    /// to a <see cref="OptixCoopVecMatrixDescription"/> array plus its length. Internal:
    /// <see cref="OptixCoopVecMatrixBuilder"/> marshals this from a managed array, it is
    /// never constructed directly by a caller.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct OptixNetworkDescriptionNative
    {
        public IntPtr Layers;
        public uint NumLayers;
    }
}

#pragma warning restore CA1008 // Enums should have zero value
#pragma warning restore CA1051 // Do not declare visible instance fields
#pragma warning restore CA1815 // Override equals and operator equals on value types
