// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixOpacityMicromap.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Runtime.InteropServices;

#pragma warning disable CA1008 // Enums should have zero value
#pragma warning disable CA1028 // Enum Storage should be Int32
#pragma warning disable CA1051 // Do not declare visible instance fields
#pragma warning disable CA1815 // Override equals and operator equals on value types

namespace ILGPU.OptiX.AccelStructures
{
    /// <summary>
    /// Mirrors OptixOpacityMicromapFormat (optix_types.h). Only the 2-state format is exercised by this library so far (no
    /// "unknown" state, so the driver never needs to fall back to invoking an any-hit
    /// program for ambiguity - the whole point of an opacity micromap).
    /// </summary>
    [CLSCompliant(false)]
    public enum OptixOpacityMicromapFormat : ushort
    {
        None = 0,
        TwoState = 1,
        FourState = 2,
    }

    /// <summary>
    /// Mirrors OptixOpacityMicromapArrayIndexingMode (optix_types.h).
    /// </summary>
    [CLSCompliant(false)]
    public enum OptixOpacityMicromapArrayIndexingMode
    {
        None = 0,

        /// <summary>triangle[i] uses opacityMicromapArray[i] - no index buffer needed.</summary>
        Linear = 1,

        Indexed = 2,
    }

    /// <summary>
    /// Mirrors OptixOpacityMicromapDesc (optix_types.h) - one entry per micromap in an
    /// array's descriptor buffer.
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixOpacityMicromapDesc
    {
        /// <summary>Byte offset into the array build input's raw data buffer.</summary>
        public uint ByteOffset;

        /// <summary>Micro-triangle count is 4^level; valid levels are [0, 12].</summary>
        public ushort SubdivisionLevel;

        public OptixOpacityMicromapFormat Format;
    }

    /// <summary>
    /// Mirrors OptixOpacityMicromapHistogramEntry (optix_types.h) - how many micromaps
    /// of a given format/subdivision combination are present in an array build's raw
    /// input data (not how many are actually referenced by triangles - see
    /// <see cref="OptixOpacityMicromapUsageCount"/> for that).
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixOpacityMicromapHistogramEntry
    {
        public uint Count;
        public uint SubdivisionLevel;
        public OptixOpacityMicromapFormat Format;
    }

    /// <summary>
    /// Mirrors OptixOpacityMicromapArrayBuildInput (optix_types.h) - input to
    /// <c>optixOpacityMicromapArrayComputeMemoryUsage</c>/<c>optixOpacityMicromapArrayBuild</c>.
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixOpacityMicromapArrayBuildInput
    {
        /// <summary>See <see cref="AccelStructures.OptixOpacityMicromapFlags"/>.</summary>
        public uint Flags;

        /// <summary>128-byte-aligned raw packed micro-triangle state data.</summary>
        public IntPtr InputBuffer;

        /// <summary>8-byte-aligned array of <see cref="OptixOpacityMicromapDesc"/>, one per micromap.</summary>
        public IntPtr PerMicromapDescBuffer;

        public uint PerMicromapDescStrideInBytes;

        public uint NumMicromapHistogramEntries;

        public IntPtr MicromapHistogramEntries;
    }

    /// <summary>Mirrors OptixOpacityMicromapFlags (optix_types.h).</summary>
    [CLSCompliant(false)]
    [Flags]
    public enum OptixOpacityMicromapFlags : uint
    {
        None = 0,
        PreferFastTrace = 1u << 0,
        PreferFastBuild = 1u << 1,
    }

    /// <summary>
    /// Mirrors OptixMicromapBufferSizes (optix_types.h).
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixMicromapBufferSizes
    {
        public ulong OutputSizeInBytes;
        public ulong TempSizeInBytes;
    }

    /// <summary>
    /// Mirrors OptixMicromapBuffers (optix_types.h).
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixMicromapBuffers
    {
        public IntPtr Output;
        public ulong OutputSizeInBytes;
        public IntPtr Temp;
        public ulong TempSizeInBytes;
    }

    /// <summary>
    /// Mirrors OptixOpacityMicromapUsageCount (optix_types.h) - how many micromaps of a
    /// format/subdivision combination are actually referenced by triangles in a
    /// triangle build input (distinct from <see cref="OptixOpacityMicromapHistogramEntry"/>,
    /// which counts what's present in the array build itself).
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixOpacityMicromapUsageCount
    {
        public uint Count;
        public uint SubdivisionLevel;
        public OptixOpacityMicromapFormat Format;
    }

    /// <summary>
    /// Mirrors OptixBuildInputOpacityMicromap (optix_types.h) - embedded in
    /// <see cref="OptixBuildInputTriangleArray.OpacityMicromap"/> to attach a built
    /// opacity micromap array to a triangle build input.
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixBuildInputOpacityMicromap
    {
        public OptixOpacityMicromapArrayIndexingMode IndexingMode;

        /// <summary>Required (non-zero) when <see cref="IndexingMode"/> is Linear or Indexed.</summary>
        public IntPtr OpacityMicromapArray;

        /// <summary>Required when <see cref="IndexingMode"/> is Indexed; must be zero otherwise.</summary>
        public IntPtr IndexBuffer;

        /// <summary>0, 2, or 4 (unused/16-bit/32-bit) - non-zero only when Indexed.</summary>
        public uint IndexSizeInBytes;

        public uint IndexStrideInBytes;

        public uint IndexOffset;

        public uint NumMicromapUsageCounts;

        public IntPtr MicromapUsageCounts;
    }
}

#pragma warning restore CA1008 // Enums should have zero value
#pragma warning restore CA1028 // Enum Storage should be Int32
#pragma warning restore CA1051 // Do not declare visible instance fields
#pragma warning restore CA1815 // Override equals and operator equals on value types
