// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixClusterAccel.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Runtime.InteropServices;

#pragma warning disable CA1008 // Enums should have zero value
#pragma warning disable CA1051 // Do not declare visible instance fields
#pragma warning disable CA1815 // Override equals and operator equals on value types

namespace ILGPU.OptiX.AccelStructures
{
    /// <summary>
    /// Mirrors OptixClusterAccelBuildFlags (optix_types.h).
    /// </summary>
    [Flags]
    public enum OptixClusterAccelBuildFlags
    {
        None = 0,
        PreferFastTrace = 1 << 0,
        PreferFastBuild = 1 << 1,
        AllowOpacityMicromaps = 1 << 2,
    }

    /// <summary>
    /// Mirrors OptixClusterAccelClusterFlags (optix_types.h) - per-cluster flags in
    /// the device-side args.
    /// </summary>
    [Flags]
    public enum OptixClusterAccelClusterFlags
    {
        None = 0,

        /// <summary>
        /// Required if the CLAS is in an instance with
        /// OPTIX_INSTANCE_FLAG_DISABLE_OPACITY_MICROMAPS set.
        /// </summary>
        AllowDisableOpacityMicromaps = 1 << 0,
    }

    /// <summary>
    /// Mirrors OptixClusterAccelPrimitiveFlags (optix_types.h) - the 3-bit
    /// primitiveFlags field of <see cref="OptixClusterAccelPrimitiveInfo"/>.
    /// </summary>
    [Flags]
    public enum OptixClusterAccelPrimitiveFlags
    {
        None = 0,
        DisableTriangleFaceCulling = 1 << 0,
        RequireSingleAnyhitCall = 1 << 1,
        DisableAnyhit = 1 << 2,
    }

    /// <summary>
    /// Mirrors OptixClusterAccelBuildType (optix_types.h) - what kind of cluster
    /// object a build produces.
    /// </summary>
    public enum OptixClusterAccelBuildType
    {
        GasesFromClusters = 0x2545,
        ClustersFromTriangles = 0x2546,
        TemplatesFromTriangles = 0x2547,
        ClustersFromTemplates = 0x2548,
        TemplatesFromGrids = 0x2549,
    }

    /// <summary>
    /// Mirrors OptixClusterAccelBuildMode (optix_types.h).
    /// </summary>
    public enum OptixClusterAccelBuildMode
    {
        /// <summary>All outputs packed into one buffer sized via ClusterAccelComputeMemoryUsage.</summary>
        ImplicitDestinations = 0,

        /// <summary>Per-object destination addresses supplied in device memory.</summary>
        ExplicitDestinations = 1,

        /// <summary>Computes per-object output sizes only (for a later explicit build).</summary>
        GetSizes = 2,
    }

    /// <summary>
    /// Mirrors OptixClusterAccelIndicesFormat (optix_types.h) - values equal the
    /// index byte count.
    /// </summary>
    public enum OptixClusterAccelIndicesFormat
    {
        Bits8 = 1,
        Bits16 = 2,
        Bits32 = 4,
    }

    /// <summary>
    /// Mirrors OptixClusterAccelBuildModeDescImplicitDest (optix_types.h).
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixClusterAccelBuildModeDescImplicitDest
    {
        /// <summary>128-byte-aligned output buffer (clusters/GASes) or 32-byte-aligned (templates).</summary>
        public ulong OutputBuffer;
        public ulong OutputBufferSizeInBytes;

        /// <summary>128-byte-aligned temp buffer.</summary>
        public ulong TempBuffer;
        public ulong TempBufferSizeInBytes;

        /// <summary>Receives a traversable handle per GAS or a device pointer per cluster/template.</summary>
        public ulong OutputHandlesBuffer;

        /// <summary>Minimum 8; 0 means 8.</summary>
        public uint OutputHandlesStrideInBytes;

        /// <summary>Optional uint32 array receiving per-object sizes.</summary>
        public ulong OutputSizesBuffer;

        /// <summary>Minimum 4; 0 means 4.</summary>
        public uint OutputSizesStrideInBytes;
    }

    /// <summary>
    /// Mirrors OptixClusterAccelBuildModeDescExplicitDest (optix_types.h).
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixClusterAccelBuildModeDescExplicitDest
    {
        public ulong TempBuffer;
        public ulong TempBufferSizeInBytes;

        /// <summary>Per-object destination addresses (aligned per output type).</summary>
        public ulong DestAddressesBuffer;

        /// <summary>Minimum 8; 0 means 8.</summary>
        public uint DestAddressesStrideInBytes;

        /// <summary>May alias <see cref="DestAddressesBuffer"/> to overwrite it in place.</summary>
        public ulong OutputHandlesBuffer;

        /// <summary>Minimum 8; 0 means 8.</summary>
        public uint OutputHandlesStrideInBytes;

        /// <summary>Optional uint32 array receiving per-object sizes.</summary>
        public ulong OutputSizesBuffer;

        /// <summary>Minimum 4; 0 means 4.</summary>
        public uint OutputSizesStrideInBytes;
    }

    /// <summary>
    /// Mirrors OptixClusterAccelBuildModeDescGetSize (optix_types.h).
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixClusterAccelBuildModeDescGetSize
    {
        /// <summary>Required uint32 array receiving per-object sizes.</summary>
        public ulong OutputSizesBuffer;

        /// <summary>Minimum 4; 0 means 4.</summary>
        public uint OutputSizesStrideInBytes;

        public ulong TempBuffer;
        public ulong TempBufferSizeInBytes;
    }

    /// <summary>
    /// Mirrors OptixClusterAccelBuildModeDesc (optix_types.h) - the build-mode tag
    /// plus a union of the per-mode destination descriptions. The union members
    /// contain 8-byte fields, so the union sits at offset 8 after the 4-byte enum
    /// (C alignment).
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Explicit)]
    public struct OptixClusterAccelBuildModeDesc
    {
        [FieldOffset(0)]
        public OptixClusterAccelBuildMode Mode;

        [FieldOffset(8)]
        public OptixClusterAccelBuildModeDescImplicitDest ImplicitDest;

        [FieldOffset(8)]
        public OptixClusterAccelBuildModeDescExplicitDest ExplicitDest;

        [FieldOffset(8)]
        public OptixClusterAccelBuildModeDescGetSize GetSize;
    }

    /// <summary>
    /// Mirrors OptixClusterAccelBuildInputTriangles (optix_types.h) - host-side
    /// limits over all triangle-cluster args of a build.
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixClusterAccelBuildInputTriangles
    {
        public OptixClusterAccelBuildFlags Flags;

        /// <summary>Max number of args provided at build time.</summary>
        public uint MaxArgCount;

        public OptixVertexFormat VertexFormat;

        /// <summary>
        /// Highest used SBT index over all clusters (including base offsets and any
        /// template-instantiation offset).
        /// </summary>
        public uint MaxSbtIndexValue;

        /// <summary>Number of unique SBT indices per cluster (1 when uniform).</summary>
        public uint MaxUniqueSbtIndexCountPerArg;

        public uint MaxTriangleCountPerArg;
        public uint MaxVertexCountPerArg;

        /// <summary>Optional; 0 means MaxTriangleCountPerArg * MaxArgCount.</summary>
        public uint MaxTotalTriangleCount;

        /// <summary>Optional; 0 means MaxVertexCountPerArg * MaxArgCount.</summary>
        public uint MaxTotalVertexCount;

        /// <summary>Lower bound on the number of position mantissa bits truncated.</summary>
        public uint MinPositionTruncateBitCount;
    }

    /// <summary>
    /// Mirrors OptixClusterAccelBuildInputGrids (optix_types.h).
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixClusterAccelBuildInputGrids
    {
        public OptixClusterAccelBuildFlags Flags;
        public uint MaxArgCount;
        public OptixVertexFormat VertexFormat;
        public uint MaxSbtIndexValue;

        /// <summary>Maximum number of edge segments along the grid width.</summary>
        public uint MaxWidth;

        /// <summary>Maximum number of edge segments along the grid height.</summary>
        public uint MaxHeight;
    }

    /// <summary>
    /// Mirrors OptixClusterAccelBuildInputClusters (optix_types.h).
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixClusterAccelBuildInputClusters
    {
        public OptixClusterAccelBuildFlags Flags;
        public uint MaxArgCount;
        public uint MaxTotalClusterCount;
        public uint MaxClusterCountPerArg;
    }

    /// <summary>
    /// Mirrors OptixClusterAccelBuildInput (optix_types.h) - the build-type tag plus
    /// a union of the per-type limit structs. All union members hold only 4-byte
    /// fields, so the union sits at offset 4 right after the enum (C alignment) -
    /// unlike <see cref="OptixBuildInput"/>, whose members force 8-byte alignment.
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Explicit)]
    public struct OptixClusterAccelBuildInput
    {
        [FieldOffset(0)]
        public OptixClusterAccelBuildType Type;

        [FieldOffset(4)]
        public OptixClusterAccelBuildInputClusters Clusters;

        [FieldOffset(4)]
        public OptixClusterAccelBuildInputTriangles Triangles;

        [FieldOffset(4)]
        public OptixClusterAccelBuildInputGrids Grids;
    }

    /// <summary>
    /// Mirrors OptixClusterAccelPrimitiveInfo (optix_types.h) - a C bitfield
    /// (sbtIndex : 24, reserved : 5, primitiveFlags : 3) represented as its raw
    /// 32-bit word; build values via <see cref="Pack"/>.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixClusterAccelPrimitiveInfo
    {
        /// <summary>Packs the sbtIndex (24 bits) and primitive flags (3 bits) bitfield word.</summary>
        public static uint Pack(
            uint sbtIndex,
            OptixClusterAccelPrimitiveFlags primitiveFlags = OptixClusterAccelPrimitiveFlags.None)
        {
            if (sbtIndex >= 1u << 24)
                throw new ArgumentOutOfRangeException(
                    nameof(sbtIndex), "Cluster SBT indices are limited to 24 bits.");
            return (sbtIndex & 0xFFFFFF) | ((uint)primitiveFlags << 29);
        }
    }

    /// <summary>
    /// Mirrors OptixClusterAccelBuildInputTrianglesArgs (optix_types.h) - one
    /// per-cluster argument record, consumed from device memory by
    /// ClusterAccelBuild. The C bitfield word (triangleCount : 9, vertexCount : 9,
    /// positionTruncateBitCount : 6, indexFormat : 4, opacityMicromapIndexFormat : 4)
    /// is exposed as <see cref="PackedCounts"/>; build it via
    /// <see cref="PackCounts"/>.
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixClusterAccelBuildInputTrianglesArgs
    {
        /// <summary>32-bit user-defined cluster id (readable via optixGetClusterId).</summary>
        public uint ClusterId;

        /// <summary>Combination of <see cref="OptixClusterAccelClusterFlags"/>.</summary>
        public uint ClusterFlags;

        /// <summary>The packed counts/formats bitfield word - see <see cref="PackCounts"/>.</summary>
        public uint PackedCounts;

        /// <summary>Base per-cluster SBT info - see <see cref="OptixClusterAccelPrimitiveInfo.Pack"/>.</summary>
        public uint BasePrimitiveInfo;

        /// <summary>0 means natural stride.</summary>
        public ushort IndexBufferStrideInBytes;
        public ushort VertexBufferStrideInBytes;
        public ushort PrimitiveInfoBufferStrideInBytes;
        public ushort OpacityMicromapIndexBufferStrideInBytes;

        public ulong IndexBuffer;

        /// <summary>Mandatory for ClustersFromTriangles; optional hint for TemplatesFromTriangles.</summary>
        public ulong VertexBuffer;

        /// <summary>Optional per-primitive array of packed primitive-info words.</summary>
        public ulong PrimitiveInfoBuffer;

        public ulong OpacityMicromapArray;
        public ulong OpacityMicromapIndexBuffer;

        /// <summary>Optional (TemplatesFromTriangles only): 32-byte-aligned OptixAabb per cluster.</summary>
        public ulong InstantiationBoundingBoxLimit;

        /// <summary>
        /// Packs the counts/formats bitfield word (triangleCount and vertexCount max
        /// 256, positionTruncateBitCount max 63).
        /// </summary>
        public static uint PackCounts(
            uint triangleCount,
            uint vertexCount,
            OptixClusterAccelIndicesFormat indexFormat,
            uint positionTruncateBitCount = 0,
            OptixClusterAccelIndicesFormat opacityMicromapIndexFormat = default)
        {
            if (triangleCount > 256)
                throw new ArgumentOutOfRangeException(nameof(triangleCount));
            if (vertexCount > 256)
                throw new ArgumentOutOfRangeException(nameof(vertexCount));
            if (positionTruncateBitCount >= 1u << 6)
                throw new ArgumentOutOfRangeException(nameof(positionTruncateBitCount));

            return (triangleCount & 0x1FF)
                | ((vertexCount & 0x1FF) << 9)
                | ((positionTruncateBitCount & 0x3F) << 18)
                | (((uint)indexFormat & 0xF) << 24)
                | (((uint)opacityMicromapIndexFormat & 0xF) << 28);
        }
    }

    /// <summary>
    /// Mirrors OptixClusterAccelBuildInputGridsArgs (optix_types.h). The packed
    /// word holds positionTruncateBitCount in its low 6 bits; Dimensions packs the
    /// two grid dimensions bytes plus 16 reserved bits.
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixClusterAccelBuildInputGridsArgs
    {
        public uint BaseClusterId;

        /// <summary>Combination of <see cref="OptixClusterAccelClusterFlags"/>.</summary>
        public uint ClusterFlags;

        /// <summary>See <see cref="OptixClusterAccelPrimitiveInfo.Pack"/>.</summary>
        public uint BasePrimitiveInfo;

        /// <summary>positionTruncateBitCount in bits 0..5; rest reserved.</summary>
        public uint PackedTruncate;

        /// <summary>dimensions[0] in bits 0..7, dimensions[1] in bits 8..15; rest reserved.</summary>
        public uint PackedDimensions;

        /// <summary>Packs the two grid edge-segment dimensions.</summary>
        public static uint PackDimensions(byte width, byte height) =>
            (uint)width | ((uint)height << 8);
    }

    /// <summary>
    /// Mirrors OptixClusterAccelBuildInputTemplatesArgs (optix_types.h).
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixClusterAccelBuildInputTemplatesArgs
    {
        /// <summary>Offset applied to the template's base cluster id.</summary>
        public uint ClusterIdOffset;

        /// <summary>Offset applied to the template's base SBT index (result limited to 24 bits).</summary>
        public uint SbtIndexOffset;

        /// <summary>Opaque device pointer to the template.</summary>
        public ulong ClusterTemplate;

        /// <summary>Vertex data to instantiate with; vertex order must match template creation.</summary>
        public ulong VertexBuffer;

        public uint VertexStrideInBytes;
        public uint Reserved;
    }

    /// <summary>
    /// Mirrors OptixClusterAccelBuildInputClustersArgs (optix_types.h).
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixClusterAccelBuildInputClustersArgs
    {
        public uint ClusterHandlesCount;

        /// <summary>0 means natural stride (8).</summary>
        public uint ClusterHandlesBufferStrideInBytes;

        /// <summary>
        /// Device pointer to the cluster handles - can come directly from a CLAS
        /// build's OutputHandlesBuffer.
        /// </summary>
        public ulong ClusterHandlesBuffer;
    }
}

#pragma warning restore CA1008 // Enums should have zero value
#pragma warning restore CA1051 // Do not declare visible instance fields
#pragma warning restore CA1815 // Override equals and operator equals on value types
