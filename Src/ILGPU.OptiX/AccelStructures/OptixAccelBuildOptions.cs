// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixAccelBuildOptions.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Diagnostics.CodeAnalysis;

#pragma warning disable CA1008 // Enums should have zero value
#pragma warning disable CA1028 // Enum Storage should be Int32
#pragma warning disable CA1051 // Do not declare visible instance fields
#pragma warning disable CA1707 // Identifiers should not contain underscores
#pragma warning disable CA1815 // Override equals and operator equals on value types

namespace ILGPU.OptiX.AccelStructures
{
    [CLSCompliant(false)]
    [Flags]
    [SuppressMessage(
        "Naming",
        "CA1711:Identifiers should not have incorrect suffix")]
    public enum OptixBuildFlags : uint
    {
        None = 0,
        AllowUpdate = 1u << 0,
        AllowCompaction = 1u << 1,
        PreferFastTrace = 1u << 2,
        PreferFastBuild = 1u << 3,
        AllowRandomVertexAccess = 1u << 4,
        AllowRandomInstanceAccess = 1u << 5,
    }

    public enum OptixBuildOperation
    {
        Build = 0x2161,
        Update = 0x2162,
    }

    public enum OptixAccelPropertyType
    {
        CompactedSize = 0x2181,
        Aabbs = 0x2182,
    }

    [CLSCompliant(false)]
    [Flags]
    [SuppressMessage(
        "Naming",
        "CA1711:Identifiers should not have incorrect suffix")]
    public enum OptixMotionFlags : uint
    {
        None = 0u,
        StartVanish = 1u << 0,
        EndVanish = 1u << 1
    }

    [CLSCompliant(false)]
    public struct OptixMotionOptions
    {
        public ushort NumKeys;
        public ushort Flags;
        public float TimeBegin;
        public float TimeEnd;
    }

    [CLSCompliant(false)]
    public struct OptixAccelBuildOptions
    {
        public OptixBuildFlags BuildFlags;
        public OptixBuildOperation Operation;
        public OptixMotionOptions MotionOptions;
    }

    [CLSCompliant(false)]
    public struct OptixAccelBufferSizes
    {
        public ulong OutputSizeInBytes;
        public ulong TempSizeInBytes;
        public ulong TempUpdateSizeInBytes;
    }

    public struct OptixAccelEmitDesc
    {
        public IntPtr Result;
        public OptixAccelPropertyType Type;
    }
}

#pragma warning restore CA1008 // Enums should have zero value
#pragma warning restore CA1051 // Do not declare visible instance fields
#pragma warning restore CA1028 // Enum Storage should be Int32
#pragma warning restore CA1707 // Identifiers should not contain underscores
#pragma warning restore CA1815 // Override equals and operator equals on value types
