// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixAccelRelocation.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Runtime.InteropServices;

#pragma warning disable CA1051 // Do not declare visible instance fields
#pragma warning disable CA1815 // Override equals and operator equals on value types

namespace ILGPU.OptiX.AccelStructures
{
    /// <summary>
    /// Mirrors OptiX's OptixRelocationInfo (optix_types.h) - opaque data identifying a
    /// built acceleration structure's device/driver compatibility for relocation.
    /// Produced by OptixDeviceContextExtensions.AccelGetRelocationInfo, consumed by
    /// CheckRelocationCompatibility and AccelRelocate.
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixRelocationInfo
    {
        public ulong Info0;
        public ulong Info1;
        public ulong Info2;
        public ulong Info3;
    }

    /// <summary>
    /// Mirrors OptixRelocateInputOpacityMicromap (optix_types.h). Always zero (no OMM
    /// support wired yet) until opacity micromaps are implemented.
    /// </summary>
    [CLSCompliant(false)]
    public struct OptixRelocateInputOpacityMicromap
    {
        public IntPtr OpacityMicromapArray;
    }

    /// <summary>
    /// Mirrors OptixRelocateInputTriangleArray (optix_types.h).
    /// </summary>
    [CLSCompliant(false)]
    public struct OptixRelocateInputTriangleArray
    {
        public uint NumSbtRecords;
        public OptixRelocateInputOpacityMicromap OpacityMicromap;
    }

    /// <summary>
    /// Mirrors OptixRelocateInputInstanceArray (optix_types.h).
    /// </summary>
    [CLSCompliant(false)]
    public struct OptixRelocateInputInstanceArray
    {
        public uint NumInstances;
        public IntPtr TraversableHandles;
    }

    /// <summary>
    /// Mirrors OptixRelocateInput (optix_types.h) - one entry per original build input,
    /// in the same order the source acceleration structure was built with. Only
    /// Triangles and Instances need a populated union member; other build input types
    /// (custom primitives) require no relocation data beyond the Type tag.
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Explicit)]
    public struct OptixRelocateInput
    {
        [FieldOffset(0)]
        public OptixBuildInputType Type;

        [FieldOffset(8)]
        public OptixRelocateInputInstanceArray InstanceArray;

        [FieldOffset(8)]
        public OptixRelocateInputTriangleArray TriangleArray;
    }
}

#pragma warning restore CA1051 // Do not declare visible instance fields
#pragma warning restore CA1815 // Override equals and operator equals on value types
