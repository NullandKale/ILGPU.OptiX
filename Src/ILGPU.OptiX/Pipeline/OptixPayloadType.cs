// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixPayloadType.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Diagnostics.CodeAnalysis;

#pragma warning disable CA1008 // Enums should have zero value
#pragma warning disable CA1028 // Enum Storage should be Int32
#pragma warning disable CA1051 // Do not declare visible instance fields
#pragma warning disable CA1815 // Override equals and operator equals on value types

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Mirrors OptixPayloadTypeID (optix_types.h) - identifies one of up to 8 distinct
    /// payload layouts a pipeline can declare via <see cref="OptixModuleCompileOptions.PayloadTypes"/>,
    /// selected per <c>optixTrace</c> call site via <see cref="DeviceApi.OptixTrace.Typed0"/>/
    /// <see cref="DeviceApi.OptixTrace.Typed1"/>.
    /// </summary>
    [CLSCompliant(false)]
    [SuppressMessage("Naming", "CA1711:Identifiers should not have incorrect suffix")]
    public enum OptixPayloadTypeID : uint
    {
        /// <summary>Legacy untyped payload dispatch - the only ID every existing sample uses.</summary>
        Default = 0,
        Id0 = 1u << 0,
        Id1 = 1u << 1,
        Id2 = 1u << 2,
        Id3 = 1u << 3,
        Id4 = 1u << 4,
        Id5 = 1u << 5,
        Id6 = 1u << 6,
        Id7 = 1u << 7,
    }

    /// <summary>
    /// Mirrors OptixPayloadSemantics (optix_types.h) - per-payload-word read/write
    /// permissions for each shader stage. Combine with bitwise OR; a word needs to be
    /// writable by the trace caller or at least one shader stage, and readable by the
    /// caller or at least one stage after being written.
    /// </summary>
    [CLSCompliant(false)]
    [Flags]
    public enum OptixPayloadSemantics : uint
    {
        None = 0,

        TraceCallerRead = 1u << 0,
        TraceCallerWrite = 2u << 0,
        TraceCallerReadWrite = 3u << 0,

        ClosestHitRead = 1u << 2,
        ClosestHitWrite = 2u << 2,
        ClosestHitReadWrite = 3u << 2,

        MissRead = 1u << 4,
        MissWrite = 2u << 4,
        MissReadWrite = 3u << 4,

        AnyHitRead = 1u << 6,
        AnyHitWrite = 2u << 6,
        AnyHitReadWrite = 3u << 6,

        IntersectionRead = 1u << 8,
        IntersectionWrite = 2u << 8,
        IntersectionReadWrite = 3u << 8,

        /// <summary>
        /// Full read/write access from the trace caller and every shader stage - the
        /// permissive default this library's <see cref="RayTypeBuilder{TLaunchParams}.Payload{TPayload}(OptixPayloadTypeID)"/>
        /// uses for every payload word, matching how untyped payloads already behave
        /// (no compiler-enforced per-stage restriction).
        /// </summary>
        FullAccess = TraceCallerReadWrite | ClosestHitReadWrite | MissReadWrite | AnyHitReadWrite | IntersectionReadWrite,
    }

    /// <summary>
    /// Mirrors OptixPayloadType (optix_types.h) - one declared payload layout, referenced
    /// by <see cref="OptixModuleCompileOptions.NumPayloadTypes"/>/<see cref="OptixModuleCompileOptions.PayloadTypes"/>.
    /// </summary>
    [CLSCompliant(false)]
    public struct OptixPayloadType
    {
        public uint NumPayloadValues;

        /// <summary>Native pointer to a host array of <see cref="NumPayloadValues"/> semantics words.</summary>
        public IntPtr PayloadSemantics;
    }
}

#pragma warning restore CA1008 // Enums should have zero value
#pragma warning restore CA1028 // Enum Storage should be Int32
#pragma warning restore CA1051 // Do not declare visible instance fields
#pragma warning restore CA1815 // Override equals and operator equals on value types
