// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: SbtRecord.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

#pragma warning disable CA1051 // Do not declare visible instance fields

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// The native layout of a single SBT record: a fixed-size header (written by
    /// <see cref="OptixAPI.SbtRecordPackHeader(IntPtr, IntPtr)"/>) followed immediately
    /// by the record's per-shader data. Using this type instead of hand-writing
    /// <c>fixed byte Header[32]</c> boilerplate means no consumer ever has to measure or
    /// declare a record's total byte size again - the SBT's stride is computed at
    /// runtime by <see cref="OptixSbtRecords.StrideOf{T}"/>.
    /// </summary>
    /// <typeparam name="T">The unmanaged record data type read by the shader.</typeparam>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct SbtRecord<T> where T : unmanaged
    {
        /// <summary>
        /// Opaque header written by the OptiX runtime; never touched by consumer code.
        /// </summary>
        public fixed byte Header[OptixAPI.OPTIX_SBT_RECORD_HEADER_SIZE];

        /// <summary>
        /// The record's shader-visible data.
        /// </summary>
        public T Data;

        static SbtRecord()
        {
            // The header must sit at offset 0 and Data immediately after it - if either
            // ever drifts (e.g. from a future field being added above Header), every
            // hit program's `(T*)OptixGetSbtDataPointer.Value` cast silently reads
            // garbage instead of failing loudly, so assert the exact layout once here.
            // Marshal.OffsetOf<T>()/SizeOf<T>() both throw ArgumentException ("must not
            // be a generic type") for ANY generic type argument, even a fully closed
            // instantiation like SbtRecord<MaterialSbtData> - not just open ones - so
            // layout must be measured via raw pointer arithmetic on a local instance
            // instead of the interop marshaler.
            SbtRecord<T> instance = default;
            byte* basePtr = (byte*)&instance;
            byte* headerPtr = instance.Header;
            byte* dataPtr = (byte*)&instance.Data;

            Debug.Assert(
                headerPtr - basePtr == 0,
                "SbtRecord<T>.Header must be the first field.");
            Debug.Assert(
                dataPtr - basePtr == OptixAPI.OPTIX_SBT_RECORD_HEADER_SIZE,
                "SbtRecord<T>.Data must immediately follow the header with no padding " +
                "between them.");
        }
    }
}

#pragma warning restore CA1051 // Do not declare visible instance fields
