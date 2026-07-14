// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixSbtRecords.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using ILGPU.OptiX.Native;

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Packs <see cref="SbtRecord{T}"/> arrays into stride-aligned, upload-ready byte
    /// buffers. Because the SBT takes an explicit per-table stride
    /// (<see cref="Interop.OptixShaderBindingTable.MissRecordStrideInBytes"/>,
    /// <see cref="Interop.OptixShaderBindingTable.HitgroupRecordStrideInBytes"/>),
    /// records need not be tightly packed in memory - the padding required to satisfy
    /// <see cref="OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT"/> lives between records in the
    /// packed buffer, computed here, so no consumer ever declares
    /// <c>[StructLayout(Size = ...)]</c> or measures a record's size by hand again.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixSbtRecords
    {
        /// <summary>
        /// The per-record stride, in bytes, for an <see cref="SbtRecord{T}"/> array -
        /// the natural size of the record rounded up to
        /// <see cref="OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT"/>.
        /// </summary>
        public static int StrideOf<T>() where T : unmanaged
        {
            // Marshal.SizeOf<T>() throws ArgumentException ("must not be a generic
            // type") for ANY generic type argument, even a fully closed instantiation
            // like SbtRecord<MaterialSbtData> - not just open ones. Unsafe.SizeOf<T>()
            // has no such restriction and is what we actually want here anyway: the
            // real CLR layout size, not a COM-marshaled size.
            int rawSize = Unsafe.SizeOf<SbtRecord<T>>();
            int alignment = OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT;
            return ((rawSize + alignment - 1) / alignment) * alignment;
        }

        /// <summary>
        /// Packs one header (from each of <paramref name="kernels"/>) and one data
        /// value (from the matching entry of <paramref name="data"/>) per record, into
        /// a single stride-aligned buffer ready to upload as an SBT record table.
        /// </summary>
        /// <param name="kernels">
        /// One kernel per record, in order; each contributes its program group's
        /// header via <c>optixSbtRecordPackHeader</c>.
        /// </param>
        /// <param name="data">One data value per record, in the same order.</param>
        public static byte[] Pack<T>(IReadOnlyList<OptixKernel> kernels, ReadOnlySpan<T> data)
            where T : unmanaged
        {
            if (kernels == null)
                throw new ArgumentNullException(nameof(kernels));
            if (kernels.Count != data.Length)
                throw new ArgumentException(
                    $"Kernel count ({kernels.Count}) must match data count ({data.Length}); " +
                    "each SBT record needs exactly one kernel (for its header) and one data value.",
                    nameof(data));

            int stride = StrideOf<T>();
            var buffer = new byte[stride * kernels.Count];

            unsafe
            {
                fixed (byte* bufferBase = buffer)
                fixed (T* dataBase = data)
                {
                    for (var i = 0; i < kernels.Count; i++)
                    {
                        var recordPtr = bufferBase + (i * stride);

                        OptixAPI.Current.SbtRecordPackHeader(
                            kernels[i].ProgramGroup.ProgramGroupPtr,
                            new IntPtr(recordPtr));

                        *(T*)(recordPtr + OptixAPI.OPTIX_SBT_RECORD_HEADER_SIZE) = dataBase[i];
                    }
                }
            }

            return buffer;
        }
    }
}
