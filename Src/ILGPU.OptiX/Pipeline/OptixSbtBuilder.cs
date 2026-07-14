// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixSbtBuilder.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.OptiX.Interop;
using ILGPU.Util;
using ILGPU.OptiX.Native;
using System;
using System.Runtime.InteropServices;

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Packs SBT records with automatic alignment, sizing, and buffer management.
    /// Supports generic record types packed via OptixSbt.PackRecords.
    /// </summary>
    public sealed class OptixSbtBuilder : DisposeBase
    {
        private class BufferData
        {
            public MemoryBuffer Buffer { get; set; }
            public uint Count { get; set; }
            public uint Stride { get; set; }
        }

        private CudaAccelerator? accelerator;
        private BufferData? raygenData;
        private BufferData? missData;
        private BufferData? hitgroupData;

        /// <summary>
        /// Sets the accelerator for GPU memory allocation.
        /// </summary>
        public OptixSbtBuilder WithAccelerator(CudaAccelerator accel)
        {
            if (accel == null)
                throw new ArgumentNullException(nameof(accel), "Accelerator must not be null.");

            accelerator = accel;
            return this;
        }

        // OptixSbt.PackRecords<TRecord>() already validates this for the (only) record
        // construction path every current sample uses, but a caller could in principle
        // hand-build a records array without going through PackRecords - validate here
        // too so a misaligned record is always a clean C# exception, never an unchecked
        // native OptiX failure. Real alignment requirement is 16 bytes
        // (OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT), not 32 - the original design doc's "must
        // be 32-byte aligned" text was never checked against the actual OptiX constant.
        private static void ValidateAlignment<TRecord>() where TRecord : unmanaged
        {
            int size = Marshal.SizeOf<TRecord>();
            if (size % OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT != 0)
                throw new ArgumentException(
                    $"Record type {typeof(TRecord).Name} is not {OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT}-byte " +
                    $"aligned; expected sizeof({typeof(TRecord).Name}) % {OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT} == 0, " +
                    $"got {size} bytes. Use [StructLayout(LayoutKind.Sequential, Size = ...)] to declare alignment.");
        }

        /// <summary>
        /// Sets raygen records (already packed array).
        /// For kernel collections, call OptixSbt.PackRecords{TRecord}(kernels) first.
        /// </summary>
        public OptixSbtBuilder SetRaygenRecords<TRecord>(TRecord[] records) where TRecord : unmanaged
        {
            if (records == null)
                throw new ArgumentNullException(nameof(records));
            if (accelerator == null)
                throw new InvalidOperationException("Accelerator must be set via WithAccelerator() first.");
            ValidateAlignment<TRecord>();

            // Free a previous call's buffer rather than leaking it - Set*Records()
            // allocates immediately (not lazily at Build() time), so calling this twice
            // on the same builder would otherwise orphan the first buffer.
            raygenData?.Buffer.Dispose();

            var buffer = accelerator.Allocate1D(records);
            raygenData = new BufferData
            {
                Buffer = buffer,
                Count = (uint)records.Length,
                Stride = (uint)Marshal.SizeOf<TRecord>()
            };
            return this;
        }

        /// <summary>
        /// Sets the raygen record from a buffer packed by
        /// <see cref="OptixSbtRecords.Pack{T}(System.Collections.Generic.IReadOnlyList{OptixKernel}, System.ReadOnlySpan{T})"/>.
        /// Unlike the generic <see cref="SetRaygenRecords{TRecord}(TRecord[])"/>
        /// overload, the stride is explicit, so <typeparamref name="T"/> never needs a
        /// hand-declared <c>[StructLayout(Size = ...)]</c> to satisfy SBT alignment.
        /// </summary>
        public OptixSbtBuilder SetRaygenRecords<T>(byte[] packedRecordBytes) where T : unmanaged
        {
            if (packedRecordBytes == null)
                throw new ArgumentNullException(nameof(packedRecordBytes));
            if (accelerator == null)
                throw new InvalidOperationException("Accelerator must be set via WithAccelerator() first.");

            raygenData?.Buffer.Dispose();

            var buffer = accelerator.Allocate1D(packedRecordBytes);
            raygenData = new BufferData
            {
                Buffer = buffer,
                Count = 1,
                Stride = (uint)OptixSbtRecords.StrideOf<T>()
            };
            return this;
        }

        /// <summary>
        /// Sets miss records from a buffer packed by
        /// <see cref="OptixSbtRecords.Pack{T}(System.Collections.Generic.IReadOnlyList{OptixKernel}, System.ReadOnlySpan{T})"/>.
        /// </summary>
        public OptixSbtBuilder SetMissRecords<T>(byte[] packedRecordBytes) where T : unmanaged
        {
            if (packedRecordBytes == null)
                throw new ArgumentNullException(nameof(packedRecordBytes));
            if (accelerator == null)
                throw new InvalidOperationException("Accelerator must be set via WithAccelerator() first.");

            missData?.Buffer.Dispose();

            var stride = (uint)OptixSbtRecords.StrideOf<T>();
            var buffer = accelerator.Allocate1D(packedRecordBytes);
            missData = new BufferData
            {
                Buffer = buffer,
                Count = (uint)(packedRecordBytes.Length / stride),
                Stride = stride
            };
            return this;
        }

        /// <summary>
        /// Adds hitgroup records from a buffer packed by
        /// <see cref="OptixSbtRecords.Pack{T}(System.Collections.Generic.IReadOnlyList{OptixKernel}, System.ReadOnlySpan{T})"/>.
        /// </summary>
        public OptixSbtBuilder AddHitgroupRecords<T>(byte[] packedRecordBytes) where T : unmanaged
        {
            if (packedRecordBytes == null)
                throw new ArgumentNullException(nameof(packedRecordBytes));
            if (accelerator == null)
                throw new InvalidOperationException("Accelerator must be set via WithAccelerator() first.");

            hitgroupData?.Buffer.Dispose();

            var stride = (uint)OptixSbtRecords.StrideOf<T>();
            var buffer = accelerator.Allocate1D(packedRecordBytes);
            hitgroupData = new BufferData
            {
                Buffer = buffer,
                Count = (uint)(packedRecordBytes.Length / stride),
                Stride = stride
            };
            return this;
        }

        /// <summary>
        /// Sets miss records (already packed array).
        /// For kernel collections, call OptixSbt.PackRecords{TRecord}(kernels) first.
        /// </summary>
        public OptixSbtBuilder SetMissRecords<TRecord>(TRecord[] records) where TRecord : unmanaged
        {
            if (records == null)
                throw new ArgumentNullException(nameof(records));
            if (accelerator == null)
                throw new InvalidOperationException("Accelerator must be set via WithAccelerator() first.");
            ValidateAlignment<TRecord>();

            missData?.Buffer.Dispose();

            var buffer = accelerator.Allocate1D(records);
            missData = new BufferData
            {
                Buffer = buffer,
                Count = (uint)records.Length,
                Stride = (uint)Marshal.SizeOf<TRecord>()
            };
            return this;
        }

        /// <summary>
        /// Adds hitgroup records (already packed array).
        /// For kernel collections, call OptixSbt.PackRecords{TRecord}(kernels) first.
        /// </summary>
        public OptixSbtBuilder AddHitgroupRecords<TRecord>(TRecord[] records) where TRecord : unmanaged
        {
            if (records == null)
                throw new ArgumentNullException(nameof(records));
            if (accelerator == null)
                throw new InvalidOperationException("Accelerator must be set via WithAccelerator() first.");
            ValidateAlignment<TRecord>();

            hitgroupData?.Buffer.Dispose();

            var buffer = accelerator.Allocate1D(records);
            hitgroupData = new BufferData
            {
                Buffer = buffer,
                Count = (uint)records.Length,
                Stride = (uint)Marshal.SizeOf<TRecord>()
            };
            return this;
        }

        /// <summary>
        /// Builds the SBT with allocated GPU buffers.
        /// </summary>
        public BuiltSbt Build(CudaStream? stream = null)
        {
            if (accelerator == null)
                throw new InvalidOperationException("Accelerator must be set via WithAccelerator().");
            if (raygenData == null)
                throw new InvalidOperationException("Raygen records must be set via SetRaygenRecords().");

            try
            {
                var sbt = new OptixShaderBindingTable
                {
                    RaygenRecord = raygenData.Buffer.NativePtr,
                    MissRecordBase = missData?.Buffer.NativePtr ?? IntPtr.Zero,
                    MissRecordStrideInBytes = missData?.Stride ?? 0,
                    MissRecordCount = missData?.Count ?? 0,
                    HitgroupRecordBase = hitgroupData?.Buffer.NativePtr ?? IntPtr.Zero,
                    HitgroupRecordStrideInBytes = hitgroupData?.Stride ?? 0,
                    HitgroupRecordCount = hitgroupData?.Count ?? 0,
                };

                var built = new BuiltSbt(sbt, raygenData.Buffer, missData?.Buffer, hitgroupData?.Buffer, accelerator, stream);

                // Ownership of the buffers has transferred to `built` - clear our own
                // references so Dispose() below (if this now-spent builder is later
                // disposed) doesn't also try to free them.
                raygenData = null;
                missData = null;
                hitgroupData = null;

                return built;
            }
            catch
            {
                // Free whatever this builder had already allocated - matches the
                // documented "any GPU memory allocated during a failed build is freed
                // before the exception propagates" contract (see OptixAccelBuilder.Build()
                // for the same pattern). Previously this catch block did nothing.
                raygenData?.Buffer.Dispose();
                missData?.Buffer.Dispose();
                hitgroupData?.Buffer.Dispose();
                raygenData = null;
                missData = null;
                hitgroupData = null;
                throw;
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                // Only reaches live buffers if Build() was never called on this
                // instance - anything already handed to a BuiltSbt has had its
                // reference cleared by Build() above, so this can't double-dispose.
                // Previously this method did nothing, leaking any buffer allocated via
                // Set*Records()/AddHitgroupRecords() if Build() was never reached.
                raygenData?.Buffer.Dispose();
                missData?.Buffer.Dispose();
                hitgroupData?.Buffer.Dispose();
                raygenData = null;
                missData = null;
                hitgroupData = null;
            }
            base.Dispose(disposing);
        }
    }

    /// <summary>
    /// Result of SBT building, owning all GPU buffers.
    /// </summary>
    public sealed class BuiltSbt : IDisposable
    {
        private bool disposed;

        public OptixShaderBindingTable Sbt { get; }
        private readonly MemoryBuffer raygenBuffer;
        private readonly MemoryBuffer? missBuffer;
        private readonly MemoryBuffer? hitgroupBuffer;
        private readonly CudaAccelerator accelerator;

        internal BuiltSbt(OptixShaderBindingTable sbt, MemoryBuffer raygenBuffer, MemoryBuffer? missBuffer,
            MemoryBuffer? hitgroupBuffer, CudaAccelerator accelerator, CudaStream? stream)
        {
            Sbt = sbt;
            this.raygenBuffer = raygenBuffer;
            this.missBuffer = missBuffer;
            this.hitgroupBuffer = hitgroupBuffer;
            this.accelerator = accelerator;

            // If stream is null, synchronize device
            if (stream == null)
            {
                accelerator.Synchronize();
            }
        }

        public void Dispose()
        {
            if (!disposed)
            {
                disposed = true;
                hitgroupBuffer?.Dispose();
                missBuffer?.Dispose();
                raygenBuffer?.Dispose();
                GC.SuppressFinalize(this);
            }
        }
    }
}
