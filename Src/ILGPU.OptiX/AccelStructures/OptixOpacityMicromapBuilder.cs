// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixOpacityMicromapBuilder.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.OptiX.Pipeline;
using ILGPU.OptiX.Util;
using System;

namespace ILGPU.OptiX.AccelStructures
{
    /// <summary>
    /// A single micromap's opacity state, for the common "one uniform state per whole
    /// triangle, no actual subdivision" case - lets
    /// the driver skip any-hit invocations entirely for that triangle (2-state format,
    /// no "unknown" ambiguity possible).
    /// </summary>
    [CLSCompliant(false)]
    public enum OptixOpacityMicromapState : byte
    {
        Transparent = 0,
        Opaque = 1,
    }

    /// <summary>
    /// Builds opacity micromap arrays - lets triangle
    /// geometry mark regions transparent/opaque without an any-hit program (the driver
    /// tests against the micromap directly). Only the simplest, most common case is
    /// exposed here: one whole-triangle uniform state per micromap (subdivision level
    /// 0 - "4^0 = 1 micro-triangle", i.e. no actual sub-triangle subdivision), 2-state
    /// format (unambiguous, no any-hit fallback ever needed). Real per-micro-triangle
    /// subdivided patterns (level > 0, encoding a texture-driven cutout shape via
    /// OptiX's micro-triangle index/Bird-curve ordering) remain out of scope - add
    /// that only when a concrete consumer needs finer-than-per-triangle cutouts, not
    /// speculatively.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixOpacityMicromapBuilder
    {
        /// <summary>
        /// Builds an opacity micromap array with one whole-triangle-uniform-state
        /// micromap per entry in <paramref name="states"/>, in LINEAR indexing order
        /// (triangle[i] in the eventual <see cref="OptixAccelBuilder.AddTriangleMesh"/>
        /// call uses <c>states[i]</c> - no index buffer needed).
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="accelerator">The accelerator to allocate device buffers from.</param>
        /// <param name="states">One state per triangle, in triangle order.</param>
        /// <param name="stream">The CUDA stream, or null to build synchronously.</param>
        public static BuiltOpacityMicromapArray BuildUniformStateArray(
            OptixDeviceContext deviceContext,
            CudaAccelerator accelerator,
            OptixOpacityMicromapState[] states,
            CudaStream stream = null)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (accelerator == null)
                throw new ArgumentNullException(nameof(accelerator));
            if (states == null || states.Length == 0)
                throw new ArgumentException("At least one state must be provided.", nameof(states));

            // One byte per micromap (a single 2-state bit would suffice, but a whole
            // byte per entry is trivially small here and avoids any risk of getting
            // sub-byte bit-packing wrong for no real memory benefit at this scale).
            var inputBytes = new byte[states.Length];
            var descs = new OptixOpacityMicromapDesc[states.Length];
            var usageCounts = new OptixOpacityMicromapUsageCount[]
            {
                new OptixOpacityMicromapUsageCount
                {
                    Count = (uint)states.Length,
                    SubdivisionLevel = 0,
                    Format = OptixOpacityMicromapFormat.TwoState,
                },
            };

            for (int i = 0; i < states.Length; i++)
            {
                inputBytes[i] = (byte)states[i];
                descs[i] = new OptixOpacityMicromapDesc
                {
                    ByteOffset = (uint)i,
                    SubdivisionLevel = 0,
                    Format = OptixOpacityMicromapFormat.TwoState,
                };
            }

            var histogram = new[]
            {
                new OptixOpacityMicromapHistogramEntry
                {
                    Count = (uint)states.Length,
                    SubdivisionLevel = 0,
                    Format = OptixOpacityMicromapFormat.TwoState,
                },
            };

            MemoryBuffer1D<byte, Stride1D.Dense> inputBuffer = null;
            MemoryBuffer1D<OptixOpacityMicromapDesc, Stride1D.Dense> descBuffer = null;
            MemoryBuffer1D<byte, Stride1D.Dense> outputBuffer = null;
            MemoryBuffer1D<byte, Stride1D.Dense> tempBuffer = null;
            SafeHGlobal usageCountsHandle = null;

            try
            {
                inputBuffer = accelerator.Allocate1D(inputBytes);
                descBuffer = accelerator.Allocate1D(descs);
                // MicromapHistogramEntries is host-readable (a plain pointer in
                // optix_types.h, not CUdeviceptr, same as OptixBuildInputTriangleArray's
                // own Flags field) - only needs to survive this one synchronous call.
                using var histogramHandle = SafeHGlobal.AllocFrom<OptixOpacityMicromapHistogramEntry>(histogram);

                var buildInput = new OptixOpacityMicromapArrayBuildInput
                {
                    Flags = (uint)OptixOpacityMicromapFlags.PreferFastTrace,
                    InputBuffer = inputBuffer.NativePtr,
                    PerMicromapDescBuffer = descBuffer.NativePtr,
                    PerMicromapDescStrideInBytes = 0,
                    NumMicromapHistogramEntries = (uint)histogram.Length,
                    MicromapHistogramEntries = histogramHandle,
                };

                var bufferSizes = deviceContext.OpacityMicromapArrayComputeMemoryUsage(buildInput);

                outputBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.OutputSizeInBytes);
                tempBuffer = accelerator.Allocate1D<byte>((long)Math.Max(1UL, bufferSizes.TempSizeInBytes));

                var buildStream = stream ?? accelerator.DefaultStream;
                deviceContext.OpacityMicromapArrayBuild(
                    buildStream,
                    buildInput,
                    new OptixMicromapBuffers
                    {
                        Output = outputBuffer.NativePtr,
                        OutputSizeInBytes = (ulong)outputBuffer.LengthInBytes,
                        Temp = tempBuffer.NativePtr,
                        TempSizeInBytes = (ulong)tempBuffer.LengthInBytes,
                    });

                if (stream == null)
                    accelerator.Synchronize();

                // MicromapUsageCounts (on OptixBuildInputOpacityMicromap, set by
                // AddTriangleMesh later) is also host-readable, but must stay alive
                // through the eventual OptixAccelBuilder.Build() call - persisted on
                // the returned object, not disposed here.
                usageCountsHandle = SafeHGlobal.AllocFrom<OptixOpacityMicromapUsageCount>(usageCounts);

                var result = new BuiltOpacityMicromapArray(
                    outputBuffer, usageCountsHandle, (uint)usageCounts.Length);
                outputBuffer = null; // Result now owns it
                usageCountsHandle = null;
                return result;
            }
            finally
            {
                inputBuffer?.Dispose();
                descBuffer?.Dispose();
                tempBuffer?.Dispose();
                outputBuffer?.Dispose();
                usageCountsHandle?.Dispose();
            }
        }
    }

    /// <summary>
    /// A built opacity micromap array, owning its device buffer - pass to
    /// <see cref="OptixAccelBuilder.AddTriangleMesh"/> to attach it to a triangle mesh.
    /// </summary>
    [CLSCompliant(false)]
    public sealed class BuiltOpacityMicromapArray : IDisposable
    {
        private bool disposed;
        private readonly MemoryBuffer1D<byte, Stride1D.Dense> outputBuffer;
        private readonly SafeHGlobal usageCountsHandle;

        /// <summary>Device pointer to the built micromap array.</summary>
        public IntPtr ArrayDevicePointer => outputBuffer.NativePtr;

        internal IntPtr UsageCountsPtr => usageCountsHandle;
        internal uint NumUsageCounts { get; }

        internal BuiltOpacityMicromapArray(
            MemoryBuffer1D<byte, Stride1D.Dense> outputBuffer,
            SafeHGlobal usageCountsHandle,
            uint numUsageCounts)
        {
            this.outputBuffer = outputBuffer;
            this.usageCountsHandle = usageCountsHandle;
            NumUsageCounts = numUsageCounts;
        }

        public void Dispose()
        {
            if (!disposed)
            {
                disposed = true;
                outputBuffer?.Dispose();
                usageCountsHandle?.Dispose();
                GC.SuppressFinalize(this);
            }
        }
    }
}
