// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixMotionTransformBuilder.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.OptiX.Pipeline;
using System;

namespace ILGPU.OptiX.AccelStructures
{
    /// <summary>
    /// Builds rigid-body motion transform traversables - the standard OptiX approach
    /// for "spinning/translating object" motion blur,
    /// wrapping a static child traversable (typically a GAS built normally, with no
    /// motion options of its own) in a time-varying transform interpolated between
    /// motion keys. This is distinct from vertex/deforming motion blur (multiple
    /// per-key vertex buffers on the GAS build input itself), which remains
    /// unimplemented - out of scope until a concrete consumer needs it.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixMotionTransformBuilder
    {
        /// <summary>
        /// Builds a 2-key matrix motion transform wrapping <paramref name="childTraversableHandle"/>.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="accelerator">The accelerator to allocate the transform's device buffer from.</param>
        /// <param name="childTraversableHandle">The static traversable (typically a GAS) this transform moves.</param>
        /// <param name="timeBegin">The point in time motion key 0 applies at.</param>
        /// <param name="timeEnd">The point in time motion key 1 applies at (must be greater than <paramref name="timeBegin"/>).</param>
        /// <param name="transformKey0">Object-to-world 3x4 row-major matrix (12 floats) at <paramref name="timeBegin"/>.</param>
        /// <param name="transformKey1">Object-to-world 3x4 row-major matrix (12 floats) at <paramref name="timeEnd"/>.</param>
        /// <param name="flags">Motion flags (start/end vanish), default none.</param>
        public static BuiltMotionTransform BuildMatrixMotionTransform(
            OptixDeviceContext deviceContext,
            CudaAccelerator accelerator,
            IntPtr childTraversableHandle,
            float timeBegin,
            float timeEnd,
            float[] transformKey0,
            float[] transformKey1,
            OptixMotionFlags flags = OptixMotionFlags.None)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (accelerator == null)
                throw new ArgumentNullException(nameof(accelerator));
            if (transformKey0 == null || transformKey0.Length != 12)
                throw new ArgumentException("Must have exactly 12 elements (row-major 3x4).", nameof(transformKey0));
            if (transformKey1 == null || transformKey1.Length != 12)
                throw new ArgumentException("Must have exactly 12 elements (row-major 3x4).", nameof(transformKey1));
            if (timeEnd <= timeBegin)
                throw new ArgumentOutOfRangeException(nameof(timeEnd), "timeEnd must be greater than timeBegin.");

            var native = new OptixMatrixMotionTransform
            {
                Child = unchecked((ulong)childTraversableHandle.ToInt64()),
                MotionOptions = new OptixMotionOptions
                {
                    NumKeys = 2,
                    Flags = (ushort)flags,
                    TimeBegin = timeBegin,
                    TimeEnd = timeEnd,
                },
            };
            unsafe
            {
                for (int i = 0; i < 12; i++)
                {
                    native.Transform[i] = transformKey0[i];
                    native.Transform[12 + i] = transformKey1[i];
                }
            }

            // CUDA device allocations are consistently aligned well past the 64-byte
            // OPTIX_TRANSFORM_BYTE_ALIGNMENT this struct requires - same assumption
            // OptixAccelBuilder already relies on for its own (128-byte-aligned) AS
            // output buffers, with no extra padding logic needed there either.
            MemoryBuffer1D<OptixMatrixMotionTransform, Stride1D.Dense> buffer = null;
            try
            {
                buffer = accelerator.Allocate1D(new[] { native });
                accelerator.Synchronize();

                var handle = deviceContext.ConvertPointerToTraversableHandle(
                    buffer.NativePtr, OptixTraversableType.MatrixMotionTransform);

                var result = new BuiltMotionTransform(handle, buffer);
                buffer = null; // Result now owns it
                return result;
            }
            catch
            {
                buffer?.Dispose();
                throw;
            }
        }
    }

    /// <summary>
    /// A built motion transform traversable, owning its device buffer.
    /// </summary>
    [CLSCompliant(false)]
    public sealed class BuiltMotionTransform : IDisposable
    {
        private bool disposed;

        public IntPtr TraversableHandle { get; }
        internal MemoryBuffer1D<OptixMatrixMotionTransform, Stride1D.Dense> Buffer { get; }

        internal BuiltMotionTransform(IntPtr handle, MemoryBuffer1D<OptixMatrixMotionTransform, Stride1D.Dense> buffer)
        {
            TraversableHandle = handle;
            Buffer = buffer;
        }

        public void Dispose()
        {
            if (!disposed)
            {
                disposed = true;
                Buffer?.Dispose();
                GC.SuppressFinalize(this);
            }
        }
    }
}
