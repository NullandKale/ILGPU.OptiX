// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixDeviceView.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime;
using System.Diagnostics;

namespace ILGPU.OptiX.Device
{
    /// <summary>
    /// A minimal, GPU-blittable view over a contiguous device buffer, safe to embed as
    /// a field in a LaunchParams struct passed to OptiX device programs (raygen/
    /// closesthit/miss/anyhit). Neither ILGPU's own <c>ArrayView{T}</c> nor its internal
    /// <c>ILGPU.Backends.PointerViews.ViewImplementation{T}</c> work for this: the
    /// former carries a live managed <c>MemoryBuffer</c> reference (fails the
    /// <c>unmanaged</c> constraint <c>OptixLaunch</c> requires for its raw byte-copy of
    /// the whole LaunchParams struct to the device), and the latter's indexer uses
    /// <c>Unsafe.AsRef{T}</c> internally, which this library's OptiX-targeting compile
    /// path does not support. This type wraps a raw pointer with plain pointer
    /// dereference indexing instead - confirmed by direct testing to compile and run
    /// correctly through the OptiX pipeline, where the other two do not.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public unsafe readonly struct OptixDeviceView<T>
        where T : unmanaged
    {
        readonly T* ptr;

        /// <summary>
        /// The number of elements in this view.
        /// </summary>
        public readonly long Length;

        /// <summary>
        /// Constructs a view directly from a raw pointer and length.
        /// </summary>
        public OptixDeviceView(T* ptr, long length)
        {
            this.ptr = ptr;
            Length = length;
        }

        /// <summary>
        /// True if this view points at an allocated buffer - false for a scene/sample
        /// that leaves an optional buffer unallocated (matching the null-pointer
        /// sentinel this replaces; see <see cref="From"/>).
        /// </summary>
        public bool IsValid => ptr != null;

        /// <summary>
        /// Accesses the element at the given index.
        /// </summary>
        public ref T this[long index]
        {
            get
            {
                Debug.Assert(index >= 0 && index < Length, "OptixDeviceView index out of range");
                return ref ptr[index];
            }
        }

        /// <summary>
        /// Builds a view from a device buffer's own native pointer/length - the one
        /// place a caller needs to know a <c>MemoryBuffer1D{T}</c> has a
        /// <c>NativePtr</c> at all; every other call site just passes the buffer
        /// itself. Returns <see cref="default"/> (an invalid, zero-length view) for a
        /// null buffer, representing an optional/absent buffer.
        /// </summary>
        public static OptixDeviceView<T> From(MemoryBuffer1D<T, Stride1D.Dense> buffer) =>
            buffer == null ? default : new OptixDeviceView<T>((T*)buffer.NativePtr, buffer.Length);

        /// <summary>
        /// Builds a view from a raw-byte buffer, reinterpreting it as <typeparamref
        /// name="T"/> - for the case where the same physical buffer is written as
        /// <typeparamref name="T"/> by a raygen program but also consumed as raw bytes
        /// elsewhere (e.g. a row-flip kernel taking <c>ArrayView{byte}</c>), so the
        /// buffer itself is allocated as <c>byte</c> rather than <typeparamref name="T"/>.
        /// </summary>
        public static OptixDeviceView<T> FromBytes(MemoryBuffer1D<byte, Stride1D.Dense> buffer) =>
            buffer == null ? default : new OptixDeviceView<T>((T*)buffer.NativePtr, buffer.Length / sizeof(T));
    }
}
