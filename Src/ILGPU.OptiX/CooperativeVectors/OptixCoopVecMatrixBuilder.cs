// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixCoopVecMatrixBuilder.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX.Native;
using ILGPU.OptiX.Pipeline;
using ILGPU.OptiX.Util;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.CooperativeVectors
{
    /// <summary>
    /// Safe fluent entry points for the OptiX cooperative-vector host API - computing matrix buffer sizes and converting
    /// matrices between layouts (typically row-major, as uploaded by the host, into
    /// InferencingOptimal, as consumed by <see cref="DeviceApi.OptixCoopVec.MatVecMul"/>).
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixCoopVecMatrixBuilder
    {
        /// <summary>
        /// Computes the required buffer size, in bytes, for a single matrix of the
        /// given shape/type/layout. The driver rounds the result up to a multiple of
        /// 64 bytes so matrices can be tightly packed while staying aligned.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="N">Matrix row count.</param>
        /// <param name="K">Matrix column count.</param>
        /// <param name="elementType">The matrix element type.</param>
        /// <param name="layout">The matrix layout.</param>
        /// <param name="rowColumnStrideInBytes">Ignored for the two "optimal" layouts; 0 assumes tight packing.</param>
        /// <returns>The required size in bytes.</returns>
        public static uint ComputeMatrixSizeInBytes(
            this OptixDeviceContext deviceContext,
            uint N,
            uint K,
            OptixCoopVecElemType elementType,
            OptixCoopVecMatrixLayout layout,
            uint rowColumnStrideInBytes = 0)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));

            OptixException.ThrowIfFailed(
                OptixAPI.Current.CoopVecMatrixComputeSize(
                    deviceContext.DeviceContextPtr,
                    N,
                    K,
                    elementType,
                    layout,
                    rowColumnStrideInBytes,
                    out var sizeInBytes));
            return (uint)sizeInBytes;
        }

        /// <summary>
        /// Converts one or more identically-shaped "networks" (each network is an
        /// ordered array of matrix layers, per <c>OptixNetworkDescription</c>) from the
        /// layout/type described by <paramref name="inputLayerDescriptions"/> into the
        /// layout/type described by <paramref name="outputLayerDescriptions"/> - the
        /// full <c>optixCoopVecMatrixConvert</c> surface, including the batch case
        /// (<paramref name="numNetworks"/> &gt; 1, e.g. per-object weight sets packed
        /// side by side and converted in one call via
        /// <paramref name="inputNetworkStrideInBytes"/>/<paramref name="outputNetworkStrideInBytes"/>).
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="numNetworks">Number of network instances to convert.</param>
        /// <param name="inputLayerDescriptions">Describes each matrix layer already present at <paramref name="inputBuffer"/> (same topology for every network instance).</param>
        /// <param name="inputBuffer">Device buffer holding the input network(s).</param>
        /// <param name="inputNetworkStrideInBytes">Byte stride between consecutive network instances in <paramref name="inputBuffer"/>; ignored if <paramref name="numNetworks"/> is 1.</param>
        /// <param name="outputLayerDescriptions">Describes each matrix layer's desired layout at <paramref name="outputBuffer"/> (same topology for every network instance).</param>
        /// <param name="outputBuffer">Device buffer to receive the converted network(s).</param>
        /// <param name="outputNetworkStrideInBytes">Byte stride between consecutive network instances in <paramref name="outputBuffer"/>; ignored if <paramref name="numNetworks"/> is 1.</param>
        public static void ConvertMatrices(
            this OptixDeviceContext deviceContext,
            AcceleratorStream stream,
            uint numNetworks,
            OptixCoopVecMatrixDescription[] inputLayerDescriptions,
            MemoryBuffer inputBuffer,
            ulong inputNetworkStrideInBytes,
            OptixCoopVecMatrixDescription[] outputLayerDescriptions,
            MemoryBuffer outputBuffer,
            ulong outputNetworkStrideInBytes)
        {
            if (deviceContext == null)
                throw new ArgumentNullException(nameof(deviceContext));
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (stream is not CudaStream cudaStream)
                throw new ArgumentOutOfRangeException(nameof(stream));
            if (inputLayerDescriptions == null || inputLayerDescriptions.Length == 0)
                throw new ArgumentException("At least one layer description must be provided.", nameof(inputLayerDescriptions));
            if (outputLayerDescriptions == null || outputLayerDescriptions.Length == 0)
                throw new ArgumentException("At least one layer description must be provided.", nameof(outputLayerDescriptions));
            if (inputBuffer == null)
                throw new ArgumentNullException(nameof(inputBuffer));
            if (outputBuffer == null)
                throw new ArgumentNullException(nameof(outputBuffer));

            // OptixNetworkDescription.layers is itself a pointer, so this needs two
            // marshaling levels: the per-layer descriptor array first, then the network
            // descriptor struct embedding a pointer to it (same two-level pattern
            // OptixAccelBuilder.Build uses for its per-build-input pointer arrays).
            using var inputLayersHandle = SafeHGlobal.AllocFrom<OptixCoopVecMatrixDescription>(inputLayerDescriptions);
            using var outputLayersHandle = SafeHGlobal.AllocFrom<OptixCoopVecMatrixDescription>(outputLayerDescriptions);

            var inputNetwork = new OptixNetworkDescriptionNative { Layers = inputLayersHandle, NumLayers = (uint)inputLayerDescriptions.Length };
            var outputNetwork = new OptixNetworkDescriptionNative { Layers = outputLayersHandle, NumLayers = (uint)outputLayerDescriptions.Length };
            using var inputNetworkHandle = SafeHGlobal.AllocFrom(inputNetwork);
            using var outputNetworkHandle = SafeHGlobal.AllocFrom(outputNetwork);

            OptixException.ThrowIfFailed(
                OptixAPI.Current.CoopVecMatrixConvert(
                    deviceContext.DeviceContextPtr,
                    cudaStream.StreamPtr,
                    numNetworks,
                    inputNetworkHandle,
                    inputBuffer.GetDeviceAddress(),
                    inputNetworkStrideInBytes,
                    outputNetworkHandle,
                    outputBuffer.GetDeviceAddress(),
                    outputNetworkStrideInBytes));
        }

        /// <summary>
        /// Converts a single matrix (a one-layer, one-instance network) already uploaded
        /// at <paramref name="inputBuffer"/> into <paramref name="outputBuffer"/> - a
        /// convenience wrapper over <see cref="ConvertMatrices"/> for the common case
        /// (e.g. one weight matrix at a time, as Sample24 does for each MLP layer).
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="inputDescription">Describes the matrix already present at <paramref name="inputBuffer"/>.</param>
        /// <param name="inputBuffer">Device buffer holding the input matrix.</param>
        /// <param name="outputDescription">Describes the desired matrix layout at <paramref name="outputBuffer"/>.</param>
        /// <param name="outputBuffer">Device buffer to receive the converted matrix.</param>
        public static void ConvertMatrix(
            this OptixDeviceContext deviceContext,
            AcceleratorStream stream,
            OptixCoopVecMatrixDescription inputDescription,
            MemoryBuffer inputBuffer,
            OptixCoopVecMatrixDescription outputDescription,
            MemoryBuffer outputBuffer) =>
            deviceContext.ConvertMatrices(
                stream,
                numNetworks: 1,
                new[] { inputDescription },
                inputBuffer,
                inputNetworkStrideInBytes: 0,
                new[] { outputDescription },
                outputBuffer,
                outputNetworkStrideInBytes: 0);
    }

    /// <summary>
    /// Extension for getting a cooperative-vector-usable device address (a plain
    /// <see cref="ulong"/> "CUdeviceptr") out of an ILGPU device buffer, without a
    /// sample ever touching <see cref="IntPtr"/>/<c>unsafe</c> directly (hard
    /// constraint). The coop-vec device intrinsics
    /// (<see cref="DeviceApi.OptixCoopVec"/>) take these addresses as ordinary
    /// <see cref="ulong"/> kernel parameters - flowing through a LaunchParams struct
    /// exactly like an <c>OptixTraversableHandle</c> already does - rather than trying
    /// to compute a buffer's address from inside device code (ILGPU has no supported
    /// device-side intrinsic for that; the closest host-only helper is
    /// <c>ArrayView&lt;T&gt;.LoadEffectiveAddressAsPtr</c>, explicitly disabled inside
    /// kernels).
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixCoopVecBufferExtensions
    {
        /// <summary>Returns the device address of <paramref name="buffer"/> as a plain <see cref="ulong"/>.</summary>
        public static ulong GetDeviceAddress(this MemoryBuffer buffer)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));
            return unchecked((ulong)buffer.NativePtr.ToInt64());
        }
    }
}
