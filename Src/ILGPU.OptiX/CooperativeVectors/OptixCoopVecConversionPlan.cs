// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixCoopVecConversionPlan.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX.Native;
using ILGPU.OptiX.Pipeline;
using ILGPU.OptiX.Util;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Util;
using System;

namespace ILGPU.OptiX.CooperativeVectors
{
    /// <summary>
    /// A reusable, pre-marshaled matrix-conversion plan for a fixed network topology -
    /// the persistent counterpart to
    /// <see cref="OptixCoopVecMatrixBuilder.ConvertMatrices"/>, which marshals and
    /// frees its input/output network descriptions (four native allocations plus
    /// per-element marshaling) on every single call. A caller converting the same
    /// topology repeatedly - e.g. weight/gradient layout conversions once or more per
    /// frame - builds one plan per (input layout, output layout) pair up front and
    /// calls <see cref="Convert"/> with whatever buffers apply, allocation-free.
    ///
    /// <para>
    /// The descriptions describe layouts, not storage: the same plan converts any
    /// buffer pair matching its topology (e.g. one RowMajor-to-TrainingOptimal plan
    /// serves both a training-weights and an inference-weights conversion).
    /// </para>
    /// </summary>
    [CLSCompliant(false)]
    public sealed class OptixCoopVecConversionPlan : DisposeBase
    {
        private readonly OptixDeviceContext deviceContext;
        private readonly SafeHGlobal inputLayersHandle;
        private readonly SafeHGlobal outputLayersHandle;
        private readonly SafeHGlobal inputNetworkHandle;
        private readonly SafeHGlobal outputNetworkHandle;

        /// <summary>
        /// Marshals the input/output network descriptions once, for reuse by every
        /// subsequent <see cref="Convert"/> call.
        /// </summary>
        /// <param name="deviceContext">The OptiX device context.</param>
        /// <param name="inputLayerDescriptions">Describes each matrix layer present at a Convert call's input buffer.</param>
        /// <param name="outputLayerDescriptions">Describes each matrix layer's desired layout at a Convert call's output buffer.</param>
        public OptixCoopVecConversionPlan(
            OptixDeviceContext deviceContext,
            OptixCoopVecMatrixDescription[] inputLayerDescriptions,
            OptixCoopVecMatrixDescription[] outputLayerDescriptions)
        {
            this.deviceContext = deviceContext
                ?? throw new ArgumentNullException(nameof(deviceContext));
            if (inputLayerDescriptions == null || inputLayerDescriptions.Length == 0)
                throw new ArgumentException("At least one layer description must be provided.", nameof(inputLayerDescriptions));
            if (outputLayerDescriptions == null || outputLayerDescriptions.Length == 0)
                throw new ArgumentException("At least one layer description must be provided.", nameof(outputLayerDescriptions));

            // OptixNetworkDescription.layers is itself a pointer, so this needs two
            // marshaling levels: the per-layer descriptor array first, then the network
            // descriptor struct embedding a pointer to it (same two-level pattern
            // OptixAccelBuilder.Build uses for its per-build-input pointer arrays).
            inputLayersHandle = SafeHGlobal.AllocFrom<OptixCoopVecMatrixDescription>(inputLayerDescriptions);
            outputLayersHandle = SafeHGlobal.AllocFrom<OptixCoopVecMatrixDescription>(outputLayerDescriptions);

            var inputNetwork = new OptixNetworkDescriptionNative { Layers = inputLayersHandle, NumLayers = (uint)inputLayerDescriptions.Length };
            var outputNetwork = new OptixNetworkDescriptionNative { Layers = outputLayersHandle, NumLayers = (uint)outputLayerDescriptions.Length };
            inputNetworkHandle = SafeHGlobal.AllocFrom(inputNetwork);
            outputNetworkHandle = SafeHGlobal.AllocFrom(outputNetwork);
        }

        /// <summary>
        /// Converts <paramref name="numNetworks"/> network instances from
        /// <paramref name="inputBuffer"/> into <paramref name="outputBuffer"/> using
        /// this plan's pre-marshaled descriptions - no per-call allocations.
        /// </summary>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="numNetworks">Number of network instances to convert.</param>
        /// <param name="inputBuffer">Device buffer holding the input network(s).</param>
        /// <param name="inputNetworkStrideInBytes">Byte stride between consecutive network instances in <paramref name="inputBuffer"/>; ignored if <paramref name="numNetworks"/> is 1.</param>
        /// <param name="outputBuffer">Device buffer to receive the converted network(s).</param>
        /// <param name="outputNetworkStrideInBytes">Byte stride between consecutive network instances in <paramref name="outputBuffer"/>; ignored if <paramref name="numNetworks"/> is 1.</param>
        public void Convert(
            AcceleratorStream stream,
            uint numNetworks,
            MemoryBuffer inputBuffer,
            ulong inputNetworkStrideInBytes,
            MemoryBuffer outputBuffer,
            ulong outputNetworkStrideInBytes)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (stream is not CudaStream cudaStream)
                throw new ArgumentOutOfRangeException(nameof(stream));
            if (inputBuffer == null)
                throw new ArgumentNullException(nameof(inputBuffer));
            if (outputBuffer == null)
                throw new ArgumentNullException(nameof(outputBuffer));

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

        /// <summary cref="DisposeBase.Dispose(bool)"/>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                inputLayersHandle?.Dispose();
                outputLayersHandle?.Dispose();
                inputNetworkHandle?.Dispose();
                outputNetworkHandle?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
