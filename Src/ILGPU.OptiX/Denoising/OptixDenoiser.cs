// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixDenoiser.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX.Util;
using ILGPU.Util;
using ILGPU.OptiX.Native;
using System;
using System.Runtime.InteropServices;

namespace ILGPU.OptiX.Denoising
{
    /// <summary>
    /// Wrapper over an OptiX denoiser. Create via
    /// <see cref="OptixDeviceContextExtensions.CreateDenoiser"/>.
    /// </summary>
    [CLSCompliant(false)]
    public sealed class OptixDenoiser : DisposeBase
    {
        #region Properties

        /// <summary>
        /// The native OptiX denoiser.
        /// </summary>
        public IntPtr DenoiserPtr { get; private set; }

        #endregion

        #region Fields

        // Persistent native scratch for Invoke()/ComputeIntensity()'s marshaled
        // struct arguments - overwritten in place every call instead of the
        // struct-taking OptixAPI overloads' per-call AllocHGlobal/FreeHGlobal
        // (SafeHGlobal.AllocFrom), since Invoke is typically called every frame in a
        // real-time denoiser loop. layersHandle is resized only if the layer count
        // itself changes (usually 1, but this stays correct if a caller varies it).
        SafeHGlobal? parametersHandle;
        SafeHGlobal? guideLayerHandle;
        SafeHGlobal? layersHandle;
        int layersHandleCount;
        SafeHGlobal? intensityImageHandle;

        #endregion

        #region Instance

        /// <summary>
        /// Constructs a new denoiser wrapper.
        /// </summary>
        /// <param name="denoiserPtr">The OptiX denoiser.</param>
        public OptixDenoiser(IntPtr denoiserPtr)
        {
            if (denoiserPtr == IntPtr.Zero)
                throw new ArgumentNullException(nameof(denoiserPtr));
            DenoiserPtr = denoiserPtr;
        }

        #endregion

        #region Methods

        /// <summary>
        /// Computes the memory resources (state/scratch buffer sizes) required by this
        /// denoiser for the given maximum input resolution.
        /// </summary>
        /// <param name="maximumInputWidth">The maximum input image width.</param>
        /// <param name="maximumInputHeight">The maximum input image height.</param>
        /// <returns>The required memory sizes.</returns>
        public OptixDenoiserSizes ComputeMemoryResources(uint maximumInputWidth, uint maximumInputHeight)
        {
            OptixException.ThrowIfFailed(
                OptixAPI.Current.DenoiserComputeMemoryResources(
                    DenoiserPtr,
                    maximumInputWidth,
                    maximumInputHeight,
                    out var sizes));
            return sizes;
        }

        /// <summary>
        /// Sets up the denoiser's internal state for a given input resolution. Must be
        /// called (again, if the resolution changed) before <see cref="Invoke"/>.
        /// </summary>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="inputWidth">The input image width.</param>
        /// <param name="inputHeight">The input image height.</param>
        /// <param name="state">The denoiser state buffer (sized per ComputeMemoryResources).</param>
        /// <param name="stateSizeInBytes">The denoiser state buffer size.</param>
        /// <param name="scratch">The denoiser scratch buffer (sized per ComputeMemoryResources).</param>
        /// <param name="scratchSizeInBytes">The denoiser scratch buffer size.</param>
        public void Setup(
            IntPtr stream,
            uint inputWidth,
            uint inputHeight,
            IntPtr state,
            ulong stateSizeInBytes,
            IntPtr scratch,
            ulong scratchSizeInBytes)
        {
            OptixException.ThrowIfFailed(
                OptixAPI.Current.DenoiserSetup(
                    DenoiserPtr,
                    stream,
                    inputWidth,
                    inputHeight,
                    state,
                    stateSizeInBytes,
                    scratch,
                    scratchSizeInBytes));
        }

        /// <summary>
        /// Runs the denoiser over the given input/output layers.
        /// </summary>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="parameters">The denoiser parameters (blend factor, autoexposure).</param>
        /// <param name="denoiserState">The denoiser state buffer, as set up via <see cref="Setup"/>.</param>
        /// <param name="denoiserStateSizeInBytes">The denoiser state buffer size.</param>
        /// <param name="guideLayer">The guide layer (albedo/normal images), may be default/empty.</param>
        /// <param name="layers">The input/output layer(s) to denoise.</param>
        /// <param name="scratch">The denoiser scratch buffer, as set up via <see cref="Setup"/>.</param>
        /// <param name="scratchSizeInBytes">The denoiser scratch buffer size.</param>
        /// <param name="inputOffsetX">
        /// X offset of the layers' input images within the full frame being denoised
        /// - 0 for whole-frame denoising, per-tile offsets for tiled denoising (see
        /// <see cref="OptixDenoiserTiling.InvokeTiled"/>).
        /// </param>
        /// <param name="inputOffsetY">Y offset - see <paramref name="inputOffsetX"/>.</param>
        public void Invoke(
            IntPtr stream,
            OptixDenoiserParams parameters,
            IntPtr denoiserState,
            ulong denoiserStateSizeInBytes,
            OptixDenoiserGuideLayer guideLayer,
            ReadOnlySpan<OptixDenoiserLayer> layers,
            IntPtr scratch,
            ulong scratchSizeInBytes,
            uint inputOffsetX = 0,
            uint inputOffsetY = 0)
        {
            parametersHandle ??= SafeHGlobal.Alloc<OptixDenoiserParams>();
            Marshal.StructureToPtr(parameters, parametersHandle.NativePtr, false);

            guideLayerHandle ??= SafeHGlobal.Alloc<OptixDenoiserGuideLayer>();
            Marshal.StructureToPtr(guideLayer, guideLayerHandle.NativePtr, false);

            if (layersHandle == null || layersHandleCount != layers.Length)
            {
                layersHandle?.Dispose();
                layersHandle = SafeHGlobal.Alloc<OptixDenoiserLayer>(layers.Length);
                layersHandleCount = layers.Length;
            }
            var layerSize = Marshal.SizeOf<OptixDenoiserLayer>();
            var nextLayerPtr = layersHandle.NativePtr;
            foreach (var layer in layers)
            {
                Marshal.StructureToPtr(layer, nextLayerPtr, false);
                nextLayerPtr += layerSize;
            }

            OptixException.ThrowIfFailed(
                OptixAPI.Current.DenoiserInvoke(
                    DenoiserPtr,
                    stream,
                    parametersHandle.NativePtr,
                    denoiserState,
                    denoiserStateSizeInBytes,
                    guideLayerHandle.NativePtr,
                    layersHandle.NativePtr,
                    (uint)layers.Length,
                    inputOffsetX,
                    inputOffsetY,
                    scratch,
                    scratchSizeInBytes));
        }

        /// <summary>
        /// Computes the average log intensity of an input image, for use as
        /// OptixDenoiserParams.HdrIntensity (autoexposure input for tiled rendering).
        /// </summary>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="outputIntensity">The output device buffer (a single float).</param>
        /// <param name="scratch">The denoiser scratch buffer.</param>
        /// <param name="scratchSizeInBytes">The denoiser scratch buffer size.</param>
        public void ComputeIntensity(
            IntPtr stream,
            OptixImage2D inputImage,
            IntPtr outputIntensity,
            IntPtr scratch,
            ulong scratchSizeInBytes)
        {
            intensityImageHandle ??= SafeHGlobal.Alloc<OptixImage2D>();
            Marshal.StructureToPtr(inputImage, intensityImageHandle.NativePtr, false);

            OptixException.ThrowIfFailed(
                OptixAPI.Current.DenoiserComputeIntensity(
                    DenoiserPtr,
                    stream,
                    intensityImageHandle.NativePtr,
                    outputIntensity,
                    scratch,
                    scratchSizeInBytes));
        }

        /// <summary>
        /// Computes the average log color of an input image, for use as
        /// OptixDenoiserParams.HdrAverageColor (AOV model kind only).
        /// </summary>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="outputAverageColor">The output device buffer (three floats).</param>
        /// <param name="scratch">The denoiser scratch buffer.</param>
        /// <param name="scratchSizeInBytes">The denoiser scratch buffer size.</param>
        public void ComputeAverageColor(
            IntPtr stream,
            OptixImage2D inputImage,
            IntPtr outputAverageColor,
            IntPtr scratch,
            ulong scratchSizeInBytes)
        {
            OptixException.ThrowIfFailed(
                OptixAPI.Current.DenoiserComputeAverageColor(
                    DenoiserPtr,
                    stream,
                    inputImage,
                    outputAverageColor,
                    scratch,
                    scratchSizeInBytes));
        }

        #endregion

        #region IDisposable

        /// <summary cref="DisposeBase.Dispose(bool)"/>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (DenoiserPtr != IntPtr.Zero)
                {
                    OptixAPI.Current.DenoiserDestroy(DenoiserPtr);
                    DenoiserPtr = IntPtr.Zero;
                }

                parametersHandle?.Dispose();
                guideLayerHandle?.Dispose();
                layersHandle?.Dispose();
                intensityImageHandle?.Dispose();
            }
            base.Dispose(disposing);
        }

        #endregion
    }
}
