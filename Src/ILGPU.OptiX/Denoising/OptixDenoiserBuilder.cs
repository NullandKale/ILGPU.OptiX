// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixDenoiserBuilder.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Util;
using System;

namespace ILGPU.OptiX.Denoising
{
    /// <summary>
    /// Wraps the full denoiser lifecycle.
    /// </summary>
    public sealed class OptixDenoiserBuilder : DisposeBase
    {
        private OptixDeviceContext context;
        private CudaAccelerator accelerator;
        private uint width;
        private uint height;
        private OptixDenoiserOptions? denoiserOptions;
        private OptixDenoiserModelKind modelKind = OptixDenoiserModelKind.OPTIX_DENOISER_MODEL_KIND_LDR;

        /// <summary>
        /// Sets the device context.
        /// </summary>
        public OptixDenoiserBuilder WithDeviceContext(OptixDeviceContext ctx)
        {
            if (ctx == null)
                throw new ArgumentNullException(nameof(ctx), "Device context must not be null.");

            context = ctx;
            return this;
        }

        /// <summary>
        /// Sets the accelerator for GPU memory operations.
        /// </summary>
        public OptixDenoiserBuilder WithAccelerator(CudaAccelerator accel)
        {
            if (accel == null)
                throw new ArgumentNullException(nameof(accel), "Accelerator must not be null.");

            accelerator = accel;
            return this;
        }

        /// <summary>
        /// Sets the image dimensions.
        /// </summary>
        public OptixDenoiserBuilder WithImageDimensions(uint w, uint h)
        {
            if (w == 0 || h == 0)
                throw new ArgumentOutOfRangeException(
                    nameof(w), "Image dimensions must be > 0.");

            width = w;
            height = h;
            return this;
        }

        /// <summary>
        /// Sets the denoiser options.
        /// </summary>
        public OptixDenoiserBuilder WithDenoiserOptions(OptixDenoiserOptions opts)
        {
            denoiserOptions = opts;
            return this;
        }

        /// <summary>
        /// Sets the denoiser model kind. Defaults to LDR (simple color denoising with
        /// no extra required inputs). TEMPORAL requires motion-vector/flow guide-layer
        /// data and previous-frame layers that this builder does not supply - only
        /// switch to it if the caller wires those up itself.
        /// </summary>
        public OptixDenoiserBuilder WithModelKind(OptixDenoiserModelKind kind)
        {
            modelKind = kind;
            return this;
        }

        /// <summary>
        /// Builds the denoiser with allocated state and scratch buffers.
        /// </summary>
        public BuiltDenoiser Build(CudaStream stream = null)
        {
            if (context == null)
                throw new InvalidOperationException("Context must be set via WithContext().");
            if (accelerator == null)
                throw new InvalidOperationException("Accelerator must be set via WithAccelerator().");
            if (width == 0 || height == 0)
                throw new InvalidOperationException("Image dimensions must be set via WithImageDimensions().");

            MemoryBuffer stateBuffer = null;
            MemoryBuffer scratchBuffer = null;
            MemoryBuffer guidanceBuffer = null;

            try
            {
                // Create denoiser
                var opts = denoiserOptions ?? new OptixDenoiserOptions { GuideAlbedo = 0, GuideNormal = 0 };
                var denoiser = context.CreateDenoiser(modelKind, opts);

                // Compute memory resources needed
                var sizes = denoiser.ComputeMemoryResources(width, height);

                // Allocate state and scratch buffers
                stateBuffer = accelerator.Allocate1D<byte>((long)sizes.StateSizeInBytes);
                scratchBuffer = accelerator.Allocate1D<byte>((long)Math.Max(
                    sizes.WithoutOverlapScratchSizeInBytes,
                    sizes.WithOverlapScratchSizeInBytes));

                // Get the stream for setup
                CudaStream cudaStream;
                if (stream != null)
                {
                    if (stream is not CudaStream cs)
                        throw new ArgumentException("Stream must be a CudaStream.", nameof(stream));
                    cudaStream = cs;
                }
                else
                {
                    cudaStream = accelerator.DefaultStream as CudaStream;
                    if (cudaStream == null)
                        throw new InvalidOperationException("Failed to get CUDA stream from accelerator.");
                }

                // Call Setup to initialize the denoiser
                denoiser.Setup(
                    cudaStream.StreamPtr,
                    width,
                    height,
                    stateBuffer.NativePtr,
                    (ulong)stateBuffer.LengthInBytes,
                    scratchBuffer.NativePtr,
                    (ulong)scratchBuffer.LengthInBytes);

                // Synchronize if stream is null (synchronous mode)
                if (stream == null)
                {
                    accelerator.Synchronize();
                }

                var built = new BuiltDenoiser(denoiser, stateBuffer, scratchBuffer, guidanceBuffer, accelerator, stream);
                return built;
            }
            catch
            {
                stateBuffer?.Dispose();
                scratchBuffer?.Dispose();
                guidanceBuffer?.Dispose();
                throw;
            }
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
        }
    }

    /// <summary>
    /// Result of denoiser building, ready to use for denoising operations.
    /// </summary>
    public sealed class BuiltDenoiser : IDisposable
    {
        private bool disposed;

        public OptixDenoiser Denoiser { get; }
        public MemoryBuffer StateBuffer { get; }
        public MemoryBuffer ScratchBuffer { get; }
        public MemoryBuffer GuidanceBuffer { get; }
        private readonly CudaAccelerator accelerator;

        internal BuiltDenoiser(OptixDenoiser denoiser, MemoryBuffer stateBuffer,
            MemoryBuffer scratchBuffer, MemoryBuffer guidanceBuffer,
            CudaAccelerator accelerator, CudaStream stream)
        {
            Denoiser = denoiser;
            StateBuffer = stateBuffer;
            ScratchBuffer = scratchBuffer;
            GuidanceBuffer = guidanceBuffer;
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
                GuidanceBuffer?.Dispose();
                ScratchBuffer?.Dispose();
                StateBuffer?.Dispose();
                Denoiser?.Dispose();
                GC.SuppressFinalize(this);
            }
        }
    }
}
