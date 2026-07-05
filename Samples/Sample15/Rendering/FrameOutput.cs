using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.Denoising;
using ILGPU.OptiX.Interop;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Diagnostics;

namespace Sample15
{
    /// <summary>
    /// Owns the framebuffers (HDR color + accum-count history + denoiser AOV/flow
    /// guides + denoised) and the interop display buffer (see
    /// CudaGlInteropDisplayBuffer), the OptiX temporal denoiser, and the tonemap
    /// kernel - everything between "the trace wrote HDR pixels" and "a frame is
    /// drawable as a GL texture", with no CPU round-trip anywhere (unlike Sample13's
    /// FrameOutput, which reads the display buffer back to a CPU byte[] for WPF - see
    /// docs/SAMPLE14_PLAN.md's FrameOutput/SampleRenderer changes section).
    /// </summary>
    public sealed class FrameOutput : IDisposable
    {
        readonly GpuContext gpu;
        readonly Action<Index1D, int, int, float, int, ArrayView<Vec4>, ArrayView<byte>> tonemapAndFlip;
        readonly Stopwatch stepStopwatch = new Stopwatch();

        int width;
        int height;

        // Ping-ponged raw HDR accumulation - RaygenProgram.cs reprojects the *previous*
        // frame's color/count (via Flow) and blends into the *current* frame's slot, so
        // "current" and "previous" swap every frame (see SwapColorHistory). Replaces
        // the old single-buffer, FrameID-keyed running mean, which could only blend by
        // pixel index and so had to hard-reset on any camera move (see
        // SampleRenderer.cs's MaxHistoryFrames).
        readonly MemoryBuffer1D<Vec4, Stride1D.Dense>[] hdrColorBuffers = new MemoryBuffer1D<Vec4, Stride1D.Dense>[2];
        readonly MemoryBuffer1D<float, Stride1D.Dense>[] accumCountBuffers = new MemoryBuffer1D<float, Stride1D.Dense>[2];
        int currentColorIndex;

        public MemoryBuffer1D<Vec4, Stride1D.Dense> CurrentColorBuffer => hdrColorBuffers[currentColorIndex];
        public MemoryBuffer1D<Vec4, Stride1D.Dense> PreviousColorBuffer => hdrColorBuffers[1 - currentColorIndex];
        public MemoryBuffer1D<float, Stride1D.Dense> CurrentAccumCountBuffer => accumCountBuffers[currentColorIndex];
        public MemoryBuffer1D<float, Stride1D.Dense> PreviousAccumCountBuffer => accumCountBuffers[1 - currentColorIndex];

        public MemoryBuffer1D<Vec4, Stride1D.Dense> AlbedoBuffer { get; private set; }
        public MemoryBuffer1D<Vec4, Stride1D.Dense> NormalBuffer { get; private set; }

        // Per-pixel screen-space motion vector + trust mask guiding the temporal
        // denoiser (see LaunchParams.FlowBuffer's own doc comment) - written by
        // RaygenProgram.cs every frame, read here only.
        public MemoryBuffer1D<Vec2, Stride1D.Dense> FlowBuffer { get; private set; }
        public MemoryBuffer1D<float, Stride1D.Dense> FlowTrustworthinessBuffer { get; private set; }

        // Ping-ponged denoised output - this frame's Output becomes next frame's
        // PreviousOutput (OPTIX_DENOISER_MODEL_KIND_TEMPORAL requires the previous
        // frame's own *denoised* result, not the raw HDR color, as history). This is a
        // separate ping-pong from hdrColorBuffers above - it only swaps while the
        // denoiser is actually running (see DenoiseAndTonemap).
        readonly MemoryBuffer1D<Vec4, Stride1D.Dense>[] denoisedColorBuffers = new MemoryBuffer1D<Vec4, Stride1D.Dense>[2];
        int currentDenoisedIndex;

        MemoryBuffer1D<byte, Stride1D.Dense> denoiserIntensity;
        BuiltDenoiser builtDenoiser;

        CudaGlInteropDisplayBuffer interopBuffer;
        public int GlTextureHandle => interopBuffer.GlTextureHandle;

        public FrameOutput(GpuContext gpu)
        {
            this.gpu = gpu;

            // Denoiser will be created in Resize() when dimensions are known
            tonemapAndFlip = gpu.Accelerator.LoadAutoGroupedStreamKernel<Index1D, int, int, float, int, ArrayView<Vec4>, ArrayView<byte>>(TonemapKernel.tonemapAndFlip);
        }

        public unsafe void Resize(int width, int height)
        {
            this.width = width;
            this.height = height;
            var accelerator = gpu.Accelerator;

            hdrColorBuffers[0]?.Dispose();
            hdrColorBuffers[1]?.Dispose();
            hdrColorBuffers[0] = accelerator.Allocate1D<Vec4>(width * height);
            hdrColorBuffers[0].MemSetToZero();
            hdrColorBuffers[1] = accelerator.Allocate1D<Vec4>(width * height);
            hdrColorBuffers[1].MemSetToZero();
            accumCountBuffers[0]?.Dispose();
            accumCountBuffers[1]?.Dispose();
            accumCountBuffers[0] = accelerator.Allocate1D<float>(width * height);
            accumCountBuffers[0].MemSetToZero();
            accumCountBuffers[1] = accelerator.Allocate1D<float>(width * height);
            accumCountBuffers[1].MemSetToZero();
            currentColorIndex = 0;

            AlbedoBuffer = accelerator.Allocate1D<Vec4>(width * height);
            AlbedoBuffer.MemSetToZero();
            NormalBuffer = accelerator.Allocate1D<Vec4>(width * height);
            NormalBuffer.MemSetToZero();
            FlowBuffer = accelerator.Allocate1D<Vec2>(width * height);
            FlowBuffer.MemSetToZero();
            FlowTrustworthinessBuffer = accelerator.Allocate1D<float>(width * height);
            FlowTrustworthinessBuffer.MemSetToZero();

            denoisedColorBuffers[0]?.Dispose();
            denoisedColorBuffers[1]?.Dispose();
            denoisedColorBuffers[0] = accelerator.Allocate1D<Vec4>(width * height);
            denoisedColorBuffers[0].MemSetToZero();
            denoisedColorBuffers[1] = accelerator.Allocate1D<Vec4>(width * height);
            denoisedColorBuffers[1].MemSetToZero();
            currentDenoisedIndex = 0;

            interopBuffer?.Dispose();
            interopBuffer = new CudaGlInteropDisplayBuffer(width, height, accelerator);

            denoiserIntensity?.Dispose();
            denoiserIntensity = accelerator.Allocate1D<byte>(sizeof(float));

            builtDenoiser?.Dispose();
            builtDenoiser = new OptixDenoiserBuilder()
                .WithDeviceContext(gpu.DeviceContext)
                .WithAccelerator(accelerator)
                .WithImageDimensions((uint)width, (uint)height)
                .WithDenoiserOptions(new OptixDenoiserOptions { GuideAlbedo = 1, GuideNormal = 1 })
                // TEMPORAL, not the builder's LDR default - DenoiseAndTonemap() below
                // feeds GuideLayer.Flow/FlowTrustworthiness and Layer.PreviousOutput/
                // TemporalModeUsePreviousLayers, which only have meaning to OptiX under
                // the TEMPORAL model kind (see docs/API_BUILDER_PLAN.md's 2026-07-04 note).
                .WithModelKind(OptixDenoiserModelKind.OPTIX_DENOISER_MODEL_KIND_TEMPORAL)
                .Build();
        }

        // Restarts progressive accumulation (scene switch / resize) - clears both
        // slots of both ping-ponged history pairs, since the previous scene/resolution's
        // content doesn't apply to the new one (SampleRenderer.cs also sets
        // PrevFrameValid=0 alongside this, so nothing would have sampled the stale
        // contents anyway - this is defensive/for visual cleanliness).
        public void ResetAccumulation()
        {
            hdrColorBuffers[0].MemSetToZero();
            hdrColorBuffers[1].MemSetToZero();
            accumCountBuffers[0].MemSetToZero();
            accumCountBuffers[1].MemSetToZero();
            currentColorIndex = 0;
        }

        // Flips which ping-ponged hdrColorBuffers/accumCountBuffers slot is "current" -
        // call once per frame, after the raygen launch has finished writing to
        // CurrentColorBuffer/CurrentAccumCountBuffer (so next frame's launch reads this
        // frame's result as its Previous*). Unconditional (unlike the denoised ping-pong
        // below), since raw accumulation happens whether or not the AI denoiser runs.
        public void SwapColorHistory() => currentColorIndex = 1 - currentColorIndex;

        // Runs the (optional) denoiser and the tonemap kernel, writing straight into
        // the mapped CUDA-GL interop buffer (no CPU byte[] anywhere) then blitting it
        // to the display texture. frameId is the post-increment FrameID (frames since
        // scene load) used for the denoiser's own blend factor. hasValidPreviousFrame
        // gates OPTIX_DENOISER_MODEL_KIND_TEMPORAL's own PreviousOutput/
        // TemporalModeUsePreviousLayers (false right after a scene switch or resize,
        // since the ping-ponged denoised buffer from before doesn't apply - see
        // SampleRenderer.cs).
        public unsafe void DenoiseAndTonemap(bool denoiserOn, bool accumulate, int frameId, bool hasValidPreviousFrame, float exposureStops, int tonemapOperator, out double denoiseMs, out double tonemapMs)
        {
            OptixImage2D MakeImage(MemoryBuffer1D<Vec4, Stride1D.Dense> source) => new OptixImage2D
            {
                Data = (ulong)source.NativePtr.ToInt64(),
                Width = (uint)width,
                Height = (uint)height,
                RowStrideInBytes = (uint)(width * sizeof(Vec4)),
                PixelStrideInBytes = (uint)sizeof(Vec4),
                Format = OptixPixelFormat.OPTIX_PIXEL_FORMAT_FLOAT4
            };

            MemoryBuffer1D<Vec4, Stride1D.Dense> tonemapSource;
            denoiseMs = 0;
            if (denoiserOn && builtDenoiser != null)
            {
                var colorImage = MakeImage(CurrentColorBuffer);
                var cudaStream = (CudaStream)gpu.Accelerator.DefaultStream;

                stepStopwatch.Restart();
                builtDenoiser.Denoiser.ComputeIntensity(
                    cudaStream.StreamPtr,
                    colorImage,
                    denoiserIntensity.NativePtr,
                    builtDenoiser.ScratchBuffer.NativePtr,
                    (ulong)builtDenoiser.ScratchBuffer.LengthInBytes);

                // BlendFactor: 0 = fully denoised, 1 = unmodified raw input. Now that
                // FrameID no longer resets on camera move (RaygenProgram.cs's own
                // reprojected accumulation is the real convergence mechanism), this
                // simply settles toward "mostly denoised" over the session instead of
                // spiking back to fully-raw on every camera move like it used to.
                float blendFactor = accumulate ? System.Math.Min(0.5f, 1f / frameId) : 0f;
                var denoiserParams = new OptixDenoiserParams
                {
                    HdrIntensity = (ulong)denoiserIntensity.NativePtr.ToInt64(),
                    HdrAverageColor = 0,
                    BlendFactor = blendFactor,
                    TemporalModeUsePreviousLayers = hasValidPreviousFrame ? 1u : 0u
                };

                var flowImage = new OptixImage2D
                {
                    Data = (ulong)FlowBuffer.NativePtr.ToInt64(),
                    Width = (uint)width,
                    Height = (uint)height,
                    RowStrideInBytes = (uint)(width * sizeof(Vec2)),
                    PixelStrideInBytes = (uint)sizeof(Vec2),
                    Format = OptixPixelFormat.OPTIX_PIXEL_FORMAT_FLOAT2
                };
                var flowTrustImage = new OptixImage2D
                {
                    Data = (ulong)FlowTrustworthinessBuffer.NativePtr.ToInt64(),
                    Width = (uint)width,
                    Height = (uint)height,
                    RowStrideInBytes = (uint)(width * sizeof(float)),
                    PixelStrideInBytes = (uint)sizeof(float),
                    Format = OptixPixelFormat.OPTIX_PIXEL_FORMAT_FLOAT1
                };
                var guideLayer = new OptixDenoiserGuideLayer
                {
                    Albedo = MakeImage(AlbedoBuffer),
                    Normal = MakeImage(NormalBuffer),
                    Flow = flowImage,
                    FlowTrustworthiness = flowTrustImage
                };

                int outputIndex = currentDenoisedIndex;
                int previousIndex = 1 - currentDenoisedIndex;
                var layer = new OptixDenoiserLayer
                {
                    Input = colorImage,
                    Output = MakeImage(denoisedColorBuffers[outputIndex]),
                    PreviousOutput = MakeImage(denoisedColorBuffers[previousIndex])
                };

                builtDenoiser.Denoiser.Invoke(
                    cudaStream.StreamPtr,
                    denoiserParams,
                    builtDenoiser.StateBuffer.NativePtr,
                    (ulong)builtDenoiser.StateBuffer.LengthInBytes,
                    guideLayer,
                    new[] { layer },
                    builtDenoiser.ScratchBuffer.NativePtr,
                    (ulong)builtDenoiser.ScratchBuffer.LengthInBytes);
                gpu.Accelerator.Synchronize();
                denoiseMs = stepStopwatch.Elapsed.TotalMilliseconds;
                tonemapSource = denoisedColorBuffers[outputIndex];
                currentDenoisedIndex = previousIndex;
            }
            else
            {
                tonemapSource = CurrentColorBuffer;
            }

            stepStopwatch.Restart();
            var stream = (CudaStream)gpu.Accelerator.DefaultStream;
            interopBuffer.MapCuda(stream);
            var dest = interopBuffer.GetCudaArrayView();
            tonemapAndFlip(new Index1D(width * height), width, height, exposureStops, tonemapOperator, tonemapSource.View, dest);
            gpu.Accelerator.Synchronize();
            // Must not unmap while GPU work touching the resource is still in flight -
            // the Synchronize() above already guarantees that, so unmap can follow
            // immediately (see docs/SAMPLE14_PLAN.md's note on this ordering).
            interopBuffer.UnmapCuda(stream);
            tonemapMs = stepStopwatch.Elapsed.TotalMilliseconds;
        }

        // GPU-internal PBO -> texture blit (no CPU copy) - call once per frame after
        // DenoiseAndTonemap, before drawing the fullscreen quad.
        public void BlitToTexture() => interopBuffer.BlitToTexture();

        public void Dispose()
        {
            denoiserIntensity?.Dispose();
            builtDenoiser?.Dispose();

            interopBuffer?.Dispose();
            denoisedColorBuffers[0]?.Dispose();
            denoisedColorBuffers[1]?.Dispose();
            FlowTrustworthinessBuffer.Dispose();
            FlowBuffer.Dispose();
            NormalBuffer.Dispose();
            AlbedoBuffer.Dispose();
            accumCountBuffers[0]?.Dispose();
            accumCountBuffers[1]?.Dispose();
            hdrColorBuffers[0]?.Dispose();
            hdrColorBuffers[1]?.Dispose();
        }
    }
}
