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
    /// Owns the framebuffers (HDR color + accum-count history + denoiser AOV guides +
    /// denoised) and the interop display buffer (see CudaGlInteropDisplayBuffer), the
    /// OptiX AI denoiser, and the tonemap kernel - everything between "the trace wrote
    /// HDR pixels" and "a frame is drawable as a GL texture", with no CPU round-trip
    /// anywhere (unlike Sample13's FrameOutput, which reads the display buffer back to
    /// a CPU byte[] for WPF - see docs/SAMPLE14_PLAN.md's FrameOutput/SampleRenderer
    /// changes section). Runs in HDR (non-temporal) model kind, not TEMPORAL - temporal
    /// accumulation is already handled by RaygenProgram.cs's own per-pixel reprojected
    /// history, so the denoiser's job is purely spatial cleanup of whatever noise
    /// remains at the current accumulation level; running a second, independent
    /// temporal integrator on top (with its own history and its own convergence rate)
    /// risked the two disagreeing on disocclusions instead of adding real quality.
    /// </summary>
    public sealed class FrameOutput : IDisposable
    {
        readonly GpuContext gpu;
        readonly Action<Index1D, int, int, float, int, ArrayView<Vec4>, ArrayView<byte>> tonemapAndFlip;
        readonly Action<Index1D, int, int, int, float, float, ArrayView<Vec4>, ArrayView<Vec2>, ArrayView<Vec4>, ArrayView<float>, ArrayView<Vec4>, ArrayView<float>> resolveTaa;
        readonly Stopwatch stepStopwatch = new Stopwatch();

        int width;
        int height;

        // Ping-ponged raw HDR accumulation - TaaResolveKernel.cs reprojects the
        // *previous* frame's color/count and blends into the *current* frame's slot, so
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

        // This frame's raw (not yet history-blended) output, written by RaygenProgram.cs
        // and consumed only by TaaResolveKernel.cs's own neighborhood clamp/blend pass
        // (see LaunchParams.RawColorBuffer's own doc comment) - scratch, not
        // ping-ponged, since nothing needs last frame's copy once resolved.
        public MemoryBuffer1D<Vec4, Stride1D.Dense> RawColorBuffer { get; private set; }
        public MemoryBuffer1D<Vec2, Stride1D.Dense> ReprojCoordBuffer { get; private set; }

        public MemoryBuffer1D<Vec4, Stride1D.Dense> AlbedoBuffer { get; private set; }
        public MemoryBuffer1D<Vec4, Stride1D.Dense> NormalBuffer { get; private set; }

        // The denoiser's own output buffer - HDR (non-temporal) model kind needs no
        // ping-pong/PreviousOutput history, unlike TEMPORAL (see this class's own doc
        // comment for why HDR was chosen over TEMPORAL).
        MemoryBuffer1D<Vec4, Stride1D.Dense> denoisedColorBuffer;

        MemoryBuffer1D<byte, Stride1D.Dense> denoiserIntensity;
        BuiltDenoiser builtDenoiser;

        CudaGlInteropDisplayBuffer interopBuffer;
        public int GlTextureHandle => interopBuffer.GlTextureHandle;

        public FrameOutput(GpuContext gpu)
        {
            this.gpu = gpu;

            // Denoiser will be created in Resize() when dimensions are known
            tonemapAndFlip = gpu.Accelerator.LoadAutoGroupedStreamKernel<Index1D, int, int, float, int, ArrayView<Vec4>, ArrayView<byte>>(TonemapKernel.tonemapAndFlip);
            resolveTaa = gpu.Accelerator.LoadAutoGroupedStreamKernel<Index1D, int, int, int, float, float, ArrayView<Vec4>, ArrayView<Vec2>, ArrayView<Vec4>, ArrayView<float>, ArrayView<Vec4>, ArrayView<float>>(TaaResolveKernel.resolve);
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

            RawColorBuffer?.Dispose();
            RawColorBuffer = accelerator.Allocate1D<Vec4>(width * height);
            RawColorBuffer.MemSetToZero();
            ReprojCoordBuffer?.Dispose();
            ReprojCoordBuffer = accelerator.Allocate1D<Vec2>(width * height);
            ReprojCoordBuffer.MemSetToZero();

            AlbedoBuffer = accelerator.Allocate1D<Vec4>(width * height);
            AlbedoBuffer.MemSetToZero();
            NormalBuffer = accelerator.Allocate1D<Vec4>(width * height);
            NormalBuffer.MemSetToZero();

            denoisedColorBuffer?.Dispose();
            denoisedColorBuffer = accelerator.Allocate1D<Vec4>(width * height);
            denoisedColorBuffer.MemSetToZero();

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
                // HDR (non-temporal), not TEMPORAL - see this class's own doc comment
                // for why RaygenProgram.cs's own reprojected accumulation already
                // covers temporal integration, so the denoiser doesn't need its own
                // Flow-guided PreviousOutput history on top of it.
                .WithModelKind(OptixDenoiserModelKind.OPTIX_DENOISER_MODEL_KIND_HDR)
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
            RawColorBuffer.MemSetToZero();
            ReprojCoordBuffer.MemSetToZero();
        }

        // Flips which ping-ponged hdrColorBuffers/accumCountBuffers slot is "current" -
        // call once per frame, after the raygen launch has finished writing to
        // CurrentColorBuffer/CurrentAccumCountBuffer (so next frame's launch reads this
        // frame's result as its Previous*). Unconditional (unlike the denoised ping-pong
        // below), since raw accumulation happens whether or not the AI denoiser runs.
        public void SwapColorHistory() => currentColorIndex = 1 - currentColorIndex;

        // Runs TaaResolveKernel.resolve - call once per frame after RaygenProgram's
        // OptiX launch has finished writing RawColorBuffer/ReprojCoordBuffer for every
        // pixel (see LaunchParams.RawColorBuffer's own doc comment for why the
        // neighborhood-clamped blend can't happen inside that OptiX launch itself), and
        // before DenoiseAndTonemap (which reads CurrentColorBuffer as its input).
        public void ResolveTaa(int maxHistoryFrames, float historyDecayHalfLifeSeconds, float deltaTimeSeconds)
        {
            resolveTaa(
                new Index1D(width * height),
                width,
                height,
                maxHistoryFrames,
                historyDecayHalfLifeSeconds,
                deltaTimeSeconds,
                RawColorBuffer.View,
                ReprojCoordBuffer.View,
                PreviousColorBuffer.View,
                PreviousAccumCountBuffer.View,
                CurrentColorBuffer.View,
                CurrentAccumCountBuffer.View);
        }

        // Runs the (optional) denoiser and the tonemap kernel, writing straight into
        // the mapped CUDA-GL interop buffer (no CPU byte[] anywhere) then blitting it
        // to the display texture. frameId is the post-increment FrameID (frames since
        // scene load) used for the denoiser's own blend factor.
        public unsafe void DenoiseAndTonemap(bool denoiserOn, bool accumulate, int frameId, float exposureStops, int tonemapOperator, out double denoiseMs, out double tonemapMs)
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
                    BlendFactor = blendFactor
                };

                var guideLayer = new OptixDenoiserGuideLayer
                {
                    Albedo = MakeImage(AlbedoBuffer),
                    Normal = MakeImage(NormalBuffer)
                };

                var layer = new OptixDenoiserLayer
                {
                    Input = colorImage,
                    Output = MakeImage(denoisedColorBuffer)
                };

                Span<OptixDenoiserLayer> layers = stackalloc OptixDenoiserLayer[1] { layer };
                builtDenoiser.Denoiser.Invoke(
                    cudaStream.StreamPtr,
                    denoiserParams,
                    builtDenoiser.StateBuffer.NativePtr,
                    (ulong)builtDenoiser.StateBuffer.LengthInBytes,
                    guideLayer,
                    layers,
                    builtDenoiser.ScratchBuffer.NativePtr,
                    (ulong)builtDenoiser.ScratchBuffer.LengthInBytes);
                gpu.Accelerator.Synchronize();
                denoiseMs = stepStopwatch.Elapsed.TotalMilliseconds;
                tonemapSource = denoisedColorBuffer;
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
            denoisedColorBuffer?.Dispose();
            NormalBuffer.Dispose();
            AlbedoBuffer.Dispose();
            ReprojCoordBuffer?.Dispose();
            RawColorBuffer?.Dispose();
            accumCountBuffers[0]?.Dispose();
            accumCountBuffers[1]?.Dispose();
            hdrColorBuffers[0]?.Dispose();
            hdrColorBuffers[1]?.Dispose();
        }
    }
}
