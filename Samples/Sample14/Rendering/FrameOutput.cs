using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.Interop;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Diagnostics;

namespace Sample14
{
    /// <summary>
    /// Owns the framebuffers (HDR color + denoiser AOV guides + denoised) and the
    /// interop display buffer (see CudaGlInteropDisplayBuffer), the OptiX denoiser, and
    /// the tonemap kernel - everything between "the trace wrote HDR pixels" and "a
    /// frame is drawable as a GL texture", with no CPU round-trip anywhere (unlike
    /// Sample13's FrameOutput, which reads the display buffer back to a CPU byte[] for
    /// WPF - see docs/SAMPLE14_PLAN.md's FrameOutput/SampleRenderer changes section).
    /// </summary>
    public sealed class FrameOutput : IDisposable
    {
        readonly GpuContext gpu;
        readonly OptixDenoiser denoiser;
        readonly Action<Index1D, int, int, ArrayView<Vec4>, ArrayView<byte>> tonemapAndFlip;
        readonly Stopwatch stepStopwatch = new Stopwatch();

        int width;
        int height;

        public MemoryBuffer1D<Vec4, Stride1D.Dense> HdrColorBuffer { get; private set; }
        public MemoryBuffer1D<Vec4, Stride1D.Dense> AlbedoBuffer { get; private set; }
        public MemoryBuffer1D<Vec4, Stride1D.Dense> NormalBuffer { get; private set; }
        MemoryBuffer1D<Vec4, Stride1D.Dense> denoisedColorBuffer;
        MemoryBuffer1D<byte, Stride1D.Dense> denoiserIntensity;
        MemoryBuffer1D<byte, Stride1D.Dense> denoiserState;
        MemoryBuffer1D<byte, Stride1D.Dense> denoiserScratch;

        CudaGlInteropDisplayBuffer interopBuffer;
        public int GlTextureHandle => interopBuffer.GlTextureHandle;

        public FrameOutput(GpuContext gpu)
        {
            this.gpu = gpu;

            // Guide-layer denoiser (matches Sample12/Sample13's OPTIX_DENOISER_MODEL_KIND_LDR +
            // GuideAlbedo/GuideNormal setup) - one denoiser instance works across every
            // scene switch since it only depends on width/height (set up in Resize),
            // not on scene content.
            denoiser = gpu.DeviceContext.CreateDenoiser(
                OptixDenoiserModelKind.OPTIX_DENOISER_MODEL_KIND_LDR,
                new OptixDenoiserOptions { GuideAlbedo = 1, GuideNormal = 1 });

            tonemapAndFlip = gpu.Accelerator.LoadAutoGroupedStreamKernel<Index1D, int, int, ArrayView<Vec4>, ArrayView<byte>>(TonemapKernel.tonemapAndFlip);
        }

        public unsafe void Resize(int width, int height)
        {
            this.width = width;
            this.height = height;
            var accelerator = gpu.Accelerator;

            HdrColorBuffer = accelerator.Allocate1D<Vec4>(width * height);
            HdrColorBuffer.MemSetToZero();
            AlbedoBuffer = accelerator.Allocate1D<Vec4>(width * height);
            AlbedoBuffer.MemSetToZero();
            NormalBuffer = accelerator.Allocate1D<Vec4>(width * height);
            NormalBuffer.MemSetToZero();
            denoisedColorBuffer = accelerator.Allocate1D<Vec4>(width * height);
            denoisedColorBuffer.MemSetToZero();

            interopBuffer?.Dispose();
            interopBuffer = new CudaGlInteropDisplayBuffer(width, height, accelerator);

            denoiserIntensity?.Dispose();
            denoiserIntensity = accelerator.Allocate1D<byte>(sizeof(float));

            denoiserState?.Dispose();
            denoiserScratch?.Dispose();
            var denoiserSizes = denoiser.ComputeMemoryResources((uint)width, (uint)height);
            denoiserState = accelerator.Allocate1D<byte>((long)denoiserSizes.StateSizeInBytes);
            ulong denoiserScratchSize = Math.Max(denoiserSizes.WithOverlapScratchSizeInBytes, denoiserSizes.WithoutOverlapScratchSizeInBytes);
            denoiserScratch = accelerator.Allocate1D<byte>((long)denoiserScratchSize);
            denoiser.Setup(
                gpu.DefaultStreamPtr,
                (uint)width,
                (uint)height,
                denoiserState.NativePtr,
                (ulong)denoiserState.LengthInBytes,
                denoiserScratch.NativePtr,
                (ulong)denoiserScratch.LengthInBytes);
        }

        // Restarts progressive accumulation (scene switch / camera move).
        public void ResetAccumulation() => HdrColorBuffer.MemSetToZero();

        // Runs the (optional) denoiser and the tonemap kernel, writing straight into
        // the mapped CUDA-GL interop buffer (no CPU byte[] anywhere) then blitting it
        // to the display texture. frameId is the post-increment FrameID (number of
        // accumulated frames) used for the denoiser's accumulation blend factor.
        public unsafe void DenoiseAndTonemap(bool denoiserOn, bool accumulate, int frameId, out double denoiseMs, out double tonemapMs)
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
            if (denoiserOn)
            {
                var colorImage = MakeImage(HdrColorBuffer);

                stepStopwatch.Restart();
                denoiser.ComputeIntensity(
                    gpu.DefaultStreamPtr,
                    colorImage,
                    denoiserIntensity.NativePtr,
                    denoiserScratch.NativePtr,
                    (ulong)denoiserScratch.LengthInBytes);

                var denoiserParams = new OptixDenoiserParams
                {
                    HdrIntensity = (ulong)denoiserIntensity.NativePtr.ToInt64(),
                    HdrAverageColor = 0,
                    BlendFactor = accumulate ? 1f / frameId : 0f,
                    TemporalModeUsePreviousLayers = 0
                };
                var guideLayer = new OptixDenoiserGuideLayer { Albedo = MakeImage(AlbedoBuffer), Normal = MakeImage(NormalBuffer) };
                var layer = new OptixDenoiserLayer { Input = colorImage, Output = MakeImage(denoisedColorBuffer) };

                denoiser.Invoke(
                    gpu.DefaultStreamPtr,
                    denoiserParams,
                    denoiserState.NativePtr,
                    (ulong)denoiserState.LengthInBytes,
                    guideLayer,
                    new[] { layer },
                    denoiserScratch.NativePtr,
                    (ulong)denoiserScratch.LengthInBytes);
                gpu.Accelerator.Synchronize();
                denoiseMs = stepStopwatch.Elapsed.TotalMilliseconds;
                tonemapSource = denoisedColorBuffer;
            }
            else
            {
                tonemapSource = HdrColorBuffer;
            }

            stepStopwatch.Restart();
            var stream = (CudaStream)gpu.Accelerator.DefaultStream;
            interopBuffer.MapCuda(stream);
            var dest = interopBuffer.GetCudaArrayView();
            tonemapAndFlip(new Index1D(width * height), width, height, tonemapSource.View, dest);
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
            denoiserScratch?.Dispose();
            denoiserState?.Dispose();
            denoiser.Dispose();

            interopBuffer?.Dispose();
            denoisedColorBuffer.Dispose();
            NormalBuffer.Dispose();
            AlbedoBuffer.Dispose();
            HdrColorBuffer.Dispose();
        }
    }
}
