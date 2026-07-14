using ILGPU;
using ILGPU.Algorithms;

namespace Sample15
{
    /// <summary>
    /// Second half of the TAA pipeline, run as a plain ILGPU kernel (not an OptiX
    /// program) after RaygenProgram's OptiX launch has finished writing every pixel's
    /// raw sample. OptiX raygen threads have no ordering/synchronization guarantee
    /// across pixels in the same launch, so a pixel can't safely read a neighbor's
    /// *this-frame* color there (see RaygenProgram.cs's own note at its reprojection
    /// block) - by the time this kernel runs, RawColorBuffer is fully written for the
    /// whole frame, so a 3x3 neighborhood read around any pixel is safe.
    ///
    /// Deliberately has no neighborhood/variance color clamp. That's a standard
    /// real-time-TAA technique that assumes the "raw" per-frame color is
    /// near-ground-truth (true for a rasterizer, where raw variation is just AA
    /// jitter) and clamps stale/ghosted history back into this frame's local color
    /// range. Here the raw signal is a single noisy Monte Carlo path-tracer sample,
    /// so clamping a converging running average into one frame's noisy neighborhood
    /// would re-inject raw per-sample noise into the output every frame, capping
    /// achievable quality at roughly single-sample noise regardless of how many
    /// frames have been accumulated - tried and removed for exactly this reason.
    /// RaygenProgram's own depth-based disocclusion check (trustHistory) is the only
    /// defense against stale/incorrect history now.
    /// </summary>
    public static class TaaResolveKernel
    {
        // Sentinel for ReprojCoordBuffer meaning "don't reproject this pixel" - written
        // by RaygenProgram.cs whenever its own trustHistory is false (background,
        // disocclusion, off-screen, specular, or no previous frame yet). A valid
        // reprojected coordinate is always >= -0.5 (RaygenProgram's own off-screen
        // rejection check), so anything far below that is unambiguous.
        public const float NoHistorySentinel = -1e8f;

        // Same bilinear/nearest sampling as RaygenProgram.cs's own helpers, just against
        // ArrayView (this is a plain ILGPU kernel, not an OptiX device program working
        // off unsafe pointers/LaunchParams) - see RaygenProgram.cs's SampleBilinearColor
        // doc comment for why bilinear (not nearest) matters for the color history.
        static Vec4 SampleBilinearColor(ArrayView<Vec4> buffer, int width, int height, float px, float py)
        {
            float cx = XMath.Clamp(px, 0f, width - 1f);
            float cy = XMath.Clamp(py, 0f, height - 1f);
            int x0 = (int)cx, y0 = (int)cy;
            int x1 = XMath.Min(x0 + 1, width - 1);
            int y1 = XMath.Min(y0 + 1, height - 1);
            float fx = cx - x0, fy = cy - y0;

            Vec4 c00 = buffer[x0 + (y0 * width)];
            Vec4 c10 = buffer[x1 + (y0 * width)];
            Vec4 c01 = buffer[x0 + (y1 * width)];
            Vec4 c11 = buffer[x1 + (y1 * width)];

            float r = ((c00.x * (1f - fx)) + (c10.x * fx)) * (1f - fy) + (((c01.x * (1f - fx)) + (c11.x * fx)) * fy);
            float g = ((c00.y * (1f - fx)) + (c10.y * fx)) * (1f - fy) + (((c01.y * (1f - fx)) + (c11.y * fx)) * fy);
            float b = ((c00.z * (1f - fx)) + (c10.z * fx)) * (1f - fy) + (((c01.z * (1f - fx)) + (c11.z * fx)) * fy);
            float a = ((c00.w * (1f - fx)) + (c10.w * fx)) * (1f - fy) + (((c01.w * (1f - fx)) + (c11.w * fx)) * fy);
            return new Vec4(r, g, b, a);
        }

        static float SampleNearestScalar(ArrayView<float> buffer, int width, int height, float px, float py)
        {
            int x = XMath.Clamp((int)XMath.Round(px), 0, width - 1);
            int y = XMath.Clamp((int)XMath.Round(py), 0, height - 1);
            return buffer[x + (y * width)];
        }

        public static void resolve(
            Index1D index,
            int width,
            int height,
            int maxHistoryFrames,
            float historyDecayHalfLifeSeconds,
            float deltaTimeSeconds,
            ArrayView<Vec4> rawColor,
            ArrayView<Vec2> reprojCoord,
            ArrayView<Vec4> prevColor,
            ArrayView<float> prevAccumCount,
            ArrayView<Vec4> outColor,
            ArrayView<float> outAccumCount)
        {
            Vec4 raw = rawColor[index];
            Vec3 pixelColor = new Vec3(raw.x, raw.y, raw.z);
            float currentDepth = raw.w;

            // Sentinel check - see this class's own NoHistorySentinel doc comment.
            // Comparing against the sentinel's own half rather than 0/a small epsilon
            // keeps this correct however far below -0.5 the sentinel is defined as.
            Vec2 coord = reprojCoord[index];
            bool trustHistory = coord.x > (NoHistorySentinel * 0.5f);

            Vec3 blendedColor = pixelColor;
            float newCount = 1f;
            if (trustHistory)
            {
                Vec4 prevSample = SampleBilinearColor(prevColor, width, height, coord.x, coord.y);
                float prevCount = SampleNearestScalar(prevAccumCount, width, height, coord.x, coord.y);
                Vec3 prevSampleColor = new Vec3(prevSample.x, prevSample.y, prevSample.z);

                // Real-time decay (see SampleRenderer.HistoryDecayHalfLifeSeconds's own
                // doc comment) - ages the effective count down every frame on a
                // wall-clock schedule, so even a static/well-reprojected pixel keeps
                // slowly forgetting old contributions instead of freezing at a fixed
                // weight once it hits maxHistoryFrames.
                if (historyDecayHalfLifeSeconds > 0f)
                    prevCount *= XMath.Pow(0.5f, deltaTimeSeconds / historyDecayHalfLifeSeconds);

                newCount = XMath.Min(prevCount + 1f, (float)maxHistoryFrames);
                blendedColor = prevSampleColor + ((pixelColor - prevSampleColor) / newCount);
            }

            outColor[index] = new Vec4(blendedColor.x, blendedColor.y, blendedColor.z, currentDepth);
            outAccumCount[index] = newCount;
        }
    }
}
