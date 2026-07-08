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
    /// Clamps the reprojected history sample to the min/max color this frame's own
    /// nearby pixels actually produced before blending it in - the standard
    /// "neighborhood/variance clamping" technique production TAA implementations use to
    /// catch ghosting/fireflies that survive RaygenProgram's own depth-based
    /// disocclusion check (which only catches gross surface changes, not fine detail,
    /// thin geometry, or a single stray bright sample).
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
            int ix = index % width;
            int iy = index / width;

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

                // Neighborhood color clamp (3x3, clamped to buffer edges) - bounds the
                // reprojected history sample to what this frame's own nearby pixels
                // actually produced, so a history value that's ghosting or fireflying
                // past that range gets pulled back into range instead of blended in
                // wholesale. See this class's own doc comment.
                int x0 = XMath.Max(ix - 1, 0), x1 = XMath.Min(ix + 1, width - 1);
                int y0 = XMath.Max(iy - 1, 0), y1 = XMath.Min(iy + 1, height - 1);
                Vec3 neighborMin = pixelColor;
                Vec3 neighborMax = pixelColor;
                for (int ny = y0; ny <= y1; ny++)
                {
                    for (int nx = x0; nx <= x1; nx++)
                    {
                        Vec4 n = rawColor[nx + (ny * width)];
                        neighborMin = new Vec3(XMath.Min(neighborMin.x, n.x), XMath.Min(neighborMin.y, n.y), XMath.Min(neighborMin.z, n.z));
                        neighborMax = new Vec3(XMath.Max(neighborMax.x, n.x), XMath.Max(neighborMax.y, n.y), XMath.Max(neighborMax.z, n.z));
                    }
                }
                Vec3 clampedPrevColor = new Vec3(
                    XMath.Clamp(prevSampleColor.x, neighborMin.x, neighborMax.x),
                    XMath.Clamp(prevSampleColor.y, neighborMin.y, neighborMax.y),
                    XMath.Clamp(prevSampleColor.z, neighborMin.z, neighborMax.z));

                // Real-time decay (see SampleRenderer.HistoryDecayHalfLifeSeconds's own
                // doc comment) - ages the effective count down every frame on a
                // wall-clock schedule, so even a static/well-reprojected pixel keeps
                // slowly forgetting old contributions instead of freezing at a fixed
                // weight once it hits maxHistoryFrames.
                if (historyDecayHalfLifeSeconds > 0f)
                    prevCount *= XMath.Pow(0.5f, deltaTimeSeconds / historyDecayHalfLifeSeconds);

                newCount = XMath.Min(prevCount + 1f, (float)maxHistoryFrames);
                blendedColor = clampedPrevColor + ((pixelColor - clampedPrevColor) / newCount);
            }

            outColor[index] = new Vec4(blendedColor.x, blendedColor.y, blendedColor.z, currentDepth);
            outAccumCount[index] = newCount;
        }
    }
}
