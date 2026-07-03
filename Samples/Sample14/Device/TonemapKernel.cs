using ILGPU;
using ILGPU.Algorithms;

namespace Sample14
{
    /// <summary>
    /// Plain ILGPU kernel (not an OptiX program) that converts the HDR buffer to the
    /// BGRA display buffer using proper tone mapping.
    ///
    /// Proper tone mapping pipeline:
    /// 1. Exposure adjustment - scales the HDR values
    /// 2. Tone mapping operator (Reinhard) - compresses HDR to LDR range
    /// 3. Gamma correction - linearize to sRGB space
    /// 4. Display output - convert to byte values
    /// </summary>
    public static class TonemapKernel
    {
        // Exposure adjustment (2^stops, where each stop = 2x brightness)
        // -1 stop = 0.5 (half brightness), 0 stops = 1.0, +1 stop = 2.0
        private const float Exposure = 0.5f;

        // Gamma correction (standard sRGB)
        private const float InvGamma = 1.0f / 2.2f;

        // Tone mapping: Reinhard operator
        // Simple reinhard: color / (1 + color)
        // This compresses bright values smoothly without hard clipping
        private static Vec3 ReinhardTonemap(Vec3 color)
        {
            return color / (new Vec3(1f, 1f, 1f) + color);
        }

        // Tone mapping + row flip for display
        public static void tonemapAndFlip(Index1D index, int width, int height, ArrayView<Vec4> hdr, ArrayView<byte> dest)
        {
            int x = index % width;
            int y = (height - 1) - (index / width);
            int newIndex = ((y * width) + x) * 4;

            Vec4 color = hdr[index];
            Vec3 rgb = new Vec3(color.x, color.y, color.z);

            // Step 1: Apply exposure
            rgb = rgb * Exposure;

            // Step 2: Apply Reinhard tone mapping
            rgb = ReinhardTonemap(rgb);

            // Step 3: Apply gamma correction (sRGB)
            rgb = new Vec3(
                XMath.Pow(rgb.x, InvGamma),
                XMath.Pow(rgb.y, InvGamma),
                XMath.Pow(rgb.z, InvGamma)
            );

            // Step 4: Convert to byte values
            byte r = (byte)(XMath.Clamp(rgb.x, 0f, 1f) * 255f);
            byte g = (byte)(XMath.Clamp(rgb.y, 0f, 1f) * 255f);
            byte b = (byte)(XMath.Clamp(rgb.z, 0f, 1f) * 255f);

            dest[newIndex] = b;
            dest[newIndex + 1] = g;
            dest[newIndex + 2] = r;
            dest[newIndex + 3] = 0xff;
        }
    }
}
