using ILGPU;
using ILGPU.Algorithms;

namespace Sample15
{
    /// <summary>
    /// Plain ILGPU kernel (not an OptiX program) that converts the HDR buffer to the
    /// BGRA display buffer using proper tone mapping.
    ///
    /// Proper tone mapping pipeline:
    /// 1. Exposure adjustment - scales the HDR values
    /// 2. Tone mapping operator (Reinhard or ACES, runtime-selectable) - compresses
    ///    HDR to LDR range
    /// 3. sRGB OETF (piecewise, not an approximation - Milestone M8) - linear to
    ///    display-referred sRGB
    /// 4. Display output - convert to byte values
    /// </summary>
    public static class TonemapKernel
    {
        public const int OperatorReinhard = 0;
        public const int OperatorAces = 1;

        // Tone mapping: Reinhard operator - color / (1 + color). Compresses bright
        // values smoothly without hard clipping.
        private static Vec3 ReinhardTonemap(Vec3 color) => color / (new Vec3(1f, 1f, 1f) + color);

        // Narkowicz's fitted ACES filmic curve ("ACES Filmic Tone Mapping Curve", 2016)
        // - a per-channel rational polynomial fit to the ACES reference rendering
        // transform, not the full ACES pipeline (no color-space/gamut mapping) - the
        // same "close approximation, not the real thing" tradeoff every game/renderer
        // using this exact curve makes. Punchier contrast and a more filmic shoulder
        // than Reinhard, at the cost of a slight built-in saturation boost.
        private static Vec3 AcesFilmTonemap(Vec3 color)
        {
            const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
            Vec3 numerator = color * ((a * color) + new Vec3(b, b, b));
            Vec3 denominator = (color * ((c * color) + new Vec3(d, d, d))) + new Vec3(e, e, e);
            return new Vec3(
                XMath.Clamp(numerator.x / denominator.x, 0f, 1f),
                XMath.Clamp(numerator.y / denominator.y, 0f, 1f),
                XMath.Clamp(numerator.z / denominator.z, 0f, 1f));
        }

        // True piecewise sRGB OETF (IEC 61966-2-1), replacing the flat pow(x, 1/2.2)
        // approximation used through M7 - the linear segment near black matters for
        // shadow-detail banding that a flat gamma curve doesn't reproduce exactly.
        private static float LinearToSrgb(float c)
        {
            c = XMath.Clamp(c, 0f, 1f);
            return c <= 0.0031308f ? c * 12.92f : (1.055f * XMath.Pow(c, 1f / 2.4f)) - 0.055f;
        }

        // Tone mapping + row flip for display. exposureStops follows the standard
        // photographic-stops convention (each +1 multiplies linear radiance by 2x);
        // tonemapOperator selects OperatorReinhard/OperatorAces (see SampleRenderer's
        // own Exposure/TonemapOperator properties and the UI panel's controls).
        public static void tonemapAndFlip(Index1D index, int width, int height, float exposureStops, int tonemapOperator, ArrayView<Vec4> hdr, ArrayView<byte> dest)
        {
            int x = index % width;
            int y = (height - 1) - (index / width);
            int newIndex = ((y * width) + x) * 4;

            Vec4 color = hdr[index];
            Vec3 rgb = new Vec3(color.x, color.y, color.z);

            // Step 1: Apply exposure (2^stops).
            rgb *= XMath.Pow(2f, exposureStops);

            // Step 2: Apply the selected tone mapping operator.
            rgb = tonemapOperator == OperatorAces ? AcesFilmTonemap(rgb) : ReinhardTonemap(rgb);

            // Step 3: sRGB OETF.
            rgb = new Vec3(LinearToSrgb(rgb.x), LinearToSrgb(rgb.y), LinearToSrgb(rgb.z));

            // Step 4: Convert to byte values.
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
