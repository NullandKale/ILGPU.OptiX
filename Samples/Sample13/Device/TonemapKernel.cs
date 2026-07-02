using ILGPU;
using ILGPU.Algorithms;

namespace Sample13
{
    /// <summary>
    /// Plain ILGPU kernel (not an OptiX program) that converts the HDR buffer to the
    /// BGRA display buffer.
    /// </summary>
    public static class TonemapKernel
    {
        // Gamma-corrected (sqrt) tonemap + row flip for display, matching Sample12's.
        public static void tonemapAndFlip(Index1D index, int width, int height, ArrayView<Vec4> hdr, ArrayView<byte> dest)
        {
            int x = index % width;
            int y = (height - 1) - (index / width);
            int newIndex = ((y * width) + x) * 4;

            Vec4 color = hdr[index];
            byte r = (byte)(XMath.Clamp(XMath.Sqrt(color.x), 0f, 1f) * 255f);
            byte g = (byte)(XMath.Clamp(XMath.Sqrt(color.y), 0f, 1f) * 255f);
            byte b = (byte)(XMath.Clamp(XMath.Sqrt(color.z), 0f, 1f) * 255f);

            dest[newIndex] = b;
            dest[newIndex + 1] = g;
            dest[newIndex + 2] = r;
            dest[newIndex + 3] = 0xff;
        }
    }
}
