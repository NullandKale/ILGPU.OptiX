using System.Numerics;
using ILGPU;
using ILGPU.Algorithms;

namespace Sample19
{
    /// <summary>
    /// Plain ILGPU kernel (not an OptiX program) that packs a denoised float4 tile
    /// directly into the shared BGRA interop display buffer - no CPU round-trip.
    /// Denoiser output here is already display-range [0,1] (the raygen shading
    /// clamps before denoising), so this is a straight clamp+pack, not a full HDR
    /// tonemap (contrast Sample13's Device/TonemapKernel.cs, which handles genuine
    /// HDR input).
    /// </summary>
    public static class PackKernel
    {
        public static void PackToDisplay(
            Index1D index, int tileWidth, int xOffset, int stride,
            ArrayView<Vector4> src, ArrayView<byte> dest)
        {
            int x = index % tileWidth;
            int y = index / tileWidth;

            var c = src[index];
            byte r = (byte)(XMath.Clamp(c.X, 0f, 1f) * 255f);
            byte g = (byte)(XMath.Clamp(c.Y, 0f, 1f) * 255f);
            byte b = (byte)(XMath.Clamp(c.Z, 0f, 1f) * 255f);

            long pixelIndex = (x + xOffset) + (long)y * stride;
            long byteIndex = pixelIndex * 4;
            dest[byteIndex + 0] = b;
            dest[byteIndex + 1] = g;
            dest[byteIndex + 2] = r;
            dest[byteIndex + 3] = 0xff;
        }
    }
}
