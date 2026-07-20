using ILGPU.OptiX.Cuda;
using StbImageSharp;
using System.IO;

namespace Sample13
{
    // Uses StbImageSharp (no WPF dependency). StbImageSharp already returns
    // tightly-packed RGBA8 data, row 0 = top of the image, in R,G,B,A channel order -
    // exactly what CudaTextureObject expects, so no channel swap or row-flip is
    // needed here.
    internal static class TextureLoader
    {
        public static byte[] LoadRgba8(string path, out int width, out int height)
        {
            using var stream = File.OpenRead(path);
            var image = ImageResult.FromStream(stream, ColorComponents.RedGreenBlueAlpha);
            width = image.Width;
            height = image.Height;
            return image.Data;
        }
    }
}
