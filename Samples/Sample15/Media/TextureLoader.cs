using ILGPU.OptiX.Cuda;
using StbImageSharp;
using System.IO;

namespace Sample15
{
    // WPF-free replacement for Sample13's Media/TextureLoader.cs (which used
    // System.Windows.Media.Imaging.BitmapFrame - a hard WPF/PresentationCore
    // dependency that doesn't exist once WPF is dropped. StbImageSharp already returns
    // tightly-packed RGBA8 data, row 0 = top of the image, in R,G,B,A channel order -
    // exactly what CudaTextureObject expects, so no channel swap (unlike Sample13's
    // BGRA->RGBA swap for WPF's Bgra32) or row-flip is needed here.
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
