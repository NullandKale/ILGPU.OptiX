// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: TextureLoader.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace Sample13
{
    internal static class TextureLoader
    {
        // Returns tightly-packed RGBA8 pixel data, row-major, top-to-bottom - the
        // layout CudaTextureObject expects. WPF's imaging pipeline only exposes
        // 8-bit-per-channel BGRA (no plain RGBA), so channels are swapped while
        // copying out.
        public static byte[] LoadRgba8(string path, out int width, out int height)
        {
            var frame = BitmapFrame.Create(
                new Uri(path),
                BitmapCreateOptions.None,
                BitmapCacheOption.OnLoad);
            var converted = new FormatConvertedBitmap(frame, PixelFormats.Bgra32, null, 0);

            width = converted.PixelWidth;
            height = converted.PixelHeight;

            int stride = width * 4;
            var bgra = new byte[stride * height];
            converted.CopyPixels(bgra, stride, 0);

            var rgba = new byte[bgra.Length];
            unsafe
            {
                fixed (byte* src = bgra)
                    BgraToRgba(src, rgba, width * height);
            }
            return rgba;
        }

        // Swaps BGRA byte order into the tightly-packed RGBA8 layout CudaTextureObject
        // expects - shared by the static-image path above and VideoTexture's per-frame
        // refresh (whose source is an unmanaged ffmpeg frame buffer, hence the pointer).
        public static unsafe void BgraToRgba(byte* bgra, byte[] rgba, int pixelCount)
        {
            fixed (byte* dst = rgba)
            {
                for (int i = 0; i < pixelCount; i++)
                {
                    int o = i * 4;
                    dst[o + 0] = bgra[o + 2]; // R <- B
                    dst[o + 1] = bgra[o + 1]; // G
                    dst[o + 2] = bgra[o + 0]; // B <- R
                    dst[o + 3] = bgra[o + 3]; // A
                }
            }
        }
    }
}
