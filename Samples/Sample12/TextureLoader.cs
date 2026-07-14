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
using ILGPU.OptiX.Cuda;

namespace Sample12
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
            for (int i = 0; i < bgra.Length; i += 4)
            {
                rgba[i + 0] = bgra[i + 2];
                rgba[i + 1] = bgra[i + 1];
                rgba[i + 2] = bgra[i + 0];
                rgba[i + 3] = bgra[i + 3];
            }
            return rgba;
        }
    }
}
