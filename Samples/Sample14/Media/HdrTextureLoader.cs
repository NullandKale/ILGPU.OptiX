// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: HdrTextureLoader.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using StbImageSharp;
using System.IO;

namespace Sample14
{
    // HDR (Radiance .hdr) loader for the environment map - StbImageSharp's
    // ImageResultFloat wraps stb_image's stbi_loadf,
    // which auto-detects the Radiance format and decodes directly to linear float RGB
    // (no manual RGBE decode, no gamma correction needed - unlike TextureLoader.cs's
    // 8-bit sRGB path, this data is already linear).
    internal static class HdrTextureLoader
    {
        public struct HdrImage
        {
            // Tightly packed, row 0 = top of the image, R,G,B float triplets per texel
            // (matches ImageResultFloat's own layout for ColorComponents.RedGreenBlue).
            public float[] Pixels;
            public int Width;
            public int Height;
        }

        public static HdrImage Load(string path)
        {
            using var stream = File.OpenRead(path);
            ImageResultFloat image = ImageResultFloat.FromStream(stream, ColorComponents.RedGreenBlue);
            return new HdrImage { Pixels = image.Data, Width = image.Width, Height = image.Height };
        }
    }
}
