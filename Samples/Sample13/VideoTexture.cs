// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: VideoTexture.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX;
using System;

namespace Sample13
{
    // Bridges an AsyncFfmpegVideoReader (raw BGRA frame stream, decoded on its own
    // background thread) to a single persistent CudaTextureObject, refreshed once per
    // rendered frame - the video equivalent of TextureLoader.LoadRgba8 for static PNGs.
    // The CudaTextureObject's array/handle are created once at construction and never
    // recreated; only its pixel contents change (via CudaTextureObject.Update), so the
    // ulong TextureObject handle already baked into a material's HitgroupRecord at scene
    // switch stays valid for the scene's whole lifetime - SampleRenderer just needs to
    // call Refresh() once per frame for every active video texture.
    internal sealed class VideoTexture : IDisposable
    {
        private readonly AsyncFfmpegVideoReader reader;
        private readonly CudaTextureObject textureObject;
        private readonly byte[] rgbaScratch;
        private readonly int pixelCount;

        public ulong TextureObject => textureObject.TextureObject;

        public VideoTexture(string videoFilePath)
        {
            // useRGBA requests ffmpeg's "bgra" pixel format (4 bytes/pixel, alpha
            // forced to 255) - still channel-swapped relative to CudaTextureObject's
            // RGBA8 assumption, fixed up per-frame in Refresh below. Both ffmpeg's raw
            // video output and this codebase's static-texture loader (TextureLoader.
            // LoadRgba8) are top-down row order, so no vertical flip is needed here.
            reader = new AsyncFfmpegVideoReader(videoFilePath, useRGBA: true);
            pixelCount = reader.Width * reader.Height;
            rgbaScratch = new byte[pixelCount * 4];
            textureObject = new CudaTextureObject(rgbaScratch, reader.Width, reader.Height);
        }

        public unsafe void Refresh()
        {
            IntPtr framePtr = reader.GetCurrentFramePtr();
            byte* src = (byte*)framePtr;
            fixed (byte* dst = rgbaScratch)
            {
                for (int i = 0; i < pixelCount; i++)
                {
                    int o = i * 4;
                    dst[o + 0] = src[o + 2]; // R <- B
                    dst[o + 1] = src[o + 1]; // G
                    dst[o + 2] = src[o + 0]; // B <- R
                    dst[o + 3] = src[o + 3]; // A
                }
            }
            textureObject.Update(rgbaScratch);
        }

        public void Dispose()
        {
            reader.Dispose();
            textureObject.Dispose();
        }
    }
}
