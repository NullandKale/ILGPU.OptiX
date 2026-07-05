// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: devicePrograms.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;

namespace Sample04
{
    public static class devicePrograms
    {
        public unsafe static void __raygen__renderFrame(LaunchParams launchParams)
        {
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;
            var camera = launchParams.camera;

            float screenX = (2f * ((ix + 0.5f) * camera.reciprocalWidth)) - 1f;
            float screenY = (2f * ((iy + 0.5f) * camera.reciprocalHeight)) - 1f;

            Vec3 rayDir = Vec3.unitVector(camera.axis.transform(
                new Vec3(screenX * camera.aspectRatio, screenY, camera.cameraPlaneDist)));

            // per-ray color is passed back via payload registers as bit-reinterpreted
            // floats (ILGPU has no support for pointer<->integer conversions inside a
            // kernel, so the usual optix7course "packed pointer to a PRD" trick doesn't
            // work here).
            uint p0 = 0, p1 = 0, p2 = 0;

            OptixTrace.Trace(
                launchParams.traversable,
                (camera.origin.x, camera.origin.y, camera.origin.z),
                (rayDir.x, rayDir.y, rayDir.z),
                1e-3f,
                1e20f,
                0.0f,
                0xff,
                OptixRayFlags.OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                0,
                1,
                0,
                ref p0,
                ref p1,
                ref p2);

            Vec3 pixelColor = new Vec3(
                Interop.IntAsFloat(p0),
                Interop.IntAsFloat(p1),
                Interop.IntAsFloat(p2));

            uint r = (uint)(XMath.Clamp(pixelColor.x, 0f, 1f) * 255f);
            uint g = (uint)(XMath.Clamp(pixelColor.y, 0f, 1f) * 255f);
            uint b = (uint)(XMath.Clamp(pixelColor.z, 0f, 1f) * 255f);

            // convert to 32-bit bgra value (we explicitly set alpha to 0xff
            // to make stb_image_write happy ...
            uint rgba = 0xff000000 | (b << 0) | (g << 8) | (r << 16);

            // and write to frame buffer ...
            long fbIndex = ix + iy * launchParams.camera.width;
            launchParams.ColorBuffer[fbIndex] = rgba;
        }

        public static void __miss__radiance(LaunchParams launchParams)
        {
            SetPRD(launchParams.camera.noHitColor);
        }

        public static void __closest__radiance(LaunchParams launchParams)
        {
            // no attribute/hit-info wrappers are exposed by ILGPU.OptiX yet
            // (e.g. barycentrics, geometric normal), so hits are shaded flat. Uses a
            // color distinct from camera.noHitColor so hit vs. miss is visually
            // distinguishable.
            SetPRD(new Vec3(0.9f, 0.2f, 0.2f));
        }

        public static void __anyhit__radiance(LaunchParams launchParams)
        { }

        private static void SetPRD(Vec3 color)
        {
            OptixPayloadInterop.SetFloat(0, color.x);
            OptixPayloadInterop.SetFloat(1, color.y);
            OptixPayloadInterop.SetFloat(2, color.z);
        }

        public static void flipBitmap(Index1D index, int width, int height, ArrayView<byte> source, ArrayView<byte> dest)
        {


            int x = index % width;
            int y = (height - 1) - (index / width);

            int newIndex = ((y * width) + x) * 4;
            int oldIndexStart = index * 4;

            dest[newIndex] = source[oldIndexStart];
            dest[newIndex + 1] = source[oldIndexStart + 1];
            dest[newIndex + 2] = source[oldIndexStart + 2];
            dest[newIndex + 3] = source[oldIndexStart + 3];
        }
    }
}
