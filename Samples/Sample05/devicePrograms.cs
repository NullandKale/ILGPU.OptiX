using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;
using System.Numerics;

namespace Sample05
{
    public static class devicePrograms
    {
        public static void __raygen__renderFrame(LaunchParams launchParams)
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
            // ILGPU.OptiX doesn't wrap optixGetSbtDataPointer (it returns a raw
            // device pointer, and ILGPU has no support for reinterpreting an
            // integer/register value as a pointer inside a kernel - the same
            // limitation that ruled out the packed-pointer PRD trick in raygen).
            // Per-object data is instead passed as plain device pointers on
            // LaunchParams, indexed directly by primitive index - this gets the
            // same result (per-triangle color + geometric-normal shading) as
            // optix7course's example05_firstSBTData.
            uint primId = OptixGetPrimitiveIndex.Value;
            Vec3i tri = launchParams.indices[primId];
            Vec3 a = launchParams.vertices[tri.x];
            Vec3 b = launchParams.vertices[tri.y];
            Vec3 c = launchParams.vertices[tri.z];
            var normalV3 = OptixHitProgramHelpers.GetGeometricNormal(
                new Vector3(a.x, a.y, a.z),
                new Vector3(b.x, b.y, b.z),
                new Vector3(c.x, c.y, c.z));
            Vec3 normal = new Vec3(normalV3.X, normalV3.Y, normalV3.Z);

            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            Vec3 rayDir = new Vec3(dx, dy, dz);

            float cosDN = 0.2f + (0.8f * XMath.Abs(Vec3.dot(rayDir, normal)));
            Vec3 color = launchParams.primitiveColors[primId];
            SetPRD(cosDN * color);
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
