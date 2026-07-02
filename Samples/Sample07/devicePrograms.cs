using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;

namespace Sample07
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

            // Per-ray color is passed back via payload registers as bit-reinterpreted
            // floats - see Sample06's devicePrograms.cs for why (no pointer<->integer
            // conversion support inside an ILGPU kernel).
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

            uint rgba = 0xff000000 | (b << 0) | (g << 8) | (r << 16);

            long fbIndex = ix + iy * launchParams.camera.width;
            launchParams.ColorBuffer[fbIndex] = rgba;
        }

        public static void __miss__radiance(LaunchParams launchParams)
        {
            SetPRD(launchParams.camera.noHitColor);
        }

        public unsafe static void __closest__radiance(LaunchParams launchParams)
        {
            // One merged GAS build input for the whole model geometry (see Model.cs),
            // but the hit triangle's material comes from the current hitgroup record's
            // custom data - OptiX picks the record per triangle via the build input's
            // SbtIndexOffsetBuffer (Model's TriangleMaterialIds), same as the C++
            // reference's per-mesh SBT records.
            uint primId = OptixGetPrimitiveIndex.Value;

            Vec3i tri = launchParams.Indices[primId];
            Vec3 a = launchParams.Vertices[tri.x];
            Vec3 b = launchParams.Vertices[tri.y];
            Vec3 c = launchParams.Vertices[tri.z];
            Vec3 geometricNormal = Vec3.unitVector(Vec3.cross(b - a, c - a));

            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            Vec3 rayDir = new Vec3(dx, dy, dz);

            MaterialSbtData* sbtData = (MaterialSbtData*)OptixGetSbtDataPointer.Value;
            Vec3 color = sbtData->Color;

            float cosDN = 0.2f + (0.8f * XMath.Abs(Vec3.dot(rayDir, geometricNormal)));
            SetPRD(cosDN * color);
        }

        public static void __anyhit__radiance(LaunchParams launchParams)
        { }

        private static void SetPRD(Vec3 color)
        {
            OptixPayload.Payload0 = Interop.FloatAsInt(color.x);
            OptixPayload.Payload1 = Interop.FloatAsInt(color.y);
            OptixPayload.Payload2 = Interop.FloatAsInt(color.z);
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
