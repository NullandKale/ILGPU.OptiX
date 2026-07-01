using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;

namespace Sample06
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
                0, // SBTstride=0: all meshes share the one hitgroup record (index
                   // = SBToffset + SBTstride*OptixGetSbtGASIndex() must stay 0
                   // regardless of which of the two build inputs was hit, since we
                   // only have one hitgroup record and pick per-mesh data via
                   // OptixGetSbtGASIndex ourselves in the shader instead)
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

        public unsafe static void __closest__radiance(LaunchParams launchParams)
        {
            // ILGPU.OptiX doesn't wrap optixGetSbtDataPointer (it returns a raw
            // device pointer, and ILGPU has no support for reinterpreting an
            // integer/register value as a pointer inside a kernel). Sample06 has
            // multiple independent meshes (one OptixBuildInput each, see
            // SampleRenderer.buildAccel), so - unlike Sample05's single merged mesh -
            // we need to know WHICH mesh was hit, not just which triangle.
            // OptixGetSbtGASIndex identifies the build input (mesh) without needing a
            // pointer, playing the same role optixGetSbtDataPointer's per-record data
            // would in the reference example06_multipleObjects. Per-mesh data comes
            // from LaunchParams' fixed mesh0Xxx/mesh1Xxx fields (see LaunchParams.cs
            // for why an array-of-pointers indexed by mesh ID isn't used).
            uint meshId = OptixGetSbtGASIndex.Value;
            uint primId = OptixGetPrimitiveIndex.Value;

            Vec3* vertices = meshId == 0 ? launchParams.mesh0Vertices : launchParams.mesh1Vertices;
            Vec3i* indices = meshId == 0 ? launchParams.mesh0Indices : launchParams.mesh1Indices;
            Vec3 color = meshId == 0 ? launchParams.mesh0Color : launchParams.mesh1Color;

            Vec3i tri = indices[primId];
            Vec3 a = vertices[tri.x];
            Vec3 b = vertices[tri.y];
            Vec3 c = vertices[tri.z];
            Vec3 normal = Vec3.unitVector(Vec3.cross(b - a, c - a));

            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            Vec3 rayDir = new Vec3(dx, dy, dz);

            float cosDN = 0.2f + (0.8f * XMath.Abs(Vec3.dot(rayDir, normal)));
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
