using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;
using ILGPU.OptiX.Device;
using ILGPU.OptiX.DeviceApi;
using ILGPU.OptiX.Cuda;
using System.Numerics;

namespace Sample10
{
    public static class devicePrograms
    {
        private const uint RADIANCE_RAY_TYPE = OptixPayloadDefaults.RADIANCE_RAY_TYPE;
        private const uint SHADOW_RAY_TYPE = OptixPayloadDefaults.SHADOW_RAY_TYPE;
        private const uint RAY_TYPE_COUNT = OptixPayloadDefaults.RAY_TYPE_COUNT;
        private const int NUM_PIXEL_SAMPLES = 16;

        public static void __raygen__renderFrame(LaunchParams launchParams)
        {
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;
            var camera = launchParams.camera;

            // Seeded per pixel per frame - matches the C++ reference's
            // prd.random.init(ix + accumID*width, iy + accumID*height), with FrameID
            // standing in for accumID (this sample re-renders from scratch every frame
            // rather than accumulating across frames, so FrameID only needs to change
            // the seed, not weight an accumulation buffer).
            LCG rng = new LCG(
                (uint)(ix + (launchParams.FrameID * camera.width)),
                (uint)(iy + (launchParams.FrameID * camera.height)));

            Vec3 pixelColor = new Vec3(0f, 0f, 0f);
            for (int sampleID = 0; sampleID < NUM_PIXEL_SAMPLES; sampleID++)
            {
                float screenX = (2f * ((ix + rng.Next()) * camera.reciprocalWidth)) - 1f;
                float screenY = (2f * ((iy + rng.Next()) * camera.reciprocalHeight)) - 1f;

                Vec3 rayDir = Vec3.unitVector(camera.axis.transform(
                    new Vec3(screenX * camera.aspectRatio, screenY, camera.cameraPlaneDist)));

                uint p0 = 0, p1 = 0, p2 = 0, p3 = rng.State;

                OptixTrace.Trace(
                    launchParams.traversable,
                    (camera.origin.x, camera.origin.y, camera.origin.z),
                    (rayDir.x, rayDir.y, rayDir.z),
                    1e-3f,
                    1e20f,
                    0.0f,
                    0xff,
                    OptixRayFlags.DisableAnyHit,
                    RADIANCE_RAY_TYPE,
                    RAY_TYPE_COUNT,
                    RADIANCE_RAY_TYPE,
                    ref p0,
                    ref p1,
                    ref p2,
                    ref p3);

                pixelColor += new Vec3(Interop.IntAsFloat(p0), Interop.IntAsFloat(p1), Interop.IntAsFloat(p2));
                rng.State = p3;
            }
            pixelColor /= (float)NUM_PIXEL_SAMPLES;

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
            uint primId = OptixGetPrimitiveIndex.Value;
            Vec3i tri = launchParams.Indices[primId];
            var (bw, bu, bv) = OptixHitProgramHelpers.GetTriangleBarycentrics();

            Vec3 a = launchParams.Vertices[tri.x];
            Vec3 b = launchParams.Vertices[tri.y];
            Vec3 c = launchParams.Vertices[tri.z];
            var geometricNormalV3 = OptixHitProgramHelpers.GetGeometricNormal(
                new Vector3(a.x, a.y, a.z),
                new Vector3(b.x, b.y, b.z),
                new Vector3(c.x, c.y, c.z));
            Vec3 geometricNormal = new Vec3(geometricNormalV3.X, geometricNormalV3.Y, geometricNormalV3.Z);

            Vec3 n0 = launchParams.Normals[tri.x];
            Vec3 n1 = launchParams.Normals[tri.y];
            Vec3 n2 = launchParams.Normals[tri.z];
            var shadingNormalV3 = OptixHitProgramHelpers.InterpolateAttribute(
                new Vector3(n0.x, n0.y, n0.z),
                new Vector3(n1.x, n1.y, n1.z),
                new Vector3(n2.x, n2.y, n2.z),
                bw, bu, bv);
            Vec3 shadingNormal = new Vec3(shadingNormalV3.X, shadingNormalV3.Y, shadingNormalV3.Z);

            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            Vec3 rayDir = new Vec3(dx, dy, dz);

            // Face-forward the geometric normal, then pull the shading normal into the
            // same hemisphere - matches the reference's handling of low-poly shading
            // normals that can otherwise point away from the visible side of the
            // triangle.
            var orientedGeomNormalV3 = OptixHitProgramHelpers.OrientGeometricNormal(geometricNormalV3, rayDir);
            geometricNormal = new Vec3(orientedGeomNormalV3.X, orientedGeomNormalV3.Y, orientedGeomNormalV3.Z);

            if (Vec3.dot(geometricNormal, shadingNormal) < 0f)
                shadingNormal -= 2f * Vec3.dot(geometricNormal, shadingNormal) * geometricNormal;
            shadingNormal = Vec3.unitVector(shadingNormal);

            MaterialSbtData* sbtData = (MaterialSbtData*)OptixGetSbtDataPointer.Value;
            Vec3 diffuseColor = sbtData->Color;
            ulong textureObject = sbtData->TextureObject;

            if (textureObject != 0)
            {
                Vec2 t0 = launchParams.TexCoords[tri.x];
                Vec2 t1 = launchParams.TexCoords[tri.y];
                Vec2 t2 = launchParams.TexCoords[tri.z];
                float tu = (bw * t0.x) + (bu * t1.x) + (bv * t2.x);
                float tv = (bw * t0.y) + (bu * t1.y) + (bv * t2.y);

                var (tr, tg, tb, _) = CudaTex2D.Sample(textureObject, tu, tv);
                diffuseColor = new Vec3(diffuseColor.x * tr, diffuseColor.y * tg, diffuseColor.z * tb);
            }

            Vec3 pixelColor = (0.1f + (0.2f * XMath.Abs(Vec3.dot(shadingNormal, rayDir)))) * diffuseColor;

            Vec3 surfPos = (bw * a) + (bu * b) + (bv * c);

            // RNG state travels between raygen and closesthit through a payload
            // register (see LCG.cs) rather than a PRD pointer, since ILGPU.OptiX
            // doesn't support pointer<->int payload packing.
            LCG rng = default;
            rng.State = OptixPayload.Payload3;

            Vec3 lightPos = launchParams.LightOrigin + (rng.Next() * launchParams.LightDu) + (rng.Next() * launchParams.LightDv);
            Vec3 lightDir = lightPos - surfPos;
            float lightDist = lightDir.length();
            lightDir = Vec3.unitVector(lightDir);

            float NdotL = Vec3.dot(lightDir, shadingNormal);
            if (NdotL >= 0f)
            {
                uint shadowPayload = Interop.FloatAsInt(0f);
                OptixTrace.Trace(
                    launchParams.traversable,
                    (surfPos.x + (1e-3f * geometricNormal.x), surfPos.y + (1e-3f * geometricNormal.y), surfPos.z + (1e-3f * geometricNormal.z)),
                    (lightDir.x, lightDir.y, lightDir.z),
                    1e-3f,
                    lightDist * (1f - 1e-3f),
                    0.0f,
                    0xff,
                    OptixRayFlags.DisableAnyHit
                        | OptixRayFlags.TerminateOnFirstHit
                        | OptixRayFlags.DisableClosestHit,
                    SHADOW_RAY_TYPE,
                    RAY_TYPE_COUNT,
                    SHADOW_RAY_TYPE,
                    ref shadowPayload);

                float lightVisibility = Interop.IntAsFloat(shadowPayload);
                pixelColor += lightVisibility * launchParams.LightPower * diffuseColor * (NdotL / (lightDist * lightDist));
            }

            SetPRD(pixelColor);
            OptixPayload.Payload3 = rng.State;
        }

        public static void __anyhit__radiance(LaunchParams launchParams)
        { }

        public static void __closesthit__shadow(LaunchParams launchParams)
        { }

        public static void __anyhit__shadow(LaunchParams launchParams)
        { }

        public static void __miss__shadow(LaunchParams launchParams)
        {
            // Nothing occluded the shadow ray, so the light sample is visible.
            OptixPayloadInterop.SetFloat(0, 1f);
        }

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
