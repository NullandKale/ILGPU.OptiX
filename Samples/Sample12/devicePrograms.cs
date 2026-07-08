using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;
using ILGPU.OptiX.Device;
using System.Numerics;

namespace Sample12
{
    public static class devicePrograms
    {
        private const uint RADIANCE_RAY_TYPE = OptixPayloadDefaults.RADIANCE_RAY_TYPE;
        private const uint SHADOW_RAY_TYPE = OptixPayloadDefaults.SHADOW_RAY_TYPE;
        private const uint RAY_TYPE_COUNT = OptixPayloadDefaults.RAY_TYPE_COUNT;
        private const int NUM_LIGHT_SAMPLES = 4;

        public static void __raygen__renderFrame(LaunchParams launchParams)
        {
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;
            var camera = launchParams.camera;

            LCG rng = new LCG((uint)(ix + (camera.width * iy)), (uint)launchParams.FrameID);

            int numPixelSamples = launchParams.NumPixelSamples;
            Vec3 pixelColor = new Vec3(0f, 0f, 0f);
            Vec3 pixelNormal = new Vec3(0f, 0f, 0f);
            Vec3 pixelAlbedo = new Vec3(0f, 0f, 0f);
            for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
            {
                float screenX = (2f * ((ix + rng.Next()) * camera.reciprocalWidth)) - 1f;
                float screenY = (2f * ((iy + rng.Next()) * camera.reciprocalHeight)) - 1f;

                Vec3 rayDir = Vec3.unitVector(camera.axis.transform(
                    new Vec3(screenX * camera.aspectRatio, screenY, camera.cameraPlaneDist)));

                // Color (p0-2), shading normal (p3-5), diffuse albedo (p6-8) - 9
                // payloads is the practical ceiling for this binding's OptixTrace
                // wrapper (see OptixTrace.cs), so unlike Sample10/11 the RNG state
                // isn't threaded through a payload here; __closest__radiance reseeds
                // its own RNG instead (see below).
                uint p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0, p8 = 0;

                OptixTrace.Trace(
                    launchParams.traversable,
                    (camera.origin.x, camera.origin.y, camera.origin.z),
                    (rayDir.x, rayDir.y, rayDir.z),
                    1e-3f,
                    1e20f,
                    0.0f,
                    0xff,
                    OptixRayFlags.OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                    RADIANCE_RAY_TYPE,
                    RAY_TYPE_COUNT,
                    RADIANCE_RAY_TYPE,
                    ref p0, ref p1, ref p2,
                    ref p3, ref p4, ref p5,
                    ref p6, ref p7, ref p8);

                pixelColor += new Vec3(Interop.IntAsFloat(p0), Interop.IntAsFloat(p1), Interop.IntAsFloat(p2));
                pixelNormal += new Vec3(Interop.IntAsFloat(p3), Interop.IntAsFloat(p4), Interop.IntAsFloat(p5));
                pixelAlbedo += new Vec3(Interop.IntAsFloat(p6), Interop.IntAsFloat(p7), Interop.IntAsFloat(p8));
            }
            pixelColor /= (float)numPixelSamples;
            pixelNormal /= (float)numPixelSamples;
            pixelAlbedo /= (float)numPixelSamples;

            long fbIndex = ix + (iy * camera.width);

            if (launchParams.FrameID > 0)
            {
                Vec4 previous = launchParams.ColorBuffer[fbIndex];
                float weight = launchParams.FrameID;
                pixelColor = new Vec3(
                    ((weight * previous.x) + pixelColor.x) / (weight + 1f),
                    ((weight * previous.y) + pixelColor.y) / (weight + 1f),
                    ((weight * previous.z) + pixelColor.z) / (weight + 1f));
            }

            launchParams.ColorBuffer[fbIndex] = new Vec4(pixelColor.x, pixelColor.y, pixelColor.z, 1f);
            launchParams.AlbedoBuffer[fbIndex] = new Vec4(pixelAlbedo.x, pixelAlbedo.y, pixelAlbedo.z, 1f);
            launchParams.NormalBuffer[fbIndex] = new Vec4(pixelNormal.x, pixelNormal.y, pixelNormal.z, 1f);
        }

        public static void __miss__radiance(LaunchParams launchParams)
        {
            SetPRD(new Vec3(1f, 1f, 1f), new Vec3(0f, 0f, 0f), new Vec3(0f, 0f, 0f));
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

            // No RNG payload continuity from raygen here (see __raygen__renderFrame) -
            // reseed from launch index/frame/primitive instead. Slightly less
            // decorrelated across the outer pixel-sample loop than Sample10/11, but
            // still decorrelated across pixels, frames, and hit surfaces.
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;
            LCG rng = new LCG((uint)(ix + (1000003 * iy)), (uint)((launchParams.FrameID * 7919) + primId));

            for (int lightSampleID = 0; lightSampleID < NUM_LIGHT_SAMPLES; lightSampleID++)
            {
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
                        OptixRayFlags.OPTIX_RAY_FLAG_DISABLE_ANYHIT
                            | OptixRayFlags.OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                            | OptixRayFlags.OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                        SHADOW_RAY_TYPE,
                        RAY_TYPE_COUNT,
                        SHADOW_RAY_TYPE,
                        ref shadowPayload);

                    float lightVisibility = Interop.IntAsFloat(shadowPayload);
                    pixelColor += lightVisibility * launchParams.LightPower * diffuseColor * (NdotL / (lightDist * lightDist * NUM_LIGHT_SAMPLES));
                }
            }

            SetPRD(pixelColor, shadingNormal, diffuseColor);
        }

        public static void __anyhit__radiance(LaunchParams launchParams)
        { }

        public static void __closesthit__shadow(LaunchParams launchParams)
        { }

        public static void __anyhit__shadow(LaunchParams launchParams)
        { }

        public static void __miss__shadow(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 1f);
        }

        private static void SetPRD(Vec3 color, Vec3 normal, Vec3 albedo)
        {
            OptixPayloadVec3Helper.SetVec3Registers(0, new Vector3(color.x, color.y, color.z));
            OptixPayloadVec3Helper.SetVec3Registers(3, new Vector3(normal.x, normal.y, normal.z));
            OptixPayloadVec3Helper.SetVec3Registers(6, new Vector3(albedo.x, albedo.y, albedo.z));
        }

        // Matches example12_denoiseSeparateChannels/toneMap.cu's
        // computeFinalPixelColorsKernel: applies a sqrt gamma approximation (absent
        // from Sample11's tonemap) before packing to display bytes and flipping rows.
        public static void tonemapAndFlip(Index1D index, int width, int height, ArrayView<Vec4> hdr, ArrayView<byte> dest)
        {
            int x = index % width;
            int y = (height - 1) - (index / width);
            int newIndex = ((y * width) + x) * 4;

            Vec4 color = hdr[index];
            byte r = (byte)(XMath.Clamp(XMath.Sqrt(color.x), 0f, 1f) * 255f);
            byte g = (byte)(XMath.Clamp(XMath.Sqrt(color.y), 0f, 1f) * 255f);
            byte b = (byte)(XMath.Clamp(XMath.Sqrt(color.z), 0f, 1f) * 255f);

            dest[newIndex] = b;
            dest[newIndex + 1] = g;
            dest[newIndex + 2] = r;
            dest[newIndex + 3] = 0xff;
        }
    }
}
