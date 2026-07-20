using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.DeviceApi;
using System.Numerics;

namespace Sample13
{
    /// <summary>
    /// The ray-generation program: camera ray setup, the iterative bounce loop, and
    /// progressive frame accumulation into the HDR color buffer plus the denoiser's
    /// AOV guide buffers.
    /// </summary>
    public static class RaygenProgram
    {
        // Hard safety cap on total Trace() calls per sample - the per-kind counters
        // from LaunchParams already bound the path length to at most
        // MaxMirrorBounces + MaxRefractionBounces + MaxDiffuseBounces continuations,
        // so this is never the limiting factor in practice.
        private const int MaxTotalBounces = 8;

        public unsafe static void __raygen__renderFrame(LaunchParams launchParams)
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

                Vec3 rayOrigin = camera.origin;
                Vec3 rayDir = Vec3.unitVector(camera.axis.transform(
                    new Vec3(screenX * camera.aspectRatio, screenY, camera.cameraPlaneDist)));

                Vec3 throughput = new Vec3(1f, 1f, 1f);
                Vec3 sampleRadiance = new Vec3(0f, 0f, 0f);
                Vec3 sampleNormal = new Vec3(0f, 0f, 0f);
                Vec3 sampleAlbedo = new Vec3(0f, 0f, 0f);
                int mirrorBounces = 0;
                int refractionBounces = 0;
                int diffuseBounces = 0;

                for (int bounce = 0; bounce < MaxTotalBounces; bounce++)
                {
                    uint p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0, p8 = 0, p9 = 0, p10 = 0, p11 = 0, p12 = 0;
                    uint p13 = 0, p14 = 0, p15 = 0, p16 = 0, p17 = 0, p18 = 0;
                    OptixTrace.Trace(
                        launchParams.traversable,
                        (rayOrigin.x, rayOrigin.y, rayOrigin.z),
                        (rayDir.x, rayDir.y, rayDir.z),
                        1e-3f,
                        1e20f,
                        0.0f,
                        0xff,
                        // Any-hit must run on radiance rays because of alpha-cutout
                        // materials (Sponza's leaf geometry) - __anyhit__radiance is
                        // what actually ignores the intersection for a
                        // below-threshold-alpha sample. DISABLE_ANYHIT would silently
                        // skip that test and shade the nearest triangle regardless of
                        // alpha.
                        OptixRayFlags.None,
                        Payloads.RADIANCE_RAY_TYPE,
                        Payloads.RAY_TYPE_COUNT,
                        Payloads.RADIANCE_RAY_TYPE,
                        ref p0, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7, ref p8, ref p9, ref p10, ref p11, ref p12,
                        ref p13, ref p14, ref p15, ref p16, ref p17, ref p18);

                    // Payload results come back through the ref-parameters passed to
                    // Trace() above, not through OptixPayloadInterop/GetVec3Registers -
                    // those call the optixGetPayload device intrinsic, which is only
                    // legal inside the hit/miss/anyhit programs that ran during this
                    // trace, not here in raygen after Trace() has already returned.
                    Vec3 radiance = new Vec3(
                        ILGPU.Interop.IntAsFloat(p0),
                        ILGPU.Interop.IntAsFloat(p1),
                        ILGPU.Interop.IntAsFloat(p2));
                    sampleRadiance += throughput * radiance;

                    // AOV guide buffers only ever reflect the primary ray's own hit
                    // (bounce 0) - a mirror/glass primary hit still contributes its own
                    // normal/tint here, which is exactly what the denoiser needs to
                    // recognize that surface.
                    if (bounce == 0)
                    {
                        sampleNormal = new Vec3(
                            ILGPU.Interop.IntAsFloat(p13),
                            ILGPU.Interop.IntAsFloat(p14),
                            ILGPU.Interop.IntAsFloat(p15));
                        sampleAlbedo = new Vec3(
                            ILGPU.Interop.IntAsFloat(p16),
                            ILGPU.Interop.IntAsFloat(p17),
                            ILGPU.Interop.IntAsFloat(p18));
                    }

                    uint flag = p3;
                    if (flag == Payloads.BOUNCE_TERMINAL)
                        break;

                    if (flag == Payloads.BOUNCE_CONTINUE_MIRROR)
                    {
                        mirrorBounces++;
                        if (mirrorBounces > launchParams.MaxMirrorBounces)
                            break;
                    }
                    else if (flag == Payloads.BOUNCE_CONTINUE_DIELECTRIC)
                    {
                        refractionBounces++;
                        if (refractionBounces > launchParams.MaxRefractionBounces)
                            break;
                    }
                    else if (flag == Payloads.BOUNCE_CONTINUE_DIFFUSE)
                    {
                        diffuseBounces++;
                        if (diffuseBounces > launchParams.MaxDiffuseBounces)
                            break;
                    }

                    throughput *= new Vec3(
                        ILGPU.Interop.IntAsFloat(p10),
                        ILGPU.Interop.IntAsFloat(p11),
                        ILGPU.Interop.IntAsFloat(p12));
                    rayOrigin = new Vec3(
                        ILGPU.Interop.IntAsFloat(p4),
                        ILGPU.Interop.IntAsFloat(p5),
                        ILGPU.Interop.IntAsFloat(p6));
                    rayDir = new Vec3(
                        ILGPU.Interop.IntAsFloat(p7),
                        ILGPU.Interop.IntAsFloat(p8),
                        ILGPU.Interop.IntAsFloat(p9));

                    if (throughput.lengthSquared() < 1e-6f)
                        break;
                }

                pixelColor += sampleRadiance;
                pixelNormal += sampleNormal;
                pixelAlbedo += sampleAlbedo;
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

            // Unlike ColorBuffer, AlbedoBuffer/NormalBuffer are NOT blended across
            // frames - each frame overwrites them fresh (these are the denoiser's AOV
            // guide inputs, not part of the progressively-accumulated image).
            launchParams.NormalBuffer[fbIndex] = new Vec4(pixelNormal.x, pixelNormal.y, pixelNormal.z, 1f);
            launchParams.AlbedoBuffer[fbIndex] = new Vec4(pixelAlbedo.x, pixelAlbedo.y, pixelAlbedo.z, 1f);
        }
    }
}
