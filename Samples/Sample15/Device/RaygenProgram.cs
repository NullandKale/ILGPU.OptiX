using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;
using System.Numerics;

namespace Sample15
{
    /// <summary>
    /// The ray-generation program: camera ray setup, the iterative bounce loop, and
    /// progressive frame accumulation into the HDR color buffer plus the denoiser's
    /// AOV guide buffers.
    /// </summary>
    public static class RaygenProgram
    {
        // Hard safety cap on total Trace() calls per sample (docs/SAMPLE15_PLAN.md
        // Milestone M3) - launchParams.MaxBounces and Russian roulette termination are
        // what actually bound path length in practice; this is only a runaway
        // safety net (e.g. a pathological RR draw sequence) and is set well above any
        // reasonable MaxBounces the UI allows.
        private const int MaxTotalBounces = 32;

        // Bounce index (0-based) at which Russian roulette termination starts
        // considering killing the path - below this, every path continues
        // unconditionally, matching the reference path tracers' convention of not
        // rouletting very short paths (where the variance/cost tradeoff isn't worth it).
        private const int RussianRouletteStartBounce = 3;

        // Clamped survival-probability range - the lower bound keeps a path with
        // near-zero throughput (e.g. deep inside a dark diffuse cavity) from being kept
        // alive by an unbounded 1/p throughput correction; the upper bound guarantees
        // roulette can still terminate even a fully-white-throughput path.
        private const float MinSurvivalProbability = 0.05f;
        private const float MaxSurvivalProbability = 0.95f;

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
            Vec3 pixelHitPos = new Vec3(0f, 0f, 0f);
            float pixelSpecular = 0f;
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
                Vec3 sampleHitPos = new Vec3(0f, 0f, 0f);
                float sampleSpecular = 0f;

                // Sentinel until the first Trace() call returns a real value - the
                // primary/camera ray has no previous bounce to MIS against
                // (docs/SAMPLE15_PLAN.md Milestone M4, Payloads.DeltaOrPrimarySentinel).
                float bsdfPdf = Payloads.DeltaOrPrimarySentinel;

                for (int bounce = 0; bounce < MaxTotalBounces; bounce++)
                {
                    uint p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0, p8 = 0, p9 = 0, p10 = 0, p11 = 0, p12 = 0;
                    uint p13 = 0, p14 = 0, p15 = 0, p16 = 0, p17 = 0, p18 = 0, p19 = rng.State, p20 = Interop.FloatAsInt(bsdfPdf);
                    uint p21 = 0, p22 = 0, p23 = 0;
                    OptixTrace.Trace(
                        launchParams.traversable,
                        (rayOrigin.x, rayOrigin.y, rayOrigin.z),
                        (rayDir.x, rayDir.y, rayDir.z),
                        1e-3f,
                        1e20f,
                        0.0f,
                        0xff,
                        // Any-hit must run on radiance rays now that alpha-cutout
                        // materials exist (Sponza's leaf geometry) - __anyhit__radiance
                        // is what actually ignores the intersection for a
                        // below-threshold-alpha sample. DISABLE_ANYHIT (the old value
                        // here, from before alpha-cutout existed) would silently skip
                        // that test and shade the nearest triangle regardless of alpha.
                        OptixRayFlags.OPTIX_RAY_FLAG_NONE,
                        Payloads.RADIANCE_RAY_TYPE,
                        Payloads.RAY_TYPE_COUNT,
                        Payloads.RADIANCE_RAY_TYPE,
                        ref p0, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7, ref p8, ref p9, ref p10, ref p11, ref p12,
                        ref p13, ref p14, ref p15, ref p16, ref p17, ref p18, ref p19, ref p20, ref p21, ref p22, ref p23);

                    // Payload results come back through the ref-parameters passed to
                    // Trace() above, not through OptixPayloadInterop/GetVec3Registers -
                    // those call the optixGetPayload device intrinsic, which is only
                    // legal inside the hit/miss/anyhit programs that ran during this
                    // trace, not here in raygen after Trace() has already returned.
                    Vec3 radianceVec = new Vec3(
                        ILGPU.Interop.IntAsFloat(p0),
                        ILGPU.Interop.IntAsFloat(p1),
                        ILGPU.Interop.IntAsFloat(p2));
                    sampleRadiance += throughput * radianceVec;

                    // AOV guide buffers only ever reflect the primary ray's own hit
                    // (bounce 0), matching Sample11/12's convention - a mirror/glass
                    // primary hit still contributes its own normal/tint here, which is
                    // exactly what the denoiser needs to recognize that surface.
                    uint flag = p3;
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
                        sampleHitPos = new Vec3(
                            ILGPU.Interop.IntAsFloat(p21),
                            ILGPU.Interop.IntAsFloat(p22),
                            ILGPU.Interop.IntAsFloat(p23));
                        // Mirror/glass reflections and refractions have apparent motion
                        // that doesn't match the reflecting/refracting surface's own
                        // geometric motion - position-based reprojection is simply
                        // wrong for them (ghosting/smearing what should be a fresh
                        // view-dependent reflection every frame). BOUNCE_CONTINUE_MIRROR
                        // also covers the general GGX path's stochastic specular-lobe
                        // pick on a merely glossy (not perfectly mirrored) surface, not
                        // just true delta mirrors - an accepted approximation rather
                        // than threading a separate "is this a true delta lobe" payload.
                        sampleSpecular = (flag == Payloads.BOUNCE_CONTINUE_MIRROR || flag == Payloads.BOUNCE_CONTINUE_DIELECTRIC) ? 1f : 0f;
                    }

                    if (flag == Payloads.BOUNCE_TERMINAL)
                        break;

                    // Resume the RNG stream from wherever the closest-hit program's
                    // sampling left it (docs/SAMPLE15_PLAN.md Milestone M3) - a single
                    // continuous stream shared between raygen's own draws (pixel
                    // jitter, Russian roulette below) and every shading call.
                    rng.State = p19;
                    bsdfPdf = ILGPU.Interop.IntAsFloat(p20);

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

                    if (bounce + 1 >= launchParams.MaxBounces)
                        break;

                    // Russian roulette termination (docs/SAMPLE15_PLAN.md Design
                    // Decision 7) - replaces the old per-material-kind bounce budgets
                    // and the lengthSquared < 1e-6 throughput cutoff with a single
                    // unbiased stochastic rule: survival probability tracks the path's
                    // remaining energy, and a surviving path's throughput is corrected
                    // by 1/p so the estimator stays unbiased.
                    if (bounce >= RussianRouletteStartBounce)
                    {
                        float survival = XMath.Clamp(
                            XMath.Max(throughput.x, XMath.Max(throughput.y, throughput.z)),
                            MinSurvivalProbability, MaxSurvivalProbability);
                        if (rng.Next() > survival)
                            break;
                        throughput /= survival;
                    }
                }

                pixelColor += sampleRadiance;
                pixelNormal += sampleNormal;
                pixelAlbedo += sampleAlbedo;
                pixelHitPos += sampleHitPos;
                pixelSpecular += sampleSpecular;
            }
            pixelColor /= (float)numPixelSamples;
            pixelNormal /= (float)numPixelSamples;
            pixelAlbedo /= (float)numPixelSamples;
            pixelHitPos /= (float)numPixelSamples;

            long fbIndex = ix + (iy * camera.width);

            // Unlike ColorBuffer, AlbedoBuffer/NormalBuffer are NOT blended across
            // frames - each frame overwrites them fresh, matching Sample12's
            // devicePrograms.cs (these are the denoiser's AOV guide inputs, not part of
            // the progressively-accumulated image).
            launchParams.NormalBuffer[fbIndex] = new Vec4(pixelNormal.x, pixelNormal.y, pixelNormal.z, 1f);
            launchParams.AlbedoBuffer[fbIndex] = new Vec4(pixelAlbedo.x, pixelAlbedo.y, pixelAlbedo.z, 1f);

            // Reproject this pixel's primary-hit world position against the *previous*
            // frame's camera - the inverse of how this method's own screenX/screenY ->
            // rayDir construction above works. Single source of truth reused below for
            // both the new per-pixel reprojected color/count accumulation (replacing
            // the old FrameID-keyed running mean) and the Flow/FlowTrustworthiness
            // guide buffers for the OptiX temporal denoiser. A near-zero pixelNormal
            // means the primary ray missed all geometry (the miss program's own
            // normal=(0,0,0) sentinel, averaged across samples) - background pixels
            // have no world position to reproject, so they're always untrustworthy.
            bool primaryHit = pixelNormal.lengthSquared() > 1e-6f;

            // This frame's own view-space depth (distance along the camera's forward
            // axis) - stashed in ColorBuffer's alpha channel below so NEXT frame can
            // compare its own depth against it for disocclusion rejection, and reused
            // here for the current-vs-previous reprojection delta.
            Vec3 localDirCur = pixelHitPos - camera.origin;
            float camZCur = primaryHit ? Vec3.dot(localDirCur, camera.axis.z) : 0f;
            float currentDepth = camZCur;

            // Mirror/glass reflections and refractions have apparent motion that
            // doesn't match the reflecting/refracting surface's own geometric motion -
            // position-based reprojection is simply wrong for them, so never trust
            // history there (always a fresh sample). Costs nothing for a true delta
            // mirror (a perfect reflection has zero sample-to-sample variance to
            // reduce anyway) and only slightly increases noise on glass.
            bool isSpecularHit = pixelSpecular > 0f;

            Vec3 localDir = pixelHitPos - launchParams.PrevCameraOrigin;
            float camZ = Vec3.dot(localDir, launchParams.PrevCameraAxisZ);
            bool trustHistory = launchParams.PrevFrameValid != 0 && primaryHit && !isSpecularHit && camZ > 1e-4f && camZCur > 1e-4f;

            float prevPixelX = 0f, prevPixelY = 0f;
            if (trustHistory)
            {
                float camX = Vec3.dot(localDir, launchParams.PrevCameraAxisX);
                float camY = Vec3.dot(localDir, launchParams.PrevCameraAxisY);
                float prevScreenX = (camX * launchParams.PrevCameraPlaneDist) / (launchParams.PrevCameraAspectRatio * camZ);
                float prevScreenY = (camY * launchParams.PrevCameraPlaneDist) / camZ;
                float prevRawPixelX = ((prevScreenX + 1f) * 0.5f * camera.width) - 0.5f;
                float prevRawPixelY = ((prevScreenY + 1f) * 0.5f * camera.height) - 0.5f;

                // Reproject the SAME world-space hit through THIS frame's own camera
                // too, and use only the delta between the two projections rather than
                // prevRawPixel directly. Reprojecting via the previous camera alone
                // reconstructs a jitter-noisy fractional coordinate even on a
                // perfectly static camera (the primary ray itself was jittered for AA,
                // so the hit position never lands exactly at the pixel center) - that
                // would force a blurring resample of neighboring pixels every single
                // frame, and since each frame's history already contains the previous
                // frame's blur, it compounds into visible progressive softening. The
                // jitter term is present in both projections (same world point) and
                // cancels out in the subtraction, leaving just the camera-motion-
                // induced screen displacement - added onto this pixel's own exact
                // integer coordinate, so a static camera reprojects back to exactly
                // (ix, iy).
                float camXCur = Vec3.dot(localDirCur, camera.axis.x);
                float camYCur = Vec3.dot(localDirCur, camera.axis.y);
                float curScreenX = (camXCur * camera.cameraPlaneDist) / (camera.aspectRatio * camZCur);
                float curScreenY = (camYCur * camera.cameraPlaneDist) / camZCur;
                float curRawPixelX = ((curScreenX + 1f) * 0.5f * camera.width) - 0.5f;
                float curRawPixelY = ((curScreenY + 1f) * 0.5f * camera.height) - 0.5f;

                prevPixelX = ix + (prevRawPixelX - curRawPixelX);
                prevPixelY = iy + (prevRawPixelY - curRawPixelY);

                // Reject reprojected coordinates that landed off-screen (camera panned
                // past this content last frame) - sampling below would otherwise just
                // clamp to the edge and smear stale edge pixels inward.
                if (prevPixelX < -0.5f || prevPixelX > camera.width - 0.5f || prevPixelY < -0.5f || prevPixelY > camera.height - 0.5f)
                    trustHistory = false;
            }

            // Per-pixel reprojected accumulation (bounded/capped incremental mean -
            // "SMA-then-EMA", the standard production-TAA convergence technique):
            // history keeps growing (and the new-sample weight keeps shrinking) up to
            // MaxHistoryFrames, then holds at a fixed small blend-in rate forever -
            // unlike the old unbounded running mean, this never needs a hard reset on
            // camera motion, since untrustworthy pixels just reset their own count to 1
            // individually instead of the whole frame reverting to raw noise.
            Vec3 blendedColor = pixelColor;
            float newCount = 1f;
            if (trustHistory)
            {
                Vec4 prevSample = SampleBilinearColor(launchParams.PrevColorBuffer, camera.width, camera.height, prevPixelX, prevPixelY);
                float prevCount = SampleNearestScalar(launchParams.PrevAccumCountBuffer, camera.width, camera.height, prevPixelX, prevPixelY);

                // Disocclusion rejection: the reprojected screen location used to show
                // a surface at prevSample.w's depth - if that's very different from
                // this frame's own depth at the same location, the previous frame was
                // looking at a different surface entirely (a foreground object moved
                // and revealed background, or vice versa), and blending it in produces
                // exactly the ghosting/smearing this guards against. Relative threshold
                // (not absolute), since depth values scale with scene/object size.
                float depthDiff = XMath.Abs(currentDepth - prevSample.w);
                float depthThreshold = launchParams.DepthRejectionThreshold * XMath.Max(currentDepth, prevSample.w);
                if (depthDiff > depthThreshold)
                    trustHistory = false;

                if (trustHistory)
                {
                    Vec3 prevColor = new Vec3(prevSample.x, prevSample.y, prevSample.z);

                    // Real-time decay (see LaunchParams.HistoryDecayHalfLifeSeconds's
                    // own doc comment) - ages the effective count down every frame on a
                    // wall-clock schedule, so even a static/well-reprojected pixel keeps
                    // slowly forgetting old contributions instead of freezing at a
                    // fixed weight once it hits MaxHistoryFrames.
                    if (launchParams.HistoryDecayHalfLifeSeconds > 0f)
                        prevCount *= XMath.Pow(0.5f, launchParams.DeltaTimeSeconds / launchParams.HistoryDecayHalfLifeSeconds);

                    newCount = XMath.Min(prevCount + 1f, (float)launchParams.MaxHistoryFrames);
                    blendedColor = prevColor + ((pixelColor - prevColor) / newCount);
                }
            }

            launchParams.ColorBuffer[fbIndex] = new Vec4(blendedColor.x, blendedColor.y, blendedColor.z, currentDepth);
            launchParams.AccumCountBuffer[fbIndex] = newCount;

            Vec2 flow = trustHistory ? new Vec2(ix - prevPixelX, iy - prevPixelY) : new Vec2(0f, 0f);
            launchParams.FlowBuffer[fbIndex] = flow;
            launchParams.FlowTrustworthinessBuffer[fbIndex] = trustHistory ? 1f : 0f;
        }

        // 4-tap bilinear sample of the color+depth history at a (generally fractional)
        // pixel coordinate, clamped to the buffer edges. Nearest-neighbor sampling was
        // tried here first to kill a compounding-blur bug, but it traded that for a
        // worse artifact: nearest-neighbor resampling of a high-frequency repeating
        // pattern (e.g. a checkerboard floor) under any camera motion is the classic
        // cause of moire/aliasing - picking the "wrong" neighboring pixel right at a
        // checker-cell boundary flips to the wrong color entirely. Bilinear is safe
        // for the static-camera case now that the current-vs-previous delta
        // cancellation above makes a static camera reproject to an EXACT integer pixel
        // coordinate (fx=fy=0 exactly, so this degenerates to a pure single-tap fetch
        // with zero blending) - the original compounding-blur bug was the jitter term
        // not being cancelled, not bilinear filtering itself. Vec4/float have no
        // operator overloads in this sample, so this is written component-by-component
        // rather than via a generic lerp.
        static unsafe Vec4 SampleBilinearColor(Vec4* buffer, int width, int height, float px, float py)
        {
            float cx = XMath.Clamp(px, 0f, width - 1f);
            float cy = XMath.Clamp(py, 0f, height - 1f);
            int x0 = (int)cx, y0 = (int)cy;
            int x1 = XMath.Min(x0 + 1, width - 1);
            int y1 = XMath.Min(y0 + 1, height - 1);
            float fx = cx - x0, fy = cy - y0;

            Vec4 c00 = buffer[x0 + (y0 * width)];
            Vec4 c10 = buffer[x1 + (y0 * width)];
            Vec4 c01 = buffer[x0 + (y1 * width)];
            Vec4 c11 = buffer[x1 + (y1 * width)];

            float r = ((c00.x * (1f - fx)) + (c10.x * fx)) * (1f - fy) + (((c01.x * (1f - fx)) + (c11.x * fx)) * fy);
            float g = ((c00.y * (1f - fx)) + (c10.y * fx)) * (1f - fy) + (((c01.y * (1f - fx)) + (c11.y * fx)) * fy);
            float b = ((c00.z * (1f - fx)) + (c10.z * fx)) * (1f - fy) + (((c01.z * (1f - fx)) + (c11.z * fx)) * fy);
            float a = ((c00.w * (1f - fx)) + (c10.w * fx)) * (1f - fy) + (((c01.w * (1f - fx)) + (c11.w * fx)) * fy);
            return new Vec4(r, g, b, a);
        }

        // Nearest-neighbor sample for AccumCount - a confidence/history-length scalar,
        // not visible color, so there's no moire concern; kept simple rather than
        // blending "how many frames" across a boundary.
        static unsafe float SampleNearestScalar(float* buffer, int width, int height, float px, float py)
        {
            int x = XMath.Clamp((int)XMath.Round(px), 0, width - 1);
            int y = XMath.Clamp((int)XMath.Round(py), 0, height - 1);
            return buffer[x + (y * width)];
        }
    }
}
