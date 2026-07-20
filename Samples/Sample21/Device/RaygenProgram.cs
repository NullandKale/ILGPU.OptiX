using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;
using ILGPU.OptiX.Device;
using ILGPU.OptiX.DeviceApi;
using Sample21.Core;
using System;
using System.Numerics;

namespace Sample21
{
    /// <summary>
    /// The ray-generation program: camera ray setup, the iterative bounce loop, and
    /// progressive frame accumulation into the HDR color buffer plus the denoiser's
    /// AOV guide buffers.
    /// </summary>
    public static class RaygenProgram
    {
        // Hard safety cap on total Trace() calls per sample -
        // launchParams.MaxBounces and Russian roulette termination are
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

        // NRC adaptive-footprint-handoff epsilon floors - no VkNRC equivalent needed
        // there, since its pure GGX/VNDF sampling never produces a near-zero
        // cosine or pdf the way this sample's normal-mapped geometry/delta lobes can).
        // An unguarded divide here risks NaN entering nrcSqrtASum, which silently
        // disables the adaptive stop test for the rest of that path (bounded only by
        // MaxBounces, not incorrect but wasteful) and would otherwise propagate into
        // EvalRecord/TrainRecord and from there into gradients.
        private const float NrcCosineEpsilon = 1e-4f;
        private const float NrcPdfEpsilon = 1e-6f;

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
            Vec3 pixelHitPos = new Vec3(0f, 0f, 0f);
            float pixelSpecular = 0f;

            // NRC - purely additive, riding along the existing path tracer as a side
            // channel; does not change pixelColor. See Device/Records.cs. Only the LAST
            // pixel sample's snapshot is kept when NumPixelSamples > 1 - a deliberate
            // simplification, not a correctness issue (NumPixelSamples defaults to 1;
            // TAA/reprojection is the sample's actual noise-reduction mechanism across
            // frames either way).
            bool nrcEvalActive = false;
            Vec3 nrcEvalThroughput = default, nrcEvalPosition = default, nrcEvalScatteredDir = default, nrcEvalNormal = default, nrcEvalAlbedo = default;
            float nrcEvalRoughness = 0f, nrcEvalMetallic = 0f;
            bool nrcTrainEmitted = false;
            Vec3 nrcTrainBias = default, nrcTrainFactor = default;
            Vec3 nrcTrainEndpointPos = default, nrcTrainEndpointScatteredDir = default, nrcTrainEndpointNormal = default, nrcTrainEndpointAlbedo = default;
            float nrcTrainEndpointRoughness = 0f, nrcTrainEndpointMetallic = 0f;
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
                // (Payloads.DeltaOrPrimarySentinel).
                float bsdfPdf = Payloads.DeltaOrPrimarySentinel;

                // NRC per-sample suffix state (see the pixel-level nrc* locals above) -
                // suffixFactor/suffixBias track radiance/throughput RELATIVE to the
                // handoff point (starting at identity), not the camera-relative
                // throughput/sampleRadiance the rest of this loop already computes -
                // this is what lets the self-training target's cache query be a clean
                // multiply with no division-by-near-zero-throughput risk.
                bool nrcInTrainSuffix = false;
                Vec3 nrcSuffixFactor = new Vec3(1f, 1f, 1f);
                Vec3 nrcSuffixBias = new Vec3(0f, 0f, 0f);

                // NRC adaptive ray-footprint handoff state (VkNRC path_tracer.comp's
                // c_a0/sqrt_a_sum) - nrcHandoffDone latches true the bounce the primary
                // path hands off to the cache (main radiance accumulation stops there);
                // nrcSuffixSqrtASum is a SEPARATE accumulator reusing the same nrcC_a0
                // threshold for the training suffix's own continuation, mirroring
                // VkNRC's ExtendedPathTrace resetting sqrt_a_sum but not c_a0.
                bool nrcHandoffDone = false;
                float nrcC_a0 = 0f, nrcSqrtASum = 0f, nrcSuffixSqrtASum = 0f;

                for (int bounce = 0; bounce < MaxTotalBounces; bounce++)
                {
                    // hit_{k-1} (or the camera at bounce 0) - captured before Trace()
                    // updates rayOrigin/rayDir below, so the NRC adaptive-footprint
                    // block later in this iteration can compute this bounce's own
                    // segment length (hit_{k-1} -> hit_k) against it.
                    Vec3 nrcHitPrev = rayOrigin;

                    var payload = new Payloads.RadiancePayload
                    {
                        RngState = rng.State,
                        BsdfPdf = bsdfPdf,
                    };

                    Vector3 origin = new Vector3(rayOrigin.x, rayOrigin.y, rayOrigin.z);
                    Vector3 direction = new Vector3(rayDir.x, rayDir.y, rayDir.z);

                    // Shader Execution Reordering (SER), always on: traverse first
                    // (finds the hit but does not yet run any hit/miss program), let the
                    // driver batch coherent hits together, then invoke the resulting
                    // hitgroup/miss program - same SBT wiring a plain OptixTrace.Trace
                    // call would use. Purely a scheduling hint - identical payload
                    // result either way, just a throughput win on Sponza's divergent
                    // mixed-material bounces.
                    OptixHitObject.Traverse(
                        launchParams.traversable,
                        origin,
                        direction,
                        1e-3f,
                        1e20f,
                        ref payload,
                        // Any-hit must run on radiance rays since alpha-cutout materials
                        // exist (Sponza's leaf geometry) - __anyhit__radiance is what
                        // actually ignores the intersection for a below-threshold-alpha
                        // sample. DISABLE_ANYHIT here would silently skip that test and
                        // shade the nearest triangle regardless of alpha.
                        rayFlags: OptixRayFlags.None,
                        sbtOffset: Payloads.RADIANCE_RAY_TYPE,
                        sbtStride: Payloads.RAY_TYPE_COUNT,
                        missSbtIndex: Payloads.RADIANCE_RAY_TYPE);
                    OptixReorder.Invoke();
                    OptixHitObject.Invoke(ref payload);

                    // Payload results come back through the struct passed by ref to
                    // Trace() above, not through OptixPayloadInterop/GetVec3Registers -
                    // those call the optixGetPayload device intrinsic, which is only
                    // legal inside the hit/miss/anyhit programs that ran during this
                    // trace, not here in raygen after Trace() has already returned.
                    Vec3 radianceVec = new Vec3(payload.RadianceX, payload.RadianceY, payload.RadianceZ);
                    // Gated on NOT having handed off yet - once the adaptive test below
                    // hands this path off to the NRC cache, CompositeProgram.cs adds the
                    // cache's own prediction on top of whatever RawColorBuffer holds at
                    // that point; letting the path keep contributing its own radiance
                    // past the handoff would double-count everything the cache is meant
                    // to replace. Training-suffix bounces (nrcInTrainSuffix) always run
                    // AFTER nrcHandoffDone is already true, so this naturally excludes
                    // them too - their radiance only ever feeds nrcSuffixBias below, not
                    // the primary render.
                    if (!nrcHandoffDone)
                        sampleRadiance += throughput * radianceVec;

                    // NRC: accumulate this bounce's contribution into the training
                    // suffix's bias, in the same "throughput BEFORE this bounce's own
                    // tint" convention sampleRadiance itself just used above, but
                    // relative to the handoff point (nrcSuffixFactor) instead of the
                    // camera (throughput).
                    if (nrcInTrainSuffix)
                        nrcSuffixBias += nrcSuffixFactor * radianceVec;

                    // AOV guide buffers only ever reflect the primary ray's own hit
                    // (bounce 0) - a mirror/glass primary hit still contributes its own
                    // normal/tint here, which is exactly what the denoiser needs to
                    // recognize that surface.
                    uint flag = payload.Flag;
                    if (bounce == 0)
                    {
                        sampleNormal = new Vec3(payload.NormalX, payload.NormalY, payload.NormalZ);
                        sampleAlbedo = new Vec3(payload.AlbedoX, payload.AlbedoY, payload.AlbedoZ);
                        sampleHitPos = new Vec3(payload.HitPosX, payload.HitPosY, payload.HitPosZ);
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
                    {
                        // Path ended naturally (miss/pure-emissive hit) mid-suffix -
                        // bias already holds the exact terminal contribution (added
                        // above, before this check), so nothing left to predict.
                        if (nrcInTrainSuffix)
                        {
                            nrcTrainEmitted = true;
                            nrcTrainBias = nrcSuffixBias;
                            nrcTrainFactor = new Vec3(0f, 0f, 0f);
                            nrcInTrainSuffix = false;
                        }
                        break;
                    }

                    // Resume the RNG stream from wherever the closest-hit program's
                    // sampling left it - a single
                    // continuous stream shared between raygen's own draws (pixel
                    // jitter, Russian roulette below) and every shading call.
                    rng.State = payload.RngState;
                    bsdfPdf = payload.BsdfPdf;

                    Vec3 bounceTint = new Vec3(payload.TintX, payload.TintY, payload.TintZ);
                    throughput *= bounceTint;
                    rayOrigin = new Vec3(payload.NewOriginX, payload.NewOriginY, payload.NewOriginZ);
                    rayDir = new Vec3(payload.NewDirX, payload.NewDirY, payload.NewDirZ);

                    // NRC: adaptive ray-footprint handoff (VkNRC path_tracer.comp's
                    // c_a0/sqrt_a_sum heuristic, ported exactly) - the path hands off
                    // to the cache once accumulated footprint growth
                    // exceeds a threshold derived from the primary hit's own projected
                    // footprint (nrcC_a0, computed once at bounce 0), or the bounce
                    // budget runs out - whichever comes first, matching VkNRC's own
                    // `bounce < MAX_BOUNCE && sqrt_a_sum^2 <= c_a0` loop condition
                    // (MAX_BOUNCE ported as this sample's own live-tunable
                    // launchParams.MaxBounces rather than a second hardcoded ceiling).
                    //
                    // dist2/cosine both describe THIS bounce's own segment
                    // (hitPrev -> hit_bounce) and the sample that produced it
                    // (bsdfPdf/the normal+direction at hit_bounce) - VkNRC's invariant is
                    // that the distance INTO a hit pairs with the pdf/cosine of the
                    // sample OUT of that SAME hit, both of which fall within this one
                    // iteration in this sample's uniform per-bounce loop structure
                    // (unlike VkNRC's own "one BRDFStep before the loop" structure).
                    if (launchParams.NrcEnabled != 0)
                    {
                        Vec3 hitNormal = new Vec3(payload.NormalX, payload.NormalY, payload.NormalZ);
                        float dist2 = (rayOrigin - nrcHitPrev).lengthSquared();
                        float cosine = XMath.Max(XMath.Abs(Vec3.dot(hitNormal, rayDir)), NrcCosineEpsilon);

                        if (!nrcHandoffDone)
                        {
                            if (bounce == 0)
                            {
                                // c_a0: the primary hit's own projected footprint -
                                // VkNRC: C * dist2(camera->hit0) / (4*PI*cosine_at_hit0),
                                // no pdf term (this is the base unit the loop's later
                                // pdf-divided terms compare against, not itself an
                                // importance-sampled quantity).
                                nrcC_a0 = launchParams.NrcFootprintConstant * dist2 / (4f * MathF.PI * cosine);
                                nrcSqrtASum = 0f;
                            }
                            else
                            {
                                // Delta/mirror lobes (bsdfPdf == Payloads.DeltaOrPrimarySentinel,
                                // i.e. <= 0) contribute 0 - physically correct (a perfect
                                // mirror has no footprint growth), not just a safe default.
                                if (bsdfPdf > 0f)
                                    nrcSqrtASum += XMath.Sqrt(dist2 / (XMath.Max(bsdfPdf, NrcPdfEpsilon) * cosine));

                                bool footprintExceeded = (nrcSqrtASum * nrcSqrtASum) > nrcC_a0;
                                bool outOfBudget = (bounce + 1) >= launchParams.MaxBounces;
                                if (footprintExceeded || outOfBudget)
                                {
                                    // Handoff: snapshot the hit this path is continuing
                                    // FROM - its own BRDF tint is already folded into
                                    // `throughput` above. sampleRadiance's own gate
                                    // above already stops accumulating starting NEXT
                                    // iteration (this bounce's own radiance, added
                                    // earlier this iteration, is correctly included -
                                    // matches VkNRC's `bias` including the handoff hit).
                                    nrcHandoffDone = true;
                                    nrcEvalActive = true;
                                    nrcEvalThroughput = throughput;
                                    nrcEvalPosition = rayOrigin;
                                    nrcEvalScatteredDir = rayDir;
                                    nrcEvalNormal = hitNormal;
                                    nrcEvalRoughness = payload.RoughnessOut;
                                    nrcEvalMetallic = payload.MetallicOut;
                                    nrcEvalAlbedo = new Vec3(payload.AlbedoX, payload.AlbedoY, payload.AlbedoZ);

                                    if (rng.Next() < launchParams.NrcTrainProbability)
                                    {
                                        nrcInTrainSuffix = true;
                                        nrcSuffixFactor = new Vec3(1f, 1f, 1f);
                                        nrcSuffixBias = new Vec3(0f, 0f, 0f);
                                        nrcSuffixSqrtASum = 0f;   // reuses nrcC_a0, not a fresh threshold
                                    }
                                }
                            }
                        }
                        else if (nrcInTrainSuffix)
                        {
                            // Training-suffix continuation: same per-iteration dist2/
                            // cosine/pdf, same nrcC_a0 threshold, separate accumulator -
                            // mirrors VkNRC's ExtendedPathTrace second loop (sqrt_a_sum
                            // reset, c_a0 reused).
                            nrcSuffixFactor *= bounceTint;
                            if (bsdfPdf > 0f)
                                nrcSuffixSqrtASum += XMath.Sqrt(dist2 / (XMath.Max(bsdfPdf, NrcPdfEpsilon) * cosine));

                            bool suffixFootprintExceeded = (nrcSuffixSqrtASum * nrcSuffixSqrtASum) > nrcC_a0;
                            bool suffixOutOfBudget = (bounce + 1) >= launchParams.MaxBounces;
                            if (suffixFootprintExceeded || suffixOutOfBudget)
                            {
                                // Suffix completed naturally (didn't terminate early) -
                                // the endpoint is THIS bounce's hit, and the
                                // self-training-target stage will query the cache there.
                                nrcTrainEmitted = true;
                                nrcTrainBias = nrcSuffixBias;
                                nrcTrainFactor = nrcSuffixFactor;
                                nrcTrainEndpointPos = rayOrigin;
                                nrcTrainEndpointScatteredDir = rayDir;
                                nrcTrainEndpointNormal = hitNormal;
                                nrcTrainEndpointRoughness = payload.RoughnessOut;
                                nrcTrainEndpointMetallic = payload.MetallicOut;
                                nrcTrainEndpointAlbedo = new Vec3(payload.AlbedoX, payload.AlbedoY, payload.AlbedoZ);
                                nrcInTrainSuffix = false;
                            }
                        }
                    }

                    // Primary path resolved via the cache and no training suffix is (or
                    // was) in flight for this sample - nothing left to trace.
                    if (nrcHandoffDone && !nrcInTrainSuffix)
                        break;

                    if (bounce + 1 >= launchParams.MaxBounces)
                    {
                        // Path terminated by the bounce budget before the suffix
                        // finished naturally - bias already holds the exact remaining
                        // radiance (nothing left to predict), so emit with Factor=0 (the
                        // self-training-target stage's cache query is multiplied away).
                        if (nrcInTrainSuffix)
                        {
                            nrcTrainEmitted = true;
                            nrcTrainBias = nrcSuffixBias;
                            nrcTrainFactor = new Vec3(0f, 0f, 0f);
                            nrcInTrainSuffix = false;
                        }
                        break;
                    }

                    // Russian roulette termination - a single unbiased stochastic rule:
                    // survival probability tracks the path's remaining energy, and a
                    // surviving path's throughput is corrected by 1/p so the estimator
                    // stays unbiased.
                    if (bounce >= RussianRouletteStartBounce)
                    {
                        float survival = XMath.Clamp(
                            XMath.Max(throughput.x, XMath.Max(throughput.y, throughput.z)),
                            MinSurvivalProbability, MaxSurvivalProbability);
                        if (rng.Next() > survival)
                        {
                            // Killed by Russian roulette mid-suffix - same "bias is
                            // exact, no cache query needed" treatment as the other two
                            // early-exit sites above.
                            if (nrcInTrainSuffix)
                            {
                                nrcTrainEmitted = true;
                                nrcTrainBias = nrcSuffixBias;
                                nrcTrainFactor = new Vec3(0f, 0f, 0f);
                                nrcInTrainSuffix = false;
                            }
                            break;
                        }
                        throughput /= survival;
                        if (nrcInTrainSuffix)
                            nrcSuffixFactor /= survival;
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

            // NRC record emission - see this method's own NRC comment near the top.
            // EvalRecords has exactly one slot per pixel;
            // TrainRecords is a pooled buffer with an atomic counter since only a
            // TrainProbability fraction of pixels emit one.
            if (launchParams.NrcEvalRecords.IsValid)
            {
                // Positions are remapped into VkNRC's [-1,1] scene-local range HERE, the
                // single point where they enter NRC records (gradient/composite/
                // self-training all encode these stored fields as-is) - see
                // LaunchParams.NrcSceneCenter's doc comment for why this remap exists
                // (VkNRC bakes it into the vertices at model load; this port doesn't).
                Vec3 nrcEvalPositionScaled =
                    (nrcEvalPosition - launchParams.NrcSceneCenter) * launchParams.NrcSceneInvExtent;

                launchParams.NrcEvalRecords[fbIndex] = nrcEvalActive
                    ? new EvalRecord
                    {
                        Active = 1,
                        Throughput = nrcEvalThroughput,
                        Position = nrcEvalPositionScaled,
                        ScatteredDir = nrcEvalScatteredDir,
                        Normal = nrcEvalNormal,
                        Roughness = nrcEvalRoughness,
                        Metallic = nrcEvalMetallic,
                        Albedo = nrcEvalAlbedo,
                    }
                    : default;

                if (nrcTrainEmitted)
                {
                    // Random batch assignment (VkNRC path_tracer.comp:
                    // `batch = clamp(uint(RNGNext()*4), 0, 3)`) - disjoint per-frame
                    // training batches, not one pool reprocessed N times. One
                    // counter/slice per batch.
                    // The batch count is the runtime performance knob
                    // LaunchParams.NrcTrainBatchCount (<= NrcConstants.TrainBatchesPerFrame,
                    // which still sizes the buffers), so fewer trained batches per frame
                    // concentrates records rather than wasting them on untrained slices.
                    int batch = (int)(rng.Next() * launchParams.NrcTrainBatchCount);
                    if (batch >= launchParams.NrcTrainBatchCount)
                        batch = launchParams.NrcTrainBatchCount - 1;
                    if (batch < 0)
                        batch = 0;

                    int slot = Atomic.Add(ref launchParams.NrcTrainCounter[batch], 1);
                    if (slot < NrcConstants.TrainBatchSize)
                    {
                        launchParams.NrcTrainRecords[(batch * NrcConstants.TrainBatchSize) + slot] = new TrainRecord
                        {
                            Bias = nrcTrainBias,
                            Factor = nrcTrainFactor,
                            EndpointPosition =
                                (nrcTrainEndpointPos - launchParams.NrcSceneCenter) * launchParams.NrcSceneInvExtent,
                            EndpointScatteredDir = nrcTrainEndpointScatteredDir,
                            EndpointNormal = nrcTrainEndpointNormal,
                            EndpointRoughness = nrcTrainEndpointRoughness,
                            EndpointMetallic = nrcTrainEndpointMetallic,
                            EndpointAlbedo = nrcTrainEndpointAlbedo,
                            StartPosition = nrcEvalPositionScaled,
                            StartScatteredDir = nrcEvalScatteredDir,
                            StartNormal = nrcEvalNormal,
                            StartRoughness = nrcEvalRoughness,
                            StartMetallic = nrcEvalMetallic,
                            StartAlbedo = nrcEvalAlbedo,
                        };
                    }
                }
            }

            // Unlike ColorBuffer, AlbedoBuffer/NormalBuffer are NOT blended across
            // frames - each frame overwrites them fresh (these are the denoiser's AOV
            // guide inputs, not part of the progressively-accumulated image).
            launchParams.NormalBuffer[fbIndex] = new Vec4(pixelNormal.x, pixelNormal.y, pixelNormal.z, 1f);
            launchParams.AlbedoBuffer[fbIndex] = new Vec4(pixelAlbedo.x, pixelAlbedo.y, pixelAlbedo.z, 1f);

            // Reproject this pixel's primary-hit world position against the *previous*
            // frame's camera - the inverse of how this method's own screenX/screenY ->
            // rayDir construction above works. Feeds the per-pixel reprojected
            // color/count accumulation TaaResolveKernel.cs finishes. A near-zero
            // pixelNormal means the primary ray
            // missed all geometry (the miss program's own normal=(0,0,0) sentinel,
            // averaged across samples) - background pixels have no world position to
            // reproject, so they're always untrustworthy.
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

            // Disocclusion rejection resolves trustHistory the rest of the way here
            // (needs this pixel's own reprojection math above plus the previous
            // frame's stored depth), but the actual color blend - and the neighborhood
            // clamp that guards it - is deferred to TaaResolveKernel.cs, run as a
            // separate pass once every pixel in this frame has finished writing below.
            // OptiX raygen threads have no ordering/synchronization guarantee across
            // pixels in the same launch, so reading a neighbor's *this-frame* color
            // here (which the clamp needs) would be a race; TaaResolveKernel.cs runs
            // only after this whole launch has completed, when that read is safe.
            if (trustHistory)
            {
                Vec4 prevSample = SampleBilinearColor(launchParams.PrevColorBuffer, camera.width, camera.height, prevPixelX, prevPixelY);

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
            }

            launchParams.RawColorBuffer[fbIndex] = new Vec4(pixelColor.x, pixelColor.y, pixelColor.z, currentDepth);
            launchParams.ReprojCoordBuffer[fbIndex] = trustHistory
                ? new Vec2(prevPixelX, prevPixelY)
                : new Vec2(TaaResolveKernel.NoHistorySentinel, TaaResolveKernel.NoHistorySentinel);
        }

        // 4-tap bilinear sample of the color+depth history at a (generally fractional)
        // pixel coordinate, clamped to the buffer edges. Bilinear, not nearest-neighbor:
        // nearest-neighbor resampling of a high-frequency repeating pattern (e.g. a
        // checkerboard floor) under any camera motion is the classic cause of
        // moire/aliasing - picking the "wrong" neighboring pixel right at a checker-cell
        // boundary flips to the wrong color entirely. Bilinear costs nothing extra for
        // the static-camera case, since the current-vs-previous delta cancellation
        // above makes a static camera reproject to an EXACT integer pixel coordinate
        // (fx=fy=0 exactly, so this degenerates to a pure single-tap fetch with zero
        // blending). Vec4/float have no operator overloads in this sample, so this is
        // written component-by-component rather than via a generic lerp.
        static Vec4 SampleBilinearColor(OptixDeviceView<Vec4> buffer, int width, int height, float px, float py)
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
    }
}
