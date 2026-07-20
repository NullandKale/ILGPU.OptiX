using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.DeviceApi;
using System.Numerics;

namespace Sample21
{
    /// <summary>
    /// Everything about the shadow ray type in one place: the trace-call-site wrapper
    /// (<see cref="Trace"/>, called by <see cref="NextEventEstimation"/>) and the three
    /// OptiX device programs it invokes. Multi-hit colored transmittance - walks
    /// through transparent occluders via
    /// <see cref="__anyhit__shadow"/>'s optixIgnoreIntersection, accumulating
    /// TransmissionColor*Transmission per hit up to MaxRefractions, unlike a plain
    /// binary occlusion test.
    ///
    /// PAYLOAD CONTRACT - ShadowPayload (transmittance xyz + hit count):
    ///   Seed (Trace, before OptixTrace.Trace):  (1,1,1), hitCount=0
    ///   __anyhit__shadow   (cutout/glass materials ONLY - opaque materials have
    ///                       DISABLE_ANYHIT geometry flags, see AccelStructureBuilder):
    ///     (cutout below threshold):  no write, IgnoreIntersection()
    ///     (cutout-opaque hit):       writes (0,0,0), Terminate()
    ///     (glass pass-through):      writes accumulated transmittance, IgnoreIntersection()
    ///     (glass terminal):          writes accumulated transmittance, hit accepted
    ///   __closesthit__shadow (the committed hit, TerminateOnFirstHit = the first
    ///                         accepted one):
    ///     (opaque material):         writes (0,0,0) - THE occlusion answer for every
    ///                                DISABLE_ANYHIT material, whose hits run no anyhit
    ///     (glass material):          no write - anyhit's accumulated value stands
    ///   __miss__shadow:              no write - reads back the seed/accumulated value
    ///   Reader: Trace's return value / NextEventEstimation.SampleDirectLighting
    ///
    /// Payload registers are read/write-in-place for the whole trace (see
    /// example/optix7course/example09_shadowRays/devicePrograms.cu, the reference this
    /// was ported from): a hit that runs no program at all leaves whatever value the
    /// caller seeded before the trace untouched. The reference exploits this by seeding
    /// its payload to 0 ("occluded") and only ever writing 1 from its miss program.
    /// This port instead seeds transmittance to 1 (so __anyhit__shadow's glass-
    /// accumulation multiply has an identity starting value to work from) - which is
    /// why the opaque closesthit row and __miss__shadow both need their own explicit
    /// rows above instead of relying on an implicit default: omitting the opaque write
    /// would report light as fully visible straight through solid geometry, and
    /// letting __miss__shadow overwrite the seed would discard any already-accumulated
    /// glass transmittance whenever the ray reaches tmax without a further hit.
    ///
    /// Ray flags: TerminateOnFirstHit - any accepted hit answers an occlusion query, no
    /// closest-hit sorting needed - plus closesthit ENABLED, since it's the only
    /// program that runs at all for opaque (DISABLE_ANYHIT) hits.
    /// </summary>
    public static class ShadowRay
    {
        internal unsafe static Vec3 Trace(LaunchParams launchParams, Vec3 surfPos, Vec3 outwardNormal, Vec3 lightDir, float lightDist)
        {
            var payload = new Payloads.ShadowPayload
            {
                TransmittanceX = 1f,
                TransmittanceY = 1f,
                TransmittanceZ = 1f,
            };

            Vector3 origin = new Vector3(
                surfPos.x + (1e-3f * outwardNormal.x),
                surfPos.y + (1e-3f * outwardNormal.y),
                surfPos.z + (1e-3f * outwardNormal.z));
            Vector3 direction = new Vector3(lightDir.x, lightDir.y, lightDir.z);

            OptixTrace.Trace(
                launchParams.traversable,
                origin,
                direction,
                1e-3f,
                lightDist * (1f - 1e-3f),
                ref payload,
                rayFlags:
                    // Occlusion query: ANY accepted hit blocks (fully, or with the
                    // glass transmittance anyhit already accumulated), so stop
                    // traversal at the first one instead of searching for the closest.
                    // Closest-hit stays ENABLED - it is the only program that runs for
                    // hits on DISABLE_ANYHIT (opaque) materials, and carries their
                    // occlusion answer (see this class's payload contract).
                    OptixRayFlags.TerminateOnFirstHit,
                sbtOffset: Payloads.SHADOW_RAY_TYPE,
                sbtStride: Payloads.RAY_TYPE_COUNT,
                missSbtIndex: Payloads.SHADOW_RAY_TYPE);

            return new Vec3(payload.TransmittanceX, payload.TransmittanceY, payload.TransmittanceZ);
        }

        // Accumulates colored transmittance through transparent occluders instead of a
        // plain binary occlusion test. Alpha-cutout geometry (see
        // RadianceRay.__anyhit__radiance) gets the same treatment here first, so
        // alpha-cut leaves correctly let light/shadow rays pass through their
        // punched-out areas too, not just camera rays.
        //
        // Opaque hits must explicitly zero the payload and terminate here - see this
        // class's own payload-contract doc comment above for why an implicit "accept
        // and fall through" isn't enough. OptixTerminateRay.Terminate() (rather than
        // just letting the accepted hit fall out of any further traversal on its own)
        // also skips searching for a closer hit that can't change this ray's answer
        // anyway - an opaque hit fully blocks regardless of which opaque surface was
        // nearest.
        public unsafe static void __anyhit__shadow(LaunchParams launchParams)
        {
            MaterialSbtData* sbtData = (MaterialSbtData*)OptixGetSbtDataPointer.Value;

            if (sbtData->AlphaCutoff > 0f && sbtData->BaseColorTexture != 0 &&
                MaterialShading.TryGetTriangleUV(launchParams, out Vec2 uv) &&
                MaterialShading.SampleAlpha(sbtData, uv) < sbtData->AlphaCutoff)
            {
                OptixIgnoreIntersection.Ignore();
                return;
            }

            if (sbtData->Transmission <= 0f)
            {
                OptixPayload.Write(new Payloads.ShadowPayload
                {
                    TransmittanceX = 0f,
                    TransmittanceY = 0f,
                    TransmittanceZ = 0f,
                });
                OptixTerminateRay.Terminate();
                return;
            }

            const uint maxRefractions = 2;
            const float transmittanceEpsilon = 1e-6f;

            var payload = OptixPayload.Read<Payloads.ShadowPayload>();
            Vec3 transmittance = new Vec3(payload.TransmittanceX, payload.TransmittanceY, payload.TransmittanceZ);
            uint hitCount = payload.HitCount + 1;

            transmittance *= sbtData->TransmissionColor * sbtData->Transmission;

            payload.TransmittanceX = transmittance.x;
            payload.TransmittanceY = transmittance.y;
            payload.TransmittanceZ = transmittance.z;
            payload.HitCount = hitCount;
            OptixPayload.Write(in payload);

            if (hitCount < maxRefractions && transmittance.lengthSquared() > transmittanceEpsilon)
                OptixIgnoreIntersection.Ignore();
            // else: let this hit stick (accept & terminate) - the accumulated
            // transmittance above is the final answer, matching the reference's
            // early-out-on-negligible-transmittance / MaxRefractions-exceeded behavior.
        }

        // Runs for the committed (first accepted, per TerminateOnFirstHit) hit. Opaque
        // materials carry DISABLE_ANYHIT geometry flags (AccelStructureBuilder), so for
        // them this is the ONLY program that runs - it must write the "fully blocked"
        // answer. Glass materials (Transmission > 0) only commit a hit when their
        // anyhit accepted a terminal one, having already written the accumulated
        // transmittance as the final answer - leave the payload untouched for those.
        public unsafe static void __closesthit__shadow(LaunchParams launchParams)
        {
            MaterialSbtData* sbtData = (MaterialSbtData*)OptixGetSbtDataPointer.Value;
            if (sbtData->Transmission <= 0f)
            {
                OptixPayload.Write(new Payloads.ShadowPayload
                {
                    TransmittanceX = 0f,
                    TransmittanceY = 0f,
                    TransmittanceZ = 0f,
                });
            }
        }

        // A true no-op - see this class's own payload-contract doc comment above for
        // why overwriting the payload here would be wrong. Reaching miss means the ray
        // reached its tmax without hitting any further
        // opaque blocker - but it may have already passed through one or more glass
        // surfaces, whose accumulated (< 1) transmittance __anyhit__shadow already
        // wrote into the payload. The caller (Trace, above) already seeds the payload
        // to (1,1,1) before the trace, so a ray that hits nothing at all reads back
        // exactly that same untouched seed with no write needed here.
        public static void __miss__shadow(LaunchParams launchParams)
        { }
    }
}
