using ILGPU;
using ILGPU.OptiX;

namespace Sample13
{
    /// <summary>
    /// The radiance miss program (sky gradient) and the shadow-ray program set
    /// implementing colored multi-hit transmittance shadows.
    /// </summary>
    public static class MissAndShadowPrograms
    {
        public static void __miss__radiance(LaunchParams launchParams)
        {
            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            float t = 0.5f * (dy + 1f);
            Vec3 sky = launchParams.BackgroundBottom + (t * (launchParams.BackgroundTop - launchParams.BackgroundBottom));
            Payloads.SetTerminalPayload(sky, new Vec3(0f, 0f, 0f), new Vec3(0f, 0f, 0f));
        }

        public static void __anyhit__radiance(LaunchParams launchParams)
        { }

        public static void __closesthit__shadow(LaunchParams launchParams)
        { }

        // Accumulates colored transmittance through transparent occluders instead of a
        // plain binary occlusion test (docs/SAMPLE13_PLAN.md design (f)). Opaque hits are
        // a no-op here, which leaves OptiX's default any-hit behavior in effect: accept
        // the hit and terminate the ray, i.e. fully block the light - exactly matching
        // the reference's "opaque fully blocks" rule.
        public unsafe static void __anyhit__shadow(LaunchParams launchParams)
        {
            MaterialSbtData* sbtData = (MaterialSbtData*)OptixGetSbtDataPointer.Value;
            if (sbtData->Transparency <= 0f)
                return;

            const uint maxRefractions = 2;
            const float transmittanceEpsilon = 1e-6f;

            Vec3 transmittance = new Vec3(
                Interop.IntAsFloat(OptixPayload.Payload0),
                Interop.IntAsFloat(OptixPayload.Payload1),
                Interop.IntAsFloat(OptixPayload.Payload2));
            uint hitCount = OptixPayload.Payload3 + 1;

            transmittance *= sbtData->TransmissionColor * sbtData->Transparency;

            OptixPayload.Payload0 = Interop.FloatAsInt(transmittance.x);
            OptixPayload.Payload1 = Interop.FloatAsInt(transmittance.y);
            OptixPayload.Payload2 = Interop.FloatAsInt(transmittance.z);
            OptixPayload.Payload3 = hitCount;

            if (hitCount < maxRefractions && transmittance.lengthSquared() > transmittanceEpsilon)
                OptixIgnoreIntersection.Ignore();
            // else: let this hit stick (accept & terminate) - the accumulated
            // transmittance above is the final answer, matching the reference's
            // early-out-on-negligible-transmittance / MaxRefractions-exceeded behavior.
        }

        public static void __miss__shadow(LaunchParams launchParams)
        {
            OptixPayload.Payload0 = Interop.FloatAsInt(1f);
            OptixPayload.Payload1 = Interop.FloatAsInt(1f);
            OptixPayload.Payload2 = Interop.FloatAsInt(1f);
        }
    }
}
