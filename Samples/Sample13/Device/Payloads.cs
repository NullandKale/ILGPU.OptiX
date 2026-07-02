using ILGPU;
using ILGPU.OptiX;

namespace Sample13
{
    /// <summary>
    /// Shared ray-type/bounce-flag constants and the payload packing helpers used by
    /// every OptiX program in this sample. The 19-register payload convention:
    /// 0-2 radiance contribution, 3 continuation flag, 4-6 new origin, 7-9 new
    /// direction, 10-12 throughput tint, 13-15 AOV normal, 16-18 AOV albedo.
    /// </summary>
    public static class Payloads
    {
        internal const uint RADIANCE_RAY_TYPE = 0;
        internal const uint SHADOW_RAY_TYPE = 1;
        internal const uint RAY_TYPE_COUNT = 2;

        // Continuation flags returned via Payload3 by __closest__radiance - matches the
        // reference's MaxMirrorBounces/MaxRefractions caps (both 2), tracked as separate
        // counters in raygen's own loop (closesthit doesn't need to know the running
        // counts - it only reports which kind of surface was hit).
        internal const uint BOUNCE_TERMINAL = 0;
        internal const uint BOUNCE_CONTINUE_MIRROR = 1;
        internal const uint BOUNCE_CONTINUE_DIELECTRIC = 2;

        // normal/albedo (payloads 13-18) are only ever read back by raygen from the
        // bounce==0 Trace() call (see __raygen__renderFrame) - matching Sample11/12's
        // AOV guide-buffer convention of reflecting only the primary-ray hit, not
        // anything accumulated across bounces - but every shading branch still
        // populates them on every call, since raygen has no way to know in advance
        // which bounce a given closest-hit invocation corresponds to.
        internal static void SetTerminalPayload(Vec3 contribution, Vec3 normal, Vec3 albedo)
        {
            OptixPayload.Payload0 = Interop.FloatAsInt(contribution.x);
            OptixPayload.Payload1 = Interop.FloatAsInt(contribution.y);
            OptixPayload.Payload2 = Interop.FloatAsInt(contribution.z);
            OptixPayload.Payload3 = BOUNCE_TERMINAL;
            SetAovPayload(normal, albedo);
        }

        internal static void SetContinuePayload(Vec3 contribution, uint flag, Vec3 newOrigin, Vec3 newDir, Vec3 tint, Vec3 normal, Vec3 albedo)
        {
            OptixPayload.Payload0 = Interop.FloatAsInt(contribution.x);
            OptixPayload.Payload1 = Interop.FloatAsInt(contribution.y);
            OptixPayload.Payload2 = Interop.FloatAsInt(contribution.z);
            OptixPayload.Payload3 = flag;
            OptixPayload.Payload4 = Interop.FloatAsInt(newOrigin.x);
            OptixPayload.Payload5 = Interop.FloatAsInt(newOrigin.y);
            OptixPayload.Payload6 = Interop.FloatAsInt(newOrigin.z);
            OptixPayload.Payload7 = Interop.FloatAsInt(newDir.x);
            OptixPayload.Payload8 = Interop.FloatAsInt(newDir.y);
            OptixPayload.Payload9 = Interop.FloatAsInt(newDir.z);
            OptixPayload.Payload10 = Interop.FloatAsInt(tint.x);
            OptixPayload.Payload11 = Interop.FloatAsInt(tint.y);
            OptixPayload.Payload12 = Interop.FloatAsInt(tint.z);
            SetAovPayload(normal, albedo);
        }

        internal static void SetAovPayload(Vec3 normal, Vec3 albedo)
        {
            OptixPayload.Payload13 = Interop.FloatAsInt(normal.x);
            OptixPayload.Payload14 = Interop.FloatAsInt(normal.y);
            OptixPayload.Payload15 = Interop.FloatAsInt(normal.z);
            OptixPayload.Payload16 = Interop.FloatAsInt(albedo.x);
            OptixPayload.Payload17 = Interop.FloatAsInt(albedo.y);
            OptixPayload.Payload18 = Interop.FloatAsInt(albedo.z);
        }
    }
}
