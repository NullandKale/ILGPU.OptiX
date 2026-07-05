using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.Device;
using System.Numerics;

namespace Sample14
{
    /// <summary>
    /// Shared ray-type/bounce-flag constants and the payload packing helpers used by
    /// every OptiX program in this sample. The 19-register payload convention:
    /// 0-2 radiance contribution, 3 continuation flag, 4-6 new origin, 7-9 new
    /// direction, 10-12 throughput tint, 13-15 AOV normal, 16-18 AOV albedo.
    /// </summary>
    public static class Payloads
    {
        internal const uint RADIANCE_RAY_TYPE = OptixPayloadDefaults.RADIANCE_RAY_TYPE;
        internal const uint SHADOW_RAY_TYPE = OptixPayloadDefaults.SHADOW_RAY_TYPE;
        internal const uint RAY_TYPE_COUNT = OptixPayloadDefaults.RAY_TYPE_COUNT;

        // Continuation flags returned via Payload3 by __closest__radiance - matches the
        // reference's MaxMirrorBounces/MaxRefractions/MaxDiffuseBounces caps (all 2),
        // tracked as separate counters in raygen's own loop (closesthit doesn't need to
        // know the running counts - it only reports which kind of surface was hit).
        internal const uint BOUNCE_TERMINAL = OptixPayloadDefaults.BOUNCE_TERMINAL;
        internal const uint BOUNCE_CONTINUE_MIRROR = OptixPayloadDefaults.BOUNCE_CONTINUE_MIRROR;
        internal const uint BOUNCE_CONTINUE_DIELECTRIC = OptixPayloadDefaults.BOUNCE_CONTINUE_DIELECTRIC;
        internal const uint BOUNCE_CONTINUE_DIFFUSE = OptixPayloadDefaults.BOUNCE_CONTINUE_DIFFUSE;

        // normal/albedo (payloads 13-18) are only ever read back by raygen from the
        // bounce==0 Trace() call (see __raygen__renderFrame) - matching Sample11/12's
        // AOV guide-buffer convention of reflecting only the primary-ray hit, not
        // anything accumulated across bounces - but every shading branch still
        // populates them on every call, since raygen has no way to know in advance
        // which bounce a given closest-hit invocation corresponds to.
        internal static void SetTerminalPayload(Vec3 contribution, Vec3 normal, Vec3 albedo)
        {
            OptixPayloadVec3Helper.SetVec3Registers(0, new Vector3(contribution.x, contribution.y, contribution.z));
            OptixPayloadInterop.SetUint(3, BOUNCE_TERMINAL);
            SetAovPayload(normal, albedo);
        }

        internal static void SetContinuePayload(Vec3 contribution, uint flag, Vec3 newOrigin, Vec3 newDir, Vec3 tint, Vec3 normal, Vec3 albedo)
        {
            OptixPayloadVec3Helper.SetVec3Registers(0, new Vector3(contribution.x, contribution.y, contribution.z));
            OptixPayloadInterop.SetUint(3, flag);
            OptixPayloadVec3Helper.SetVec3Registers(4, new Vector3(newOrigin.x, newOrigin.y, newOrigin.z));
            OptixPayloadVec3Helper.SetVec3Registers(7, new Vector3(newDir.x, newDir.y, newDir.z));
            OptixPayloadVec3Helper.SetVec3Registers(10, new Vector3(tint.x, tint.y, tint.z));
            SetAovPayload(normal, albedo);
        }

        internal static void SetAovPayload(Vec3 normal, Vec3 albedo)
        {
            OptixPayloadVec3Helper.SetVec3Registers(13, new Vector3(normal.x, normal.y, normal.z));
            OptixPayloadVec3Helper.SetVec3Registers(16, new Vector3(albedo.x, albedo.y, albedo.z));
        }
    }
}
