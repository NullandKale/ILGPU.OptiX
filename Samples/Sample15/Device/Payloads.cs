using ILGPU.OptiX;
using ILGPU.OptiX.Device;
using System.Runtime.InteropServices;

namespace Sample15
{
    /// <summary>
    /// Shared ray-type/bounce-flag constants and the payload packing helpers used by
    /// every OptiX program in this sample. The 24-register payload convention:
    /// 0-2 radiance contribution, 3 continuation flag, 4-6 new origin, 7-9 new
    /// direction, 10-12 throughput tint, 13-15 AOV normal, 16-18 AOV albedo, 19 carried
    /// RNG state (docs/SAMPLE15_PLAN.md Milestone M3), 20 carried BSDF pdf
    /// (docs/SAMPLE15_PLAN.md Milestone M4 - the sampled direction's BSDF pdf from the
    /// bounce that produced this ray, or the sentinel <see cref="DeltaOrPrimarySentinel"/>
    /// if that bounce was a delta lobe or this is the primary/camera ray; read by the
    /// next closest-hit call to power-heuristic-MIS-weight its own material's emission
    /// against the light list's pdf for the same direction, instead of adding full
    /// emission unconditionally every time a BSDF-sampled ray happens to land on a
    /// light), 21-23 world-space hit position (read back by raygen only from the
    /// bounce==0 call, same "only bounce 0 matters" convention as the AOV normal/albedo
    /// above - used to compute this pixel's motion vector for the temporal denoiser;
    /// see RaygenProgram.cs's reprojection block).
    /// </summary>
    public static class Payloads
    {
        // Exact device-side layout of the radiance ray's 24-register payload, used by
        // OptixPayload.Read{T}/Write{T} (this file) and OptixTrace.Trace{T}
        // (RaygenProgram.cs) - see docs/API_USABILITY_PLAN.md section 2. Flat
        // float/uint fields (rather than nesting Vec3, which has no
        // [StructLayout] of its own and so has no *guaranteed* field order) avoid
        // depending on an unrelated type's undocumented layout for correctness here.
        // OptixPayloadLayout.CountOf<T> (host-side only) validates this struct is
        // exactly 24 words with no padding.
        [StructLayout(LayoutKind.Sequential)]
        internal struct RadiancePayload
        {
            public float RadianceX, RadianceY, RadianceZ;    // 0-2
            public uint Flag;                                 // 3
            public float NewOriginX, NewOriginY, NewOriginZ;  // 4-6
            public float NewDirX, NewDirY, NewDirZ;           // 7-9
            public float TintX, TintY, TintZ;                 // 10-12
            public float NormalX, NormalY, NormalZ;           // 13-15
            public float AlbedoX, AlbedoY, AlbedoZ;           // 16-18
            public uint RngState;                             // 19
            public float BsdfPdf;                             // 20
            public float HitPosX, HitPosY, HitPosZ;           // 21-23
        }

        // The shadow ray only ever uses payload0-3 (transmittance xyz + hit count) -
        // see Rays/ShadowRay.cs's Trace/__anyhit__shadow/__miss__shadow (and its own
        // payload-contract doc comment). Same flat-fields-not-nested-Vec3 rationale as
        // RadiancePayload above.
        [StructLayout(LayoutKind.Sequential)]
        internal struct ShadowPayload
        {
            public float TransmittanceX, TransmittanceY, TransmittanceZ; // 0-2
            public uint HitCount;                                        // 3
        }

        // A real BSDF pdf is always > 0 - this sentinel flags "no valid MIS competitor
        // for this bounce" (either the previous bounce sampled a delta lobe, whose pdf
        // is not a finite comparable value, or there was no previous bounce at all -
        // this is the primary/camera ray). Either way, an emissive surface hit under
        // this sentinel should show its full, unweighted emission.
        internal const float DeltaOrPrimarySentinel = -1f;

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
        internal static void SetTerminalPayload(Vec3 contribution, Vec3 normal, Vec3 albedo, Vec3 hitPos)
        {
            // NewOrigin/NewDirection/Tint/RngState/BsdfPdf are left at their struct
            // default (0) - raygen's bounce loop breaks as soon as it sees
            // BOUNCE_TERMINAL, before it would ever read any of them back out (see
            // RaygenProgram.cs's __raygen__renderFrame), so writing 0 instead of
            // leaving the incoming register value untouched (what the old
            // per-register OptixPayload.Set(...) calls did) is not observable.
            var payload = new RadiancePayload
            {
                RadianceX = contribution.x,
                RadianceY = contribution.y,
                RadianceZ = contribution.z,
                Flag = BOUNCE_TERMINAL,
                NormalX = normal.x,
                NormalY = normal.y,
                NormalZ = normal.z,
                AlbedoX = albedo.x,
                AlbedoY = albedo.y,
                AlbedoZ = albedo.z,
                HitPosX = hitPos.x,
                HitPosY = hitPos.y,
                HitPosZ = hitPos.z,
            };
            OptixPayload.Write(in payload);
        }

        internal static void SetContinuePayload(Vec3 contribution, uint flag, Vec3 newOrigin, Vec3 newDir, Vec3 tint, Vec3 normal, Vec3 albedo, uint rngState, float bsdfPdf, Vec3 hitPos)
        {
            var payload = new RadiancePayload
            {
                RadianceX = contribution.x,
                RadianceY = contribution.y,
                RadianceZ = contribution.z,
                Flag = flag,
                NewOriginX = newOrigin.x,
                NewOriginY = newOrigin.y,
                NewOriginZ = newOrigin.z,
                NewDirX = newDir.x,
                NewDirY = newDir.y,
                NewDirZ = newDir.z,
                TintX = tint.x,
                TintY = tint.y,
                TintZ = tint.z,
                NormalX = normal.x,
                NormalY = normal.y,
                NormalZ = normal.z,
                AlbedoX = albedo.x,
                AlbedoY = albedo.y,
                AlbedoZ = albedo.z,
                RngState = rngState,
                BsdfPdf = bsdfPdf,
                HitPosX = hitPos.x,
                HitPosY = hitPos.y,
                HitPosZ = hitPos.z,
            };
            OptixPayload.Write(in payload);
        }

        // Reads the RNG state carried in via Payload19 before this closest-hit call -
        // the caller (raygen) seeds it once per sample before the bounce loop starts.
        internal static uint GetCarriedRngState() => OptixPayload.Read<RadiancePayload>().RngState;

        // Reads the BSDF pdf (or DeltaOrPrimarySentinel) carried in via Payload20 -
        // see this class's own doc comment.
        internal static float GetCarriedBsdfPdf() => OptixPayload.Read<RadiancePayload>().BsdfPdf;
    }
}
