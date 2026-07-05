using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.Device;
using System;
using System.Numerics;

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
        // Single source of truth for the radiance payload's register count - read by
        // RendererPipeline.cs's NumPayloadValues so the pipeline's declared count and
        // the actual OptixTrace.Trace(...) call site in RaygenProgram.cs can never
        // drift out of sync again (docs/SAMPLE15_PLAN.md Milestone M3 - this drift is
        // exactly what caused "20 payload values specified in optixTrace, but only 19
        // values are configured for payload type 0" the first time this was bumped).
        // 24 is within the true 26-register ceiling (see the OptixTrace payload
        // ceiling note - bounded by CudaAsm.EmitRef's 44-ref generated overload cap,
        // not OptiX's own 32-payload limit), so no .tt regeneration is needed.
        internal const int RadiancePayloadCount = 24;

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
            // Span<T>/stackalloc + OptixPayload.SetRange are host-only conveniences -
            // ILGPU's device-code IL frontend cannot compile Span<T> (it hits an
            // internal compiler error resolving its metadata token), so payload
            // registers are set one at a time here instead.
            OptixPayload.Set(0, Interop.FloatAsInt(contribution.x));
            OptixPayload.Set(1, Interop.FloatAsInt(contribution.y));
            OptixPayload.Set(2, Interop.FloatAsInt(contribution.z));
            OptixPayload.Set(3, BOUNCE_TERMINAL);
            SetAovPayload(normal, albedo);
            SetHitPositionPayload(hitPos);
            // Payload19 (RNG state) deliberately left untouched - raygen's loop breaks
            // on BOUNCE_TERMINAL before it would read a next-bounce seed back out.
        }

        internal static void SetContinuePayload(Vec3 contribution, uint flag, Vec3 newOrigin, Vec3 newDir, Vec3 tint, Vec3 normal, Vec3 albedo, uint rngState, float bsdfPdf, Vec3 hitPos)
        {
            OptixPayload.Set(0, Interop.FloatAsInt(contribution.x));
            OptixPayload.Set(1, Interop.FloatAsInt(contribution.y));
            OptixPayload.Set(2, Interop.FloatAsInt(contribution.z));
            OptixPayload.Set(3, flag);
            OptixPayload.Set(4, Interop.FloatAsInt(newOrigin.x));
            OptixPayload.Set(5, Interop.FloatAsInt(newOrigin.y));
            OptixPayload.Set(6, Interop.FloatAsInt(newOrigin.z));
            OptixPayload.Set(7, Interop.FloatAsInt(newDir.x));
            OptixPayload.Set(8, Interop.FloatAsInt(newDir.y));
            OptixPayload.Set(9, Interop.FloatAsInt(newDir.z));
            OptixPayload.Set(10, Interop.FloatAsInt(tint.x));
            OptixPayload.Set(11, Interop.FloatAsInt(tint.y));
            OptixPayload.Set(12, Interop.FloatAsInt(tint.z));
            SetAovPayload(normal, albedo);
            OptixPayload.Payload19 = rngState;
            OptixPayload.Payload20 = Interop.FloatAsInt(bsdfPdf);
            SetHitPositionPayload(hitPos);
        }

        // Reads the RNG state carried in via Payload19 before this closest-hit call -
        // the caller (raygen) seeds it once per sample before the bounce loop starts.
        internal static uint GetCarriedRngState() => OptixPayload.Payload19;

        // Reads the BSDF pdf (or DeltaOrPrimarySentinel) carried in via Payload20 -
        // see this class's own doc comment.
        internal static float GetCarriedBsdfPdf() => Interop.IntAsFloat(OptixPayload.Payload20);

        internal static void SetAovPayload(Vec3 normal, Vec3 albedo)
        {
            OptixPayload.Set(13, Interop.FloatAsInt(normal.x));
            OptixPayload.Set(14, Interop.FloatAsInt(normal.y));
            OptixPayload.Set(15, Interop.FloatAsInt(normal.z));
            OptixPayload.Set(16, Interop.FloatAsInt(albedo.x));
            OptixPayload.Set(17, Interop.FloatAsInt(albedo.y));
            OptixPayload.Set(18, Interop.FloatAsInt(albedo.z));
        }

        internal static void SetHitPositionPayload(Vec3 hitPos)
        {
            OptixPayload.Set(21, Interop.FloatAsInt(hitPos.x));
            OptixPayload.Set(22, Interop.FloatAsInt(hitPos.y));
            OptixPayload.Set(23, Interop.FloatAsInt(hitPos.z));
        }
    }
}
