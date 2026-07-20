using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;
using ILGPU.OptiX.DeviceApi;
using System.Numerics;

namespace Sample21
{
    /// <summary>
    /// Everything about the radiance ray type in one place: the three OptiX device
    /// programs it invokes, in invocation order (any-hit during traversal, then
    /// whichever of closest-hit/miss is the ray's terminal outcome). RaygenProgram.cs's
    /// own trace-call-site loop is NOT folded in here, unlike ShadowRay.Trace - it's a
    /// 351-line multi-bounce driver with accumulation/denoiser-AOV/motion-vector
    /// responsibilities far beyond seeding a payload, so merging it would bloat this
    /// file without adding contract clarity (the radiance payload's contract is
    /// genuinely spread across "raygen seeds/continues it" and "closest-hit/miss write
    /// it" - that's inherent to it being a multi-bounce loop, not a scattering bug).
    ///
    /// PAYLOAD CONTRACT - RadiancePayload (contribution/flag/new-ray/tint/AOVs/RNG/pdf/hitpos):
    ///   Seed (RaygenProgram, before the first OptixTrace.Trace of a sample):  RngState/BsdfPdf only
    ///   __anyhit__radiance   (alpha-cutout below threshold): no write, IgnoreIntersection()
    ///   __anyhit__radiance   (otherwise):                    no write, hit accepted normally
    ///   __closest__radiance:  full write via MaterialShading.ShadeSurface/ShadeDielectric's
    ///                         Payloads.SetContinuePayload/SetTerminalPayload - every
    ///                         field, every call, regardless of which bounce this is
    ///   __miss__radiance:     full write via Payloads.SetTerminalPayload (background/
    ///                         envmap radiance, BOUNCE_TERMINAL)
    ///   Reader: RaygenProgram's own bounce loop, between successive Trace() calls
    /// </summary>
    public static class RadianceRay
    {
        // Alpha-cutout test (Sponza's leaf geometry): a triangle whose sampled texture
        // alpha falls below the material's AlphaCutoff is invisible to radiance rays -
        // ignoring the intersection lets the ray continue through to whatever is behind
        // it, instead of shading the leaf quad's transparent background pixels as solid
        // geometry. AlphaCutoff defaults to 0, which short-circuits this for every
        // other (opaque-textured) material without an extra texture fetch. Never
        // touches the payload either way - see this class's own payload-contract doc
        // comment above.
        public unsafe static void __anyhit__radiance(LaunchParams launchParams)
        {
            MaterialSbtData* sbtData = (MaterialSbtData*)OptixGetSbtDataPointer.Value;
            if (sbtData->AlphaCutoff <= 0f || sbtData->BaseColorTexture == 0)
                return;

            if (MaterialShading.TryGetTriangleUV(launchParams, out Vec2 uv) &&
                MaterialShading.SampleAlpha(sbtData, uv) < sbtData->AlphaCutoff)
                OptixIgnoreIntersection.Ignore();
        }

        public unsafe static void __closest__radiance(LaunchParams launchParams)
        {
            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            Vec3 rayDir = new Vec3(dx, dy, dz);

            // RNG state carried in via Payload19 - seeded once per sample by raygen and
            // threaded continuously through every closest-hit call across all bounces,
            // rather than each shading branch reseeding its own TEA hash from (pixel,
            // frame, primitive).
            uint rngStateIn = Payloads.GetCarriedRngState();

            // BSDF pdf carried in via Payload20 - see Payloads.cs's own doc comment.
            float bsdfPdfIn = Payloads.GetCarriedBsdfPdf();

            uint primId = OptixGetPrimitiveIndex.Value;
            float lightPickAreaPdf = launchParams.NeeLightAreaPdf[primId];
            Vec3i tri = launchParams.Indices[primId];
            var (bw, bu, bv) = OptixHitProgramHelpers.GetTriangleBarycentrics();

            Vec3 a = launchParams.Vertices[tri.x];
            Vec3 b = launchParams.Vertices[tri.y];
            Vec3 c = launchParams.Vertices[tri.z];
            var rawGeometricNormalV3 = OptixHitProgramHelpers.GetGeometricNormal(
                new Vector3(a.x, a.y, a.z),
                new Vector3(b.x, b.y, b.z),
                new Vector3(c.x, c.y, c.z));
            Vec3 rawGeometricNormal = new Vec3(rawGeometricNormalV3.X, rawGeometricNormalV3.Y, rawGeometricNormalV3.Z);

            Vec3 n0 = launchParams.Normals[tri.x];
            Vec3 n1 = launchParams.Normals[tri.y];
            Vec3 n2 = launchParams.Normals[tri.z];
            var shadingNormalV3 = OptixHitProgramHelpers.InterpolateAttribute(
                new Vector3(n0.x, n0.y, n0.z),
                new Vector3(n1.x, n1.y, n1.z),
                new Vector3(n2.x, n2.y, n2.z),
                bw, bu, bv);
            // Normalize once, not once per extracted component - the original code
            // called Vector3.Normalize(shadingNormalV3) three separate times (once per
            // .X/.Y/.Z), each redoing the same sqrt+divide on identical input.
            Vector3 shadingNormalNormalized = Vector3.Normalize(shadingNormalV3);
            Vec3 shadingNormal = new Vec3(shadingNormalNormalized.X, shadingNormalNormalized.Y, shadingNormalNormalized.Z);

            Vec3 tan0 = launchParams.Tangents[tri.x];
            Vec3 tan1 = launchParams.Tangents[tri.y];
            Vec3 tan2 = launchParams.Tangents[tri.z];
            Vec3 tangent = (bw * tan0) + (bu * tan1) + (bv * tan2);

            Vec3 surfPos = (bw * a) + (bu * b) + (bv * c);

            Vec2 t0 = launchParams.TexCoords[tri.x];
            Vec2 t1 = launchParams.TexCoords[tri.y];
            Vec2 t2 = launchParams.TexCoords[tri.z];
            Vec2 uv = new Vec2((bw * t0.x) + (bu * t1.x) + (bv * t2.x), (bw * t0.y) + (bu * t1.y) + (bv * t2.y));

            // outwardNormal always faces against the incoming ray (frontFace == true
            // means the ray hit the side the raw cross-product normal already faces) -
            // needed as-is (not just for shading) by the dielectric branch to determine
            // which medium the ray is entering/leaving.
            bool frontFace = Vec3.dot(rayDir, rawGeometricNormal) < 0f;
            Vec3 outwardNormal = frontFace ? rawGeometricNormal : -rawGeometricNormal;

            if (Vec3.dot(outwardNormal, shadingNormal) < 0f)
                shadingNormal -= 2f * Vec3.dot(outwardNormal, shadingNormal) * outwardNormal;
            shadingNormal = Vec3.unitVector(shadingNormal);

            // Re-orthogonalize the (barycentric-interpolated, not necessarily unit or
            // exactly perpendicular) tangent against the final shadingNormal - same
            // Gram-Schmidt shape as Model.cs's own ComputeTangents, just done per-hit
            // here since interpolation can nudge it slightly off.
            tangent -= shadingNormal * Vec3.dot(shadingNormal, tangent);
            tangent = tangent.lengthSquared() > 1e-12f
                ? Vec3.unitVector(tangent)
                : Vec3.unitVector(Vec3.cross(XMath.Abs(shadingNormal.y) < 0.999f ? new Vec3(0f, 1f, 0f) : new Vec3(1f, 0f, 0f), shadingNormal));

            MaterialSbtData* sbtData = (MaterialSbtData*)OptixGetSbtDataPointer.Value;

            // Dispatch: Transmission > 0 routes to ShadeDielectric's Fresnel-Schlick
            // glass path (perfect-specular unless TransmissionRoughness > 0). Every
            // other material, regardless of Metallic value, goes through the single
            // unified GGX metallic-roughness BSDF in Bsdf.cs/ShadeSurface - Roughness
            // continuously drives mirror-like through fully diffuse-like response, and
            // Metallic continuously blends Lambertian diffuse against Fresnel-tinted
            // GGX specular.
            if (sbtData->Transmission > 0f)
            {
                MaterialShading.ShadeDielectric(launchParams, sbtData, rayDir, outwardNormal, frontFace, surfPos, rngStateIn);
                return;
            }

            Vec3 baseColor = MaterialShading.SampleAlbedo(sbtData, surfPos, uv);
            if (sbtData->NormalTexture != 0)
                shadingNormal = MaterialShading.ApplyNormalMap(sbtData, uv, shadingNormal, tangent);
            MaterialShading.ShadeSurface(launchParams, *sbtData, baseColor, surfPos, shadingNormal, outwardNormal, rayDir, rngStateIn, bsdfPdfIn, lightPickAreaPdf, uv);
        }

        public static void __miss__radiance(LaunchParams launchParams)
        {
            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;

            // Scene-dependent HDRI environment map - EnvMapWidth == 0 means this scene
            // has none (SceneData.EnvMapPath
            // unset), keeping the original flat analytic gradient sky below.
            if (launchParams.EnvMapWidth > 0)
            {
                Vec3 rayDir = new Vec3(dx, dy, dz);
                Vec3 envRadiance = EnvironmentMapSampling.Evaluate(launchParams, rayDir, out float pdfSolidAngle);

                // MIS-weight a BSDF-sampled ray escaping directly into the environment
                // (symmetric with MaterialShading.ShadeSurface's own emissive-triangle
                // MIS weighting) - full unweighted radiance for the primary/camera ray
                // or a ray that just left a delta lobe, matching that same sentinel
                // convention (Payloads.DeltaOrPrimarySentinel).
                float bsdfPdfIn = Payloads.GetCarriedBsdfPdf();
                float weight = 1f;
                if (launchParams.EnvMapLightPdf > 0f && bsdfPdfIn > 0f && pdfSolidAngle > 0f)
                    weight = NextEventEstimation.PowerHeuristic(bsdfPdfIn, launchParams.EnvMapLightPdf * pdfSolidAngle);

                Payloads.SetTerminalPayload(envRadiance * weight, new Vec3(0f, 0f, 0f), new Vec3(0f, 0f, 0f), new Vec3(0f, 0f, 0f));
                return;
            }

            float t = 0.5f * (dy + 1f);
            Vec3 sky = launchParams.BackgroundBottom + (t * (launchParams.BackgroundTop - launchParams.BackgroundBottom));
            Payloads.SetTerminalPayload(sky, new Vec3(0f, 0f, 0f), new Vec3(0f, 0f, 0f), new Vec3(0f, 0f, 0f));
        }
    }
}
