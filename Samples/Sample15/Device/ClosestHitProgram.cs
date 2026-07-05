using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;
using System.Numerics;

namespace Sample15
{
    /// <summary>
    /// The radiance closest-hit program: recovers the hit geometry via triangle
    /// interpolation and dispatches to the material shading branches in
    /// <see cref="ShadingHelpers"/>.
    /// </summary>
    public static class ClosestHitProgram
    {
        public unsafe static void __closest__radiance(LaunchParams launchParams)
        {
            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            Vec3 rayDir = new Vec3(dx, dy, dz);

            // RNG state carried in via Payload19 (docs/SAMPLE15_PLAN.md Milestone M3) -
            // seeded once per sample by raygen and threaded continuously through every
            // closest-hit call across all bounces, rather than each shading branch
            // reseeding its own TEA hash from (pixel, frame, primitive).
            uint rngStateIn = Payloads.GetCarriedRngState();

            // BSDF pdf carried in via Payload20 (docs/SAMPLE15_PLAN.md Milestone M4) -
            // see Payloads.cs's own doc comment.
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
            Vec3 shadingNormal = new Vec3(
                Vector3.Normalize(shadingNormalV3).X,
                Vector3.Normalize(shadingNormalV3).Y,
                Vector3.Normalize(shadingNormalV3).Z);

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
            // here since interpolation can nudge it slightly off (docs/SAMPLE15_PLAN.md
            // Milestone M6).
            tangent -= shadingNormal * Vec3.dot(shadingNormal, tangent);
            tangent = tangent.lengthSquared() > 1e-12f
                ? Vec3.unitVector(tangent)
                : Vec3.unitVector(Vec3.cross(XMath.Abs(shadingNormal.y) < 0.999f ? new Vec3(0f, 1f, 0f) : new Vec3(1f, 0f, 0f), shadingNormal));

            MaterialSbtData* sbtData = (MaterialSbtData*)OptixGetSbtDataPointer.Value;

            // M2 dispatch (docs/SAMPLE15_PLAN.md Milestone M2): only the dielectric
            // (transmissive) branch is still a threshold cut - Transmission > 0 routes to
            // ShadeDielectric's perfect-specular Fresnel-Schlick glass path (unchanged
            // until M7's rough BTDF). Every other material, regardless of Metallic value,
            // goes through the single unified GGX metallic-roughness BSDF in
            // Bsdf.cs/ShadeSurface - no more Metallic >= 0.9 mirror-vs-diffuse threshold;
            // Roughness continuously drives mirror-like through fully diffuse-like
            // response, and Metallic continuously blends Lambertian diffuse against
            // Fresnel-tinted GGX specular.
            if (sbtData->Transmission > 0f)
            {
                ShadingHelpers.ShadeDielectric(launchParams, sbtData, rayDir, outwardNormal, frontFace, surfPos, rngStateIn);
                return;
            }

            Vec3 baseColor = ShadingHelpers.SampleAlbedo(sbtData, surfPos, uv);
            if (sbtData->NormalTexture != 0)
                shadingNormal = ShadingHelpers.ApplyNormalMap(sbtData, uv, shadingNormal, tangent);
            ShadingHelpers.ShadeSurface(launchParams, *sbtData, baseColor, surfPos, shadingNormal, outwardNormal, rayDir, rngStateIn, bsdfPdfIn, lightPickAreaPdf, uv);
        }
    }
}
