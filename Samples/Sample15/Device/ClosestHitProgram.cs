using ILGPU.Algorithms;
using ILGPU.OptiX;

namespace Sample15
{
    /// <summary>
    /// The radiance closest-hit program: recovers the hit geometry (triangle
    /// interpolation or analytic custom-primitive reconstruction) and dispatches to
    /// the material shading branches in <see cref="ShadingHelpers"/>.
    /// </summary>
    public static class ClosestHitProgram
    {
        public unsafe static void __closest__radiance(LaunchParams launchParams)
        {
            var (dx, dy, dz) = OptixGetWorldRayDirection.Value;
            Vec3 rayDir = new Vec3(dx, dy, dz);

            uint hitKind = OptixGetHitKind.Value;

            // RNG state carried in via Payload19 (docs/SAMPLE15_PLAN.md Milestone M3) -
            // seeded once per sample by raygen and threaded continuously through every
            // closest-hit call across all bounces, rather than each shading branch
            // reseeding its own TEA hash from (pixel, frame, primitive).
            uint rngStateIn = Payloads.GetCarriedRngState();

            // BSDF pdf carried in via Payload20 (docs/SAMPLE15_PLAN.md Milestone M4) -
            // see Payloads.cs's own doc comment.
            float bsdfPdfIn = Payloads.GetCarriedBsdfPdf();

            // Volume-grid hits are handled entirely separately: material is
            // data-dependent on which voxel the intersection program's DDA loop found,
            // not on primitive index (there's only one GAS primitive for the whole
            // grid), so material lookup bypasses the per-primitive SBT convention every
            // other hit kind uses - see ShadeVolumeGrid.
            if (hitKind >= IntersectionPrograms.HitKindVolumeGridFaceBase && hitKind < IntersectionPrograms.HitKindVolumeGridFaceBase + 6)
            {
                ShadingHelpers.ShadeVolumeGrid(launchParams, rayDir, hitKind - IntersectionPrograms.HitKindVolumeGridFaceBase, rngStateIn);
                return;
            }

            // Geometry is recovered analytically here rather than threaded through
            // intersection-program attributes (docs/SAMPLE13_PLAN.md design (b)) - a
            // built-in triangle hit (hitKind >= TriangleFrontFace) interpolates from the
            // triangle buffers as before; any other hitKind is a custom primitive
            // recomputed from its own per-primitive parameter buffer plus the hit
            // distance/ray.
            Vec3 surfPos;
            Vec3 rawGeometricNormal;
            Vec3 shadingNormal;
            // Tangent-space normal mapping (docs/SAMPLE15_PLAN.md Milestone M6) -
            // custom primitives have no UV parameterization in this sample (matches
            // SampleAlbedo's own doc comment), so no material assigned to one ever has
            // a nonzero NormalTexture in practice; the arbitrary-orthogonal fallback
            // below exists only so this is never a degenerate/zero vector.
            Vec3 tangent = new Vec3(1f, 0f, 0f);
            Vec2 uv = new Vec2(0f, 0f); // custom primitives have no UV parameterization in this sample

            // Only a triangle hit can be a registered NEE light-list entry
            // (docs/SAMPLE15_PLAN.md Milestone M4 - LightList.cs only walks triangles);
            // OptixGetPrimitiveIndex.Value is local to whichever geometry kind's own
            // GAS build input the hit landed in, so indexing NeeLightAreaPdf by it for
            // a custom-primitive hit would alias an unrelated triangle's pdf.
            bool isTriangleHit = hitKind >= OptixGetHitKind.TriangleFrontFace;
            float lightPickAreaPdf = 0f;

            if (isTriangleHit)
            {
                uint primId = OptixGetPrimitiveIndex.Value;
                lightPickAreaPdf = launchParams.NeeLightAreaPdf[primId];
                Vec3i tri = launchParams.Indices[primId];
                var (bu, bv) = OptixGetTriangleBarycentrics.Value;
                float bw = 1f - bu - bv;

                Vec3 a = launchParams.Vertices[tri.x];
                Vec3 b = launchParams.Vertices[tri.y];
                Vec3 c = launchParams.Vertices[tri.z];
                rawGeometricNormal = Vec3.unitVector(Vec3.cross(b - a, c - a));

                Vec3 n0 = launchParams.Normals[tri.x];
                Vec3 n1 = launchParams.Normals[tri.y];
                Vec3 n2 = launchParams.Normals[tri.z];
                shadingNormal = Vec3.unitVector((bw * n0) + (bu * n1) + (bv * n2));

                Vec3 tan0 = launchParams.Tangents[tri.x];
                Vec3 tan1 = launchParams.Tangents[tri.y];
                Vec3 tan2 = launchParams.Tangents[tri.z];
                tangent = (bw * tan0) + (bu * tan1) + (bv * tan2);

                surfPos = (bw * a) + (bu * b) + (bv * c);

                Vec2 t0 = launchParams.TexCoords[tri.x];
                Vec2 t1 = launchParams.TexCoords[tri.y];
                Vec2 t2 = launchParams.TexCoords[tri.z];
                uv = new Vec2((bw * t0.x) + (bu * t1.x) + (bv * t2.x), (bw * t0.y) + (bu * t1.y) + (bv * t2.y));
            }
            else
            {
                // Custom primitive - hit position reconstructed from optixGetRayTmax(),
                // which inside closest-hit is the parametric distance of the accepted
                // hit; normal recomputed analytically per primitive kind (hitKind).
                var (ox, oy, oz) = OptixGetWorldRayOrigin.Value;
                Vec3 rayOrigin = new Vec3(ox, oy, oz);
                float hitT = OptixGetRayTmax.Value;

                surfPos = rayOrigin + (hitT * rayDir);
                rawGeometricNormal = ComputeCustomPrimitiveNormal(launchParams, hitKind, surfPos);
                shadingNormal = rawGeometricNormal;
            }

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

        // Analytic normal recomputation per custom-primitive kind (docs/SAMPLE13_PLAN.md
        // design (b)) - avoids threading attributes through OptixReportIntersection.
        private unsafe static Vec3 ComputeCustomPrimitiveNormal(LaunchParams launchParams, uint hitKind, Vec3 surfPos)
        {
            uint primId = OptixGetPrimitiveIndex.Value;
            switch (hitKind)
            {
                case IntersectionPrograms.HitKindSphere:
                {
                    SphereData sphere = launchParams.Spheres[primId];
                    return Vec3.unitVector((surfPos - sphere.Center) / sphere.Radius);
                }
                case IntersectionPrograms.HitKindBox:
                {
                    BoxData box = launchParams.Boxes[primId];
                    Vec3 center = (box.Min + box.Max) * 0.5f;
                    Vec3 halfExtents = (box.Max - box.Min) * 0.5f;
                    Vec3 pc = surfPos - center;
                    float dx = XMath.Abs(pc.x) - halfExtents.x;
                    float dy = XMath.Abs(pc.y) - halfExtents.y;
                    float dz = XMath.Abs(pc.z) - halfExtents.z;
                    if (dx > dy && dx > dz)
                        return new Vec3(pc.x >= 0f ? 1f : -1f, 0f, 0f);
                    if (dy > dz)
                        return new Vec3(0f, pc.y >= 0f ? 1f : -1f, 0f);
                    return new Vec3(0f, 0f, pc.z >= 0f ? 1f : -1f);
                }
                case IntersectionPrograms.HitKindCylinderY:
                {
                    CylinderYData cyl = launchParams.CylindersY[primId];
                    const float capEpsilon = 1e-3f;
                    if (XMath.Abs(surfPos.y - cyl.YMax) < capEpsilon)
                        return new Vec3(0f, 1f, 0f);
                    if (XMath.Abs(surfPos.y - cyl.YMin) < capEpsilon)
                        return new Vec3(0f, -1f, 0f);
                    return Vec3.unitVector(new Vec3(surfPos.x - cyl.Center.x, 0f, surfPos.z - cyl.Center.z));
                }
                case IntersectionPrograms.HitKindDisk:
                {
                    DiskData disk = launchParams.Disks[primId];
                    return disk.Normal;
                }
                case IntersectionPrograms.HitKindXYRect:
                    return new Vec3(0f, 0f, 1f);
                case IntersectionPrograms.HitKindXZRect:
                    return new Vec3(0f, 1f, 0f);
                default: // HitKindYZRect
                    return new Vec3(1f, 0f, 0f);
            }
        }
    }
}
