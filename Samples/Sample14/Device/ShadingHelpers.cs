using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;
using ILGPU.OptiX.DeviceApi;
using ILGPU.OptiX.Cuda;

namespace Sample14
{
    /// <summary>
    /// Material shading branches (diffuse/mirror/dielectric/volume-grid) plus the BRDF,
    /// texture sampling, and shadow-ray helpers they share. All methods are static and
    /// get inlined by ILGPU into whichever OptiX program calls them.
    /// </summary>
    public static class ShadingHelpers
    {
        // Oren-Nayar roughness, matching the reference's fixed DiffuseSigmaDeg = 25.0f
        // (RayTracing/RaytraceRenderer.cs) - the reference has no per-material roughness.
        private const float DiffuseSigmaDeg = 25f;

        // Oren-Nayar + ambient + multi-light shading, shared between the ordinary
        // diffuse dispatch branch in __closest__radiance and the volume grid's
        // per-voxel material lookup (ShadeVolumeGrid) - identical lighting math, only
        // how the material/albedo was obtained differs between the two callers. Direct
        // lighting is accumulated, then an indirect diffuse ray is sampled and traced
        // in the next bounce.
        internal unsafe static void ShadeDiffuse(LaunchParams launchParams, Vec3 albedo, Vec3 emission, Vec3 surfPos, Vec3 shadingNormal, Vec3 outwardNormal, Vec3 rayDir)
        {
            Vec3 pixelColor = emission + (launchParams.AmbientColor * launchParams.AmbientIntensity * albedo);
            Vec3 viewDir = Vec3.unitVector(-rayDir);

            for (int i = 0; i < launchParams.NumPointLights; i++)
            {
                PointLightGpu light = launchParams.PointLights[i];
                Vec3 toLight = light.Position - surfPos;
                float dist2 = toLight.lengthSquared();
                float dist = XMath.Sqrt(dist2);
                Vec3 lightDir = toLight / dist;

                float NdotL = Vec3.dot(shadingNormal, lightDir);
                if (NdotL <= 0f)
                    continue;

                Vec3 transmittance = ShadowTransmittance(launchParams, surfPos, outwardNormal, lightDir, dist);
                if (transmittance.lengthSquared() <= 0f)
                    continue;

                float atten = light.Intensity / dist2;
                Vec3 bsdf = OrenNayarBRDF(albedo, shadingNormal, viewDir, lightDir, DiffuseSigmaDeg);
                pixelColor += (atten * NdotL) * bsdf * light.Color * transmittance;
            }

            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;
            uint primId = OptixGetPrimitiveIndex.Value;
            LCG rng = new LCG((uint)(ix + (1000003 * iy)), (uint)((launchParams.FrameID * 7919) + primId));

            Vec3 diffuseDir = SampleCosineHemisphere(shadingNormal, rng);
            Vec3 newOrigin = surfPos + (1e-3f * outwardNormal);
            Payloads.SetContinuePayload(pixelColor, Payloads.BOUNCE_CONTINUE_DIFFUSE, newOrigin, diffuseDir, albedo, shadingNormal, albedo);
        }

        // The volume grid is a single GAS primitive whose per-voxel material can't be
        // expressed via OptiX's per-primitive SbtIndexOffsetBuffer (that's keyed by
        // primitive index, not by data read during intersection) - so material lookup
        // goes directly through LaunchParams.Materials/VoxelMaterialIds instead of
        // OptixGetSbtDataPointer, a deliberate deviation from every other primitive kind
        // in this sample (see the comment on LaunchParams.Materials).
        internal unsafe static void ShadeVolumeGrid(LaunchParams launchParams, Vec3 rayDir, uint faceCode)
        {
            var (ox, oy, oz) = OptixGetWorldRayOrigin.Value;
            Vec3 rayOrigin = new Vec3(ox, oy, oz);
            float hitT = OptixGetRayTmax.Value;
            Vec3 surfPos = rayOrigin + (hitT * rayDir);

            Vec3 rawGeometricNormal = VolumeGridFaceNormal(faceCode);
            bool frontFace = Vec3.dot(rayDir, rawGeometricNormal) < 0f;
            Vec3 outwardNormal = frontFace ? rawGeometricNormal : -rawGeometricNormal;

            Vec3i dims = launchParams.VolumeDims;
            Vec3 local = surfPos - launchParams.VolumeGridMin;
            int ix = XMath.Clamp((int)XMath.Floor(local.x / launchParams.VolumeVoxelSize.x), 0, dims.x - 1);
            int iy = XMath.Clamp((int)XMath.Floor(local.y / launchParams.VolumeVoxelSize.y), 0, dims.y - 1);
            int iz = XMath.Clamp((int)XMath.Floor(local.z / launchParams.VolumeVoxelSize.z), 0, dims.z - 1);
            uint voxel = launchParams.VoxelMaterialIds[(ix * dims.y * dims.z) + (iy * dims.z) + iz];
            // voxel == 0 (empty) never reaches here - the intersection program's DDA
            // only reports a hit on a non-empty voxel.
            MaterialSbtData mat = launchParams.Materials[(int)voxel - 1];

            Vec3 albedo = mat.MaterialKind == MaterialSbtData.Checker ? SampleChecker(mat, surfPos) : mat.Albedo;
            ShadeDiffuse(launchParams, albedo, mat.Emission, surfPos, outwardNormal, outwardNormal, rayDir);
        }

        private static Vec3 VolumeGridFaceNormal(uint faceCode)
        {
            switch (faceCode)
            {
                case 0: return new Vec3(1f, 0f, 0f);
                case 1: return new Vec3(-1f, 0f, 0f);
                case 2: return new Vec3(0f, 1f, 0f);
                case 3: return new Vec3(0f, -1f, 0f);
                case 4: return new Vec3(0f, 0f, 1f);
                default: return new Vec3(0f, 0f, -1f);
            }
        }

        internal unsafe static void ShadeMirror(MaterialSbtData* sbtData, Vec3 albedo, Vec3 rayDir, Vec3 shadingNormal, Vec3 outwardNormal, Vec3 surfPos)
        {
            Vec3 reflectDir = Vec3.reflect(shadingNormal, rayDir);
            Vec3 newOrigin = surfPos + (1e-3f * outwardNormal);
            Payloads.SetContinuePayload(sbtData->Emission, Payloads.BOUNCE_CONTINUE_MIRROR, newOrigin, reflectDir, albedo, shadingNormal, albedo);
        }

        // Fresnel-weighted stochastic reflect-XOR-refract pick (not the reference's
        // branch-both-paths-at-once) - converges to the same result over many
        // accumulated frames given the existing progressive accumulation. Total
        // internal reflection forces the reflect branch unconditionally (no valid
        // refraction direction exists). A fractional Transparency isn't currently
        // exercised by any Sample13 scene (materials use 1.0, matching the reference's
        // actual glass materials) - if a future scene wants partial transparency,
        // revisit how it should modulate this probability split.
        internal unsafe static void ShadeDielectric(LaunchParams launchParams, MaterialSbtData* sbtData, Vec3 rayDir, Vec3 outwardNormal, bool frontFace, Vec3 surfPos)
        {
            float etaFrom = frontFace ? 1f : sbtData->IndexOfRefraction;
            float etaTo = frontFace ? sbtData->IndexOfRefraction : 1f;
            float eta = etaFrom / etaTo;

            bool didRefract = TryRefract(rayDir, outwardNormal, eta, out Vec3 refractDir);

            float cosTheta = XMath.Clamp(Vec3.dot(-rayDir, outwardNormal), 0f, 1f);
            float fresnel = didRefract ? FresnelSchlick(cosTheta, etaFrom, etaTo) : 1f;
            fresnel = XMath.Clamp(fresnel + (sbtData->Reflectivity * (1f - fresnel)), 0f, 1f);

            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;
            uint primId = OptixGetPrimitiveIndex.Value;
            LCG rng = new LCG((uint)(ix + (1000003 * iy)), (uint)((launchParams.FrameID * 7919) + primId));

            bool chooseReflect = !didRefract || rng.Next() < fresnel;

            Vec3 newDir;
            Vec3 tint;
            Vec3 offsetNormal;
            if (chooseReflect)
            {
                newDir = Vec3.reflect(outwardNormal, rayDir);
                tint = new Vec3(1f, 1f, 1f);
                offsetNormal = outwardNormal;
            }
            else
            {
                newDir = refractDir;
                tint = sbtData->TransmissionColor;
                offsetNormal = -outwardNormal;
            }

            Vec3 newOrigin = surfPos + (1e-3f * offsetNormal);
            Payloads.SetContinuePayload(sbtData->Emission, Payloads.BOUNCE_CONTINUE_DIELECTRIC, newOrigin, newDir, tint, outwardNormal, tint);
        }

        // v = incident ray direction (pointing into the surface), n = outward normal
        // (facing against v, same side as the ray origin), niOverNt = etaFrom/etaTo -
        // matches Vectors.cs's existing Vec3.refract convention (Shirley's "Ray Tracing
        // in One Weekend" formula), reimplemented here with an explicit success flag
        // rather than relying on that helper's silent "return v unchanged" TIR fallback.
        private static bool TryRefract(Vec3 v, Vec3 n, float niOverNt, out Vec3 refracted)
        {
            Vec3 uv = Vec3.unitVector(v);
            float dt = Vec3.dot(uv, n);
            float discriminant = 1f - (niOverNt * niOverNt * (1f - (dt * dt)));
            if (discriminant > 0f)
            {
                refracted = (niOverNt * (uv - (n * dt))) - (n * XMath.Sqrt(discriminant));
                return true;
            }
            refracted = Vec3.reflect(n, v);
            return false;
        }

        private static float FresnelSchlick(float cosTheta, float etaFrom, float etaTo)
        {
            float r0 = (etaFrom - etaTo) / (etaFrom + etaTo);
            r0 *= r0;
            float x = 1f - cosTheta;
            float x2 = x * x;
            return r0 + ((1f - r0) * x2 * x2 * x);
        }

        // Multi-hit colored transmittance shadow ray - walks through transparent
        // occluders via __anyhit__shadow's optixIgnoreIntersection, accumulating
        // TransmissionColor*Transparency per hit
        // up to MaxRefractions, unlike a plain binary occlusion test.
        private unsafe static Vec3 ShadowTransmittance(LaunchParams launchParams, Vec3 surfPos, Vec3 outwardNormal, Vec3 lightDir, float lightDist)
        {
            uint t0 = Interop.FloatAsInt(1f);
            uint t1 = Interop.FloatAsInt(1f);
            uint t2 = Interop.FloatAsInt(1f);
            uint hitCount = 0;

            OptixTrace.Trace(
                launchParams.traversable,
                (surfPos.x + (1e-3f * outwardNormal.x), surfPos.y + (1e-3f * outwardNormal.y), surfPos.z + (1e-3f * outwardNormal.z)),
                (lightDir.x, lightDir.y, lightDir.z),
                1e-3f,
                lightDist * (1f - 1e-3f),
                0.0f,
                0xff,
                OptixRayFlags.None,
                Payloads.SHADOW_RAY_TYPE,
                Payloads.RAY_TYPE_COUNT,
                Payloads.SHADOW_RAY_TYPE,
                ref t0, ref t1, ref t2, ref hitCount);

            return new Vec3(Interop.IntAsFloat(t0), Interop.IntAsFloat(t1), Interop.IntAsFloat(t2));
        }

        // Cosine-weighted hemisphere sampling: generate a random direction in the
        // hemisphere around the normal, biased toward the normal itself. Uses the
        // standard approach: uniform sample on disk, then lift to hemisphere.
        private static Vec3 SampleCosineHemisphere(Vec3 normal, LCG rng)
        {
            float r1 = rng.Next();
            float r2 = rng.Next();

            float theta = XMath.Acos(XMath.Sqrt(1f - r1));
            float phi = 2f * XMath.PI * r2;

            Vec3 up = XMath.Abs(normal.y) < 0.999f ? new Vec3(0f, 1f, 0f) : new Vec3(1f, 0f, 0f);
            Vec3 right = Vec3.unitVector(Vec3.cross(up, normal));
            Vec3 forward = Vec3.cross(normal, right);

            float sinTheta = XMath.Sin(theta);
            Vec3 localDir = (right * (sinTheta * XMath.Cos(phi))) + (normal * XMath.Cos(theta)) + (forward * (sinTheta * XMath.Sin(phi)));
            return Vec3.unitVector(localDir);
        }

        // Direct port of the reference's OrenNayarBRDF (RayTracing/RaytraceRenderer.cs) -
        // NdotL is intentionally NOT baked in here (applied by the caller, matching
        // "radiance += beta * bsdf*nDotL*Li*transmittance" in the reference). Guards
        // against normalizing a near-zero perpendicular-component vector (degenerates to
        // NaN via 0/0 when L or V is nearly parallel to N) by falling back to
        // cosPhiDiff = 0, which only ever discards the (already small/vanishing) azimuthal
        // correction term.
        private static Vec3 OrenNayarBRDF(Vec3 albedo, Vec3 N, Vec3 V, Vec3 L, float sigmaDeg)
        {
            float sigma = sigmaDeg * (XMath.PI / 180f);
            float sigma2 = sigma * sigma;
            float A = 1f - (sigma2 / (2f * (sigma2 + 0.33f)));
            float B = 0.45f * sigma2 / (sigma2 + 0.09f);

            float NdotL = XMath.Clamp(Vec3.dot(N, L), 0f, 1f);
            float NdotV = XMath.Clamp(Vec3.dot(N, V), 0f, 1f);
            float thetaI = XMath.Acos(NdotL);
            float thetaR = XMath.Acos(NdotV);
            float alpha = XMath.Max(thetaI, thetaR);
            float beta = XMath.Min(thetaI, thetaR);

            Vec3 lightPerpRaw = L - (N * NdotL);
            Vec3 viewPerpRaw = V - (N * NdotV);
            float lightPerpLenSq = lightPerpRaw.lengthSquared();
            float viewPerpLenSq = viewPerpRaw.lengthSquared();

            float cosPhiDiff = 0f;
            if (lightPerpLenSq > 1e-8f && viewPerpLenSq > 1e-8f)
            {
                Vec3 lightPerp = lightPerpRaw / XMath.Sqrt(lightPerpLenSq);
                Vec3 viewPerp = viewPerpRaw / XMath.Sqrt(viewPerpLenSq);
                cosPhiDiff = XMath.Clamp(Vec3.dot(lightPerp, viewPerp), -1f, 1f);
            }

            float factor = (A + (B * cosPhiDiff * XMath.Sin(alpha) * XMath.Tan(beta))) * (1f / XMath.PI);
            return albedo * factor;
        }

        // Direct port of the reference's Checker closure (position-evaluated instead of a
        // captured Func<>) - floor(pos.X/scale) + floor(pos.Z/scale) parity picks between
        // the two flat colors.
        internal static Vec3 SampleChecker(MaterialSbtData mat, Vec3 pos)
        {
            int cx = (int)XMath.Floor(pos.x / mat.CheckerScale);
            int cz = (int)XMath.Floor(pos.z / mat.CheckerScale);
            bool even = ((cx + cz) & 1) == 0;
            return even ? mat.Albedo : mat.CheckerColorB;
        }

        // Direct port of the reference's SampleAlbedo (RaytraceRenderer.cs) -
        // TextureObject == 0 means no texture (matches Sample08's convention). UV is
        // only ever non-zero for triangle hits (custom primitives have no UV
        // parameterization in this sample), so a textured custom primitive is not
        // currently possible - not needed by any Sample13 scene.
        internal unsafe static Vec3 SampleAlbedo(MaterialSbtData* sbtData, Vec3 surfPos, Vec2 uv)
        {
            Vec3 baseAlbedo = sbtData->MaterialKind == MaterialSbtData.Checker
                ? SampleChecker(*sbtData, surfPos)
                : sbtData->Albedo;

            if (sbtData->TextureObject == 0)
                return baseAlbedo;

            var (tr, tg, tb, _) = CudaTex2D.Sample(sbtData->TextureObject, uv.x * sbtData->UVScale, uv.y * sbtData->UVScale);
            Vec3 texSample = new Vec3(tr, tg, tb);
            return (baseAlbedo * (1f - sbtData->TextureWeight)) + (texSample * sbtData->TextureWeight);
        }

        // Alpha-cutout support (Sponza's leaf geometry: a solid quad with the leaf
        // shape punched out via the diffuse texture's own alpha channel). Only called
        // from the any-hit programs, which already checked AlphaCutoff > 0 and
        // TextureObject != 0 before calling this.
        internal unsafe static float SampleAlpha(MaterialSbtData* sbtData, Vec2 uv)
        {
            var (_, _, _, ta) = CudaTex2D.Sample(sbtData->TextureObject, uv.x * sbtData->UVScale, uv.y * sbtData->UVScale);
            return ta;
        }

        // Barycentric UV recovery shared by the any-hit programs (which need only UV,
        // unlike __closest__radiance's full geometry recovery) - false for any hit kind
        // other than a built-in triangle, since custom primitives have no UV
        // parameterization in this sample (matches SampleAlbedo's own doc comment).
        internal unsafe static bool TryGetTriangleUV(LaunchParams launchParams, out Vec2 uv)
        {
            if (OptixGetHitKind.Value < OptixGetHitKind.TriangleFrontFace)
            {
                uv = default;
                return false;
            }

            uint primId = OptixGetPrimitiveIndex.Value;
            Vec3i tri = launchParams.Indices[primId];
            var (bu, bv) = OptixGetTriangleBarycentrics.Value;
            float bw = 1f - bu - bv;

            Vec2 t0 = launchParams.TexCoords[tri.x];
            Vec2 t1 = launchParams.TexCoords[tri.y];
            Vec2 t2 = launchParams.TexCoords[tri.z];
            uv = new Vec2((bw * t0.x) + (bu * t1.x) + (bv * t2.x), (bw * t0.y) + (bu * t1.y) + (bv * t2.y));
            return true;
        }
    }
}
