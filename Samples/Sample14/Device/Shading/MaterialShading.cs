using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;
using ILGPU.OptiX.DeviceApi;
using ILGPU.OptiX.Cuda;

namespace Sample14
{
    /// <summary>
    /// Material shading branches (unified GGX metallic-roughness surface/dielectric -
    /// see Bsdf.cs for the GGX math itself) plus the texture/geometry sampling helpers
    /// they share. All methods are static and get inlined by ILGPU into whichever
    /// OptiX program calls them. The shadow ray's own trace-call-site wrapper lives in
    /// Rays/ShadowRay.cs instead of here - see that file's payload-contract comment.
    /// </summary>
    public static class MaterialShading
    {
        // Unified metallic-roughness GGX BSDF shading - one continuous function
        // covering the full diffuse-to-mirror range. Direct lighting is next-event
        // estimation against the unified light list
        // (NextEventEstimation.cs); indirect lighting stochastically
        // picks one lobe to extend the path with, weighted by both lobes' combined pdf
        // at the sampled direction (see Bsdf.cs). bsdfPdfIn/lightPickAreaPdf implement the MIS side of a
        // BSDF-sampled ray landing directly on this material's own emission - see
        // Payloads.cs's own doc comment and Rays/RadianceRay.cs's callsite.
        internal unsafe static void ShadeSurface(LaunchParams launchParams, MaterialSbtData mat, Vec3 baseColor, Vec3 surfPos, Vec3 shadingNormal, Vec3 outwardNormal, Vec3 rayDir, uint rngStateIn, float bsdfPdfIn, float lightPickAreaPdf, Vec2 uv)
        {
            float roughness = mat.Roughness;
            float metallic = mat.Metallic;
            // ORM texture - occlusion.r (unused;
            // no ambient-occlusion consumption point exists in this sample yet),
            // roughness.g, metallic.b, multiplicatively blended against the material's
            // own scalar Roughness/Metallic (the glTF metallic-roughness convention:
            // effective value = scalarFactor * textureSample, so a scalar of 1
            // reproduces the texture as-is and a scalar of 0 zeroes it out regardless
            // of the texture). ORM/normal fetches stay linear (no sRGB decode) - only
            // SampleAlbedo's base-color fetch needs one.
            if (mat.OrmTexture != 0)
            {
                var (_, ormG, ormB, _) = CudaTex2D.Sample(mat.OrmTexture, uv.x * mat.UVScale, uv.y * mat.UVScale);
                roughness *= ormG;
                metallic *= ormB;
            }
            Vec3 viewDir = Vec3.unitVector(-rayDir);
            Vec3 f0 = Bsdf.SpecularF0(baseColor, metallic);
            bool isDelta = roughness < Bsdf.DeltaRoughnessThreshold;

            // Ambient only lights the diffuse response - a pure metal has no diffuse
            // albedo for a flat ambient term to reflect into. This is a separate flat
            // constant term, distinct from the HDRI environment map (EnvironmentMapSampling.cs),
            // which the specular lobe does sample correctly via NEE/BSDF-sampled misses.
            Vec3 pixelColor = launchParams.AmbientColor * launchParams.AmbientIntensity * baseColor * (1f - metallic);

            Vec3 emission = mat.Emission * mat.EmissionStrength;
            if (emission.lengthSquared() > 0f)
            {
                // MIS-weight a BSDF-sampled ray landing directly on this emissive
                // material - full unweighted
                // emission if this material isn't a registered light-list entry
                // (lightPickAreaPdf <= 0, e.g. a custom primitive/volume-grid hit,
                // which the light list never indexes) or the previous bounce sampled a
                // delta lobe/was the primary ray (bsdfPdfIn's sentinel), since neither
                // case has a valid competing NEE pdf to combine against.
                float emissionWeight = 1f;
                if (lightPickAreaPdf > 0f && bsdfPdfIn > 0f)
                {
                    float hitDist = OptixGetRayTmax.Value;
                    float cosAtLight = Vec3.dot(outwardNormal, -rayDir);
                    emissionWeight = cosAtLight > 1e-6f
                        ? NextEventEstimation.PowerHeuristic(bsdfPdfIn, lightPickAreaPdf * hitDist * hitDist / cosAtLight)
                        : 0f;
                }
                pixelColor += emission * emissionWeight;
            }

            LCG rng = new LCG(rngStateIn);

            if (!isDelta)
                pixelColor += NextEventEstimation.SampleDirectLighting(launchParams, surfPos, shadingNormal, outwardNormal, viewDir, baseColor, metallic, roughness, f0, ref rng);
            // else: a delta specular lobe's BRDF is 0 everywhere except the exact mirror
            // direction, which NEE (which only ever samples finite-solid-angle/point
            // directions) can never land on - it correctly contributes nothing here,
            // matching a real mirror.

            Vec3 newOrigin = surfPos + (1e-3f * outwardNormal);

            if (isDelta)
            {
                Vec3 reflectDir = Vec3.reflect(shadingNormal, rayDir);
                // Angle-dependent Fresnel (not just the normal-incidence f0) - without
                // this, a delta mirror never shows the grazing-angle brightening a real
                // mirror has, and there's a visible brightness pop as Roughness crosses
                // DeltaRoughnessThreshold from the rough branch below (which already
                // Fresnel-weights via EvalSpecularBRDF's own FresnelSchlick call).
                float NdotV = XMath.Clamp(Vec3.dot(shadingNormal, viewDir), 0f, 1f);
                Vec3 mirrorTint = Bsdf.FresnelSchlick(NdotV, f0);
                Payloads.SetContinuePayload(pixelColor, Payloads.BOUNCE_CONTINUE_MIRROR, newOrigin, reflectDir, mirrorTint, shadingNormal, baseColor, rng.State, Payloads.DeltaOrPrimarySentinel, surfPos);
                return;
            }

            // Single-sample stochastic lobe selection: pick diffuse or specular by a
            // fixed probability derived from the material's reflectance, then weight the
            // *combined* contribution of both lobes at whichever direction was actually
            // sampled by their *combined* pdf - the standard unbiased multi-lobe MC
            // estimator (not full per-lobe MIS, but correct and unbiased).
            float specWeight = Bsdf.SpecularLobeWeight(f0);
            bool chooseSpecular = rng.Next() < specWeight;

            Vec3 sampledDir = chooseSpecular
                ? Bsdf.SampleSpecularDirection(shadingNormal, viewDir, roughness, ref rng)
                : SampleCosineHemisphere(shadingNormal, ref rng);

            float NdotSampled = Vec3.dot(shadingNormal, sampledDir);
            if (NdotSampled <= 0f)
            {
                Payloads.SetTerminalPayload(pixelColor, shadingNormal, baseColor, surfPos);
                return;
            }

            Vec3 diffuseF = Bsdf.EvalDiffuseBRDF(baseColor, metallic);
            Vec3 specularF = Bsdf.EvalSpecularBRDF(shadingNormal, viewDir, sampledDir, roughness, f0);
            float combinedPdf = Bsdf.CombinedPdf(shadingNormal, viewDir, sampledDir, roughness, f0);
            if (combinedPdf <= 1e-6f)
            {
                Payloads.SetTerminalPayload(pixelColor, shadingNormal, baseColor, surfPos);
                return;
            }

            Vec3 tint = ((diffuseF + specularF) * NdotSampled) / combinedPdf;
            uint bounceFlag = chooseSpecular ? Payloads.BOUNCE_CONTINUE_MIRROR : Payloads.BOUNCE_CONTINUE_DIFFUSE;
            Payloads.SetContinuePayload(pixelColor, bounceFlag, newOrigin, sampledDir, tint, shadingNormal, baseColor, rng.State, combinedPdf, surfPos);
        }

        // Fresnel-weighted stochastic reflect-XOR-refract pick (not the reference's
        // branch-both-paths-at-once) - converges to the same result over many
        // accumulated frames given the existing progressive accumulation. Total
        // internal reflection forces the reflect branch unconditionally (no valid
        // refraction direction exists). A fractional Transmission isn't currently
        // exercised by any scene in this sample (materials use 1.0) - if a future
        // scene wants partial transmission, revisit how it should modulate this
        // probability split.
        //
        // Rough dielectric transmission (Walter et al. 2007) -
        // TransmissionRoughness < DeltaRoughnessThreshold (the struct default, 0)
        // reproduces perfect-specular glass as a degenerate case (same delta-lobe
        // convention as the opaque GGX lobe's own Roughness threshold). Otherwise, a
        // microfacet normal is VNDF-sampled the same way the opaque specular lobe does
        // (Bsdf.SampleVisibleNormal), then reflected or refracted *about that
        // microfacet*, not the geometric/shading normal - Fresnel, TryRefract, and the
        // reflect direction all take the sampled microfacet in place of outwardNormal.
        // The VNDF-sampled
        // lobe's unbiased throughput weight simplifies to the Smith masking ratio
        // G2/G1(v) (Heitz's well-known VNDF-sampling simplification) - identical form
        // for both the reflect and refract branches, differing only in which direction
        // NdotL is measured toward.
        internal unsafe static void ShadeDielectric(LaunchParams launchParams, MaterialSbtData* sbtData, Vec3 rayDir, Vec3 outwardNormal, bool frontFace, Vec3 surfPos, uint rngStateIn)
        {
            float etaFrom = frontFace ? 1f : sbtData->IOR;
            float etaTo = frontFace ? sbtData->IOR : 1f;
            float eta = etaFrom / etaTo;

            LCG rng = new LCG(rngStateIn);

            float roughness = sbtData->TransmissionRoughness;
            bool isDelta = roughness < Bsdf.DeltaRoughnessThreshold;

            Vec3 microNormal = outwardNormal;
            if (!isDelta)
            {
                Vec3 viewDir = Vec3.unitVector(-rayDir);
                microNormal = Bsdf.SampleVisibleNormal(outwardNormal, viewDir, roughness, ref rng);
                // VNDF sampling can occasionally return a microfacet normal on the
                // wrong side at grazing angles on a very rough surface - falling back
                // to the geometric normal is a safe, rare-case degradation (not a
                // correctness issue for the common case).
                if (Vec3.dot(microNormal, outwardNormal) <= 0f)
                    microNormal = outwardNormal;
            }

            bool didRefract = TryRefract(rayDir, microNormal, eta, out Vec3 refractDir);

            float cosTheta = XMath.Clamp(Vec3.dot(-rayDir, microNormal), 0f, 1f);
            float fresnel = didRefract ? Bsdf.FresnelDielectric(cosTheta, etaFrom, etaTo) : 1f;
            fresnel = XMath.Clamp(fresnel + (sbtData->Metallic * (1f - fresnel)), 0f, 1f);

            bool chooseReflect = !didRefract || rng.Next() < fresnel;

            Vec3 newDir;
            Vec3 tint;
            Vec3 offsetNormal;
            if (chooseReflect)
            {
                newDir = Vec3.reflect(microNormal, rayDir);
                tint = new Vec3(1f, 1f, 1f);
                offsetNormal = outwardNormal;
            }
            else
            {
                newDir = refractDir;
                tint = sbtData->TransmissionColor;
                offsetNormal = -outwardNormal;
            }

            if (!isDelta)
            {
                float NdotV = XMath.Clamp(Vec3.dot(outwardNormal, -rayDir), 1e-4f, 1f);
                float NdotL = XMath.Abs(Vec3.dot(outwardNormal, newDir));
                float g1v = Bsdf.SmithG1Public(NdotV, roughness);
                float weight = g1v > 1e-6f ? Bsdf.SmithG2(NdotL, NdotV, roughness) / g1v : 0f;
                tint *= weight;
            }

            Vec3 newOrigin = surfPos + (1e-3f * offsetNormal);
            Payloads.SetContinuePayload(sbtData->Emission, Payloads.BOUNCE_CONTINUE_DIELECTRIC, newOrigin, newDir, tint, outwardNormal, tint, rng.State, Payloads.DeltaOrPrimarySentinel, surfPos);
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

        // Cosine-weighted hemisphere sampling: generate a random direction in the
        // hemisphere around the normal, biased toward the normal itself. Uses the
        // standard approach: uniform sample on disk, then lift to hemisphere.
        // rng is by ref (matching Bsdf.SampleSpecularDirection's own convention) - LCG
        // is a struct, so a by-value rng here would silently discard this call's two
        // Next() draws from the caller's carried RNG stream instead of advancing it.
        private static Vec3 SampleCosineHemisphere(Vec3 normal, ref LCG rng)
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

        // Direct port of the reference's Checker closure (position-evaluated instead of a
        // captured Func<>) - floor(pos.X/scale) + floor(pos.Z/scale) parity picks between
        // the two flat colors.
        internal static Vec3 SampleChecker(MaterialSbtData mat, Vec3 pos)
        {
            int cx = (int)XMath.Floor(pos.x / mat.CheckerScale);
            int cz = (int)XMath.Floor(pos.z / mat.CheckerScale);
            bool even = ((cx + cz) & 1) == 0;
            return even ? mat.BaseColor : mat.CheckerColorB;
        }

        // BaseColorTexture == 0 means no texture. UV is only ever non-zero for
        // triangle hits (custom primitives have no UV parameterization in this
        // sample), so a textured custom primitive is not currently possible - not
        // needed by any scene in this sample.
        internal unsafe static Vec3 SampleAlbedo(MaterialSbtData* sbtData, Vec3 surfPos, Vec2 uv)
        {
            Vec3 baseColor = sbtData->MaterialKind == MaterialSbtData.Checker
                ? SampleChecker(*sbtData, surfPos)
                : sbtData->BaseColor;

            if (sbtData->BaseColorTexture == 0)
                return baseColor;

            var (tr, tg, tb, _) = CudaTex2D.Sample(sbtData->BaseColorTexture, uv.x * sbtData->UVScale, uv.y * sbtData->UVScale);
            // sRGB decode - base-color textures
            // are 8-bit sRGB-encoded source assets (TextureLoader.LoadRgba8 uploads
            // the raw bytes with no gamma handling at all), but every other quantity
            // in this renderer (GGX energy conservation, HDRI radiance, the scalar
            // BaseColor literals scene builders author directly) is linear - a flat
            // pow(x, 2.2) approximates the sRGB EOTF closely enough for this sample's
            // purposes (the exact piecewise curve is used for the tonemap's own OETF
            // instead - see TonemapKernel.cs). Applied only to the texture sample, not
            // the scalar BaseColor
            // fallback above (already linear) or ORM/normal textures (never sRGB data).
            Vec3 texSample = new Vec3(XMath.Pow(tr, 2.2f), XMath.Pow(tg, 2.2f), XMath.Pow(tb, 2.2f));
            return (baseColor * (1f - sbtData->TextureWeight)) + (texSample * sbtData->TextureWeight);
        }

        // Tangent-space normal map perturbation - standard OpenGL-convention normal map (texel.x/.y/.z map to the tangent/
        // bitangent/normal axes respectively after the [0,1] -> [-1,1] remap).
        // NormalStrength blends between the unperturbed shadingNormal (0) and the full
        // texture-driven perturbation (1); bitangent is reconstructed via cross(normal,
        // tangent) - this sample's minimal OBJ loader doesn't track UV-mirroring
        // handedness, so a mirrored-UV mesh could show an inverted bitangent (not
        // exercised by any bundled asset).
        internal unsafe static Vec3 ApplyNormalMap(MaterialSbtData* sbtData, Vec2 uv, Vec3 shadingNormal, Vec3 tangent)
        {
            var (nr, ng, nb, _) = CudaTex2D.Sample(sbtData->NormalTexture, uv.x * sbtData->UVScale, uv.y * sbtData->UVScale);
            Vec3 texNormal = new Vec3((2f * nr) - 1f, (2f * ng) - 1f, (2f * nb) - 1f);

            Vec3 bitangent = Vec3.cross(shadingNormal, tangent);
            Vec3 perturbed = Vec3.unitVector(
                (tangent * texNormal.x) + (bitangent * texNormal.y) + (shadingNormal * texNormal.z));

            float strength = XMath.Clamp(sbtData->NormalStrength, 0f, 1f);
            return Vec3.unitVector((shadingNormal * (1f - strength)) + (perturbed * strength));
        }

        // Alpha-cutout support (Sponza's leaf geometry: a solid quad with the leaf
        // shape punched out via the diffuse texture's own alpha channel). Only called
        // from the any-hit programs, which already checked AlphaCutoff > 0 and
        // BaseColorTexture != 0 before calling this.
        internal unsafe static float SampleAlpha(MaterialSbtData* sbtData, Vec2 uv)
        {
            var (_, _, _, ta) = CudaTex2D.Sample(sbtData->BaseColorTexture, uv.x * sbtData->UVScale, uv.y * sbtData->UVScale);
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
