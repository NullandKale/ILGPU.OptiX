using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;

namespace Sample15
{
    /// <summary>
    /// Material shading branches (unified GGX metallic-roughness surface/dielectric/
    /// volume-grid - see Bsdf.cs for the GGX math itself) plus the texture sampling and
    /// shadow-ray helpers they share. All methods are static and get inlined by ILGPU
    /// into whichever OptiX program calls them.
    /// </summary>
    public static class ShadingHelpers
    {
        // Unified metallic-roughness GGX BSDF shading (docs/SAMPLE15_PLAN.md Milestones
        // M2/M4) - replaces Sample14/M1's separate Oren-Nayar-diffuse/perfect-mirror
        // branches with one continuous function, shared between the ordinary dispatch
        // branch in __closest__radiance and the volume grid's per-voxel material lookup
        // (ShadeVolumeGrid). Direct lighting is next-event estimation against the
        // unified light list (NextEventEstimation.cs); indirect lighting stochastically
        // picks one lobe to extend the path with, weighted by both lobes' combined pdf
        // at the sampled direction (see Bsdf.cs and docs/SAMPLE15_PLAN.md Design
        // Decision 1/3). bsdfPdfIn/lightPickAreaPdf implement the MIS side of a
        // BSDF-sampled ray landing directly on this material's own emission - see
        // Payloads.cs's own doc comment and ClosestHitProgram.cs's callsite.
        internal unsafe static void ShadeSurface(LaunchParams launchParams, MaterialSbtData mat, Vec3 baseColor, Vec3 surfPos, Vec3 shadingNormal, Vec3 outwardNormal, Vec3 rayDir, uint rngStateIn, float bsdfPdfIn, float lightPickAreaPdf, Vec2 uv)
        {
            float roughness = mat.Roughness;
            float metallic = mat.Metallic;
            // ORM texture (docs/SAMPLE15_PLAN.md Milestone M6) - occlusion.r (unused;
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
            // albedo for a flat ambient term to reflect into (a proper environment would
            // still light metals via the specular lobe; this flat constant-ambient hack
            // predates GGX and isn't a real irradiance environment to reflect - M5's HDRI
            // replaces it and the specular lobe will sample it correctly once wired up).
            Vec3 pixelColor = launchParams.AmbientColor * launchParams.AmbientIntensity * baseColor * (1f - metallic);

            Vec3 emission = mat.Emission * mat.EmissionStrength;
            if (emission.lengthSquared() > 0f)
            {
                // MIS-weight a BSDF-sampled ray landing directly on this emissive
                // material (docs/SAMPLE15_PLAN.md Milestone M4) - full unweighted
                // emission if this material isn't a registered light-list entry
                // (lightPickAreaPdf <= 0, e.g. a custom primitive/volume-grid hit,
                // which the light list never indexes) or the previous bounce sampled a
                // delta lobe/was the primary ray (bsdfPdfIn's sentinel), since neither
                // case has a valid competing NEE pdf to combine against.
                float emissionWeight = 1f;
                if (launchParams.NeeEnabled != 0 && lightPickAreaPdf > 0f && bsdfPdfIn > 0f)
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

            if (!isDelta && launchParams.NeeEnabled != 0)
                pixelColor += NextEventEstimation.SampleDirectLighting(launchParams, surfPos, shadingNormal, outwardNormal, viewDir, baseColor, metallic, roughness, f0, ref rng);
            // else: a delta specular lobe's BRDF is 0 everywhere except the exact mirror
            // direction, which NEE (which only ever samples finite-solid-angle/point
            // directions) can never land on - it correctly contributes nothing here,
            // matching a real mirror.

            Vec3 newOrigin = surfPos + (1e-3f * outwardNormal);

            if (isDelta)
            {
                Vec3 reflectDir = Vec3.reflect(shadingNormal, rayDir);
                Payloads.SetContinuePayload(pixelColor, Payloads.BOUNCE_CONTINUE_MIRROR, newOrigin, reflectDir, f0, shadingNormal, baseColor, rng.State, Payloads.DeltaOrPrimarySentinel);
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
                Payloads.SetTerminalPayload(pixelColor, shadingNormal, baseColor);
                return;
            }

            Vec3 diffuseF = Bsdf.EvalDiffuseBRDF(baseColor, metallic);
            Vec3 specularF = Bsdf.EvalSpecularBRDF(shadingNormal, viewDir, sampledDir, roughness, f0);
            float combinedPdf = Bsdf.CombinedPdf(shadingNormal, viewDir, sampledDir, roughness, f0);
            if (combinedPdf <= 1e-6f)
            {
                Payloads.SetTerminalPayload(pixelColor, shadingNormal, baseColor);
                return;
            }

            Vec3 tint = ((diffuseF + specularF) * NdotSampled) / combinedPdf;
            uint bounceFlag = chooseSpecular ? Payloads.BOUNCE_CONTINUE_MIRROR : Payloads.BOUNCE_CONTINUE_DIFFUSE;
            Payloads.SetContinuePayload(pixelColor, bounceFlag, newOrigin, sampledDir, tint, shadingNormal, baseColor, rng.State, combinedPdf);
        }

        // The volume grid is a single GAS primitive whose per-voxel material can't be
        // expressed via OptiX's per-primitive SbtIndexOffsetBuffer (that's keyed by
        // primitive index, not by data read during intersection) - so material lookup
        // goes directly through LaunchParams.Materials/VoxelMaterialIds instead of
        // OptixGetSbtDataPointer, a deliberate deviation from every other primitive kind
        // in this sample (see docs/SAMPLE13_PLAN.md's note on this and the comment on
        // LaunchParams.Materials).
        internal unsafe static void ShadeVolumeGrid(LaunchParams launchParams, Vec3 rayDir, uint faceCode, uint rngStateIn)
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

            Vec3 baseColor = mat.MaterialKind == MaterialSbtData.Checker ? SampleChecker(mat, surfPos) : mat.BaseColor;
            // lightPickAreaPdf is always 0 here - volume-grid voxels are never
            // registered in the NEE light list (LightList.cs only walks triangles), so
            // ShadeSurface's own MIS-emission branch never triggers regardless of
            // bsdfPdfIn's value.
            ShadeSurface(launchParams, mat, baseColor, surfPos, outwardNormal, outwardNormal, rayDir, rngStateIn, Payloads.DeltaOrPrimarySentinel, 0f, new Vec2(0f, 0f));
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

        // Fresnel-weighted stochastic reflect-XOR-refract pick (not the reference's
        // branch-both-paths-at-once) - converges to the same result over many
        // accumulated frames given the existing progressive accumulation. Total
        // internal reflection forces the reflect branch unconditionally (no valid
        // refraction direction exists). A fractional Transmission isn't currently
        // exercised by any Sample15 scene (materials use 1.0, matching Sample14's
        // actual glass materials) - if a future scene wants partial transmission,
        // revisit how it should modulate this probability split.
        //
        // Rough dielectric transmission (docs/SAMPLE15_PLAN.md Milestone M7, Design
        // Decision 3 - Walter et al. 2007) - TransmissionRoughness < DeltaRoughnessThreshold
        // (the struct default, 0) reproduces exactly the old perfect-specular path as a
        // degenerate case (same delta-lobe convention as the opaque GGX lobe's own
        // Roughness threshold). Otherwise, a microfacet normal is VNDF-sampled the same
        // way the opaque specular lobe does (Bsdf.SampleVisibleNormal), then reflected
        // or refracted *about that microfacet*, not the geometric/shading normal -
        // Fresnel, TryRefract, and the reflect direction all take the sampled
        // microfacet in place of outwardNormal, which is the "mechanical
        // generalization" of the delta case the plan called for. The VNDF-sampled
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
            Payloads.SetContinuePayload(sbtData->Emission, Payloads.BOUNCE_CONTINUE_DIELECTRIC, newOrigin, newDir, tint, outwardNormal, tint, rng.State, Payloads.DeltaOrPrimarySentinel);
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

        // Multi-hit colored transmittance shadow ray (docs/SAMPLE13_PLAN.md design
        // decision (f)) - walks through transparent occluders via __anyhit__shadow's
        // optixIgnoreIntersection, accumulating TransmissionColor*Transparency per hit
        // up to MaxRefractions, unlike a plain binary occlusion test.
        internal unsafe static Vec3 ShadowTransmittance(LaunchParams launchParams, Vec3 surfPos, Vec3 outwardNormal, Vec3 lightDir, float lightDist)
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
                OptixRayFlags.OPTIX_RAY_FLAG_NONE,
                Payloads.SHADOW_RAY_TYPE,
                Payloads.RAY_TYPE_COUNT,
                Payloads.SHADOW_RAY_TYPE,
                ref t0, ref t1, ref t2, ref hitCount);

            return new Vec3(Interop.IntAsFloat(t0), Interop.IntAsFloat(t1), Interop.IntAsFloat(t2));
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

        // Direct port of Sample14's SampleAlbedo - BaseColorTexture == 0 means no texture
        // (matches Sample08's convention). UV is only ever non-zero for triangle hits
        // (custom primitives have no UV parameterization in this sample), so a textured
        // custom primitive is not currently possible - not needed by any Sample15 scene.
        internal unsafe static Vec3 SampleAlbedo(MaterialSbtData* sbtData, Vec3 surfPos, Vec2 uv)
        {
            Vec3 baseColor = sbtData->MaterialKind == MaterialSbtData.Checker
                ? SampleChecker(*sbtData, surfPos)
                : sbtData->BaseColor;

            if (sbtData->BaseColorTexture == 0)
                return baseColor;

            var (tr, tg, tb, _) = CudaTex2D.Sample(sbtData->BaseColorTexture, uv.x * sbtData->UVScale, uv.y * sbtData->UVScale);
            // sRGB decode (docs/SAMPLE15_PLAN.md Milestone M6) - base-color textures
            // are 8-bit sRGB-encoded source assets (TextureLoader.LoadRgba8 uploads
            // the raw bytes with no gamma handling at all), but every other quantity
            // in this renderer (GGX energy conservation, HDRI radiance, the scalar
            // BaseColor literals scene builders author directly) is linear - a flat
            // pow(x, 2.2) approximates the sRGB EOTF closely enough for this sample's
            // purposes (the exact piecewise curve is reserved for the tonemap's own
            // OETF, M8). Applied only to the texture sample, not the scalar BaseColor
            // fallback above (already linear) or ORM/normal textures (never sRGB data).
            Vec3 texSample = new Vec3(XMath.Pow(tr, 2.2f), XMath.Pow(tg, 2.2f), XMath.Pow(tb, 2.2f));
            return (baseColor * (1f - sbtData->TextureWeight)) + (texSample * sbtData->TextureWeight);
        }

        // Tangent-space normal map perturbation (docs/SAMPLE15_PLAN.md Milestone M6) -
        // standard OpenGL-convention normal map (texel.x/.y/.z map to the tangent/
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
