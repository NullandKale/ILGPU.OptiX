using ILGPU.Algorithms;

namespace Sample15
{
    // Metallic-roughness GGX microfacet BSDF (docs/SAMPLE15_PLAN.md Design Decision 3):
    // isotropic Trowbridge-Reitz/GGX normal distribution, Smith height-correlated joint
    // masking-shadowing (Heitz 2014, "Understanding the Masking-Shadowing Function in
    // Microfacet-Based BRDFs"), Schlick Fresnel with F0 = lerp(0.04, baseColor, metallic)
    // (the glTF/Disney/UE4 metallic-roughness convention), and GGX VNDF importance
    // sampling (Heitz 2018, "Sampling the GGX Distribution of Visible Normals").
    //
    // Reflection only - the dielectric transmission (glass) lobe stays on
    // MaterialShading.ShadeDielectric's perfect-specular Fresnel-Schlick path until M7's
    // rough BTDF (Walter et al. 2007) replaces it. Diffuse + specular are combined via
    // single-sample stochastic lobe selection (M2's ShadeSurface picks one lobe to
    // extend the path with, weighted by the *combined* pdf of both lobes at the sampled
    // direction - the standard unbiased multi-lobe MC estimator), not per-lobe MIS.
    internal static class Bsdf
    {
        // Roughness at or below this is treated as a perfect mirror (delta lobe) instead
        // of evaluating GGX - the D/Vis formulas below divide by alpha^2 in ways that
        // blow up (NaN/Inf) as alpha approaches 0, and a delta lobe has no well-defined
        // finite BRDF value anyway. This is what keeps MaterialPresets.Mirror() (Roughness
        // defaults to 0) rendering as an exact mirror, matching Sample14/M1's behavior.
        internal const float DeltaRoughnessThreshold = 0.02f;

        // Perceptual-roughness-to-alpha remap (roughness^2, the glTF/Disney convention) -
        // floored well above 0 since this is only ever called once DeltaRoughnessThreshold
        // has already routed the near-0 case to the mirror fast path.
        internal static float RoughnessToAlpha(float roughness) => XMath.Max(roughness * roughness, 1e-4f);

        internal static Vec3 SpecularF0(Vec3 baseColor, float metallic) =>
            new Vec3(0.04f, 0.04f, 0.04f) + (metallic * (baseColor - new Vec3(0.04f, 0.04f, 0.04f)));

        // D blows up like 1/alpha^2 as Roughness approaches DeltaRoughnessThreshold
        // from above (~2e6 at Roughness=0.02, vs ~3e3 at Roughness=0.1) - for a
        // material in that fragile near-delta band, NextEventEstimation's direct-
        // light sample toward a point light (which gets no MIS counterweight - see
        // SampleDirectLighting's isDeltaLight branch, always misWeight=1) can land
        // near the lobe's peak and produce a flickering, per-frame "firefly" pixel.
        // Clamping D bounds this for both EvalSpecularBRDF (NEE) and PdfSpecular
        // (the indirect bounce's importance-sampling denominator) consistently,
        // since both pull from this same function - their D-term cancellation (the
        // bounded specularF/PdfSpecular ratio Design Decision 3 relies on) is
        // unaffected. The cap sits well above any Roughness gtrsim 0.08's natural
        // peak, so it only engages in the fragile near-delta band, not general use.
        private const float MaxNormalDistribution = 20000f;

        // Trowbridge-Reitz/GGX normal distribution function.
        private static float DistributionGGX(float NdotH, float alpha)
        {
            float a2 = alpha * alpha;
            float d = (NdotH * NdotH * (a2 - 1f)) + 1f;
            return XMath.Min(a2 / (XMath.PI * XMath.Max(d * d, 1e-8f)), MaxNormalDistribution);
        }

        // Smith masking function for a single direction (used for the VNDF sampling pdf,
        // not the joint BRDF visibility term below - see VisibilitySmithGGXCorrelated).
        private static float SmithG1(float NdotX, float alpha)
        {
            float a2 = alpha * alpha;
            return (2f * NdotX) / (NdotX + XMath.Sqrt(a2 + ((1f - a2) * NdotX * NdotX)));
        }

        // Smith height-correlated joint masking-shadowing term, already divided by the
        // BRDF's 4*NdotL*NdotV denominator (Heitz 2014 eq. 72) - EvalSpecularBRDF
        // multiplies D*Vis*F directly, no separate division needed.
        private static float VisibilitySmithGGXCorrelated(float NdotL, float NdotV, float alpha)
        {
            float a2 = alpha * alpha;
            float lambdaV = NdotL * XMath.Sqrt((NdotV * NdotV * (1f - a2)) + a2);
            float lambdaL = NdotV * XMath.Sqrt((NdotL * NdotL * (1f - a2)) + a2);
            float denom = lambdaV + lambdaL;
            return denom > 0f ? 0.5f / denom : 0f;
        }

        internal static Vec3 FresnelSchlick(float cosTheta, Vec3 f0)
        {
            float x = XMath.Clamp(1f - cosTheta, 0f, 1f);
            float x5 = x * x * x * x * x;
            return f0 + ((new Vec3(1f, 1f, 1f) - f0) * x5);
        }

        // Full unpolarized dielectric Fresnel reflectance (docs/SAMPLE15_PLAN.md
        // Design Decision 3/Milestone M7, ported from optixIntro_06) - replaces the
        // Schlick approximation MaterialShading.ShadeDielectric used through M1-M6.
        // etaFrom/etaTo are the refractive indices of the medium containing cosThetaI's
        // direction and the medium being entered, respectively (same convention as
        // MaterialShading.TryRefract's niOverNt). Returns 1 (full reflection) for total
        // internal reflection.
        internal static float FresnelDielectric(float cosThetaI, float etaFrom, float etaTo)
        {
            float ci = XMath.Clamp(cosThetaI, 0f, 1f);
            float eta = etaFrom / etaTo;
            float sin2ThetaT = eta * eta * XMath.Max(0f, 1f - (ci * ci));
            if (sin2ThetaT >= 1f)
                return 1f;

            float cosThetaT = XMath.Sqrt(1f - sin2ThetaT);
            float rParallel = ((etaTo * ci) - (etaFrom * cosThetaT)) / ((etaTo * ci) + (etaFrom * cosThetaT));
            float rPerp = ((etaFrom * ci) - (etaTo * cosThetaT)) / ((etaFrom * ci) + (etaTo * cosThetaT));
            return ((rParallel * rParallel) + (rPerp * rPerp)) * 0.5f;
        }

        // Builds an arbitrary orthonormal (right, forward) tangent basis around a normal -
        // shared by cosine-hemisphere diffuse sampling and VNDF specular sampling so both
        // lobes agree on the same local frame.
        internal static void BuildOnb(Vec3 normal, out Vec3 right, out Vec3 forward)
        {
            Vec3 up = XMath.Abs(normal.y) < 0.999f ? new Vec3(0f, 1f, 0f) : new Vec3(1f, 0f, 0f);
            right = Vec3.unitVector(Vec3.cross(up, normal));
            forward = Vec3.cross(normal, right);
        }

        // Full specular BRDF value (D*Vis*F) for an arbitrary light direction l - used both
        // for NEE against analytic point lights (l fixed by the light) and for evaluating
        // the specular lobe's contribution at a direction sampled from the *diffuse* lobe
        // (multi-lobe combination needs both lobes' f/pdf at whichever direction was
        // actually sampled, regardless of which lobe generated it).
        internal static Vec3 EvalSpecularBRDF(Vec3 n, Vec3 v, Vec3 l, float roughness, Vec3 f0)
        {
            float NdotL = XMath.Clamp(Vec3.dot(n, l), 0f, 1f);
            float NdotV = XMath.Clamp(Vec3.dot(n, v), 1e-4f, 1f);
            if (NdotL <= 0f)
                return new Vec3(0f, 0f, 0f);

            Vec3 h = Vec3.unitVector(v + l);
            float NdotH = XMath.Clamp(Vec3.dot(n, h), 0f, 1f);
            float VdotH = XMath.Clamp(Vec3.dot(v, h), 0f, 1f);

            float alpha = RoughnessToAlpha(roughness);
            float D = DistributionGGX(NdotH, alpha);
            float vis = VisibilitySmithGGXCorrelated(NdotL, NdotV, alpha);
            Vec3 F = FresnelSchlick(VdotH, f0);
            return D * vis * F;
        }

        // Solid-angle pdf of the VNDF-sampled direction l, given view direction v - the
        // VdotH factor in the half-vector-space pdf cancels exactly against the Jacobian
        // of the half-vector-to-incident-direction reflection map, leaving this simple
        // G1(v)*D/(4*NdotV) form (Heitz 2018 section 3.4/eq. 17-19 combined).
        internal static float PdfSpecular(Vec3 n, Vec3 v, Vec3 l, float roughness)
        {
            float NdotV = XMath.Clamp(Vec3.dot(n, v), 1e-4f, 1f);
            Vec3 h = Vec3.unitVector(v + l);
            float NdotH = XMath.Clamp(Vec3.dot(n, h), 0f, 1f);

            float alpha = RoughnessToAlpha(roughness);
            float D = DistributionGGX(NdotH, alpha);
            float g1 = SmithG1(NdotV, alpha);
            return (g1 * D) / (4f * NdotV);
        }

        // GGX VNDF importance sampling (Heitz 2018, "Sampling the GGX Distribution of
        // Visible Normals", listing 1) - samples a microfacet normal distributed
        // proportional to the *visible* normal distribution as seen from v. Operates in
        // the local shading frame (n, right, forward already orthonormal, v transformed
        // into that frame by the caller) so the algorithm's own z-up convention lines up
        // with the shading normal. Split out from the old SampleSpecularDirection
        // (docs/SAMPLE15_PLAN.md Milestone M7) so the rough dielectric BTDF
        // (MaterialShading.ShadeDielectric) can sample the same microfacet distribution
        // and then reflect *or* refract about it, instead of only ever reflecting.
        internal static Vec3 SampleVisibleNormal(Vec3 n, Vec3 v, float roughness, ref LCG rng)
        {
            BuildOnb(n, out Vec3 right, out Vec3 forward);
            Vec3 vLocal = new Vec3(Vec3.dot(v, right), Vec3.dot(v, forward), Vec3.dot(v, n));
            // v must be in the upper hemisphere (NdotV > 0) for VNDF sampling to be
            // well-defined - callers only reach here after already checking NdotV > 0.
            float alpha = RoughnessToAlpha(roughness);

            Vec3 vh = Vec3.unitVector(new Vec3(alpha * vLocal.x, alpha * vLocal.y, vLocal.z));

            float lenSq = (vh.x * vh.x) + (vh.y * vh.y);
            Vec3 t1 = lenSq > 0f
                ? new Vec3(-vh.y, vh.x, 0f) / XMath.Sqrt(lenSq)
                : new Vec3(1f, 0f, 0f);
            Vec3 t2 = Vec3.cross(vh, t1);

            float u1 = rng.Next();
            float u2 = rng.Next();
            float r = XMath.Sqrt(u1);
            float phi = 2f * XMath.PI * u2;
            float p1 = r * XMath.Cos(phi);
            float p2raw = r * XMath.Sin(phi);
            float s = 0.5f * (1f + vh.z);
            float p2 = ((1f - s) * XMath.Sqrt(XMath.Max(0f, 1f - (p1 * p1)))) + (s * p2raw);

            Vec3 nh = (p1 * t1) + (p2 * t2) + (XMath.Sqrt(XMath.Max(0f, 1f - (p1 * p1) - (p2 * p2))) * vh);

            Vec3 neLocal = Vec3.unitVector(new Vec3(alpha * nh.x, alpha * nh.y, XMath.Max(0f, nh.z)));
            return (neLocal.x * right) + (neLocal.y * forward) + (neLocal.z * n);
        }

        internal static Vec3 SampleSpecularDirection(Vec3 n, Vec3 v, float roughness, ref LCG rng)
        {
            Vec3 m = SampleVisibleNormal(n, v, roughness, ref rng);
            return Vec3.reflect(m, -v);
        }

        // Smith masking term for an arbitrary direction, exposed for the rough
        // dielectric BTDF's VNDF-sampling weight (docs/SAMPLE15_PLAN.md Milestone M7) -
        // SmithG1 above is already this, just renamed/exposed at the class's public
        // internal surface rather than kept private.
        internal static float SmithG1Public(float NdotX, float roughness) => SmithG1(NdotX, RoughnessToAlpha(roughness));

        // Smith height-correlated joint masking-shadowing term (not divided by
        // 4*NdotL*NdotV, unlike VisibilitySmithGGXCorrelated above) - the "G2" a
        // VNDF-sampled microfacet lobe's unbiased throughput weight (G2/G1) needs
        // (docs/SAMPLE15_PLAN.md Milestone M7).
        internal static float SmithG2(float NdotL, float NdotV, float roughness)
        {
            float alpha = RoughnessToAlpha(roughness);
            return VisibilitySmithGGXCorrelated(NdotL, NdotV, alpha) * 4f * NdotL * NdotV;
        }

        internal static Vec3 EvalDiffuseBRDF(Vec3 baseColor, float metallic) =>
            (baseColor * (1f - metallic)) * (1f / XMath.PI);

        internal static float PdfDiffuse(Vec3 n, Vec3 l) => XMath.Max(0f, Vec3.dot(n, l)) * (1f / XMath.PI);

        // Single-sample stochastic lobe-selection weight (docs/SAMPLE15_PLAN.md Design
        // Decision 1/3) - shared by ShadeSurface's own indirect-bounce lobe pick and
        // NextEventEstimation's direct-lighting BSDF evaluation, so both always agree
        // on the same combined pdf for a given direction (required for MIS's power
        // heuristic to be meaningful - docs/SAMPLE15_PLAN.md Milestone M4).
        internal static float SpecularLobeWeight(Vec3 f0) =>
            XMath.Clamp(XMath.Max(f0.x, XMath.Max(f0.y, f0.z)), 0.05f, 0.95f);

        // Combined diffuse+specular pdf at direction l, the same multi-lobe MC
        // estimator denominator ShadeSurface's own indirect bounce uses.
        internal static float CombinedPdf(Vec3 n, Vec3 v, Vec3 l, float roughness, Vec3 f0)
        {
            float specWeight = SpecularLobeWeight(f0);
            float diffusePdf = PdfDiffuse(n, l);
            float specularPdf = PdfSpecular(n, v, l, roughness);
            return (specWeight * specularPdf) + ((1f - specWeight) * diffusePdf);
        }
    }
}
