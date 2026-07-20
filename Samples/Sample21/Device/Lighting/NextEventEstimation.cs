using ILGPU.Algorithms;

namespace Sample21
{
    /// <summary>
    /// Next-event estimation against the unified light list -
    /// picks one light via the scene's power-weighted
    /// CDF, samples a point on it, shadow-tests it, and evaluates the surface's BSDF
    /// toward it. Point lights are delta lights (no MIS competitor - a BSDF-sampled ray
    /// can never land on a measure-zero point, so their NEE sample is always the sole
    /// estimator); emissive triangles are finite-pdf area lights, MIS-combined against
    /// the surface's own BSDF pdf via the power heuristic, symmetric with the
    /// BSDF-sampled-ray-hits-a-light case handled in MaterialShading.ShadeSurface.
    /// </summary>
    internal static class NextEventEstimation
    {
        // Firefly clamp on the returned NEE contribution's luminance - standard
        // path-tracing practice, not a VkNRC-ported mechanism (this file has no VkNRC
        // equivalent to port from). Point lights here are raw, unclamped inverse-square
        // (radiance = Intensity/dist2 below) with no minimum-distance floor - a shading
        // point a few units from an Intensity=40 point light already returns a
        // single-sample radiance of 10-40, and NEE's own Vec3.dot(diag)/pdfLight
        // division can amplify that further when a light's picking probability is low,
        // compounding across bounces (RaygenProgram.cs's accumulate/nrcSuffixFactor)
        // into firefly spikes that drive NRC self-training-bootstrap divergence.
        // Applied to every NEE sample (not just NRC training records) - the value is
        // high enough that it only ever engages on genuine outliers, so it acts as a
        // bias-negligible variance reduction for the final image too (the same trick
        // most production path tracers use), not just an NRC-specific workaround.
        private const float MaxNeeLuminance = 10f;

        internal static float PowerHeuristic(float pdfA, float pdfB)
        {
            float a2 = pdfA * pdfA;
            float b2 = pdfB * pdfB;
            float denom = a2 + b2;
            return denom > 0f ? a2 / denom : 0f;
        }

        internal unsafe static Vec3 SampleDirectLighting(LaunchParams launchParams, Vec3 surfPos, Vec3 shadingNormal, Vec3 outwardNormal, Vec3 viewDir, Vec3 baseColor, float metallic, float roughness, Vec3 f0, ref LCG rng)
        {
            int numLights = launchParams.NumNeeLights;
            if (numLights == 0)
                return new Vec3(0f, 0f, 0f);

            // Binary search over the (monotonically increasing) CDF - O(log N) instead
            // of a linear scan. Scenes like Sponza register one light-list entry per
            // emissive triangle (LightList.cs), which can reach tens of thousands of
            // entries (e.g. Sponza's hand-promoted emissive vases), where a linear scan
            // would cost thousands of iterations per NEE sample per bounce per pixel.
            float u = rng.Next();
            int lo = 0, hi = numLights - 1;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (u <= launchParams.NeeLightCdf[mid])
                    hi = mid;
                else
                    lo = mid + 1;
            }
            int lightIdx = lo;

            LightGpu light = launchParams.NeeLights[lightIdx];
            if (light.Pdf <= 0f)
                return new Vec3(0f, 0f, 0f);

            bool isDeltaLight = light.Kind == LightGpu.KindPoint;
            Vec3 lightDir, radiance;
            float dist, pdfLight;

            if (isDeltaLight)
            {
                PointLightGpu pointLight = launchParams.PointLights[light.RefIndex];
                Vec3 toLight = pointLight.Position - surfPos;
                float dist2 = toLight.lengthSquared();
                dist = XMath.Sqrt(dist2);
                lightDir = toLight / dist;
                radiance = pointLight.Color * (pointLight.Intensity / dist2);
                // No solid-angle density for a delta light - dividing by the picking
                // pdf alone is the correct unbiased estimator (the light-selection
                // probability is the only randomness in how this sample was drawn).
                pdfLight = light.Pdf;
            }
            else if (light.Kind == LightGpu.KindEnvMap)
            {
                // "At infinity" - ShadowRay.Trace's shadow ray needs a tmax well
                // past any scene geometry, matching the camera/bounce rays' own 1e20f
                // ceiling in RaygenProgram.cs (one order of magnitude down so its own
                // (1 - 1e-3) tmax shrink still leaves ample margin).
                const float envMapDistance = 1e19f;
                dist = envMapDistance;
                lightDir = EnvironmentMapSampling.Sample(launchParams, ref rng, out radiance, out float pdfSolidAngle);
                pdfLight = light.Pdf * pdfSolidAngle;
                if (pdfSolidAngle <= 0f)
                    return new Vec3(0f, 0f, 0f);
            }
            else
            {
                Vec3i idx = launchParams.Indices[light.RefIndex];
                Vec3 a = launchParams.Vertices[idx.x];
                Vec3 b = launchParams.Vertices[idx.y];
                Vec3 c = launchParams.Vertices[idx.z];

                // Uniform sample on the triangle via the standard sqrt-based
                // barycentric map would need an extra sqrt - the reflect-across-the-
                // diagonal trick (fold (su, sv) back into the triangle when
                // su + sv > 1) is the cheaper equivalent for a uniform-area sample.
                float su = rng.Next();
                float sv = rng.Next();
                if (su + sv > 1f)
                {
                    su = 1f - su;
                    sv = 1f - sv;
                }
                Vec3 samplePos = a + ((b - a) * su) + ((c - a) * sv);

                Vec3 triNormal = Vec3.unitVector(Vec3.cross(b - a, c - a));
                Vec3 toLight = samplePos - surfPos;
                float dist2 = toLight.lengthSquared();
                dist = XMath.Sqrt(dist2);
                lightDir = toLight / dist;

                // Two-sided emitter (abs) - Sponza's hand-promoted emissive vase
                // materials have no reliable consistent winding to assume one-sidedness
                // from.
                float cosAtLight = XMath.Abs(Vec3.dot(triNormal, lightDir));
                if (cosAtLight <= 1e-6f)
                    return new Vec3(0f, 0f, 0f);

                float area = 0.5f * Vec3.cross(b - a, c - a).length();
                pdfLight = light.Pdf * dist2 / (area * cosAtLight);
                radiance = light.Emission;
            }

            float NdotL = Vec3.dot(shadingNormal, lightDir);
            if (NdotL <= 0f)
                return new Vec3(0f, 0f, 0f);

            Vec3 transmittance = ShadowRay.Trace(launchParams, surfPos, outwardNormal, lightDir, dist);
            if (transmittance.lengthSquared() <= 0f)
                return new Vec3(0f, 0f, 0f);

            Vec3 diffuseF = Bsdf.EvalDiffuseBRDF(baseColor, metallic);
            Vec3 specularF = Bsdf.EvalSpecularBRDF(shadingNormal, viewDir, lightDir, roughness, f0);

            float misWeight = 1f;
            if (!isDeltaLight)
            {
                float bsdfPdf = Bsdf.CombinedPdf(shadingNormal, viewDir, lightDir, roughness, f0);
                misWeight = PowerHeuristic(pdfLight, bsdfPdf);
            }

            Vec3 contribution = (diffuseF + specularF) * NdotL * radiance * transmittance * (misWeight / pdfLight);

            float contribLuminance = (0.2126f * contribution.x) + (0.7152f * contribution.y) + (0.0722f * contribution.z);
            if (contribLuminance > MaxNeeLuminance)
                contribution *= MaxNeeLuminance / contribLuminance;

            return contribution;
        }
    }
}
