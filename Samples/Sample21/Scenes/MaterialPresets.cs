using System;

namespace Sample21
{
    /// <summary>
    /// Small factory helpers for the material configurations the scenes use over and
    /// over. Each returns a plain <see cref="MaterialSbtData"/>; anything more unusual
    /// (textured, emissive-tinted, partially reflective checker, ...) is still written
    /// as an inline initializer at the call site.
    /// </summary>
    public static class MaterialPresets
    {
        // Roughness defaults to 1 (fully rough/matte) here, not MaterialSbtData's own
        // struct-default 0 - Bsdf.ShadeSurface treats Roughness <
        // Bsdf.DeltaRoughnessThreshold as a perfect-mirror delta lobe, so an un-set
        // Roughness would render every "diffuse" Solid()/Checker() material as a
        // mirror.
        public static MaterialSbtData Solid(Vec3 baseColor, float metallic = 0f, float roughness = 1f) =>
            new MaterialSbtData { BaseColor = baseColor, Metallic = metallic, Roughness = roughness, MaterialKind = MaterialSbtData.Solid };

        // The near-white mirror every scene reuses - Roughness stays at the struct
        // default (0), which is what actually makes this a perfect mirror in the GGX
        // dispatch (Metallic alone doesn't select the mirror branch - it only blends
        // diffuse vs. Fresnel-tinted specular once Roughness is above
        // Bsdf.DeltaRoughnessThreshold).
        public static MaterialSbtData Mirror(float metallic = 0.90f) =>
            new MaterialSbtData { BaseColor = new Vec3(0.98f, 0.98f, 0.98f), Metallic = metallic, MaterialKind = MaterialSbtData.Solid };

        public static MaterialSbtData Glass(float ior, Vec3 transmissionColor) =>
            new MaterialSbtData { BaseColor = new Vec3(1f, 1f, 1f), Transmission = 1f, IOR = ior, TransmissionColor = transmissionColor, MaterialKind = MaterialSbtData.Solid };

        public static MaterialSbtData Checker(Vec3 colorA, Vec3 colorB, float scale, float roughness = 1f) =>
            new MaterialSbtData { BaseColor = colorA, Roughness = roughness, MaterialKind = MaterialSbtData.Checker, CheckerColorB = colorB, CheckerScale = scale };

        // HSV (h in degrees, s/v in [0,1]) to linear RGB, matching the reference's own
        // color-randomization formula used by the demo scene's random spheres and the
        // radial museum's RandSaturated.
        public static Vec3 HsvToRgb(float h, float s, float v)
        {
            float c = v * s;
            float hPrime = h / 60f;
            float x = c * (1f - Math.Abs(hPrime % 2f - 1f));
            float m = v - c;

            Vec3 rgb;
            if (hPrime < 1f) rgb = new Vec3(c, x, 0f);
            else if (hPrime < 2f) rgb = new Vec3(x, c, 0f);
            else if (hPrime < 3f) rgb = new Vec3(0f, c, x);
            else if (hPrime < 4f) rgb = new Vec3(0f, x, c);
            else if (hPrime < 5f) rgb = new Vec3(x, 0f, c);
            else rgb = new Vec3(c, 0f, x);

            return new Vec3(rgb.x + m, rgb.y + m, rgb.z + m);
        }
    }
}
