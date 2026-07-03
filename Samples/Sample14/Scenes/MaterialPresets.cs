using System;

namespace Sample14
{
    /// <summary>
    /// Small factory helpers for the material configurations the scenes use over and
    /// over. Each returns a plain <see cref="MaterialSbtData"/>; anything more unusual
    /// (textured, emissive-tinted, partially reflective checker, ...) is still written
    /// as an inline initializer at the call site.
    /// </summary>
    public static class MaterialPresets
    {
        public static MaterialSbtData Solid(Vec3 albedo, float reflectivity = 0f) =>
            new MaterialSbtData { Albedo = albedo, Reflectivity = reflectivity, MaterialKind = MaterialSbtData.Solid };

        // The near-white mirror every scene reuses - reflectivity >= 0.9 dispatches to
        // the mirror branch in __closest__radiance; lower values render as diffuse.
        public static MaterialSbtData Mirror(float reflectivity = 0.90f) =>
            new MaterialSbtData { Albedo = new Vec3(0.98f, 0.98f, 0.98f), Reflectivity = reflectivity, MaterialKind = MaterialSbtData.Solid };

        public static MaterialSbtData Glass(float indexOfRefraction, Vec3 transmissionColor) =>
            new MaterialSbtData { Albedo = new Vec3(1f, 1f, 1f), Transparency = 1f, IndexOfRefraction = indexOfRefraction, TransmissionColor = transmissionColor, MaterialKind = MaterialSbtData.Solid };

        public static MaterialSbtData ClearGlass() => Glass(1.5f, new Vec3(1f, 1f, 1f));

        public static MaterialSbtData Emissive(Vec3 emission) =>
            new MaterialSbtData { Emission = emission, MaterialKind = MaterialSbtData.Solid };

        public static MaterialSbtData Checker(Vec3 albedoA, Vec3 albedoB, float scale) =>
            new MaterialSbtData { Albedo = albedoA, MaterialKind = MaterialSbtData.Checker, CheckerColorB = albedoB, CheckerScale = scale };

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
