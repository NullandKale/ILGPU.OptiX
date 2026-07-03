// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: EnvironmentMap.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace Sample15
{
    // Host-side 2D importance-sampling data for the HDRI environment map
    // (docs/SAMPLE15_PLAN.md Milestone M5/Design Decision 5) - one flat piecewise-
    // constant distribution over every texel (not a separate marginal+conditional pair;
    // a single flat CDF over width*height texels is simpler to build and binary-search
    // and has the same asymptotic cost, since the conditional array alone is already
    // width*height). Built once at startup and shared across every scene (see the
    // plan's own Open Questions resolution) - EnvironmentMapGpu is scene-independent.
    public static class EnvironmentMap
    {
        public struct EnvMapData
        {
            public Vec3[] Pixels;
            // Flat cumulative distribution over texels, luminance-weighted, row-major,
            // monotonically increasing to 1 - device code binary-searches this to pick
            // a texel proportional to its radiance.
            public float[] Cdf;
            // Precomputed per-texel pdf with respect to the (u,v) unit square (not yet
            // solid angle - device code applies the equirectangular Jacobian
            // 1/(2*pi^2*sin(theta)) itself, since that depends on which row a given
            // sample/eval falls in). PdfUv[i] = (weight_i / totalWeight) * texelCount -
            // storing this instead of recomputing it from adjacent Cdf entries avoids a
            // subtraction (and the size-1 edge case) on the device hot path.
            public float[] PdfUv;
            public int Width;
            public int Height;
            // Sum of per-texel luminance weights - used as this light's relative
            // "power" proxy when Scenes/LightList.cs folds it into a scene's picking
            // CDF alongside point lights and emissive triangles.
            public float TotalWeight;
        }

        public static EnvMapData Build(string hdrPath)
        {
            HdrTextureLoader.HdrImage image = HdrTextureLoader.Load(hdrPath);
            int width = image.Width;
            int height = image.Height;
            int texelCount = width * height;

            var pixels = new Vec3[texelCount];
            var weights = new float[texelCount];
            float totalWeight = 0f;

            for (int i = 0; i < texelCount; i++)
            {
                int srcIdx = i * 3;
                Vec3 color = new Vec3(image.Pixels[srcIdx], image.Pixels[srcIdx + 1], image.Pixels[srcIdx + 2]);
                pixels[i] = color;

                float weight = Luminance(color);
                weights[i] = weight;
                totalWeight += weight;
            }

            var cdf = new float[texelCount];
            var pdfUv = new float[texelCount];
            float running = 0f;
            for (int i = 0; i < texelCount; i++)
            {
                float p = totalWeight > 0f ? weights[i] / totalWeight : 0f;
                pdfUv[i] = p * texelCount;
                running += p;
                cdf[i] = running;
            }
            // Float summation can leave the last entry a hair under 1 - clamp so a
            // picking draw (u in [0, 1)) can never fall past the end of the CDF.
            if (cdf.Length > 0)
                cdf[cdf.Length - 1] = 1f;

            return new EnvMapData
            {
                Pixels = pixels,
                Cdf = cdf,
                PdfUv = pdfUv,
                Width = width,
                Height = height,
                TotalWeight = totalWeight,
            };
        }

        static float Luminance(Vec3 color) => (0.2126f * color.x) + (0.7152f * color.y) + (0.0722f * color.z);
    }
}
