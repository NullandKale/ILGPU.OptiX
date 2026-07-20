// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: LightList.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;

namespace Sample21
{
    // Builds the unified NEE light list from an already-built SceneData: point
    // lights plus power/area-weighted emissive triangles, walked from SceneData's
    // existing Indices/TriangleMaterialIds/Materials buffers - no duplicated
    // geometry. The environment map is a third light kind added to the same CDF.
    public static class LightList
    {
        // Relative "power" proxies, not physical radiant power - good enough for a
        // picking-probability weighting (brighter/bigger lights get sampled more often),
        // which is the CDF's only job. Point lights: color luminance * Intensity (the
        // same falloff-free constant already used everywhere else in this sample as a
        // point light's brightness). Emissive triangles: emitted radiance luminance *
        // triangle area (a physically-motivated proxy - total emitted power scales with
        // both).
        // envMapPower is EnvironmentMap.EnvMapData.TotalWeight (0 if no environment
        // map is loaded) - the shared HDRI is scene-independent, but its *picking*
        // probability isn't: it competes for CDF share against each scene's own point
        // lights/emissive triangles, so it's folded in here per scene rather than
        // baked into EnvironmentMap.Build itself. Returns envMapLightPdf (0 if
        // envMapPower <= 0) separately from the light list, since there's no per-
        // triangle-style array to carry a single scalar light's pdf in.
        public static (LightGpu[] Lights, float[] Cdf, float[] TriangleLightAreaPdf, float EnvMapLightPdf) Build(SceneData scene, float envMapPower)
        {
            var lights = new List<LightGpu>();
            var powers = new List<float>();

            for (int i = 0; i < scene.Lights.Length; i++)
            {
                PointLightGpu light = scene.Lights[i];
                float power = Luminance(light.Color) * light.Intensity;
                if (power <= 0f)
                    continue;

                lights.Add(new LightGpu { Kind = LightGpu.KindPoint, RefIndex = i });
                powers.Add(power);
            }

            if (envMapPower > 0f)
            {
                lights.Add(new LightGpu { Kind = LightGpu.KindEnvMap, RefIndex = -1 });
                powers.Add(envMapPower);
            }

            var triangleLightAreaPdf = new float[scene.Indices.Length];
            for (int tri = 0; tri < scene.Indices.Length; tri++)
            {
                MaterialSbtData mat = scene.Materials[(int)scene.TriangleMaterialIds[tri]];
                Vec3 emission = mat.Emission * mat.EmissionStrength;
                float emissionLuminance = Luminance(emission);
                if (emissionLuminance <= 0f)
                    continue;

                Vec3i idx = scene.Indices[tri];
                Vec3 a = scene.Vertices[idx.x], b = scene.Vertices[idx.y], c = scene.Vertices[idx.z];
                float area = 0.5f * Vec3.cross(b - a, c - a).length();
                if (area <= 1e-9f)
                    continue;

                float power = emissionLuminance * area;
                lights.Add(new LightGpu { Kind = LightGpu.KindTriangle, RefIndex = tri, Emission = emission });
                powers.Add(power);
                // TriangleLightAreaPdf[tri] is filled in below, once each light's
                // normalized Pdf is known - the loop above only reserves the slot's
                // membership (power > 0), area is recomputed then since it's cheap and
                // avoids a second parallel array just to carry it between the two passes.
            }

            float totalPower = 0f;
            foreach (float p in powers)
                totalPower += p;

            var cdf = new float[lights.Count];
            float running = 0f;
            float envMapLightPdf = 0f;
            for (int i = 0; i < lights.Count; i++)
            {
                float pdf = totalPower > 0f ? powers[i] / totalPower : 0f;
                LightGpu l = lights[i];
                l.Pdf = pdf;
                lights[i] = l;

                running += pdf;
                cdf[i] = running;

                if (l.Kind == LightGpu.KindTriangle)
                {
                    Vec3i idx = scene.Indices[l.RefIndex];
                    Vec3 a = scene.Vertices[idx.x], b = scene.Vertices[idx.y], c = scene.Vertices[idx.z];
                    float area = 0.5f * Vec3.cross(b - a, c - a).length();
                    // Combines the light-picking pdf and the per-triangle uniform-area
                    // pdf into one constant - lets the closest-hit program's per-hit
                    // MIS reweighting of a BSDF ray
                    // that lands on this triangle multiply by dist²/cosAtLight alone,
                    // with no vertex/area recomputation on the hot path.
                    triangleLightAreaPdf[l.RefIndex] = pdf / area;
                }
                else if (l.Kind == LightGpu.KindEnvMap)
                {
                    envMapLightPdf = pdf;
                }
            }
            // Float summation can leave the last entry a hair under 1 - clamp so the
            // picking draw (u in [0, 1)) can never fall past the end of the CDF.
            if (cdf.Length > 0)
                cdf[cdf.Length - 1] = 1f;

            return (lights.ToArray(), cdf, triangleLightAreaPdf, envMapLightPdf);
        }

        static float Luminance(Vec3 color) => (0.2126f * color.x) + (0.7152f * color.y) + (0.0722f * color.z);
    }
}
