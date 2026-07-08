// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: SceneTypes.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace Sample15
{
    // Device-visible per-light data, uploaded as a flat array (LaunchParams.PointLights,
    // indexed by LightGpu.RefIndex from the unified NEE light list below - never walked
    // 0..count directly). The reference (RayTracing/Objects/PointLight.cs) only has this
    // one light type - no area/quad lights exist there.
    public struct PointLightGpu
    {
        public Vec3 Position;
        public Vec3 Color;
        public float Intensity;
    }

    // One entry in the unified NEE light list (docs/SAMPLE15_PLAN.md Milestone M4,
    // Design Decision 4) - built host-side at scene-load time by Scenes/LightList.cs,
    // uploaded as LaunchParams.NeeLights/NeeLightCdf. Kind distinguishes which
    // reference array RefIndex points into; Pdf is this light's power-weighted picking
    // probability (Pdf_i = power_i / totalPower), read directly by device code so it
    // never needs to re-derive it from the cumulative Cdf array.
    public struct LightGpu
    {
        public const int KindPoint = 0;
        public const int KindTriangle = 1;
        // The shared HDRI environment map (docs/SAMPLE15_PLAN.md Milestone M5) - at
        // most one entry of this kind ever exists in a light list; RefIndex is unused
        // (-1) since there's only the one, scene-independent environment map.
        public const int KindEnvMap = 2;

        public int Kind;
        public int RefIndex;
        public float Pdf;

        // Emitted radiance - only meaningful for Kind == KindTriangle (baked in at
        // scene-load time from the triangle's material, since it's static for the
        // scene's lifetime); a point light's own PointLightGpu.Color/Intensity already
        // carries this, so this field is unused (default) for Kind == KindPoint.
        public Vec3 Emission;
    }
}
