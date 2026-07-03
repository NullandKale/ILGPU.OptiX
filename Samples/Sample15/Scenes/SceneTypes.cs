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
    // Device-visible per-light data, uploaded as a flat array (LaunchParams.PointLights /
    // NumPointLights). The reference (RayTracing/Objects/PointLight.cs) only has this one
    // light type - no area/quad lights exist there.
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

    // Per-primitive parameters for the Sphere custom-primitive type (direct port of
    // RayTracing/Objects/BoundedObjects.cs's Sphere fields). Indexed by
    // OptixGetPrimitiveIndex.Value from within the sphere build input's own local
    // indexing (see docs/SAMPLE13_PLAN.md's custom-primitive design).
    public struct SphereData
    {
        public Vec3 Center;
        public float Radius;
    }

    // Axis-aligned box (direct port of RayTracing/Objects/BoundedObjects.cs's Box,
    // ported as a real 3-axis slab-test box rather than the reference's 6-rect-face
    // proxy - see docs/SAMPLE13_PLAN.md's Box design note).
    public struct BoxData
    {
        public Vec3 Min;
        public Vec3 Max;
    }

    // Cylinder with its axis along Y (direct port of
    // RayTracing/Objects/BoundedObjects.cs's CylinderY).
    public struct CylinderYData
    {
        public Vec3 Center;
        public float Radius;
        public float YMin;
        public float YMax;
        public int Capped;
    }

    // Disk (direct port of RayTracing/Objects/Surfaces.cs's Disk).
    public struct DiskData
    {
        public Vec3 Center;
        public Vec3 Normal;
        public float Radius;
    }

    // Shared shape for the three axis-aligned rect types (direct port of
    // RayTracing/Objects/Surfaces.cs's XYRect/XZRect/YZRect) - each interpreted by its
    // own __intersection__ program (IntersectionPrograms.cs): C is the fixed-axis
    // coordinate, A0..A1/B0..B1 the span along the other two axes in a fixed order
    // (XYRect: A=X,B=Y,C=Z; XZRect: A=X,B=Z,C=Y; YZRect: A=Y,B=Z,C=X).
    public struct RectData
    {
        public float A0, A1;
        public float B0, B1;
        public float C;
    }
}
