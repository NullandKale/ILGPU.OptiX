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

namespace Sample13
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
