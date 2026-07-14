// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: MaterialSbtData.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX.DeviceApi;
namespace Sample14
{
    // Device-side view of a hitgroup record's custom data (the bytes after the header) -
    // what OptixGetSbtDataPointer.Value points at inside __closest__radiance. Field layout
    // must match HitgroupRecord in SampleRenderer.cs starting right after its fixed Header
    // array. Only the radiance hitgroup record's data is ever read (the shadow ray's
    // closest-hit is an empty stub).
    //
    // The full reference Material field set is added in one pass even though
    // Reflectivity/Transparency/IndexOfRefraction/TransmissionColor aren't read by any
    // shading branch until mirror/dielectric materials are added - avoids repeated
    // struct-layout churn across milestones.
    //
    // Shading dispatch order (evaluated in __closest__radiance, mirrors the reference's
    // Material field-value dispatch exactly): Transparency > 0 -> dielectric; else
    // Reflectivity >= 0.9 -> mirror; else -> diffuse (Oren-Nayar + ambient). MaterialKind
    // is an orthogonal second axis (Solid vs. Checker albedo sampling only - checkered
    // surfaces are always diffuse in every reference scene that uses them).
    public struct MaterialSbtData
    {
        public Vec3 Albedo;
        public float Reflectivity;
        public Vec3 Emission;
        public float Transparency;
        public float IndexOfRefraction;
        public Vec3 TransmissionColor;
        public ulong TextureObject;
        public float TextureWeight;
        public float UVScale;
        public int MaterialKind;
        public Vec3 CheckerColorB;
        public float CheckerScale;

        // Alpha-cutout threshold sampled from TextureObject's alpha channel (e.g.
        // Sponza's leaf/thorn geometry, which is a solid quad with the leaf shape
        // punched out via alpha) - 0 (the default) disables alpha testing entirely, so
        // every existing opaque-textured material is unaffected. Read by
        // __anyhit__radiance/__anyhit__shadow, which OptixIgnoreIntersection.Ignore()
        // any triangle hit whose sampled alpha falls below this value.
        public float AlphaCutoff;

        public const int Solid = 0;
        public const int Checker = 1;
    }
}
