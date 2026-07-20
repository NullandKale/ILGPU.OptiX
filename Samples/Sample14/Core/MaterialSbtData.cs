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
    // what OptixGetSbtDataPointer.Value points at inside __closest__radiance. This is T
    // in SbtBuilder.cs's SbtRecord<MaterialSbtData>/OptixSbtRecords.Pack<MaterialSbtData>
    // calls, which pack this struct immediately after each record's header - no
    // hand-measured duplicate struct to keep in sync with this one anymore. Only the
    // radiance hitgroup record's data is ever read (the shadow ray's closest-hit is an
    // empty stub).
    //
    // A glTF-style metallic-roughness PBR parameter set.
    //
    // Shading dispatch (__closest__radiance): Transmission > 0 -> ShadeDielectric's
    // perfect-specular Fresnel-Schlick glass; else -> ShadeSurface's unified GGX
    // metallic-roughness BSDF (Bsdf.cs), which internally treats Roughness below
    // Bsdf.DeltaRoughnessThreshold as a perfect mirror and otherwise continuously
    // blends Lambertian diffuse against Fresnel-tinted GGX specular by Metallic.
    // MaterialKind is an orthogonal second axis (Solid vs. Checker base-color sampling
    // only - checkered surfaces use whichever BSDF branch their Roughness/Metallic pick).
    public struct MaterialSbtData
    {
        public Vec3 BaseColor;
        public float Metallic;
        public float Roughness;
        public float IOR;
        public Vec3 Emission;
        public float EmissionStrength;

        // 0 = opaque, 1 = fully transmissive (glass).
        public float Transmission;
        public Vec3 TransmissionColor;

        // Roughness of the transmissive (BTDF) lobe specifically (Walter et al. rough
        // dielectric transmission, MaterialShading.ShadeDielectric) - the struct
        // default of 0 correctly reproduces perfect-specular glass unless a scene
        // explicitly opts into a rough/frosted look.
        public float TransmissionRoughness;

        // Texture handles, 0 = none. OrmTexture packs occlusion.r/roughness.g/
        // metallic.b; both it and NormalTexture are read by MaterialShading.cs's
        // ApplyNormalMap/ShadeSurface.
        public ulong BaseColorTexture;
        public ulong OrmTexture;
        public ulong NormalTexture;

        // Tangent-space normal-map blend strength (1 = full strength, 0 = struct
        // default). Any material that sets NormalTexture MUST also explicitly set
        // NormalStrength (1f unless a deliberately subtle effect is wanted), or the
        // texture will be sampled and then silently multiplied away to nothing by
        // ApplyNormalMap.
        public float NormalStrength;

        // Blend weight between the scalar BaseColor and BaseColorTexture's sample;
        // also used as the ORM/normal texture blend weight.
        public float TextureWeight;
        public float UVScale;
        public int MaterialKind;
        public Vec3 CheckerColorB;
        public float CheckerScale;

        // Alpha-cutout threshold sampled from BaseColorTexture's alpha channel (e.g.
        // Sponza's leaf/thorn geometry, which is a solid quad with the leaf shape
        // punched out via alpha) - 0 (the default) disables alpha testing entirely, so
        // every opaque-textured material without alpha cutout is unaffected. Read by
        // __anyhit__radiance/__anyhit__shadow, which OptixIgnoreIntersection.Ignore()
        // any triangle hit whose sampled alpha falls below this value.
        public float AlphaCutoff;

        // Index into the emissive-triangle light list for materials with nonzero
        // Emission, or -1 if this material has no baked light-list entry. Set by
        // scene-build-time light-list construction (Scenes/LightList.cs), not touched
        // by shading code directly.
        public int EmissiveLightListBase;

        public const int Solid = 0;
        public const int Checker = 1;
    }
}
