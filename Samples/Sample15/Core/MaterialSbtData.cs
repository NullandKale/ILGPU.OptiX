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

namespace Sample15
{
    // Device-side view of a hitgroup record's custom data (the bytes after the header) -
    // what OptixGetSbtDataPointer.Value points at inside __closest__radiance. This is T
    // in SbtBuilder.cs's SbtRecord<MaterialSbtData>/OptixSbtRecords.Pack<MaterialSbtData>
    // calls, which pack this struct immediately after each record's header - no
    // hand-measured duplicate struct to keep in sync with this one anymore. Only the
    // radiance hitgroup record's data is ever read (the shadow ray's closest-hit is an
    // empty stub).
    //
    // This is Sample15's PBR material model (docs/SAMPLE15_PLAN.md Design Decision 2): a
    // glTF-style metallic-roughness parameter set, replacing Sample14's Albedo/
    // Reflectivity/Transparency threshold-dispatch fields. The full field set (including
    // Roughness/TransmissionRoughness/Orm/Normal textures/EmissiveLightListBase) is added
    // in one pass even though only the M1 delta-lobe shading branches (mirror/diffuse/
    // glass, unchanged math from Sample14, just renamed fields) read from it so far -
    // avoids repeated struct-layout churn across milestones (M2 GGX, M4 NEE light-list
    // wiring, M6 texture pipeline, M7 rough transmission each populate/read more of it
    // without changing the layout again). See docs/SAMPLE15_PLAN.md Milestones for what
    // reads which field first.
    //
    // Shading dispatch (__closest__radiance, current as of M2): Transmission > 0 ->
    // ShadeDielectric's perfect-specular Fresnel-Schlick glass (unchanged since M1, until
    // M7's rough BTDF); else -> ShadeSurface's unified GGX metallic-roughness BSDF
    // (Bsdf.cs), which internally treats Roughness below Bsdf.DeltaRoughnessThreshold as
    // a perfect mirror and otherwise continuously blends Lambertian diffuse against
    // Fresnel-tinted GGX specular by Metallic - no more Metallic-threshold branching.
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

        // 0 = opaque, 1 = fully transmissive (glass) - replaces Sample14's binary
        // Transparency; still read as a > 0 threshold by the M1 dispatch shim above until
        // M2/M7 give rough-glass a continuous role.
        public float Transmission;
        public Vec3 TransmissionColor;

        // Roughness of the transmissive (BTDF) lobe specifically (docs/SAMPLE15_PLAN.md
        // Milestone M7, Walter et al. rough dielectric transmission,
        // MaterialShading.ShadeDielectric) - the struct default (0) is a deliberate,
        // desired degenerate case here (unlike the base Roughness field's own
        // struct-default trap, see docs/SAMPLE15_PLAN.md Milestone M4's postmortem):
        // it correctly reproduces M1's perfect-specular glass unless a scene
        // explicitly opts into a rough/frosted look.
        public float TransmissionRoughness;

        // Texture handles, 0 = none (Sample08's convention). BaseColorTexture is read
        // starting in M1 (renamed from Sample14's TextureObject); OrmTexture (packed
        // occlusion.r/roughness.g/metallic.b) and NormalTexture are wired up by M6's
        // texture/tangent pipeline (MaterialShading.cs's ApplyNormalMap/ShadeSurface).
        public ulong BaseColorTexture;
        public ulong OrmTexture;
        public ulong NormalTexture;

        // Tangent-space normal-map blend strength (1 = full strength, 0 = struct
        // default). Mirrors the exact trap Roughness's own struct-default 0 turned out
        // to be (docs/SAMPLE15_PLAN.md Milestone M4's "diffuse objects bounce like
        // specular rays" postmortem) - any material that sets NormalTexture MUST also
        // explicitly set NormalStrength (1f unless a deliberately subtle effect is
        // wanted), or the texture will be sampled and then silently multiplied away to
        // nothing by ApplyNormalMap.
        public float NormalStrength;

        // Blend weight between the scalar BaseColor and BaseColorTexture's sample (same
        // role as Sample14's TextureWeight); reused as the ORM/normal texture blend
        // weight too once M6 wires those in.
        public float TextureWeight;
        public float UVScale;
        public int MaterialKind;
        public Vec3 CheckerColorB;
        public float CheckerScale;

        // Alpha-cutout threshold sampled from BaseColorTexture's alpha channel (e.g.
        // Sponza's leaf/thorn geometry, which is a solid quad with the leaf shape
        // punched out via alpha) - 0 (the default) disables alpha testing entirely, so
        // every existing opaque-textured material is unaffected. Read by
        // __anyhit__radiance/__anyhit__shadow, which OptixIgnoreIntersection.Ignore()
        // any triangle hit whose sampled alpha falls below this value.
        public float AlphaCutoff;

        // Index into M4's emissive-triangle light list for materials with nonzero
        // Emission, or -1 if this material has no baked light-list entry. Set by
        // scene-build-time light-list construction (Scenes/LightList.cs, M4), not touched
        // by shading code directly. Unused (always 0) before M4.
        public int EmissiveLightListBase;

        public const int Solid = 0;
        public const int Checker = 1;
    }
}
