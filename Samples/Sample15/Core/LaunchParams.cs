// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: LaunchParams.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace Sample15
{
    // ILGPU compiles device kernels against one concrete unmanaged struct layout, so this
    // must stay a single superset shape across every scene the runtime scene-switcher can
    // select (docs/SAMPLE13_PLAN.md, "LaunchParams superset design"). Unused buffers on a
    // given scene are left null/zero-length rather than changing the struct itself.
    //
    // Triangle geometry, multi-point-light shading, unified GGX metallic-roughness
    // materials (docs/SAMPLE15_PLAN.md Milestone M2) + perfect-specular dielectric
    // materials, custom primitives (Sphere/Box/CylinderY/Disk/rects), volume grid, mesh
    // scenes (reusing the triangle buffers as-is), and AOV/denoiser buffers are all wired
    // up - inherited unchanged from Sample14's own build-up, see docs/SAMPLE13_PLAN.md/
    // docs/SAMPLE14_PLAN.md for that history.
    public unsafe struct LaunchParams
    {
        public int NumPixelSamples;
        public int FrameID;

        // Unified bounce budget (docs/SAMPLE15_PLAN.md Milestone M3) - replaces the
        // old per-material-kind MaxMirrorBounces/MaxRefractionBounces/
        // MaxDiffuseBounces counters now that every material kind shares the same
        // raygen bounce loop; Russian roulette (RaygenProgram.cs) is what actually
        // terminates most paths early in practice, this is just the hard ceiling.
        public int MaxBounces;
        public Vec4* ColorBuffer;
        // AOV guide buffers for the denoiser (see SampleRenderer.cs's OptixDenoiser
        // wiring) - overwritten fresh every frame from the primary ray's own hit, not
        // blended across frames like ColorBuffer (see devicePrograms.cs's raygen).
        public Vec4* NormalBuffer;
        public Vec4* AlbedoBuffer;
        public Camera camera;
        public ulong traversable;

        public Vec3* Vertices;
        public Vec3* Normals;
        public Vec2* TexCoords;
        public Vec3* Tangents;
        public Vec3i* Indices;

        // Custom-primitive parameter buffers - always allocated (possibly zero-length)
        // so this struct's layout never changes across scene switches.
        public SphereData* Spheres;
        public BoxData* Boxes;
        public CylinderYData* CylindersY;
        public DiskData* Disks;
        public RectData* XYRects;
        public RectData* XZRects;
        public RectData* YZRects;

        // Volume grid - only one grid is ever active at a time (unlike the arrays
        // above), so its parameters are plain scalar fields rather than a per-primitive
        // buffer. VoxelMaterialIds is a flat row-major (x*dims.y*dims.z + y*dims.z + z)
        // array; 0 means empty/air, N means launchParams.Materials[N-1] (see
        // devicePrograms.cs's ShadeVolumeGrid - this is the one place in the sample
        // where material lookup bypasses the per-primitive SBT convention, since a
        // single GAS primitive's material can't otherwise depend on which voxel the
        // intersection program's DDA loop found).
        public uint* VoxelMaterialIds;
        public Vec3 VolumeGridMin;
        public Vec3 VolumeVoxelSize;
        public Vec3i VolumeDims;

        // Device-side copy of the active scene's Materials[] palette, indexed directly
        // (not through an SBT record) - needed only for the volume grid's per-voxel
        // material lookup above.
        public MaterialSbtData* Materials;

        public PointLightGpu* PointLights;
        public int NumPointLights;
        public Vec3 AmbientColor;
        public float AmbientIntensity;

        // Unified NEE light list (docs/SAMPLE15_PLAN.md Milestone M4) - see
        // Scenes/LightList.cs. NeeLightAreaPdf is parallel to Indices/
        // TriangleMaterialIds (one entry per triangle, 0 = not a registered light) -
        // read directly by primitive index in __closest__radiance's MIS reweighting,
        // no light-list search needed on that hot path.
        public LightGpu* NeeLights;
        public float* NeeLightCdf;
        public int NumNeeLights;
        public float* NeeLightAreaPdf;

        // HDRI environment map (docs/SAMPLE15_PLAN.md Milestone M5) - scene-dependent
        // (see SceneData.EnvMapPath); EnvMapWidth == 0 means "no environment map for
        // this scene, fall back to the flat BackgroundTop/Bottom gradient" (checked by
        // __miss__radiance and NextEventEstimation before touching the other EnvMap*
        // pointers, which are null in that case). EnvMapLightPdf is this scene's own
        // NEE light-list picking probability for the environment map (computed by
        // Scenes/LightList.cs, 0 if EnvMapWidth == 0) - kept separate from NeeLights
        // since there's no per-triangle-style array to carry one scalar light's pdf in.
        public Vec3* EnvMapPixels;
        public float* EnvMapCdf;
        public float* EnvMapPdfUv;
        public int EnvMapWidth;
        public int EnvMapHeight;
        public float EnvMapIntensity;
        public float EnvMapLightPdf;
        // Azimuthal rotation in radians (docs/SAMPLE15_PLAN.md Milestone M8) - applied
        // uniformly by EnvironmentMapSampling's direction<->uv conversion.
        public float EnvMapRotation;

        // NEE/MIS on/off toggle (docs/SAMPLE15_PLAN.md Milestone M8) - int, not bool,
        // to keep this struct's device-side layout a plain blittable POD (matches
        // every other flag-like field here, e.g. MaterialKind on MaterialSbtData).
        public int NeeEnabled;

        public Vec3 BackgroundTop;
        public Vec3 BackgroundBottom;
    }
}
