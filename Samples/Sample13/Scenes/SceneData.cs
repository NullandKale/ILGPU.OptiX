// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: SceneData.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using MeshRange = ILGPU.OptiX.Pipeline.OptixMeshRange;
using System;

namespace Sample13
{
    // Host-only per-frame animation descriptors (never uploaded to the GPU as-is) -
    // SampleRenderer.ApplyAnimation reads these once per frame to mutate its own
    // working copies of SceneData.Lights/Spheres before re-uploading just the changed
    // buffers. Index fields refer to positions in the SceneData arrays they animate.
    // Direct port of the reference's OrbitingLightEntity/PulsingLightEntity/
    // BobbingSphereEntity (RayTracing/Scenes/TestScenesRandom.cs), but expressed as
    // plain data descriptors evaluated from elapsed wall-clock time rather than as
    // stateful per-frame-dt-accumulating entity objects - mathematically equivalent
    // (each entity's own "t += dt" accumulator is just elapsed time since the scene
    // started) and avoids reintroducing an entity/update-object system into a
    // renderer whose scene representation is otherwise flat data arrays.
    public struct OrbitingLightAnim
    {
        public int LightIndex;
        public Vec3 Pivot;
        public float Radius;
        public float Height;
        public float AngularSpeed;
        public float Phase;
    }

    public struct PulsingLightAnim
    {
        public int LightIndex;
        public float BaseIntensity;
        public float MinMult;
        public float MaxMult;
        public float Speed;
    }

    public struct BobbingSphereAnim
    {
        public int SphereIndex;
        public float BaseY;
        public float Amplitude;
        public float Speed;
        public float Phase;
    }

    // Host-side POCO a Scenes.cs builder method returns; SampleRenderer.SwitchToScene
    // consumes it to (re)allocate device buffers, rebuild the GAS/SBT, and reset FrameID -
    // mirrors the reference's own lazy-built, cached-per-index Scene objects
    // (docs/SAMPLE13_PLAN.md, "Where scene-build logic lives"). Grows a field per
    // milestone as new primitive kinds/volume grid/mesh support come online; unused
    // fields on a given scene are left null/default.
    public class SceneData
    {
        public string Name = "";

        public Vec3[] Vertices = Array.Empty<Vec3>();
        public Vec3[] Normals = Array.Empty<Vec3>();
        public Vec2[] TexCoords = Array.Empty<Vec2>();
        public Vec3i[] Indices = Array.Empty<Vec3i>();

        // One entry per triangle (same length as Indices), indexing into Materials -
        // also the GAS build input's SbtIndexOffsetBuffer, same convention as Sample07-12.
        public uint[] TriangleMaterialIds = Array.Empty<uint>();

        // Per-mesh triangle-index ranges within Indices/TriangleMaterialIds - one
        // entry per loaded OBJ mesh (see Scenes.cs's AddMesh/AddMeshAutoGround
        // helpers), used by SampleRenderer.BuildTrianglesGas to emit one GAS build
        // input per mesh instead of one build input for the whole scene. Empty means
        // "treat the whole Indices array as a single implicit mesh" (the original
        // behavior) - every scene that doesn't explicitly track mesh boundaries
        // (procedural/test/CSG scenes, BuildMeshScene) leaves this empty and is
        // unaffected.
        public MeshRange[] MeshRanges = Array.Empty<MeshRange>();

        public MaterialSbtData[] Materials = Array.Empty<MaterialSbtData>();

        // Optional, index-aligned with Materials - a relative path (under the output
        // directory, e.g. "models/sponza/textures/x.png") to load into a
        // CudaTextureObject and assign to that material's HitgroupRecord.TextureObject
        // (see SampleRenderer.cs's BuildHitgroupSbt). Shorter than Materials or an empty/
        // null entry means "no texture" for that material index - matches Sample08's
        // TextureObject==0 convention.
        public string[] MaterialTexturePaths = Array.Empty<string>();

        // Custom primitives - each *MaterialIds array indexes into the SAME shared
        // Materials[] list as triangles (one merged material palette per scene, not a
        // separate one per primitive kind).
        public SphereData[] Spheres = Array.Empty<SphereData>();
        public uint[] SphereMaterialIds = Array.Empty<uint>();

        public BoxData[] Boxes = Array.Empty<BoxData>();
        public uint[] BoxMaterialIds = Array.Empty<uint>();

        public CylinderYData[] CylindersY = Array.Empty<CylinderYData>();
        public uint[] CylinderYMaterialIds = Array.Empty<uint>();

        public DiskData[] Disks = Array.Empty<DiskData>();
        public uint[] DiskMaterialIds = Array.Empty<uint>();

        public RectData[] XYRects = Array.Empty<RectData>();
        public uint[] XYRectMaterialIds = Array.Empty<uint>();

        public RectData[] XZRects = Array.Empty<RectData>();
        public uint[] XZRectMaterialIds = Array.Empty<uint>();

        public RectData[] YZRects = Array.Empty<RectData>();
        public uint[] YZRectMaterialIds = Array.Empty<uint>();

        // Volume grid - one flat row-major (x*Dims.y*Dims.z + y*Dims.z + z) array; 0 =
        // empty/air, N = Materials[N-1] (see devicePrograms.cs's ShadeVolumeGrid).
        // VoxelMaterialIds.Length == 0 means "no volume grid in this scene".
        public uint[] VoxelMaterialIds = Array.Empty<uint>();
        public Vec3 VolumeGridMin;
        public Vec3 VolumeVoxelSize = new Vec3(1f, 1f, 1f);
        public Vec3i VolumeDims;

        public PointLightGpu[] Lights = Array.Empty<PointLightGpu>();
        public Vec3 AmbientColor = new Vec3(1f, 1f, 1f);
        public float AmbientIntensity = 0.05f;

        // Per-frame animation (museum/radial-museum scenes only) - see
        // SampleRenderer.ApplyAnimation. BobbingSpheres requires refitting the custom-
        // primitives GAS every frame (OPTIX_BUILD_OPERATION_UPDATE), so its presence
        // also drives BuildOrUpdateCustomPrimitivesGas's ALLOW_UPDATE flag and forces
        // progressive accumulation off for the scene (see HasAnimatedGeometry/render()).
        public OrbitingLightAnim[] OrbitingLights = Array.Empty<OrbitingLightAnim>();
        public PulsingLightAnim[] PulsingLights = Array.Empty<PulsingLightAnim>();
        public BobbingSphereAnim[] BobbingSpheres = Array.Empty<BobbingSphereAnim>();

        public bool HasAnimatedGeometry => BobbingSpheres.Length > 0;
        public bool HasAnyAnimation => OrbitingLights.Length > 0 || PulsingLights.Length > 0 || BobbingSpheres.Length > 0;

        public Vec3 BackgroundTop = new Vec3(0.4f, 0.55f, 0.8f);
        public Vec3 BackgroundBottom = new Vec3(0.05f, 0.05f, 0.08f);

        public Vec3 CameraOrigin;
        public Vec3 CameraLookAt;
        public Vec3 CameraUp = new Vec3(0f, 1f, 0f);
        public float CameraFovDeg = 50f;
        public float CameraWorldScale = 10f;
    }
}
