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
using ILGPU.OptiX.Cuda;

namespace Sample14
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

    // Host-side POCO a Scenes.cs builder method returns; SampleRenderer.SwitchToScene
    // consumes it to (re)allocate device buffers, rebuild the GAS/SBT, and reset FrameID -
    // mirrors the reference's own lazy-built, cached-per-index Scene objects. Grows a
    // field as new primitive kinds/volume grid/mesh support come online; unused
    // fields on a given scene are left null/default.
    public class SceneData
    {
        public string Name = "";

        public Vec3[] Vertices = Array.Empty<Vec3>();
        public Vec3[] Normals = Array.Empty<Vec3>();
        public Vec2[] TexCoords = Array.Empty<Vec2>();
        // Per-vertex tangent - see Model.cs's ComputeTangents/SceneBuilder's
        // TangentFromTriangle for how each geometry source computes it.
        public Vec3[] Tangents = Array.Empty<Vec3>();
        public Vec3i[] Indices = Array.Empty<Vec3i>();

        // One entry per triangle (same length as Indices), indexing into Materials -
        // also the GAS build input's SbtIndexOffsetBuffer.
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
        // CudaTextureObject and assign to that material's MaterialSbtData.BaseColorTexture
        // (see SbtBuilder.cs's Build/FillMaterialRecords). Shorter than Materials or an empty/
        // null entry means "no texture" for that material index (TextureObject==0).
        public string[] MaterialTexturePaths = Array.Empty<string>();

        // Same convention as MaterialTexturePaths, for MaterialSbtData.NormalTexture
        // (tangent-space normal map) and .OrmTexture (a single already-packed
        // occlusion.r/roughness.g/metallic.b texture - not composed from separate
        // grayscale maps at load time, see Model.cs's OBJMaterial doc comment).
        public string[] MaterialNormalTexturePaths = Array.Empty<string>();
        public string[] MaterialOrmTexturePaths = Array.Empty<string>();

        public PointLightGpu[] Lights = Array.Empty<PointLightGpu>();

        // NEE light list - computed by Scenes/LightList.cs from this SceneData's own Lights/Indices/
        // TriangleMaterialIds/Materials once they're final (see SceneBuilder.Build);
        // never populated by a scene builder directly. NeeLightAreaPdf is parallel to
        // Indices/TriangleMaterialIds (one entry per triangle, 0 = not a registered
        // light).
        public LightGpu[] NeeLights = Array.Empty<LightGpu>();
        public float[] NeeLightCdf = Array.Empty<float>();
        public float[] NeeLightAreaPdf = Array.Empty<float>();
        public Vec3 AmbientColor = new Vec3(1f, 1f, 1f);
        public float AmbientIntensity = 0.1f;

        // Per-frame animation (museum/radial-museum scenes only) - see
        // SampleRenderer.ApplyAnimation.
        public OrbitingLightAnim[] OrbitingLights = Array.Empty<OrbitingLightAnim>();
        public PulsingLightAnim[] PulsingLights = Array.Empty<PulsingLightAnim>();

        public bool HasAnyAnimation => OrbitingLights.Length > 0 || PulsingLights.Length > 0;

        public Vec3 BackgroundTop = new Vec3(0.4f, 0.55f, 0.8f);
        public Vec3 BackgroundBottom = new Vec3(0.05f, 0.05f, 0.08f);

        // HDRI environment map - a path relative to AppContext.BaseDirectory
        // (matching MaterialTexturePaths' convention), or
        // null/empty for "no environment map, use the flat BackgroundTop/Bottom
        // gradient instead" (this sample's scene-dependent design: not every scene
        // opts in). SampleRenderer caches the loaded/uploaded GPU data per unique path
        // so scenes sharing the same HDRI don't reload it on every switch.
        public string EnvMapPath;
        public float EnvMapIntensity = 1f;

        public Vec3 CameraOrigin;
        public Vec3 CameraLookAt;
        public Vec3 CameraUp = new Vec3(0f, 1f, 0f);
        public float CameraFovDeg = 50f;
        public float CameraWorldScale = 10f;
    }
}
