// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: Scenes.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Algorithms;
using System;
using System.Collections.Generic;
using System.IO;

namespace Sample13
{
    // One C# method per scene, mirroring the reference's own Scenes.cs/TestScenes.cs
    // organization (docs/SAMPLE13_PLAN.md, "Where scene-build logic lives"). BuildSceneTable
    // grows to the reference's full 15-scene roster as later milestones unlock the
    // prerequisite primitives (spheres in M3, boxes/rects/cylinders/disks in M4, the volume
    // grid in M5, meshes in M6) - for now it holds only a hand-authored smoke-test scene
    // that exercises the M1 shading pipeline (multi-point-light Oren-Nayar diffuse,
    // ambient, checker material, binary shadow rays) ahead of any real primitive geometry.
    public static class Scenes
    {
        // Fills any gaps between (and before/after) explicitly tracked mesh ranges
        // with anonymous ranges covering untracked indices (e.g. ad-hoc AddTriangle
        // calls interleaved between AddMeshAutoGround calls in BuildRadialMuseumScene)
        // - guarantees the result partitions [0, totalIndexCount) with no gaps or
        // overlaps, which SampleRenderer.BuildTrianglesGas relies on (every triangle
        // must land in exactly one GAS build input).
        static MeshRange[] FillMeshRangeGaps(List<MeshRange> tracked, int totalIndexCount)
        {
            tracked.Sort((a, b) => a.IndexStart.CompareTo(b.IndexStart));
            var result = new List<MeshRange>();
            int cursor = 0;
            foreach (var range in tracked)
            {
                if (range.IndexStart > cursor)
                    result.Add(new MeshRange { IndexStart = cursor, IndexCount = range.IndexStart - cursor });
                result.Add(range);
                cursor = range.IndexStart + range.IndexCount;
            }
            if (cursor < totalIndexCount)
                result.Add(new MeshRange { IndexStart = cursor, IndexCount = totalIndexCount - cursor });
            return result.ToArray();
        }

        public static SceneData BuildDebugOrenNayarScene()
        {
            var vertices = new List<Vec3>();
            var normals = new List<Vec3>();
            var texCoords = new List<Vec2>();
            var indices = new List<Vec3i>();
            var materialIds = new List<uint>();

            void AddQuad(Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3 normal, uint materialId)
            {
                int baseIndex = vertices.Count;
                vertices.Add(a); vertices.Add(b); vertices.Add(c); vertices.Add(d);
                normals.Add(normal); normals.Add(normal); normals.Add(normal); normals.Add(normal);
                texCoords.Add(new Vec2(0, 0)); texCoords.Add(new Vec2(1, 0));
                texCoords.Add(new Vec2(1, 1)); texCoords.Add(new Vec2(0, 1));

                indices.Add(new Vec3i(baseIndex, baseIndex + 1, baseIndex + 2));
                materialIds.Add(materialId);
                indices.Add(new Vec3i(baseIndex, baseIndex + 2, baseIndex + 3));
                materialIds.Add(materialId);
            }

            void AddTriangle(Vec3 a, Vec3 b, Vec3 c, Vec3 normal, uint materialId)
            {
                int baseIndex = vertices.Count;
                vertices.Add(a); vertices.Add(b); vertices.Add(c);
                normals.Add(normal); normals.Add(normal); normals.Add(normal);
                texCoords.Add(new Vec2(0, 0)); texCoords.Add(new Vec2(1, 0)); texCoords.Add(new Vec2(0, 1));

                indices.Add(new Vec3i(baseIndex, baseIndex + 1, baseIndex + 2));
                materialIds.Add(materialId);
            }

            // Material 0: checkered floor (always diffuse, matching the reference's
            // Checker closures - Reflectivity/Transparency stay 0).
            // Material 1: solid red diffuse back wall.
            // Material 2: solid green diffuse floating triangle.
            // Material 3: mirror panel (Reflectivity >= 0.9 dispatches to the mirror
            // branch in __closest__radiance).
            // Material 4: glass panel (Transparency > 0 dispatches to the dielectric
            // branch) - exercises the M2 bounce loop and multi-hit shadow transmittance.
            var materials = new[]
            {
                new MaterialSbtData
                {
                    Albedo = new Vec3(0.85f, 0.85f, 0.85f),
                    MaterialKind = MaterialSbtData.Checker,
                    CheckerColorB = new Vec3(0.1f, 0.1f, 0.1f),
                    CheckerScale = 1f,
                    TextureWeight = 1f,
                    UVScale = 1f,
                },
                new MaterialSbtData
                {
                    Albedo = new Vec3(0.75f, 0.15f, 0.12f),
                    MaterialKind = MaterialSbtData.Solid,
                    TextureWeight = 1f,
                    UVScale = 1f,
                },
                new MaterialSbtData
                {
                    Albedo = new Vec3(0.15f, 0.6f, 0.2f),
                    MaterialKind = MaterialSbtData.Solid,
                    TextureWeight = 1f,
                    UVScale = 1f,
                },
                new MaterialSbtData
                {
                    Albedo = new Vec3(0.95f, 0.95f, 0.95f),
                    Reflectivity = 0.95f,
                    MaterialKind = MaterialSbtData.Solid,
                },
                new MaterialSbtData
                {
                    Albedo = new Vec3(1f, 1f, 1f),
                    Transparency = 1f,
                    IndexOfRefraction = 1.5f,
                    TransmissionColor = new Vec3(0.9f, 0.97f, 0.95f),
                    MaterialKind = MaterialSbtData.Solid,
                },
                new MaterialSbtData
                {
                    Albedo = new Vec3(0.2f, 0.35f, 0.85f),
                    MaterialKind = MaterialSbtData.Solid,
                    TextureWeight = 1f,
                    UVScale = 1f,
                },
                // 6: cylinder - yellow.
                new MaterialSbtData { Albedo = new Vec3(0.9f, 0.8f, 0.1f), MaterialKind = MaterialSbtData.Solid },
                // 7: box - orange.
                new MaterialSbtData { Albedo = new Vec3(0.9f, 0.45f, 0.1f), MaterialKind = MaterialSbtData.Solid },
                // 8: disk - purple.
                new MaterialSbtData { Albedo = new Vec3(0.55f, 0.2f, 0.75f), MaterialKind = MaterialSbtData.Solid },
                // 9: XYRect - cyan.
                new MaterialSbtData { Albedo = new Vec3(0.15f, 0.75f, 0.75f), MaterialKind = MaterialSbtData.Solid },
                // 10: XZRect - magenta.
                new MaterialSbtData { Albedo = new Vec3(0.8f, 0.2f, 0.6f), MaterialKind = MaterialSbtData.Solid },
                // 11: YZRect - teal.
                new MaterialSbtData { Albedo = new Vec3(0.1f, 0.55f, 0.45f), MaterialKind = MaterialSbtData.Solid },
            };

            // Three spheres near the floor, between the mirror/glass panels - exercises
            // the M3 custom-primitive pipeline against materials already proven on
            // triangles (mirror, glass, diffuse).
            var spheres = new[]
            {
                new SphereData { Center = new Vec3(-1.2f, 0.8f, 1f), Radius = 0.8f },
                new SphereData { Center = new Vec3(1.2f, 0.8f, 1f), Radius = 0.8f },
                new SphereData { Center = new Vec3(0f, 0.6f, 2.5f), Radius = 0.6f },
            };
            var sphereMaterialIds = new uint[] { 3, 4, 5 };

            // M4 primitives - one of each new custom-primitive kind, scattered near the
            // existing geometry for visual verification.
            var boxes = new[]
            {
                new BoxData { Min = new Vec3(-5.5f, 0f, 2.5f), Max = new Vec3(-4.5f, 1f, 3.5f) },
            };
            var boxMaterialIds = new uint[] { 7 };

            var cylindersY = new[]
            {
                new CylinderYData { Center = new Vec3(5f, 0f, 3f), Radius = 0.5f, YMin = 0f, YMax = 1.5f, Capped = 1 },
            };
            var cylinderYMaterialIds = new uint[] { 6 };

            var disks = new[]
            {
                new DiskData { Center = new Vec3(0f, 0.02f, 6f), Normal = new Vec3(0f, 1f, 0f), Radius = 1f },
            };
            var diskMaterialIds = new uint[] { 8 };

            var xyRects = new[]
            {
                new RectData { A0 = -6f, A1 = -4f, B0 = 1f, B1 = 2f, C = -3f },
            };
            var xyRectMaterialIds = new uint[] { 9 };

            var xzRects = new[]
            {
                new RectData { A0 = 4f, A1 = 6f, B0 = 1f, B1 = 3f, C = 2.5f },
            };
            var xzRectMaterialIds = new uint[] { 10 };

            var yzRects = new[]
            {
                new RectData { A0 = 0f, A1 = 1f, B0 = 3f, B1 = 5f, C = 6.5f },
            };
            var yzRectMaterialIds = new uint[] { 11 };

            // Floor: X in [-8, 8], Z in [-8, 8], Y = 0, facing up.
            AddQuad(
                new Vec3(-8f, 0f, 8f), new Vec3(8f, 0f, 8f),
                new Vec3(8f, 0f, -8f), new Vec3(-8f, 0f, -8f),
                new Vec3(0f, 1f, 0f), 0);

            // Back wall: X in [-4, 4], Y in [0, 4], Z = -4, facing +Z (toward the camera).
            AddQuad(
                new Vec3(-4f, 0f, -4f), new Vec3(4f, 0f, -4f),
                new Vec3(4f, 4f, -4f), new Vec3(-4f, 4f, -4f),
                new Vec3(0f, 0f, 1f), 1);

            // Small floating triangle for shading variety.
            AddTriangle(
                new Vec3(2f, 1.5f, -1f), new Vec3(3f, 1.5f, -1f), new Vec3(2.5f, 2.5f, -1f),
                new Vec3(0f, 0f, 1f), 2);

            // Mirror panel, facing +X (toward scene center) - exercises the mirror
            // bounce branch.
            AddQuad(
                new Vec3(-3f, 0f, -3f), new Vec3(-3f, 0f, 1f),
                new Vec3(-3f, 3f, 1f), new Vec3(-3f, 3f, -3f),
                new Vec3(1f, 0f, 0f), 3);

            // Glass panel, facing -X (toward scene center) - exercises the dielectric
            // reflect/refract branch and multi-hit shadow transmittance.
            AddQuad(
                new Vec3(3f, 0f, -3f), new Vec3(3f, 0f, 1f),
                new Vec3(3f, 3f, 1f), new Vec3(3f, 3f, -3f),
                new Vec3(-1f, 0f, 0f), 4);

            var lights = new[]
            {
                new PointLightGpu { Position = new Vec3(-3f, 4f, 3f), Color = new Vec3(1f, 0.85f, 0.6f), Intensity = 40f },
                new PointLightGpu { Position = new Vec3(4f, 5f, -2f), Color = new Vec3(0.6f, 0.75f, 1f), Intensity = 40f },
            };

            return new SceneData
            {
                Name = "Debug: Oren-Nayar + multi-light",
                Vertices = vertices.ToArray(),
                Normals = normals.ToArray(),
                TexCoords = texCoords.ToArray(),
                Indices = indices.ToArray(),
                TriangleMaterialIds = materialIds.ToArray(),
                Materials = materials,
                Spheres = spheres,
                SphereMaterialIds = sphereMaterialIds,
                Boxes = boxes,
                BoxMaterialIds = boxMaterialIds,
                CylindersY = cylindersY,
                CylinderYMaterialIds = cylinderYMaterialIds,
                Disks = disks,
                DiskMaterialIds = diskMaterialIds,
                XYRects = xyRects,
                XYRectMaterialIds = xyRectMaterialIds,
                XZRects = xzRects,
                XZRectMaterialIds = xzRectMaterialIds,
                YZRects = yzRects,
                YZRectMaterialIds = yzRectMaterialIds,
                Lights = lights,
                AmbientColor = new Vec3(0.5f, 0.6f, 0.8f),
                AmbientIntensity = 0.15f,
                BackgroundTop = new Vec3(0.4f, 0.55f, 0.8f),
                BackgroundBottom = new Vec3(0.05f, 0.05f, 0.08f),
                CameraOrigin = new Vec3(0f, 3f, 9f),
                CameraLookAt = new Vec3(0f, 1.2f, -1f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 55f,
                CameraWorldScale = 10f,
            };
        }

        // Direct port of the reference's Scenes.BuildVolumeGridTestScene() (16x8x16 grid,
        // voxelSize 0.5, minCorner (-4,0,-6)) - floor+perimeter walls (both matId 1),
        // 4 colored pillars, a checker dais, 4 "game object" placeholder cells (their
        // metaId is reference-only bookkeeping our engine doesn't use - the cells
        // themselves are real, rendered voxels), plus 4 pedestals+spheres and 1 clear
        // glass sphere floating above the dais. docs/SAMPLE13_PLAN.md's own decision:
        // the DDA *technique* is ported, not the reference's procedural
        // WorldGenerator-driven BuildMinecraftLike (out of scope).
        public static SceneData BuildVolumeGridTestScene()
        {
            const int nx = 16, ny = 8, nz = 16;
            var voxels = new uint[nx * ny * nz];
            int Index(int x, int y, int z) => (x * ny * nz) + (y * nz) + z;

            // Floor: matId 1 (stone/white) everywhere at y=0.
            for (var x = 0; x < nx; x++)
                for (var z = 0; z < nz; z++)
                    voxels[Index(x, 0, z)] = 1;

            // Perimeter walls, y=1..3, also matId 1 (same stone material as the floor).
            for (var y = 1; y <= 3; y++)
            {
                for (var x = 0; x < nx; x++)
                {
                    voxels[Index(x, y, 0)] = 1;
                    voxels[Index(x, y, nz - 1)] = 1;
                }
                for (var z = 0; z < nz; z++)
                {
                    voxels[Index(0, y, z)] = 1;
                    voxels[Index(nx - 1, y, z)] = 1;
                }
            }

            void Pillar(int cx, int cz, int height, uint matId)
            {
                for (var y = 1; y < height && y < ny; y++)
                    voxels[Index(cx, y, cz)] = matId;
            }
            Pillar(4, 4, 4, 2);   // red
            Pillar(11, 4, 3, 3);  // green
            Pillar(4, 11, 5, 4);  // blue
            Pillar(11, 11, 4, 5); // mirror

            // Checker dais, x/z in [6,9), y=1.
            for (var x = 6; x < 9; x++)
                for (var z = 6; z < 9; z++)
                    voxels[Index(x, 1, z)] = ((x + z) & 1) == 0 ? 1u : 4u;

            // "Game object" placeholder cells - the reference's metaId (101-104) is
            // bookkeeping this engine has no equivalent for; the voxels themselves are
            // real and rendered with their given material.
            voxels[Index(2, 1, 2)] = 2;
            voxels[Index(13, 1, 2)] = 3;
            voxels[Index(2, 1, 13)] = 4;
            voxels[Index(13, 1, 13)] = 5;

            // Materials[i] corresponds to voxel value i+1 (see LaunchParams.VoxelMaterialIds).
            var materials = new[]
            {
                new MaterialSbtData { Albedo = new Vec3(0.82f, 0.82f, 0.85f), MaterialKind = MaterialSbtData.Solid },                     // 1: stone/white
                new MaterialSbtData { Albedo = new Vec3(0.95f, 0.15f, 0.15f), MaterialKind = MaterialSbtData.Solid },                     // 2: red
                new MaterialSbtData { Albedo = new Vec3(0.15f, 0.95f, 0.20f), MaterialKind = MaterialSbtData.Solid },                     // 3: green
                new MaterialSbtData { Albedo = new Vec3(0.15f, 0.25f, 0.95f), MaterialKind = MaterialSbtData.Solid },                     // 4: blue
                new MaterialSbtData { Albedo = new Vec3(0.98f, 0.98f, 0.98f), Reflectivity = 0.9f, MaterialKind = MaterialSbtData.Solid }, // 5: mirror
                // Extra materials (indices 5-6) for the pedestals/spheres placed around the grid, below.
                new MaterialSbtData { Albedo = new Vec3(0.85f, 0.85f, 0.85f), MaterialKind = MaterialSbtData.Solid },                     // 5: pedestal
                new MaterialSbtData
                {
                    Albedo = new Vec3(1f, 1f, 1f), Transparency = 1f, IndexOfRefraction = 1.5f,
                    TransmissionColor = new Vec3(1f, 1f, 1f), MaterialKind = MaterialSbtData.Solid,
                }, // 6: clear glass
            };

            const float pedR = 0.25f, pedH = 1.2f, sphR = 0.35f;
            Vec3 posL = new Vec3(-1.6f, 0f, -2.0f);
            Vec3 posR = new Vec3(1.6f, 0f, -2.0f);
            Vec3 posF = new Vec3(0f, 0f, -0.8f);
            Vec3 posB = new Vec3(0f, 0f, -3.2f);

            var cylindersY = new[]
            {
                new CylinderYData { Center = posL, Radius = pedR, YMin = 0f, YMax = pedH, Capped = 1 },
                new CylinderYData { Center = posR, Radius = pedR, YMin = 0f, YMax = pedH, Capped = 1 },
                new CylinderYData { Center = posF, Radius = pedR, YMin = 0f, YMax = pedH, Capped = 1 },
                new CylinderYData { Center = posB, Radius = pedR, YMin = 0f, YMax = pedH, Capped = 1 },
            };
            var cylinderYMaterialIds = new uint[] { 5, 5, 5, 5 };

            var spheres = new[]
            {
                new SphereData { Center = posL + new Vec3(0f, pedH + sphR, 0f), Radius = sphR }, // mirror
                new SphereData { Center = posR + new Vec3(0f, pedH + sphR, 0f), Radius = sphR },  // red
                new SphereData { Center = posF + new Vec3(0f, pedH + sphR, 0f), Radius = sphR },  // blue
                new SphereData { Center = posB + new Vec3(0f, pedH + sphR, 0f), Radius = sphR },  // green
                new SphereData { Center = new Vec3(0f, 2f, -2.0f), Radius = 0.5f },                // clear glass
            };
            var sphereMaterialIds = new uint[] { 4, 1, 3, 2, 6 };

            var lights = new[]
            {
                new PointLightGpu { Position = new Vec3(0f, 5f, -3f), Color = new Vec3(1f, 1f, 1f), Intensity = 220f },
                new PointLightGpu { Position = new Vec3(-2.5f, 3f, -1.8f), Color = new Vec3(1f, 0.95f, 0.9f), Intensity = 90f },
            };

            return new SceneData
            {
                Name = "Volume grid room (DDA)",
                VoxelMaterialIds = voxels,
                VolumeGridMin = new Vec3(-4f, 0f, -6f),
                VolumeVoxelSize = new Vec3(0.5f, 0.5f, 0.5f),
                VolumeDims = new Vec3i(nx, ny, nz),
                Materials = materials,
                CylindersY = cylindersY,
                CylinderYMaterialIds = cylinderYMaterialIds,
                Spheres = spheres,
                SphereMaterialIds = sphereMaterialIds,
                Lights = lights,
                AmbientColor = new Vec3(1f, 1f, 1f),
                AmbientIntensity = 0.01f,
                BackgroundTop = new Vec3(0.02f, 0.02f, 0.02f),
                BackgroundBottom = new Vec3(0.02f, 0.02f, 0.02f),
                CameraOrigin = new Vec3(0f, 1f, 0f),
                CameraLookAt = new Vec3(0f, 1f, -10f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 45f,
                CameraWorldScale = 8f,
            };
        }

        // Direct port of the reference's Scenes.BuildTestScene() - four spheres (three
        // diffuse, one mirror) over a near-black background, lit by two point lights and
        // almost no ambient. The simplest possible smoke test for the sphere primitive.
        public static SceneData BuildSimpleTestScene()
        {
            var materials = new[]
            {
                new MaterialSbtData { Albedo = new Vec3(1f, 0f, 0f), MaterialKind = MaterialSbtData.Solid },
                new MaterialSbtData { Albedo = new Vec3(0f, 1f, 0f), MaterialKind = MaterialSbtData.Solid },
                new MaterialSbtData { Albedo = new Vec3(0f, 0f, 1f), MaterialKind = MaterialSbtData.Solid },
                new MaterialSbtData { Albedo = new Vec3(0.98f, 0.98f, 0.98f), Reflectivity = 0.9f, MaterialKind = MaterialSbtData.Solid },
            };

            var spheres = new[]
            {
                new SphereData { Center = new Vec3(-1.2f, 0.9f, -2.2f), Radius = 0.9f },
                new SphereData { Center = new Vec3(1.2f, 0.9f, -2.2f), Radius = 0.9f },
                new SphereData { Center = new Vec3(-1.2f, 0.9f, -3.6f), Radius = 0.9f },
                new SphereData { Center = new Vec3(1.2f, 0.9f, -3.6f), Radius = 0.9f },
            };
            var sphereMaterialIds = new uint[] { 0, 1, 2, 3 };

            var lights = new[]
            {
                new PointLightGpu { Position = new Vec3(0f, 3.2f, -2.9f), Color = new Vec3(1f, 1f, 1f), Intensity = 140f },
                new PointLightGpu { Position = new Vec3(-2.2f, 2.0f, -2.4f), Color = new Vec3(1f, 1f, 1f), Intensity = 60f },
            };

            return new SceneData
            {
                Name = "Simple test (4 spheres)",
                Materials = materials,
                Spheres = spheres,
                SphereMaterialIds = sphereMaterialIds,
                Lights = lights,
                AmbientColor = new Vec3(1f, 1f, 1f),
                AmbientIntensity = 0.01f,
                BackgroundTop = new Vec3(0.05f, 0.05f, 0.05f),
                BackgroundBottom = new Vec3(0.05f, 0.05f, 0.05f),
                CameraOrigin = new Vec3(0f, 1f, 0f),
                CameraLookAt = new Vec3(0f, 1f, -10f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 45f,
                CameraWorldScale = 6f,
            };
        }

        // Direct port of the reference's Scenes.BuildDemoScene() - a checkered floor, an
        // emissive "sun" sphere, 3 hand-placed base spheres (red/blue/mirror), and 100
        // randomly scattered small spheres (unseeded System.Random + HSV color
        // randomization + rejection sampling against previously-placed spheres, exactly
        // matching the reference so it looks different every run). The reference's
        // position-range code comments are stale; the literal X in [-9,0), Z in
        // [-9.8,-5.2) bounds below are what the reference code actually executes.
        public static SceneData BuildDemoScene()
        {
            var materials = new List<MaterialSbtData>
            {
                new MaterialSbtData { Albedo = new Vec3(0.9f, 0.2f, 0.2f), Reflectivity = 0.2f, MaterialKind = MaterialSbtData.Solid },  // 0: red
                new MaterialSbtData { Albedo = new Vec3(0.2f, 0.2f, 0.9f), Reflectivity = 0.5f, MaterialKind = MaterialSbtData.Solid },  // 1: blue
                new MaterialSbtData { Albedo = new Vec3(0.95f, 0.95f, 0.95f), Reflectivity = 0.9f, MaterialKind = MaterialSbtData.Solid }, // 2: mirror
                new MaterialSbtData { Albedo = new Vec3(1f, 1f, 1f), Emission = new Vec3(8f, 8f, 8f), MaterialKind = MaterialSbtData.Solid }, // 3: emissive sun
            };

            var xzRects = new[]
            {
                new RectData { A0 = -10f, A1 = 10f, B0 = -11f, B1 = 2f, C = 0f },
            };
            var xzRectMaterialIds = new uint[]
            {
                (uint)materials.Count,
            };
            materials.Add(new MaterialSbtData
            {
                Albedo = new Vec3(0.8f, 0.8f, 0.8f),
                MaterialKind = MaterialSbtData.Checker,
                CheckerColorB = new Vec3(0.1f, 0.1f, 0.1f),
                CheckerScale = 0.5f,
            });

            var spheres = new List<SphereData>
            {
                new SphereData { Center = new Vec3(-1.2f, 1f, 0f), Radius = 1f },     // red
                new SphereData { Center = new Vec3(1.2f, 1f, -0.5f), Radius = 1f },    // blue
                new SphereData { Center = new Vec3(0f, 0.5f, -2.5f), Radius = 0.5f },  // mirror
                new SphereData { Center = new Vec3(0f, 5f, 2f), Radius = 0.5f },       // emissive sun
            };
            var sphereMaterialIds = new List<uint> { 0, 1, 2, 3 };

            var rng = new Random();
            const int randomSphereCount = 100;
            for (var i = 0; i < randomSphereCount; i++)
            {
                for (var attempt = 0; attempt < 32; attempt++)
                {
                    float x = -9f + (float)rng.NextDouble() * 9f;
                    float z = -9.8f + (float)rng.NextDouble() * 4.6f;
                    float radius = 0.18f + (float)rng.NextDouble() * 0.32f;
                    var center = new Vec3(x, radius, z);

                    bool overlaps = false;
                    foreach (var existing in spheres)
                    {
                        float minDist = radius + existing.Radius + 0.05f;
                        if ((center - existing.Center).length() < minDist)
                        {
                            overlaps = true;
                            break;
                        }
                    }
                    if (overlaps)
                        continue;

                    float hue = (float)rng.NextDouble() * 360f;
                    Vec3 color = HsvToRgb(hue, 0.65f, 0.9f);
                    bool isReflective = rng.NextDouble() < 0.2;
                    var material = new MaterialSbtData
                    {
                        Albedo = color,
                        Reflectivity = isReflective ? 0.6f : 0.05f,
                        MaterialKind = MaterialSbtData.Solid,
                    };

                    sphereMaterialIds.Add((uint)materials.Count);
                    materials.Add(material);
                    spheres.Add(new SphereData { Center = center, Radius = radius });
                    break;
                }
            }

            var lights = new[]
            {
                new PointLightGpu { Position = new Vec3(-2f, 4f, 3f), Color = new Vec3(1f, 0.9f, 0.8f), Intensity = 60f },
                new PointLightGpu { Position = new Vec3(2.5f, 3.5f, -1.5f), Color = new Vec3(0.8f, 0.9f, 1f), Intensity = 40f },
            };

            return new SceneData
            {
                Name = "Demo scene (random spheres)",
                Materials = materials.ToArray(),
                Spheres = spheres.ToArray(),
                SphereMaterialIds = sphereMaterialIds.ToArray(),
                XZRects = xzRects,
                XZRectMaterialIds = xzRectMaterialIds,
                Lights = lights,
                AmbientColor = new Vec3(1f, 1f, 1f),
                AmbientIntensity = 0.075f,
                BackgroundTop = new Vec3(0.6f, 0.8f, 1.0f),
                BackgroundBottom = new Vec3(0.9f, 0.95f, 1.0f),
                CameraOrigin = new Vec3(0f, 1f, 0f),
                CameraLookAt = new Vec3(0f, 1f, -10f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 45f,
                CameraWorldScale = 10f,
            };
        }

        // HSV (h in degrees, s/v in [0,1]) to linear RGB, matching the reference's own
        // color-randomization formula used by BuildDemoScene's random spheres.
        private static Vec3 HsvToRgb(float h, float s, float v)
        {
            float c = v * s;
            float hPrime = h / 60f;
            float x = c * (1f - Math.Abs(hPrime % 2f - 1f));
            float m = v - c;

            Vec3 rgb;
            if (hPrime < 1f) rgb = new Vec3(c, x, 0f);
            else if (hPrime < 2f) rgb = new Vec3(x, c, 0f);
            else if (hPrime < 3f) rgb = new Vec3(0f, c, x);
            else if (hPrime < 4f) rgb = new Vec3(0f, x, c);
            else if (hPrime < 5f) rgb = new Vec3(x, 0f, c);
            else rgb = new Vec3(c, 0f, x);

            return new Vec3(rgb.x + m, rgb.y + m, rgb.z + m);
        }

        // Direct port of the reference's Scenes.BuildCornellBox() - the classic
        // red/green/white box with a ceiling light panel and two boxes, using our
        // XYRect/XZRect/YZRect custom primitives in place of the reference's infinite
        // Planes (docs/SAMPLE13_PLAN.md decision (c): finite rects, sized to the box).
        public static SceneData BuildCornellBox()
        {
            var materials = new[]
            {
                new MaterialSbtData { Albedo = new Vec3(0.82f, 0.82f, 0.82f), MaterialKind = MaterialSbtData.Solid }, // 0: white
                new MaterialSbtData { Albedo = new Vec3(0.8f, 0.1f, 0.1f), MaterialKind = MaterialSbtData.Solid },    // 1: red
                new MaterialSbtData { Albedo = new Vec3(0.1f, 0.8f, 0.1f), MaterialKind = MaterialSbtData.Solid },    // 2: green
                new MaterialSbtData { Albedo = new Vec3(0f, 0f, 0f), Emission = new Vec3(0.6f, 0.6f, 0.6f), MaterialKind = MaterialSbtData.Solid }, // 3: light panel
            };

            var yzRects = new[]
            {
                new RectData { A0 = 0f, A1 = 5f, B0 = -5f, B1 = 0f, C = -3f }, // left wall (red)
                new RectData { A0 = 0f, A1 = 5f, B0 = -5f, B1 = 0f, C = 3f },  // right wall (green)
            };
            var yzRectMaterialIds = new uint[] { 1, 2 };

            var xzRects = new[]
            {
                new RectData { A0 = -3f, A1 = 3f, B0 = -5f, B1 = 0f, C = 0f },   // floor
                new RectData { A0 = -3f, A1 = 3f, B0 = -5f, B1 = 0f, C = 5f },   // ceiling
                new RectData { A0 = -0.9f, A1 = 0.9f, B0 = -3.2f, B1 = -2.2f, C = 4.99f }, // light panel
            };
            var xzRectMaterialIds = new uint[] { 0, 0, 3 };

            var xyRects = new[]
            {
                new RectData { A0 = -3f, A1 = 3f, B0 = 0f, B1 = 5f, C = -5f }, // back wall
            };
            var xyRectMaterialIds = new uint[] { 0 };

            var boxes = new[]
            {
                new BoxData { Min = new Vec3(-2.2f, 0f, -4.0f), Max = new Vec3(-0.8f, 1.0f, -2.8f) },
                new BoxData { Min = new Vec3(0.6f, 0f, -3.3f), Max = new Vec3(2.0f, 1.8f, -2.1f) },
            };
            var boxMaterialIds = new uint[] { 0, 0 };

            var lights = new[]
            {
                new PointLightGpu { Position = new Vec3(0f, 4.6f, -2.7f), Color = new Vec3(1f, 1f, 1f), Intensity = 20f },
            };

            return new SceneData
            {
                Name = "Cornell box",
                Materials = materials,
                YZRects = yzRects,
                YZRectMaterialIds = yzRectMaterialIds,
                XZRects = xzRects,
                XZRectMaterialIds = xzRectMaterialIds,
                XYRects = xyRects,
                XYRectMaterialIds = xyRectMaterialIds,
                Boxes = boxes,
                BoxMaterialIds = boxMaterialIds,
                Lights = lights,
                AmbientColor = new Vec3(1f, 1f, 1f),
                AmbientIntensity = 0f,
                BackgroundTop = new Vec3(0f, 0f, 0f),
                BackgroundBottom = new Vec3(0f, 0f, 0f),
                CameraOrigin = new Vec3(0f, 1f, 0f),
                CameraLookAt = new Vec3(0f, 1f, -10f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 45f,
                CameraWorldScale = 6f,
            };
        }

        // Direct port of the reference's Scenes.BuildMirrorSpheresOnChecker() - gold,
        // "glassy" (reflectivity 0.6, below the 0.9 mirror-dispatch threshold in both
        // engines - it renders as ordinary diffuse, matching the reference's own
        // behavior), and true-mirror spheres over a checkered floor.
        public static SceneData BuildMirrorSpheresOnChecker()
        {
            var materials = new[]
            {
                new MaterialSbtData
                {
                    Albedo = new Vec3(0.8f, 0.8f, 0.8f),
                    MaterialKind = MaterialSbtData.Checker,
                    CheckerColorB = new Vec3(0.15f, 0.15f, 0.15f),
                    CheckerScale = 0.6f,
                }, // 0: floor
                new MaterialSbtData { Albedo = new Vec3(1.0f, 0.85f, 0.57f), Reflectivity = 0.1f, MaterialKind = MaterialSbtData.Solid },  // 1: gold
                new MaterialSbtData { Albedo = new Vec3(0.9f, 0.95f, 1.0f), Reflectivity = 0.6f, MaterialKind = MaterialSbtData.Solid },   // 2: glassy (diffuse)
                new MaterialSbtData { Albedo = new Vec3(0.98f, 0.98f, 0.98f), Reflectivity = 0.85f, MaterialKind = MaterialSbtData.Solid }, // 3: mirror
            };

            var xzRects = new[]
            {
                new RectData { A0 = -8f, A1 = 8f, B0 = -8f, B1 = 4f, C = 0f },
            };
            var xzRectMaterialIds = new uint[] { 0 };

            var spheres = new[]
            {
                new SphereData { Center = new Vec3(-1.2f, 1f, -2.0f), Radius = 1f },
                new SphereData { Center = new Vec3(1.3f, 1f, -2.6f), Radius = 1f },
                new SphereData { Center = new Vec3(0f, 0.5f, -4.2f), Radius = 0.5f },
            };
            var sphereMaterialIds = new uint[] { 1, 2, 3 };

            var lights = new[]
            {
                new PointLightGpu { Position = new Vec3(-2.5f, 3.5f, -1.5f), Color = new Vec3(1f, 0.95f, 0.9f), Intensity = 90f },
                new PointLightGpu { Position = new Vec3(2.0f, 2.8f, -3.8f), Color = new Vec3(0.9f, 0.95f, 1.0f), Intensity = 70f },
            };

            return new SceneData
            {
                Name = "Mirror spheres on checker",
                Materials = materials,
                XZRects = xzRects,
                XZRectMaterialIds = xzRectMaterialIds,
                Spheres = spheres,
                SphereMaterialIds = sphereMaterialIds,
                Lights = lights,
                AmbientColor = new Vec3(1f, 1f, 1f),
                AmbientIntensity = 0.075f,
                BackgroundTop = new Vec3(0.55f, 0.75f, 1.0f),
                BackgroundBottom = new Vec3(0.95f, 0.98f, 1.0f),
                CameraOrigin = new Vec3(0f, 1f, 0f),
                CameraLookAt = new Vec3(0f, 1f, -10f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 45f,
                CameraWorldScale = 8f,
            };
        }

        // Direct port of the reference's Scenes.BuildCylindersDisksAndTriangles() -
        // exercises a capped Y-cylinder, a disk, and a raw triangle together with the
        // custom-primitive GAS, over a checkered floor (mixed triangle+custom-primitive
        // GAS, same combination already proven by BuildDebugOrenNayarScene).
        public static SceneData BuildCylindersDisksAndTriangles()
        {
            var materials = new[]
            {
                new MaterialSbtData
                {
                    Albedo = new Vec3(0.75f, 0.75f, 0.75f),
                    MaterialKind = MaterialSbtData.Checker,
                    CheckerColorB = new Vec3(0.2f, 0.2f, 0.2f),
                    CheckerScale = 0.8f,
                }, // 0: floor
                new MaterialSbtData { Albedo = new Vec3(0.2f, 0.35f, 0.9f), MaterialKind = MaterialSbtData.Solid }, // 1: matte blue (cylinder)
                new MaterialSbtData { Albedo = new Vec3(0.9f, 0.25f, 0.25f), MaterialKind = MaterialSbtData.Solid }, // 2: matte red (triangle)
                new MaterialSbtData { Albedo = new Vec3(0.8f, 0.8f, 0.1f), MaterialKind = MaterialSbtData.Solid },   // 3: yellow (disk)
            };

            var xzRects = new[]
            {
                new RectData { A0 = -10f, A1 = 10f, B0 = -10f, B1 = 4f, C = 0f },
            };
            var xzRectMaterialIds = new uint[] { 0 };

            var cylindersY = new[]
            {
                new CylinderYData { Center = new Vec3(-1.2f, 0f, -3.0f), Radius = 0.6f, YMin = 0f, YMax = 1.6f, Capped = 1 },
            };
            var cylinderYMaterialIds = new uint[] { 1 };

            var disks = new[]
            {
                new DiskData { Center = new Vec3(1.6f, 0.01f, -2.2f), Normal = new Vec3(0f, 1f, 0f), Radius = 0.9f },
            };
            var diskMaterialIds = new uint[] { 3 };

            var vertices = new[]
            {
                new Vec3(0.2f, 0f, -3.6f), new Vec3(1.3f, 1.4f, -3.0f), new Vec3(-0.7f, 0.7f, -2.8f),
            };
            Vec3 triNormal = Vec3.unitVector(Vec3.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
            var normals = new[] { triNormal, triNormal, triNormal };
            var texCoords = new[] { new Vec2(0f, 0f), new Vec2(1f, 0f), new Vec2(0f, 1f) };
            var indices = new[] { new Vec3i(0, 1, 2) };
            var triangleMaterialIds = new uint[] { 2 };

            var lights = new[]
            {
                new PointLightGpu { Position = new Vec3(-2.2f, 3.2f, -2.0f), Color = new Vec3(1f, 0.95f, 0.9f), Intensity = 70f },
                new PointLightGpu { Position = new Vec3(2.4f, 2.2f, -4.4f), Color = new Vec3(0.9f, 0.95f, 1.0f), Intensity = 60f },
            };

            return new SceneData
            {
                Name = "Cylinders, disks and triangles",
                Vertices = vertices,
                Normals = normals,
                TexCoords = texCoords,
                Indices = indices,
                TriangleMaterialIds = triangleMaterialIds,
                Materials = materials,
                XZRects = xzRects,
                XZRectMaterialIds = xzRectMaterialIds,
                CylindersY = cylindersY,
                CylinderYMaterialIds = cylinderYMaterialIds,
                Disks = disks,
                DiskMaterialIds = diskMaterialIds,
                Lights = lights,
                AmbientColor = new Vec3(1f, 1f, 1f),
                AmbientIntensity = 0.075f,
                BackgroundTop = new Vec3(0.58f, 0.78f, 1.0f),
                BackgroundBottom = new Vec3(0.95f, 0.98f, 1.0f),
                CameraOrigin = new Vec3(0f, 1f, 0f),
                CameraLookAt = new Vec3(0f, 1f, -10f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 45f,
                CameraWorldScale = 8f,
            };
        }

        // Direct port of the reference's Scenes.BuildBoxesShowcase() - three plain white
        // diffuse boxes over a checkered floor. The reference's box constructors pass
        // trailing reflectivity args that its own Solid() closure quirk always ignores
        // (see docs/SAMPLE13_PLAN.md) - all 3 boxes render as diffuse white, matching the
        // reference's actual behavior rather than its misleading constructor arguments.
        public static SceneData BuildBoxesShowcase()
        {
            var materials = new[]
            {
                new MaterialSbtData
                {
                    Albedo = new Vec3(0.85f, 0.85f, 0.85f),
                    MaterialKind = MaterialSbtData.Checker,
                    CheckerColorB = new Vec3(0.15f, 0.15f, 0.15f),
                    CheckerScale = 0.7f,
                }, // 0: floor
                new MaterialSbtData { Albedo = new Vec3(0.86f, 0.86f, 0.86f), MaterialKind = MaterialSbtData.Solid }, // 1: white boxes
            };

            var xzRects = new[]
            {
                new RectData { A0 = -10f, A1 = 10f, B0 = -10f, B1 = 4f, C = 0f },
            };
            var xzRectMaterialIds = new uint[] { 0 };

            var boxes = new[]
            {
                new BoxData { Min = new Vec3(-2.2f, 0f, -3.6f), Max = new Vec3(-1.0f, 1.2f, -2.4f) },
                new BoxData { Min = new Vec3(-0.6f, 0f, -4.2f), Max = new Vec3(0.6f, 0.6f, -3.0f) },
                new BoxData { Min = new Vec3(1.0f, 0f, -3.0f), Max = new Vec3(2.4f, 2.0f, -1.8f) },
            };
            var boxMaterialIds = new uint[] { 1, 1, 1 };

            var lights = new[]
            {
                new PointLightGpu { Position = new Vec3(-2.0f, 3.0f, -2.0f), Color = new Vec3(1f, 0.95f, 0.9f), Intensity = 70f },
                new PointLightGpu { Position = new Vec3(2.0f, 2.5f, -4.2f), Color = new Vec3(0.9f, 0.95f, 1.0f), Intensity = 50f },
            };

            return new SceneData
            {
                Name = "Boxes showcase",
                Materials = materials,
                XZRects = xzRects,
                XZRectMaterialIds = xzRectMaterialIds,
                Boxes = boxes,
                BoxMaterialIds = boxMaterialIds,
                Lights = lights,
                AmbientColor = new Vec3(1f, 1f, 1f),
                AmbientIntensity = 0.075f,
                BackgroundTop = new Vec3(0.6f, 0.8f, 1.0f),
                BackgroundBottom = new Vec3(0.95f, 0.98f, 1.0f),
                CameraOrigin = new Vec3(0f, 1f, 0f),
                CameraLookAt = new Vec3(0f, 1f, -10f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 45f,
                CameraWorldScale = 8f,
            };
        }

        // Shared by every mesh scene (M6) - reuses the existing triangle GAS pipeline
        // as-is (meshes are pure triangle geometry, no new subsystem needed), auto-fits
        // the camera/lights to each mesh's own bounding box since these 4 OBJs (cow,
        // stanford-bunny, teapot, xyzrgb_dragon - see docs/SAMPLE13_PLAN.md) are at very
        // different real-world scales.
        private static SceneData BuildMeshScene(string name, string objFileName, MaterialSbtData material, Vec3 lightColorA, Vec3 lightColorB)
        {
            string objPath = Path.Combine(AppContext.BaseDirectory, "models", "meshes", objFileName);
            var mesh = OBJModel.Load(objPath);

            Vec3 min = new Vec3(float.MaxValue, float.MaxValue, float.MaxValue);
            Vec3 max = new Vec3(float.MinValue, float.MinValue, float.MinValue);
            foreach (var v in mesh.Vertices)
            {
                min = new Vec3(Math.Min(v.x, min.x), Math.Min(v.y, min.y), Math.Min(v.z, min.z));
                max = new Vec3(Math.Max(v.x, max.x), Math.Max(v.y, max.y), Math.Max(v.z, max.z));
            }
            Vec3 center = (min + max) / 2f;
            float radius = (max - min).length() * 0.5f;

            // Every triangle uses material index 0 - these OBJs have no mtllib/usemtl,
            // so OBJModel.Load's own (unused) default-material bookkeeping is bypassed
            // in favor of the one material passed in per mesh scene.
            var triangleMaterialIds = new uint[mesh.Indices.Length];

            var lights = new[]
            {
                new PointLightGpu { Position = center + new Vec3(radius * 1.5f, radius * 2f, radius * 1f), Color = lightColorA, Intensity = radius * radius * 3f },
                new PointLightGpu { Position = center + new Vec3(-radius * 1.5f, radius * 1f, -radius * 1.5f), Color = lightColorB, Intensity = radius * radius * 1.5f },
            };

            return new SceneData
            {
                Name = name,
                Vertices = mesh.Vertices,
                Normals = mesh.Normals,
                TexCoords = mesh.TexCoords,
                Indices = mesh.Indices,
                TriangleMaterialIds = triangleMaterialIds,
                Materials = new[] { material },
                Lights = lights,
                AmbientColor = new Vec3(0.5f, 0.55f, 0.65f),
                AmbientIntensity = 0.15f,
                BackgroundTop = new Vec3(0.35f, 0.45f, 0.6f),
                BackgroundBottom = new Vec3(0.05f, 0.05f, 0.08f),
                CameraOrigin = center + new Vec3(radius * 1.4f, radius * 0.9f, radius * 1.8f),
                CameraLookAt = center,
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 45f,
                CameraWorldScale = radius,
            };
        }

        public static SceneData BuildCowScene() => BuildMeshScene(
            "Mesh: Cow",
            "cow.obj",
            new MaterialSbtData { Albedo = new Vec3(0.83f, 0.68f, 0.21f), MaterialKind = MaterialSbtData.Solid },
            new Vec3(1f, 0.95f, 0.85f), new Vec3(0.6f, 0.7f, 1f));

        public static SceneData BuildBunnyScene() => BuildMeshScene(
            "Mesh: Stanford Bunny",
            "stanford-bunny.obj",
            new MaterialSbtData { Albedo = new Vec3(0.2f, 0.6f, 0.35f), MaterialKind = MaterialSbtData.Solid },
            new Vec3(1f, 0.95f, 0.85f), new Vec3(0.6f, 0.7f, 1f));

        public static SceneData BuildTeapotScene() => BuildMeshScene(
            "Mesh: Teapot",
            "teapot.obj",
            new MaterialSbtData { Albedo = new Vec3(0.6f, 0.05f, 0.08f), MaterialKind = MaterialSbtData.Solid },
            new Vec3(1f, 0.95f, 0.85f), new Vec3(0.6f, 0.7f, 1f));

        // Mirror (Reflectivity >= 0.9) rather than diffuse - exercises the M2 bounce
        // loop against a real high-poly mesh, matching the reference's own
        // sapphire-mirror dragon.
        public static SceneData BuildDragonScene() => BuildMeshScene(
            "Mesh: XYZRGB Dragon (mirror)",
            "xyzrgb_dragon.obj",
            new MaterialSbtData { Albedo = new Vec3(0.85f, 0.9f, 0.95f), Reflectivity = 0.95f, MaterialKind = MaterialSbtData.Solid },
            new Vec3(1f, 0.95f, 0.85f), new Vec3(0.6f, 0.7f, 1f));

        // Direct port of the reference's MeshScenes.BuildAllMeshesScene() - all four
        // meshes normalized to a common size and placed side-by-side over one floor.
        // The reference recenters each mesh at its largest-connected-component's
        // triangle-centroid purely to compute placement (not to filter the actual
        // rendered triangles) - approximated here with a plain bounding-box center,
        // which places these well-formed meshes equivalently for rendering purposes.
        public static SceneData BuildAllMeshesScene()
        {
            var vertices = new List<Vec3>();
            var normals = new List<Vec3>();
            var texCoords = new List<Vec2>();
            var indices = new List<Vec3i>();
            var triangleMaterialIds = new List<uint>();
            var meshRanges = new List<MeshRange>();

            void AddMesh(string objFileName, Vec3 targetPos, uint materialIndex)
            {
                int indexStart = indices.Count;
                string objPath = Path.Combine(AppContext.BaseDirectory, "models", "meshes", objFileName);
                var mesh = OBJModel.Load(objPath);

                Vec3 min = new Vec3(float.MaxValue, float.MaxValue, float.MaxValue);
                Vec3 max = new Vec3(float.MinValue, float.MinValue, float.MinValue);
                foreach (var v in mesh.Vertices)
                {
                    min = new Vec3(Math.Min(v.x, min.x), Math.Min(v.y, min.y), Math.Min(v.z, min.z));
                    max = new Vec3(Math.Max(v.x, max.x), Math.Max(v.y, max.y), Math.Max(v.z, max.z));
                }
                Vec3 center = (min + max) / 2f;
                Vec3 size = max - min;
                float maxExtent = Math.Max(size.x, Math.Max(size.y, size.z));
                float scale = maxExtent > 1e-6f ? 1.6f / maxExtent : 1f;

                int baseIndex = vertices.Count;
                float minY = float.MaxValue;
                var placed = new Vec3[mesh.Vertices.Length];
                for (var i = 0; i < mesh.Vertices.Length; i++)
                {
                    var v = (mesh.Vertices[i] - center) * scale;
                    placed[i] = v;
                    minY = Math.Min(minY, v.y);
                }
                float groundOffset = targetPos.y - minY;
                for (var i = 0; i < placed.Length; i++)
                {
                    vertices.Add(new Vec3(
                        placed[i].x + targetPos.x,
                        placed[i].y + groundOffset,
                        placed[i].z + targetPos.z));
                }
                normals.AddRange(mesh.Normals);
                texCoords.AddRange(mesh.TexCoords);

                foreach (var tri in mesh.Indices)
                {
                    indices.Add(new Vec3i(tri.x + baseIndex, tri.y + baseIndex, tri.z + baseIndex));
                    triangleMaterialIds.Add(materialIndex);
                }
                meshRanges.Add(new MeshRange { IndexStart = indexStart, IndexCount = indices.Count - indexStart });
            }

            var materials = new[]
            {
                new MaterialSbtData { Albedo = new Vec3(0.80f, 0.45f, 0.25f), MaterialKind = MaterialSbtData.Solid },                     // 0: cow - copper
                new MaterialSbtData { Albedo = new Vec3(0.0f, 0.5f, 0.5f), MaterialKind = MaterialSbtData.Solid },                        // 1: bunny - jade
                new MaterialSbtData { Albedo = new Vec3(0.9f, 0.9f, 0.0f), Reflectivity = 0.06f, MaterialKind = MaterialSbtData.Solid },   // 2: teapot - gold
                new MaterialSbtData { Albedo = new Vec3(0.88f, 0.0f, 0.88f), Reflectivity = 0.65f, MaterialKind = MaterialSbtData.Solid }, // 3: dragon - amethyst (below the 0.9 mirror threshold - renders diffuse)
                new MaterialSbtData
                {
                    Albedo = new Vec3(0.82f, 0.82f, 0.82f),
                    MaterialKind = MaterialSbtData.Checker,
                    CheckerColorB = new Vec3(0.15f, 0.15f, 0.15f),
                    CheckerScale = 0.7f,
                }, // 4: floor
            };

            AddMesh("cow.obj", new Vec3(-3.2f, 0f, -4.0f), 0);
            AddMesh("stanford-bunny.obj", new Vec3(-1.0f, 0f, -3.0f), 1);
            AddMesh("teapot.obj", new Vec3(1.6f, 0f, -3.2f), 2);
            AddMesh("xyzrgb_dragon.obj", new Vec3(3.2f, 0f, -4.6f), 3);

            var xzRects = new[]
            {
                new RectData { A0 = -10f, A1 = 10f, B0 = -10f, B1 = 4f, C = 0f },
            };
            var xzRectMaterialIds = new uint[] { 4 };

            var lights = new[]
            {
                new PointLightGpu { Position = new Vec3(0f, 6f, 0f), Color = new Vec3(1f, 0.95f, 0.88f), Intensity = 65f },
                new PointLightGpu { Position = new Vec3(-3f, 3f, 2f), Color = new Vec3(0.85f, 0.90f, 1.0f), Intensity = 35f },
            };

            return new SceneData
            {
                Name = "All meshes (cow/bunny/teapot/dragon)",
                Vertices = vertices.ToArray(),
                Normals = normals.ToArray(),
                TexCoords = texCoords.ToArray(),
                Indices = indices.ToArray(),
                TriangleMaterialIds = triangleMaterialIds.ToArray(),
                MeshRanges = FillMeshRangeGaps(meshRanges, indices.Count),
                Materials = materials,
                XZRects = xzRects,
                XZRectMaterialIds = xzRectMaterialIds,
                Lights = lights,
                AmbientColor = new Vec3(1f, 1f, 1f),
                AmbientIntensity = 0.15f,
                BackgroundTop = new Vec3(0f, 0f, 0f),
                BackgroundBottom = new Vec3(0f, 0f, 0f),
                CameraOrigin = new Vec3(0f, 1.5f, 2f),
                CameraLookAt = new Vec3(0f, 0.8f, -4f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 50f,
                CameraWorldScale = 8f,
            };
        }

        // Ambient-only lit, single textured quad - reuses Sample08's CudaTextureObject/
        // CudaTex2D pattern (trivial, per docs/SAMPLE13_PLAN.md's M7 scope) via one
        // texture already bundled for Sponza, rather than fetching a new asset.
        public static SceneData BuildTextureTestScene()
        {
            var vertices = new[]
            {
                new Vec3(-3f, 0f, 3f), new Vec3(3f, 0f, 3f), new Vec3(3f, 3f, 3f), new Vec3(-3f, 3f, 3f),
            };
            var normal = new Vec3(0f, 0f, 1f);
            var normals = new[] { normal, normal, normal, normal };
            var texCoords = new[] { new Vec2(0f, 0f), new Vec2(1f, 0f), new Vec2(1f, 1f), new Vec2(0f, 1f) };
            var indices = new[] { new Vec3i(0, 1, 2), new Vec3i(0, 2, 3) };
            var materialIds = new uint[] { 0, 0 };

            var materials = new[]
            {
                new MaterialSbtData
                {
                    Albedo = new Vec3(1f, 1f, 1f),
                    MaterialKind = MaterialSbtData.Solid,
                    TextureWeight = 1f,
                    UVScale = 1f,
                },
            };

            return new SceneData
            {
                Name = "Debug: Texture (ambient-only)",
                Vertices = vertices,
                Normals = normals,
                TexCoords = texCoords,
                Indices = indices,
                TriangleMaterialIds = materialIds,
                Materials = materials,
                MaterialTexturePaths = new[] { "models/sponza/textures/spnza_bricks_a_diff.png" },
                Lights = Array.Empty<PointLightGpu>(),
                AmbientColor = new Vec3(1f, 1f, 1f),
                AmbientIntensity = 0.5f,
                BackgroundTop = new Vec3(0f, 0f, 0f),
                BackgroundBottom = new Vec3(0f, 0f, 0f),
                CameraOrigin = new Vec3(0f, 1.5f, 8f),
                CameraLookAt = new Vec3(0f, 1.5f, 3f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 45f,
                CameraWorldScale = 6f,
            };
        }

        // Direct port of the reference's TestScenes.BuildTestScene() ("Museum") -
        // several widely-spaced dioramas along a corridor: 3 Cornell rooms, a mesh
        // gallery, a pedestal quartet, a triangle showcase, a textured sphere + endcap
        // wall, a video-textured box, 2 volume-grid dioramas, and a video-cube diorama
        // with reflective/refractive spheres and meshes. One deliberate architectural
        // deviation from the reference: SceneData/LaunchParams model exactly ONE volume
        // grid region per scene (a single flat VoxelMaterialIds array + one
        // VolumeGridMin/VolumeVoxelSize/VolumeDims), so the reference's two independent
        // volume-grid dioramas are merged into one combined grid at a shared 0.5 voxel
        // size, each diorama occupying its own non-overlapping sub-region (see
        // BuildMuseumVolumeGrid) - visually equivalent, just one shared array instead of
        // two independent ones.
        public static SceneData BuildMuseumScene()
        {
            var materials = new List<MaterialSbtData>();
            var materialTexturePaths = new List<string>();
            int AddMaterial(MaterialSbtData m, string texturePath = null)
            {
                materials.Add(m);
                materialTexturePaths.Add(texturePath);
                return materials.Count - 1;
            }

            var vertices = new List<Vec3>();
            var normals = new List<Vec3>();
            var texCoords = new List<Vec2>();
            var indices = new List<Vec3i>();
            var triangleMaterialIds = new List<uint>();

            var spheres = new List<SphereData>();
            var sphereMaterialIds = new List<uint>();
            var boxes = new List<BoxData>();
            var boxMaterialIds = new List<uint>();
            var cylindersY = new List<CylinderYData>();
            var cylinderYMaterialIds = new List<uint>();
            var disks = new List<DiskData>();
            var diskMaterialIds = new List<uint>();
            var xyRects = new List<RectData>();
            var xyRectMaterialIds = new List<uint>();
            var xzRects = new List<RectData>();
            var xzRectMaterialIds = new List<uint>();
            var yzRects = new List<RectData>();
            var yzRectMaterialIds = new List<uint>();
            var lights = new List<PointLightGpu>();
            var meshRanges = new List<MeshRange>();

            void AddMeshAutoGround(string objFileName, uint materialId, float scale, Vec3 targetPos)
            {
                string objPath = Path.Combine(AppContext.BaseDirectory, "models", "meshes", objFileName);
                if (!File.Exists(objPath))
                    return;
                var mesh = OBJModel.Load(objPath);

                float minY = float.MaxValue;
                foreach (var v in mesh.Vertices)
                    minY = Math.Min(minY, v.y * scale);
                float groundOffset = targetPos.y - minY;

                int indexStart = indices.Count;
                int baseIndex = vertices.Count;
                foreach (var v in mesh.Vertices)
                    vertices.Add(new Vec3(v.x * scale + targetPos.x, (v.y * scale) + groundOffset, v.z * scale + targetPos.z));
                normals.AddRange(mesh.Normals);
                texCoords.AddRange(mesh.TexCoords);
                foreach (var tri in mesh.Indices)
                {
                    indices.Add(new Vec3i(tri.x + baseIndex, tri.y + baseIndex, tri.z + baseIndex));
                    triangleMaterialIds.Add(materialId);
                }
                meshRanges.Add(new MeshRange { IndexStart = indexStart, IndexCount = indices.Count - indexStart });
            }

            // Shared base materials.
            int mirror = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.98f, 0.98f, 0.98f), Reflectivity = 0.90f, MaterialKind = MaterialSbtData.Solid });
            int red = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.95f, 0.15f, 0.15f), Reflectivity = 0.08f, MaterialKind = MaterialSbtData.Solid });
            int green = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.15f, 0.95f, 0.20f), Reflectivity = 0.06f, MaterialKind = MaterialSbtData.Solid });
            int blue = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.15f, 0.25f, 0.95f), Reflectivity = 0.06f, MaterialKind = MaterialSbtData.Solid });
            int gold = AddMaterial(new MaterialSbtData { Albedo = new Vec3(1.00f, 0.85f, 0.57f), Reflectivity = 0.25f, MaterialKind = MaterialSbtData.Solid });
            int brass = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.78f, 0.60f, 0.20f), Reflectivity = 0.18f, MaterialKind = MaterialSbtData.Solid });
            int pedestal = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.85f, 0.85f, 0.85f), MaterialKind = MaterialSbtData.Solid });
            int glassClear = AddMaterial(new MaterialSbtData { Albedo = new Vec3(1f, 1f, 1f), Transparency = 1f, IndexOfRefraction = 1.5f, TransmissionColor = new Vec3(1f, 1f, 1f), MaterialKind = MaterialSbtData.Solid });
            int glassBlue = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.9f, 0.95f, 1.0f), Transparency = 1f, IndexOfRefraction = 1.52f, TransmissionColor = new Vec3(0.9f, 0.95f, 1.0f), MaterialKind = MaterialSbtData.Solid });

            // Global floor and distant backdrop - one large XZRect/XYRect standing in for
            // the reference's infinite Planes (see docs/SAMPLE13_PLAN.md's established
            // Plane -> finite-rect substitution), sized to cover the whole corridor
            // (anchors range from Z=-12 out to Z=-88).
            int floorMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.82f, 0.82f, 0.85f), MaterialKind = MaterialSbtData.Checker, CheckerColorB = new Vec3(0.12f, 0.12f, 0.12f), CheckerScale = 0.8f });
            xzRects.Add(new RectData { A0 = -16f, A1 = 16f, B0 = -100f, B1 = 16f, C = 0f });
            xzRectMaterialIds.Add((uint)floorMat);

            int backdropMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.02f, 0.02f, 0.03f), MaterialKind = MaterialSbtData.Solid });
            xyRects.Add(new RectData { A0 = -20f, A1 = 20f, B0 = 0f, B1 = 20f, C = -99f });
            xyRectMaterialIds.Add((uint)backdropMat);

            // Cornell rooms (3 spacing anchors, no corridor walls between them).
            var cornellAnchorA = new Vec3(-9.0f, 0.0f, -12.0f);
            var cornellAnchorB = new Vec3(9.0f, 0.0f, -28.0f);
            var cornellAnchorC = new Vec3(-9.0f, 0.0f, -48.0f);

            var meshGalleryAnchor = new Vec3(9.0f, 0.0f, -40.0f);
            var pedestalQuadAnchor = new Vec3(-8.6f, 0.0f, -30.0f);

            var volumeAnchorA = new Vec3(-9.0f, 0.0f, -72.0f);
            var volumeAnchorB = new Vec3(9.0f, 0.0f, -88.0f);

            void AddCornellRoom(Vec3 anchor, float width, float height, Vec3 leftColor, Vec3 rightColor, Vec3 whiteColor, float lightPower)
            {
                float xL = anchor.x - width;
                float xR = anchor.x;
                float yB = anchor.y;
                float yT = anchor.y + height;
                float zB = anchor.z - width;
                float zF = anchor.z;

                int white = AddMaterial(new MaterialSbtData { Albedo = whiteColor, MaterialKind = MaterialSbtData.Solid });
                int leftWall = AddMaterial(new MaterialSbtData { Albedo = leftColor, MaterialKind = MaterialSbtData.Solid });
                int rightWall = AddMaterial(new MaterialSbtData { Albedo = rightColor, MaterialKind = MaterialSbtData.Solid });
                int emissive = AddMaterial(new MaterialSbtData { Emission = new Vec3(4f, 4f, 4f), MaterialKind = MaterialSbtData.Solid });

                yzRects.Add(new RectData { A0 = yB, A1 = yT, B0 = zB, B1 = zF, C = xL }); yzRectMaterialIds.Add((uint)leftWall);
                yzRects.Add(new RectData { A0 = yB, A1 = yT, B0 = zB, B1 = zF, C = xR }); yzRectMaterialIds.Add((uint)rightWall);
                xzRects.Add(new RectData { A0 = xL, A1 = xR, B0 = zB, B1 = zF, C = yB }); xzRectMaterialIds.Add((uint)white);
                xzRects.Add(new RectData { A0 = xL, A1 = xR, B0 = zB, B1 = zF, C = yT }); xzRectMaterialIds.Add((uint)white);
                xyRects.Add(new RectData { A0 = xL, A1 = xR, B0 = yB, B1 = yT, C = zB }); xyRectMaterialIds.Add((uint)white);

                float lx0 = xL + (0.20f * width);
                float lx1 = xR - (0.20f * width);
                float lz0 = zB + (0.35f * width);
                float lz1 = zB + (0.55f * width);
                float ly = yT - 0.01f;
                xzRects.Add(new RectData { A0 = lx0, A1 = lx1, B0 = lz0, B1 = lz1, C = ly }); xzRectMaterialIds.Add((uint)emissive);
                lights.Add(new PointLightGpu { Position = new Vec3((lx0 + lx1) * 0.5f, yT - 0.2f, (lz0 + lz1) * 0.5f), Color = new Vec3(1f, 0.98f, 0.95f), Intensity = lightPower });

                float cx = (xL + xR) * 0.5f;
                float cz = (zB + zF) * 0.5f;

                int ped = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.88f, 0.88f, 0.88f), MaterialKind = MaterialSbtData.Solid });
                int objA = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.90f, 0.20f, 0.20f), Reflectivity = 0.08f, MaterialKind = MaterialSbtData.Solid });
                int objB = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.20f, 0.80f, 0.95f), Reflectivity = 0.10f, MaterialKind = MaterialSbtData.Solid });
                int mirrorish = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.98f, 0.98f, 0.98f), Reflectivity = 0.85f, MaterialKind = MaterialSbtData.Solid });
                int glassish = AddMaterial(new MaterialSbtData { Albedo = new Vec3(1f, 1f, 1f), Transparency = 1f, IndexOfRefraction = 1.5f, TransmissionColor = new Vec3(1f, 1f, 1f), MaterialKind = MaterialSbtData.Solid });

                var stand0 = new Vec3(cx - 0.8f, yB, cz + 0.2f);
                var stand1 = new Vec3(cx + 0.8f, yB, cz - 0.2f);

                disks.Add(new DiskData { Center = new Vec3(cx, yB + 0.01f, cz), Normal = new Vec3(0f, 1f, 0f), Radius = width * 0.32f }); diskMaterialIds.Add((uint)AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.90f, 0.90f, 0.92f), MaterialKind = MaterialSbtData.Solid }));

                cylindersY.Add(new CylinderYData { Center = stand0, Radius = 0.28f, YMin = 0f, YMax = 0.9f, Capped = 1 }); cylinderYMaterialIds.Add((uint)ped);
                spheres.Add(new SphereData { Center = stand0 + new Vec3(0f, 0.9f + 0.35f, 0f), Radius = 0.35f }); sphereMaterialIds.Add((uint)mirrorish);

                cylindersY.Add(new CylinderYData { Center = stand1, Radius = 0.28f, YMin = 0f, YMax = 0.9f, Capped = 1 }); cylinderYMaterialIds.Add((uint)ped);
                spheres.Add(new SphereData { Center = stand1 + new Vec3(0f, 0.9f + 0.32f, 0f), Radius = 0.32f }); sphereMaterialIds.Add((uint)glassish);

                spheres.Add(new SphereData { Center = new Vec3(cx, yB + 0.35f, cz - 0.9f), Radius = 0.35f }); sphereMaterialIds.Add((uint)objA);
                spheres.Add(new SphereData { Center = new Vec3(cx, yB + 0.22f, cz + 0.9f), Radius = 0.22f }); sphereMaterialIds.Add((uint)objB);
            }

            AddCornellRoom(cornellAnchorA, 6f, 5f, new Vec3(0.80f, 0.10f, 0.10f), new Vec3(0.10f, 0.80f, 0.10f), new Vec3(0.82f, 0.82f, 0.82f), 65f);
            AddCornellRoom(cornellAnchorB, 6f, 5f, new Vec3(0.70f, 0.10f, 0.70f), new Vec3(0.10f, 0.70f, 0.70f), new Vec3(0.82f, 0.82f, 0.82f), 75f);
            AddCornellRoom(cornellAnchorC, 6f, 5f, new Vec3(0.80f, 0.20f, 0.10f), new Vec3(0.20f, 0.80f, 0.10f), new Vec3(0.90f, 0.90f, 0.90f), 90f);

            // Mesh gallery (uniform scale=1 per-mesh literal multiplier, matching the
            // reference's own scale values for these exact OBJ assets).
            AddMeshAutoGround("cow.obj", (uint)gold, 3.0f, meshGalleryAnchor + new Vec3(-2.6f, 1.0f, -0.4f));
            AddMeshAutoGround("stanford-bunny.obj", (uint)green, 3.0f, meshGalleryAnchor + new Vec3(0.0f, 1.0f, 0.0f));
            AddMeshAutoGround("teapot.obj", (uint)red, 2.0f, meshGalleryAnchor + new Vec3(2.6f, 1.0f, 0.4f));
            AddMeshAutoGround("xyzrgb_dragon.obj", (uint)mirror, 8.0f, meshGalleryAnchor + new Vec3(5.2f, 2.0f, -0.8f));
            int yellowDisk = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.85f, 0.85f, 0.1f), MaterialKind = MaterialSbtData.Solid });
            disks.Add(new DiskData { Center = meshGalleryAnchor + new Vec3(2.6f, 0.01f, 0.4f), Normal = new Vec3(0f, 1f, 0f), Radius = 0.9f }); diskMaterialIds.Add((uint)yellowDisk);

            // Pedestal quartet with spheres.
            {
                const float pedH = 1.2f, sphR = 0.35f;
                Vec3 baseC = pedestalQuadAnchor;
                Vec3 dx = new Vec3(1.8f, 0.0f, 0.0f);
                Vec3 dz = new Vec3(0.0f, 0.0f, -1.8f);
                Vec3[] pts = { baseC, baseC + dx, baseC + dz, baseC + dx + dz };
                int[] topMats = { mirror, glassBlue, red, blue };
                for (var i = 0; i < 4; i++)
                {
                    cylindersY.Add(new CylinderYData { Center = pts[i], Radius = 0.32f, YMin = 0f, YMax = pedH, Capped = 1 });
                    cylinderYMaterialIds.Add((uint)pedestal);
                    spheres.Add(new SphereData { Center = pts[i] + new Vec3(0f, pedH + sphR, 0f), Radius = sphR });
                    sphereMaterialIds.Add((uint)topMats[i]);
                }
            }

            // Triangle showcase.
            {
                Vec3 a = new Vec3(2.2f, 0.0f, -6.0f);
                Vec3 b = new Vec3(2.8f, 1.2f, -6.4f);
                Vec3 c = new Vec3(1.6f, 0.7f, -6.8f);
                Vec3 n = Vec3.unitVector(Vec3.cross(b - a, c - a));
                int baseIndex = vertices.Count;
                vertices.Add(a); vertices.Add(b); vertices.Add(c);
                normals.Add(n); normals.Add(n); normals.Add(n);
                texCoords.Add(new Vec2(0f, 0f)); texCoords.Add(new Vec2(1f, 0f)); texCoords.Add(new Vec2(0f, 1f));
                indices.Add(new Vec3i(baseIndex, baseIndex + 1, baseIndex + 2));
                triangleMaterialIds.Add((uint)brass);
            }

            // Textured demo sphere + textured endcap wall, using a bundled copy of the
            // reference's own RGBW-8k.png (only ~105KB despite the name, so copying it
            // in directly - see Sample13.csproj - was simpler than substituting a
            // different already-bundled texture).
            {
                int textured = AddMaterial(new MaterialSbtData { Albedo = new Vec3(1f, 1f, 1f), TextureWeight = 1f, UVScale = 1.5f, MaterialKind = MaterialSbtData.Solid }, "models/museum/RGBW-8k.png");
                spheres.Add(new SphereData { Center = new Vec3(-1.6f, 0.6f, -10.0f), Radius = 0.6f });
                sphereMaterialIds.Add((uint)textured);

                int texturedWall = AddMaterial(new MaterialSbtData { Albedo = new Vec3(1f, 1f, 1f), TextureWeight = 1f, UVScale = 0.35f, MaterialKind = MaterialSbtData.Solid }, "models/museum/RGBW-8k.png");
                xyRects.Add(new RectData { A0 = -10f, A1 = 10f, B0 = 0f, B1 = 8f, C = -98f });
                xyRectMaterialIds.Add((uint)texturedWall);

                // Video-textured cube demo.
                int videoBox = AddMaterial(new MaterialSbtData { Albedo = new Vec3(1f, 1f, 1f), TextureWeight = 1f, UVScale = 1f, MaterialKind = MaterialSbtData.Solid }, "models/museum/RTConsole.mp4");
                Vec3 bMin = new Vec3(-2.2f, 0.4f, -7.8f);
                boxes.Add(new BoxData { Min = bMin, Max = bMin + new Vec3(1.2f, 1.2f, 1.2f) });
                boxMaterialIds.Add((uint)videoBox);
            }

            BuildMuseumVolumeGrids(volumeAnchorA, volumeAnchorB, red, green, blue, mirror, glassClear, pedestal,
                AddMaterial, out var voxelIds, out var voxelMin, out var voxelSize, out var voxelDims,
                cylindersY, cylinderYMaterialIds, spheres, sphereMaterialIds, lights, AddMeshAutoGround, gold);

            // Video-cube diorama (reflection/refraction tests).
            {
                Vec3 basePos = new Vec3(9.0f, 0.0f, -60.0f);
                int platform = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.90f, 0.90f, 0.92f), MaterialKind = MaterialSbtData.Solid });
                disks.Add(new DiskData { Center = basePos + new Vec3(0f, 0.01f, 0f), Normal = new Vec3(0f, 1f, 0f), Radius = 1.7f });
                diskMaterialIds.Add((uint)platform);

                int videoMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(1f, 1f, 1f), TextureWeight = 1f, UVScale = 1f, MaterialKind = MaterialSbtData.Solid }, "models/museum/RTConsole.mp4");
                Vec3 cubeC = basePos + new Vec3(0f, 0.65f, 0f);
                Vec3 half = new Vec3(0.40f, 0.40f, 0.40f);
                boxes.Add(new BoxData { Min = cubeC - half, Max = cubeC + half });
                boxMaterialIds.Add((uint)videoMat);

                spheres.Add(new SphereData { Center = basePos + new Vec3(-0.95f, 0.65f + 0.35f, 0.65f), Radius = 0.35f });
                sphereMaterialIds.Add((uint)mirror);
                spheres.Add(new SphereData { Center = basePos + new Vec3(0.95f, 0.65f + 0.32f, -0.65f), Radius = 0.32f });
                sphereMaterialIds.Add((uint)glassClear);

                Vec3 standA = basePos + new Vec3(-0.9f, 0.0f, -0.7f);
                Vec3 standB = basePos + new Vec3(0.9f, 0.0f, 0.7f);
                cylindersY.Add(new CylinderYData { Center = standA, Radius = 0.22f, YMin = 0f, YMax = 0.9f, Capped = 1 }); cylinderYMaterialIds.Add((uint)pedestal);
                cylindersY.Add(new CylinderYData { Center = standB, Radius = 0.22f, YMin = 0f, YMax = 0.9f, Capped = 1 }); cylinderYMaterialIds.Add((uint)pedestal);

                int meshMirror = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.98f, 0.98f, 0.98f), Reflectivity = 0.85f, MaterialKind = MaterialSbtData.Solid });
                AddMeshAutoGround("teapot.obj", (uint)meshMirror, 0.9f, standA + new Vec3(0f, 1.0f, 0f));

                int meshGlass = AddMaterial(new MaterialSbtData { Albedo = new Vec3(1f, 1f, 1f), Transparency = 1f, IndexOfRefraction = 1.5f, TransmissionColor = new Vec3(1f, 1f, 1f), MaterialKind = MaterialSbtData.Solid });
                AddMeshAutoGround("stanford-bunny.obj", (uint)meshGlass, 0.9f, standB + new Vec3(0f, 1.0f, 0f));

                lights.Add(new PointLightGpu { Position = basePos + new Vec3(0f, 2.0f, 0f), Color = new Vec3(1f, 0.98f, 0.95f), Intensity = 140f });
            }

            lights.Add(new PointLightGpu { Position = new Vec3(0.0f, 12.0f, -50.0f), Color = new Vec3(1f, 1f, 1f), Intensity = 900f });

            return new SceneData
            {
                Name = "Museum",
                Vertices = vertices.ToArray(),
                Normals = normals.ToArray(),
                TexCoords = texCoords.ToArray(),
                Indices = indices.ToArray(),
                TriangleMaterialIds = triangleMaterialIds.ToArray(),
                MeshRanges = FillMeshRangeGaps(meshRanges, indices.Count),
                Materials = materials.ToArray(),
                MaterialTexturePaths = materialTexturePaths.ToArray(),
                Spheres = spheres.ToArray(),
                SphereMaterialIds = sphereMaterialIds.ToArray(),
                Boxes = boxes.ToArray(),
                BoxMaterialIds = boxMaterialIds.ToArray(),
                CylindersY = cylindersY.ToArray(),
                CylinderYMaterialIds = cylinderYMaterialIds.ToArray(),
                Disks = disks.ToArray(),
                DiskMaterialIds = diskMaterialIds.ToArray(),
                XYRects = xyRects.ToArray(),
                XYRectMaterialIds = xyRectMaterialIds.ToArray(),
                XZRects = xzRects.ToArray(),
                XZRectMaterialIds = xzRectMaterialIds.ToArray(),
                YZRects = yzRects.ToArray(),
                YZRectMaterialIds = yzRectMaterialIds.ToArray(),
                VoxelMaterialIds = voxelIds,
                VolumeGridMin = voxelMin,
                VolumeVoxelSize = voxelSize,
                VolumeDims = voxelDims,
                Lights = lights.ToArray(),
                AmbientColor = new Vec3(1f, 1f, 1f),
                AmbientIntensity = 0.06f,
                BackgroundTop = new Vec3(0.06f, 0.08f, 0.10f),
                BackgroundBottom = new Vec3(0.01f, 0.01f, 0.02f),
                CameraOrigin = new Vec3(0.0f, 1.7f, 2.5f),
                CameraLookAt = new Vec3(0.0f, 1.5f, -8f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 50f,
                CameraWorldScale = 15f,
            };
        }

        // Builds the museum's two volume-grid dioramas as one combined grid (see
        // BuildMuseumScene's class-level comment on why - SceneData only supports one
        // volume-grid region per scene). Diorama A (16x8x16 @ 0.5) and diorama B
        // (rescaled to 13x7x13 @ 0.5, from the reference's 14x7x14 @ 0.45 - a
        // proportionally-equivalent footprint at the shared voxel size) are placed in
        // non-overlapping sub-regions of one combined array spanning both anchors.
        private static void BuildMuseumVolumeGrids(
            Vec3 anchorA, Vec3 anchorB,
            int red, int green, int blue, int mirror, int glassClear, int pedestal,
            Func<MaterialSbtData, string, int> addMaterial,
            out uint[] voxelIds, out Vec3 voxelMin, out Vec3 voxelSize, out Vec3i voxelDims,
            List<CylinderYData> cylindersY, List<uint> cylinderYMaterialIds,
            List<SphereData> spheres, List<uint> sphereMaterialIds,
            List<PointLightGpu> lights, Action<string, uint, float, Vec3> addMeshAutoGround, int gold)
        {
            voxelSize = new Vec3(0.5f, 0.5f, 0.5f);
            voxelMin = new Vec3(Math.Min(anchorA.x, anchorB.x), 0f, Math.Min(anchorA.z, anchorB.z));

            const int nxA = 16, nyA = 8, nzA = 16;
            const int nxB = 13, nyB = 7, nzB = 13;

            int combinedNx = (int)Math.Round((Math.Max(anchorA.x + (nxA * 0.5f), anchorB.x + (nxB * 0.5f)) - voxelMin.x) / voxelSize.x) + 1;
            int combinedNy = Math.Max(nyA, nyB);
            int combinedNz = (int)Math.Round((Math.Max(anchorA.z + (nzA * 0.5f), anchorB.z + (nzB * 0.5f)) - voxelMin.z) / voxelSize.z) + 1;
            voxelDims = new Vec3i(combinedNx, combinedNy, combinedNz);

            var voxels = new uint[combinedNx * combinedNy * combinedNz];
            uint Index(int x, int y, int z) => (uint)((x * combinedNy * combinedNz) + (y * combinedNz) + z);

            int offAX = (int)Math.Round((anchorA.x - voxelMin.x) / voxelSize.x);
            int offAZ = (int)Math.Round((anchorA.z - voxelMin.z) / voxelSize.z);
            int offBX = (int)Math.Round((anchorB.x - voxelMin.x) / voxelSize.x);
            int offBZ = (int)Math.Round((anchorB.z - voxelMin.z) / voxelSize.z);

            // Diorama A: floor + perimeter walls, 4 colored pillars, a checker dais.
            // Voxel material ids are 1-based indices into the SAME shared
            // SceneData.Materials array every other primitive kind uses (voxel value N
            // -> Materials[N-1], see BuildVolumeGridTestScene) - so the red/green/blue/
            // mirror materials already added for the rest of the museum are reused
            // directly here (matching the reference's own materialLookup closures,
            // which likewise reuse the same Material instances for voxels and solids).
            // Only a stone/white and a dark checker tone are genuinely new.
            uint vStone = (uint)(addMaterial(new MaterialSbtData { Albedo = new Vec3(0.82f, 0.82f, 0.85f), MaterialKind = MaterialSbtData.Solid }, null) + 1);
            uint vRed = (uint)(red + 1);
            uint vGreen = (uint)(green + 1);
            uint vBlue = (uint)(blue + 1);
            uint vMirror = (uint)(mirror + 1);
            uint vDark = (uint)(addMaterial(new MaterialSbtData { Albedo = new Vec3(0.15f, 0.15f, 0.18f), MaterialKind = MaterialSbtData.Solid }, null) + 1);

            for (var x = 0; x < nxA; x++)
                for (var z = 0; z < nzA; z++)
                    voxels[Index(offAX + x, 0, offAZ + z)] = vStone;
            for (var y = 1; y <= 3; y++)
            {
                for (var x = 0; x < nxA; x++)
                {
                    voxels[Index(offAX + x, y, offAZ)] = vStone;
                    voxels[Index(offAX + x, y, offAZ + nzA - 1)] = vStone;
                }
                for (var z = 0; z < nzA; z++)
                {
                    voxels[Index(offAX, y, offAZ + z)] = vStone;
                    voxels[Index(offAX + nxA - 1, y, offAZ + z)] = vStone;
                }
            }
            void PillarA(int cx, int cz, int height, uint mat)
            {
                for (var y = 1; y <= height && y < nyA; y++)
                    voxels[Index(offAX + cx, y, offAZ + cz)] = mat;
            }
            PillarA(4, 4, 4, vRed);
            PillarA(11, 4, 3, vGreen);
            PillarA(4, 11, 5, vBlue);
            PillarA(11, 11, 4, vMirror);
            for (var x = 6; x <= 9; x++)
                for (var z = 6; z <= 9; z++)
                    voxels[Index(offAX + x, 1, offAZ + z)] = (((x + z) & 1) == 0) ? vStone : vBlue;

            Vec3 pedC = anchorA + new Vec3(4.0f, 0.0f, 2.0f);
            cylindersY.Add(new CylinderYData { Center = pedC, Radius = 0.35f, YMin = 0f, YMax = 1.4f, Capped = 1 });
            cylinderYMaterialIds.Add((uint)pedestal);
            spheres.Add(new SphereData { Center = pedC + new Vec3(0f, 1.4f + 0.45f, 0f), Radius = 0.45f });
            sphereMaterialIds.Add((uint)glassClear);
            lights.Add(new PointLightGpu { Position = anchorA + new Vec3(4.0f, 3.0f, 1.0f), Color = new Vec3(0.9f, 0.95f, 1.0f), Intensity = 110f });

            // Diorama B: checker floor + 4 short pillar-pairs (rescaled from the
            // reference's 14x14 @ 0.45 footprint to 13x13 @ 0.5, same proportions).
            for (var x = 0; x < nxB; x++)
                for (var z = 0; z < nzB; z++)
                    voxels[Index(offBX + x, 0, offBZ + z)] = (((x + z) & 1) == 0) ? vStone : vDark;
            for (var i = 2; i < nxB - 2; i += 3)
            {
                for (var y = 1; y <= 3 && y < nyB; y++)
                {
                    voxels[Index(offBX + i, y, offBZ + 2)] = vRed;
                    voxels[Index(offBX + i, y, offBZ + nzB - 3)] = vGreen;
                }
            }

            Vec3 stand = anchorB + new Vec3(3.0f, 0.0f, 6.0f);
            cylindersY.Add(new CylinderYData { Center = stand, Radius = 0.30f, YMin = 0f, YMax = 1.1f, Capped = 1 });
            cylinderYMaterialIds.Add((uint)pedestal);
            addMeshAutoGround("teapot.obj", (uint)gold, 1.0f, stand + new Vec3(0f, 1.12f, 0f));
            lights.Add(new PointLightGpu { Position = anchorB + new Vec3(2.5f, 2.8f, 7.0f), Color = new Vec3(1f, 0.95f, 0.9f), Intensity = 85f });

            voxelIds = voxels;
        }

        // Direct port of the reference's TestScenesRandom.Build() ("Radial Museum") - a
        // giant refractive dragon centerpiece surrounded by a ring of 12 independently
        // themed vignettes (11 real + 1 intentionally empty slot, matching the
        // reference's own commented-out AddTexturedPanel call), plus animated
        // orbiting/pulsing point lights and bobbing spheres throughout. The reference's
        // OrbitingLightEntity/PulsingLightEntity/BobbingSphereEntity update objects are
        // expressed as SceneData.OrbitingLights/PulsingLights/BobbingSpheres
        // descriptors instead (see SceneData.cs's class comment) - applied every frame
        // by SampleRenderer.ApplyAnimation, not stored as live objects here. Uses a
        // fixed seed (matching the reference's own default Build(int seed = 1337))
        // rather than the reference's seed==-1 option, for a reproducible layout.
        public static SceneData BuildRadialMuseumScene()
        {
            var rng = new Random(1337);
            float RngRange(float a, float b) => a + ((float)rng.NextDouble() * (b - a));
            int RngInt(int a, int bInclusive) => a + rng.Next(bInclusive - a + 1);
            bool RngBool(float pTrue = 0.5f) => rng.NextDouble() < pTrue;

            var materials = new List<MaterialSbtData>();
            var materialTexturePaths = new List<string>();
            int AddMaterial(MaterialSbtData m)
            {
                materials.Add(m);
                materialTexturePaths.Add(null);
                return materials.Count - 1;
            }
            int MakeGlass()
            {
                float ior = RngRange(1.30f, 1.70f);
                var tint = new Vec3(RngRange(0.85f, 1.00f), RngRange(0.85f, 1.00f), RngRange(0.85f, 1.00f));
                return AddMaterial(new MaterialSbtData { Albedo = new Vec3(1f, 1f, 1f), Transparency = 1f, IndexOfRefraction = ior, TransmissionColor = tint, MaterialKind = MaterialSbtData.Solid });
            }
            Vec3 RandSaturated(float desat = 0f) => HsvToRgb((float)rng.NextDouble() * 360f, 0.75f - desat, 0.95f);

            var vertices = new List<Vec3>();
            var normals = new List<Vec3>();
            var texCoords = new List<Vec2>();
            var indices = new List<Vec3i>();
            var triangleMaterialIds = new List<uint>();

            var spheres = new List<SphereData>();
            var sphereMaterialIds = new List<uint>();
            var boxes = new List<BoxData>();
            var boxMaterialIds = new List<uint>();
            var cylindersY = new List<CylinderYData>();
            var cylinderYMaterialIds = new List<uint>();
            var disks = new List<DiskData>();
            var diskMaterialIds = new List<uint>();
            var xyRects = new List<RectData>();
            var xyRectMaterialIds = new List<uint>();
            var xzRects = new List<RectData>();
            var xzRectMaterialIds = new List<uint>();
            var yzRects = new List<RectData>();
            var yzRectMaterialIds = new List<uint>();
            var lights = new List<PointLightGpu>();
            var meshRanges = new List<MeshRange>();

            var orbitingLights = new List<OrbitingLightAnim>();
            var pulsingLights = new List<PulsingLightAnim>();
            var bobbingSpheres = new List<BobbingSphereAnim>();

            uint[] gridVoxelIds = Array.Empty<uint>();
            Vec3 gridVoxelMin = default;
            Vec3 gridVoxelSize = new Vec3(1f, 1f, 1f);
            Vec3i gridVoxelDims = default;

            int AddLight(Vec3 pos, Vec3 color, float intensity)
            {
                lights.Add(new PointLightGpu { Position = pos, Color = color, Intensity = intensity });
                return lights.Count - 1;
            }
            void AddOrbiting(int lightIndex, Vec3 pivot, float radius, float height, float speed, float phase) =>
                orbitingLights.Add(new OrbitingLightAnim { LightIndex = lightIndex, Pivot = pivot, Radius = radius, Height = height, AngularSpeed = speed, Phase = phase });
            void AddPulsing(int lightIndex, float ampFraction, float speed) =>
                pulsingLights.Add(new PulsingLightAnim { LightIndex = lightIndex, BaseIntensity = lights[lightIndex].Intensity, MinMult = Math.Max(0f, 1f - ampFraction), MaxMult = 1f + ampFraction, Speed = speed });

            int AddSphere(Vec3 spCenter, float radius, uint materialId)
            {
                spheres.Add(new SphereData { Center = spCenter, Radius = radius });
                sphereMaterialIds.Add(materialId);
                return spheres.Count - 1;
            }
            void AddBobbing(int sphereIndex, float amplitude, float speed, float phase) =>
                bobbingSpheres.Add(new BobbingSphereAnim { SphereIndex = sphereIndex, BaseY = spheres[sphereIndex].Center.y, Amplitude = amplitude, Speed = speed, Phase = phase });

            void AddTriangle(Vec3 ta, Vec3 tb, Vec3 tc, uint materialId)
            {
                Vec3 n = Vec3.unitVector(Vec3.cross(tb - ta, tc - ta));
                int baseIndex = vertices.Count;
                vertices.Add(ta); vertices.Add(tb); vertices.Add(tc);
                normals.Add(n); normals.Add(n); normals.Add(n);
                texCoords.Add(new Vec2(0f, 0f)); texCoords.Add(new Vec2(1f, 0f)); texCoords.Add(new Vec2(0f, 1f));
                indices.Add(new Vec3i(baseIndex, baseIndex + 1, baseIndex + 2));
                triangleMaterialIds.Add(materialId);
            }

            void AddMeshAutoGround(string objFileName, uint materialId, float scale, Vec3 targetPos)
            {
                string objPath = Path.Combine(AppContext.BaseDirectory, "models", "meshes", objFileName);
                if (!File.Exists(objPath))
                    return;
                var mesh = OBJModel.Load(objPath);

                float minY = float.MaxValue;
                foreach (var v in mesh.Vertices)
                    minY = Math.Min(minY, v.y * scale);
                float groundOffset = targetPos.y - minY;

                int indexStart = indices.Count;
                int baseIndex = vertices.Count;
                foreach (var v in mesh.Vertices)
                    vertices.Add(new Vec3((v.x * scale) + targetPos.x, (v.y * scale) + groundOffset, (v.z * scale) + targetPos.z));
                normals.AddRange(mesh.Normals);
                texCoords.AddRange(mesh.TexCoords);
                foreach (var tri in mesh.Indices)
                {
                    indices.Add(new Vec3i(tri.x + baseIndex, tri.y + baseIndex, tri.z + baseIndex));
                    triangleMaterialIds.Add(materialId);
                }
                meshRanges.Add(new MeshRange { IndexStart = indexStart, IndexCount = indices.Count - indexStart });
            }

            int matMirror = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.98f, 0.98f, 0.98f), Reflectivity = 0.90f, MaterialKind = MaterialSbtData.Solid });
            int emissiveWarm = AddMaterial(new MaterialSbtData { Emission = new Vec3(3.2f, 3.0f, 2.8f), MaterialKind = MaterialSbtData.Solid });
            int emissiveCool = AddMaterial(new MaterialSbtData { Emission = new Vec3(2.9f, 3.2f, 3.4f), MaterialKind = MaterialSbtData.Solid });

            var center = new Vec3(0.0f, 0.0f, -26.0f);
            const float ringRadius = 22.0f;
            const int ringCount = 12;

            // Global floor + distant backdrop (see Museum's identical Plane -> rect
            // substitution note in BuildMuseumScene).
            int floorMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.82f, 0.82f, 0.85f), MaterialKind = MaterialSbtData.Checker, CheckerColorB = new Vec3(0.12f, 0.12f, 0.12f), CheckerScale = 0.9f });
            xzRects.Add(new RectData { A0 = -34f, A1 = 34f, B0 = -60f, B1 = 12f, C = 0f });
            xzRectMaterialIds.Add((uint)floorMat);
            int backdropMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.02f, 0.02f, 0.03f), MaterialKind = MaterialSbtData.Solid });
            xyRects.Add(new RectData { A0 = -34f, A1 = 34f, B0 = 0f, B1 = 24f, C = -80f });
            xyRectMaterialIds.Add((uint)backdropMat);

            Vec3 AnchorOnRing(float angleDeg)
            {
                float t = angleDeg * (XMath.PI / 180.0f);
                return center + new Vec3(XMath.Cos(t) * ringRadius, 0.0f, XMath.Sin(t) * ringRadius);
            }

            // Center dragon.
            {
                int plinth = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.96f, 0.96f, 0.98f), Reflectivity = 0.88f, MaterialKind = MaterialSbtData.Solid });
                disks.Add(new DiskData { Center = center + new Vec3(0f, 0.015f, 0f), Normal = new Vec3(0f, 1f, 0f), Radius = 6.2f });
                diskMaterialIds.Add((uint)plinth);

                int giantGlass = MakeGlass();
                AddMeshAutoGround("xyzrgb_dragon.obj", (uint)giantGlass, 20.0f, center + new Vec3(0.0f, -5.5f, 0.0f));

                int a = AddLight(center + new Vec3(7.0f, 6.5f, 6.0f), new Vec3(0.98f, 0.96f, 1.06f), 260f);
                int b = AddLight(center + new Vec3(-8.0f, 6.0f, -6.5f), new Vec3(1.06f, 0.96f, 0.92f), 230f);
                int c = AddLight(center + new Vec3(0.0f, 3.2f, 0.0f), new Vec3(0.65f, 0.80f, 1.30f), 140f);
                int d = AddLight(center + new Vec3(0.0f, 3.2f, 2.2f), new Vec3(1.20f, 0.75f, 0.70f), 120f);
                AddOrbiting(a, center, 8.0f, 6.5f, 0.25f, 0.0f);
                AddOrbiting(b, center, 8.0f, 6.0f, -0.22f, 1.8f);
                AddPulsing(c, 0.25f, 0.9f);
                AddPulsing(d, 0.22f, 1.4f);
            }

            void AddMiniCornell(Vec3 a)
            {
                float w = 5.2f, h = 4.4f;
                float xL = a.x - (w * 0.5f), xR = a.x + (w * 0.5f);
                float yB = a.y, yT = a.y + h;
                float zB = a.z - (w * 0.5f), zF = a.z + (w * 0.5f);

                int left = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.95f, 0.20f, 0.20f), MaterialKind = MaterialSbtData.Solid });
                int right = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.20f, 0.28f, 0.95f), MaterialKind = MaterialSbtData.Solid });
                int white = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.82f, 0.82f, 0.82f), MaterialKind = MaterialSbtData.Solid });

                yzRects.Add(new RectData { A0 = yB, A1 = yT, B0 = zB, B1 = zF, C = xL }); yzRectMaterialIds.Add((uint)left);
                yzRects.Add(new RectData { A0 = yB, A1 = yT, B0 = zB, B1 = zF, C = xR }); yzRectMaterialIds.Add((uint)right);
                xzRects.Add(new RectData { A0 = xL, A1 = xR, B0 = zB, B1 = zF, C = yB }); xzRectMaterialIds.Add((uint)white);
                xzRects.Add(new RectData { A0 = xL, A1 = xR, B0 = zB, B1 = zF, C = yT }); xzRectMaterialIds.Add((uint)white);
                xyRects.Add(new RectData { A0 = xL, A1 = xR, B0 = yB, B1 = yT, C = zB }); xyRectMaterialIds.Add((uint)white);

                float lx0 = xL + (0.18f * w), lx1 = xR - (0.18f * w);
                float lz0 = zB + (0.38f * w), lz1 = zB + (0.58f * w);
                float ly = yT - 0.01f;
                int ceiling = RngBool(0.5f) ? emissiveWarm : emissiveCool;
                xzRects.Add(new RectData { A0 = lx0, A1 = lx1, B0 = lz0, B1 = lz1, C = ly }); xzRectMaterialIds.Add((uint)ceiling);

                Vec3 stand0 = new Vec3(a.x - 0.85f, yB, a.z + 0.25f);
                Vec3 stand1 = new Vec3(a.x + 0.85f, yB, a.z - 0.25f);
                int ped = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.90f, 0.90f, 0.92f), MaterialKind = MaterialSbtData.Solid });

                cylindersY.Add(new CylinderYData { Center = stand0, Radius = 0.26f, YMin = 0f, YMax = 0.9f, Capped = 1 }); cylinderYMaterialIds.Add((uint)ped);
                int sp0Mat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.98f, 0.98f, 0.98f), Reflectivity = 0.85f, MaterialKind = MaterialSbtData.Solid });
                int sp0 = AddSphere(stand0 + new Vec3(0f, 1.25f, 0f), 0.34f, (uint)sp0Mat);
                AddBobbing(sp0, 0.06f, 1.6f, 0.0f);

                cylindersY.Add(new CylinderYData { Center = stand1, Radius = 0.26f, YMin = 0f, YMax = 0.9f, Capped = 1 }); cylinderYMaterialIds.Add((uint)ped);
                int sp1Mat = MakeGlass();
                int sp1 = AddSphere(stand1 + new Vec3(0f, 1.18f, 0f), 0.30f, (uint)sp1Mat);
                AddBobbing(sp1, 0.05f, 1.2f, 1.1f);

                xyRects.Add(new RectData { A0 = a.x - 0.55f, A1 = a.x + 0.55f, B0 = a.y + 1.9f, B1 = a.y + 2.0f, C = a.z + 0.2f }); xyRectMaterialIds.Add((uint)emissiveWarm);
                xyRects.Add(new RectData { A0 = a.x - 0.55f, A1 = a.x + 0.55f, B0 = a.y + 1.9f, B1 = a.y + 2.0f, C = a.z - 0.2f }); xyRectMaterialIds.Add((uint)emissiveCool);

                int l0 = AddLight(new Vec3(a.x - 0.8f, yT - 0.35f, a.z + 0.6f), new Vec3(1.2f, 0.7f, 0.6f), 55f);
                int l1 = AddLight(new Vec3(a.x + 0.8f, yT - 0.35f, a.z - 0.6f), new Vec3(0.7f, 0.9f, 1.2f), 55f);
                AddPulsing(l0, 0.30f, 0.0f);
                AddPulsing(l1, 0.30f, XMath.PI);
            }

            void AddGlassGarden(Vec3 a)
            {
                const int nx = 4, nz = 3;
                const float spacing = 0.95f;
                const float rmin = 0.20f, rmax = 0.42f;
                for (var iz = 0; iz < nz; iz++)
                {
                    for (var ix = 0; ix < nx; ix++)
                    {
                        float rx = ((ix - ((nx - 1) * 0.5f)) * 1.3f) + RngRange(-0.15f, 0.15f);
                        float rz = ((iz - ((nz - 1) * 0.5f)) * 1.3f) + RngRange(-0.15f, 0.15f);
                        float rr = RngRange(rmin, rmax);
                        if (RngBool(0.35f))
                        {
                            cylindersY.Add(new CylinderYData { Center = a + new Vec3(rx * spacing, 0.0f, rz * spacing), Radius = rr * 0.75f, YMin = 0f, YMax = rr * 2.2f, Capped = 1 });
                            cylinderYMaterialIds.Add((uint)MakeGlass());
                        }
                        else
                        {
                            Vec3 c = a + new Vec3(rx * spacing, rr, rz * spacing);
                            int gsMat = MakeGlass();
                            int gs = AddSphere(c, rr, (uint)gsMat);
                            AddBobbing(gs, 0.04f, 1.2f + RngRange(-0.2f, 0.2f), (rx * 1.1f) + (rz * 0.7f));
                        }
                    }
                }

                disks.Add(new DiskData { Center = a + new Vec3(0f, 0.01f, 0f), Normal = new Vec3(0f, 1f, 0f), Radius = 2.2f });
                diskMaterialIds.Add((uint)AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.96f, 0.96f, 0.98f), Reflectivity = 0.88f, MaterialKind = MaterialSbtData.Solid }));

                int l0 = AddLight(a + new Vec3(-1.2f, 2.4f, 1.0f), new Vec3(1.15f, 0.65f, 0.55f), 80f);
                int l1 = AddLight(a + new Vec3(1.2f, 2.6f, -1.0f), new Vec3(0.55f, 0.95f, 1.25f), 80f);
                int l2 = AddLight(a + new Vec3(0.0f, 3.0f, 0.0f), new Vec3(0.95f, 0.98f, 1.05f), 60f);
                AddOrbiting(l0, a, 2.0f, 2.4f, 0.6f, 0.0f);
                AddOrbiting(l1, a, 2.0f, 2.6f, -0.6f, 0.5f);
                AddPulsing(l2, 0.25f, 0.8f);
            }

            void AddMirrorCorridor(Vec3 a)
            {
                const int n = 6;
                const float step = 1.8f;
                int mirrorish = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.98f, 0.98f, 0.98f), Reflectivity = 0.92f, MaterialKind = MaterialSbtData.Solid });
                for (var i = -n; i <= n; i++)
                {
                    Vec3 pL = a + new Vec3(-1.6f, 0.0f, i * step);
                    Vec3 pR = a + new Vec3(1.6f, 0.0f, i * step);
                    cylindersY.Add(new CylinderYData { Center = pL, Radius = 0.22f, YMin = 0f, YMax = 2.1f, Capped = 1 }); cylinderYMaterialIds.Add((uint)mirrorish);
                    cylindersY.Add(new CylinderYData { Center = pR, Radius = 0.22f, YMin = 0f, YMax = 2.1f, Capped = 1 }); cylinderYMaterialIds.Add((uint)mirrorish);
                    Vec3 lp = new Vec3(a.x, a.y + 2.9f, a.z + (i * step));
                    int pl = ((i & 1) == 0)
                        ? AddLight(lp, new Vec3(1.15f, 0.75f, 0.65f), 38f)
                        : AddLight(lp, new Vec3(0.65f, 0.95f, 1.20f), 38f);
                    AddPulsing(pl, 0.35f, i * 0.7f);
                }
                int floor = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.96f, 0.96f, 0.98f), Reflectivity = 0.85f, MaterialKind = MaterialSbtData.Solid });
                xzRects.Add(new RectData { A0 = a.x - 2.2f, A1 = a.x + 2.2f, B0 = a.z - (n * step) - 0.4f, B1 = a.z + (n * step) + 0.4f, C = a.y + 0.01f });
                xzRectMaterialIds.Add((uint)floor);
            }

            void AddVoxelGrotto(Vec3 a)
            {
                const int nx = 12, ny = 6, nz = 12;
                var localVoxels = new uint[nx * ny * nz];
                uint LocalIndex(int x, int y, int z) => (uint)((x * ny * nz) + (y * nz) + z);

                int stoneMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.80f, 0.80f, 0.82f), MaterialKind = MaterialSbtData.Solid });
                int redMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.90f, 0.20f, 0.20f), Reflectivity = 0.10f, MaterialKind = MaterialSbtData.Solid });
                int greenMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.20f, 0.85f, 0.20f), Reflectivity = 0.08f, MaterialKind = MaterialSbtData.Solid });
                int blueMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.20f, 0.30f, 0.95f), Reflectivity = 0.08f, MaterialKind = MaterialSbtData.Solid });
                int mirrorMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.98f, 0.98f, 0.98f), Reflectivity = 0.88f, MaterialKind = MaterialSbtData.Solid });
                int checkerMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.85f, 0.85f, 0.88f), MaterialKind = MaterialSbtData.Solid });
                // voxelMats[i] -> voxel value i+1 (1=stone,2=red,3=green,4=blue,5=mirror,6=checker),
                // matching the reference's own materialLookup switch id space.
                uint[] voxelMats = { (uint)(stoneMat + 1), (uint)(redMat + 1), (uint)(greenMat + 1), (uint)(blueMat + 1), (uint)(mirrorMat + 1), (uint)(checkerMat + 1) };

                for (var x = 0; x < nx; x++)
                    for (var z = 0; z < nz; z++)
                        localVoxels[LocalIndex(x, 0, z)] = voxelMats[0];

                for (var i = 0; i < 20; i++)
                {
                    int cx = RngInt(1, nx - 2);
                    int cz = RngInt(1, nz - 2);
                    int h = RngInt(2, ny - 1);
                    int matSlot = RngInt(1, 4); // red/green/blue/mirror, matching reference's rng.Choice({2,3,4,5})
                    for (var y = 1; y <= h; y++)
                        localVoxels[LocalIndex(cx, y, cz)] = voxelMats[matSlot];
                }

                for (var x = 3; x < nx - 3; x++)
                    for (var z = 3; z < nz - 3; z++)
                        if (((x + z) & 1) == 0)
                            localVoxels[LocalIndex(x, 1, z)] = voxelMats[5];

                // The only volume grid in this scene - unlike the Museum scene's two
                // dioramas, no merging is needed; these map straight onto SceneData's
                // single VoxelMaterialIds/VolumeGridMin/VolumeVoxelSize/VolumeDims.
                gridVoxelIds = localVoxels;
                gridVoxelMin = a + new Vec3(-2.6f, 0.0f, -2.6f);
                gridVoxelSize = new Vec3(0.45f, 0.45f, 0.45f);
                gridVoxelDims = new Vec3i(nx, ny, nz);

                for (var i = 0; i < 10; i++)
                {
                    Vec3 p = a + new Vec3(RngRange(-2.2f, 2.2f), RngRange(0.6f, 2.2f), RngRange(-2.2f, 2.2f));
                    Vec3 e = RandSaturated(0.25f) * 2.0f;
                    int glowMat = AddMaterial(new MaterialSbtData { Emission = e, MaterialKind = MaterialSbtData.Solid });
                    AddSphere(p, 0.08f, (uint)glowMat);
                }

                int pl = AddLight(a + new Vec3(0.0f, 2.4f, 0.0f), new Vec3(0.85f, 1.05f, 1.10f), 85f);
                AddOrbiting(pl, a, 2.2f, 2.4f, 0.7f, 0.0f);
            }

            void AddPedestalPlaza(Vec3 a)
            {
                const int count = 5;
                const float pedH = 1.1f;
                int pedMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.88f, 0.88f, 0.88f), MaterialKind = MaterialSbtData.Solid });
                for (var i = 0; i < count; i++)
                {
                    float ang = i * (2f * XMath.PI) / count;
                    Vec3 p = a + new Vec3(XMath.Cos(ang) * 1.6f, 0.0f, XMath.Sin(ang) * 1.6f);
                    cylindersY.Add(new CylinderYData { Center = p, Radius = 0.30f, YMin = 0f, YMax = pedH, Capped = 1 }); cylinderYMaterialIds.Add((uint)pedMat);

                    int topMat = RngInt(0, 4) switch
                    {
                        0 => AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.98f, 0.98f, 0.98f), Reflectivity = 0.90f, MaterialKind = MaterialSbtData.Solid }),
                        1 => MakeGlass(),
                        2 => AddMaterial(new MaterialSbtData { Albedo = new Vec3(1.00f, 0.85f, 0.57f), Reflectivity = 0.10f, MaterialKind = MaterialSbtData.Solid }),
                        3 => AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.78f, 0.60f, 0.20f), Reflectivity = 0.08f, MaterialKind = MaterialSbtData.Solid }),
                        _ => AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.20f, 0.85f, 0.20f), Reflectivity = 0.08f, MaterialKind = MaterialSbtData.Solid }),
                    };

                    int orb = AddSphere(p + new Vec3(0.0f, pedH + 0.38f, 0.0f), 0.36f, (uint)topMat);
                    AddBobbing(orb, 0.07f, 1.3f, ang);
                }

                disks.Add(new DiskData { Center = a + new Vec3(0.0f, 0.015f, 0.0f), Normal = new Vec3(0f, 1f, 0f), Radius = 2.0f });
                diskMaterialIds.Add((uint)AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.96f, 0.96f, 0.98f), Reflectivity = 0.88f, MaterialKind = MaterialSbtData.Solid }));

                int l0 = AddLight(a + new Vec3(0.0f, 3.0f, 0.0f), new Vec3(1.00f, 0.92f, 0.80f), 55f);
                int l1 = AddLight(a + new Vec3(1.6f, 2.4f, 1.2f), new Vec3(0.65f, 0.95f, 1.25f), 50f);
                int l2 = AddLight(a + new Vec3(-1.6f, 2.4f, -1.2f), new Vec3(1.20f, 0.70f, 0.70f), 50f);
                AddPulsing(l0, 0.25f, 0.0f);
                AddPulsing(l1, 0.20f, 0.7f);
                AddPulsing(l2, 0.20f, 1.9f);
            }

            void AddTriangleField(Vec3 a)
            {
                const int n = 14;
                for (var i = 0; i < n; i++)
                {
                    Vec3 p0 = a + new Vec3(RngRange(-2.0f, 2.0f), RngRange(0.0f, 0.8f), RngRange(-2.0f, 2.0f));
                    Vec3 p1 = a + new Vec3(RngRange(-2.0f, 2.0f), RngRange(0.0f, 1.0f), RngRange(-2.0f, 2.0f));
                    Vec3 p2 = a + new Vec3(RngRange(-2.0f, 2.0f), RngRange(0.0f, 1.2f), RngRange(-2.0f, 2.0f));
                    int mat = AddMaterial(new MaterialSbtData { Albedo = RandSaturated(0.15f), Reflectivity = RngRange(0.02f, 0.08f), MaterialKind = MaterialSbtData.Solid });
                    AddTriangle(p0, p1, p2, (uint)mat);
                }

                disks.Add(new DiskData { Center = a + new Vec3(0.0f, 0.012f, 0.0f), Normal = new Vec3(0f, 1f, 0f), Radius = 2.2f });
                diskMaterialIds.Add((uint)AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.90f, 0.90f, 0.94f), Reflectivity = 0.82f, MaterialKind = MaterialSbtData.Solid }));

                int l0 = AddLight(a + new Vec3(0.0f, 2.6f, -1.2f), new Vec3(1.15f, 0.80f, 0.75f), 60f);
                int l1 = AddLight(a + new Vec3(0.0f, 2.6f, 1.2f), new Vec3(0.70f, 0.95f, 1.25f), 60f);
                AddOrbiting(l0, a, 2.4f, 2.6f, 0.5f, 0.0f);
                AddOrbiting(l1, a, 2.4f, 2.6f, -0.5f, 0.9f);
            }

            void AddMeshShowcase(Vec3 a)
            {
                int cowMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(1.00f, 0.85f, 0.57f), Reflectivity = 0.10f, MaterialKind = MaterialSbtData.Solid });
                int bunnyMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.20f, 0.85f, 0.20f), Reflectivity = 0.08f, MaterialKind = MaterialSbtData.Solid });
                int teapotMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.95f, 0.15f, 0.15f), Reflectivity = 0.10f, MaterialKind = MaterialSbtData.Solid });
                int dragonMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.98f, 0.98f, 0.98f), Reflectivity = 0.90f, MaterialKind = MaterialSbtData.Solid });

                AddMeshAutoGround("cow.obj", (uint)cowMat, 1.0f, a + new Vec3(-2.4f, 0.0f, 0.2f));
                AddMeshAutoGround("stanford-bunny.obj", (uint)bunnyMat, 1.0f, a + new Vec3(0.0f, 0.0f, 0.0f));
                AddMeshAutoGround("teapot.obj", (uint)teapotMat, 1.0f, a + new Vec3(2.4f, 0.0f, -0.2f));
                AddMeshAutoGround("xyzrgb_dragon.obj", (uint)dragonMat, 1.0f, a + new Vec3(4.8f, 0.0f, -0.6f));

                disks.Add(new DiskData { Center = a + new Vec3(0.0f, 0.012f, 0.0f), Normal = new Vec3(0f, 1f, 0f), Radius = 3.0f });
                diskMaterialIds.Add((uint)AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.90f, 0.90f, 0.94f), Reflectivity = 0.82f, MaterialKind = MaterialSbtData.Solid }));

                int l0 = AddLight(a + new Vec3(-1.6f, 2.4f, 1.2f), new Vec3(1.25f, 0.65f, 0.60f), 60f);
                int l1 = AddLight(a + new Vec3(1.6f, 2.4f, -1.2f), new Vec3(0.60f, 0.95f, 1.25f), 60f);
                int l2 = AddLight(a + new Vec3(0.0f, 3.0f, 0.0f), new Vec3(0.95f, 1.00f, 0.98f), 70f);
                AddOrbiting(l2, a, 2.6f, 3.0f, 0.5f, 0.0f);
                AddPulsing(l0, 0.25f, 0.0f);
                AddPulsing(l1, 0.25f, 1.0f);
            }

            void AddLightStage(Vec3 a)
            {
                int stageMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.85f, 0.85f, 0.90f), MaterialKind = MaterialSbtData.Solid });
                disks.Add(new DiskData { Center = a + new Vec3(0.0f, 0.02f, 0.0f), Normal = new Vec3(0f, 1f, 0f), Radius = 1.8f });
                diskMaterialIds.Add((uint)stageMat);

                int canopyMat = AddMaterial(new MaterialSbtData { Emission = new Vec3(3.0f, 3.0f, 3.2f), MaterialKind = MaterialSbtData.Solid });
                disks.Add(new DiskData { Center = a + new Vec3(0.0f, 2.4f, 0.0f), Normal = new Vec3(0f, -1f, 0f), Radius = 0.7f });
                diskMaterialIds.Add((uint)canopyMat);

                int focusMat = MakeGlass();
                int focus = AddSphere(a + new Vec3(0.0f, 0.6f, 0.0f), 0.45f, (uint)focusMat);
                AddBobbing(focus, 0.10f, 1.8f, 0.0f);

                int glow0 = AddMaterial(new MaterialSbtData { Emission = new Vec3(2.8f, 0.9f, 0.7f), MaterialKind = MaterialSbtData.Solid });
                int glow1 = AddMaterial(new MaterialSbtData { Emission = new Vec3(0.7f, 2.6f, 3.0f), MaterialKind = MaterialSbtData.Solid });
                int glow2 = AddMaterial(new MaterialSbtData { Emission = new Vec3(2.6f, 2.6f, 0.9f), MaterialKind = MaterialSbtData.Solid });
                AddSphere(a + new Vec3(0.9f, 0.12f, 0.0f), 0.10f, (uint)glow0);
                AddSphere(a + new Vec3(-0.9f, 0.12f, 0.0f), 0.10f, (uint)glow1);
                AddSphere(a + new Vec3(0.0f, 0.12f, 0.9f), 0.10f, (uint)glow2);

                int a0 = AddLight(a + new Vec3(1.2f, 2.0f, 0.8f), new Vec3(1.0f, 0.95f, 0.9f), 55f);
                int a1 = AddLight(a + new Vec3(-1.2f, 2.0f, -0.8f), new Vec3(0.9f, 0.95f, 1.0f), 55f);
                AddOrbiting(a0, a, 1.6f, 2.0f, 0.9f, 0.0f);
                AddOrbiting(a1, a, 1.6f, 2.0f, -0.9f, 0.6f);
            }

            void AddMetalRing(Vec3 a)
            {
                const int count = 10;
                const float R = 1.9f;
                for (var i = 0; i < count; i++)
                {
                    float t = (i * (2f * XMath.PI) / count) + RngRange(-0.02f, 0.02f);
                    Vec3 c = a + new Vec3(XMath.Cos(t) * R, 0.5f, XMath.Sin(t) * R);
                    int m = RngInt(0, 3) switch
                    {
                        0 => AddMaterial(new MaterialSbtData { Albedo = new Vec3(1.00f, 0.85f, 0.57f), Reflectivity = 0.10f, MaterialKind = MaterialSbtData.Solid }),
                        1 => AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.78f, 0.60f, 0.20f), Reflectivity = 0.08f, MaterialKind = MaterialSbtData.Solid }),
                        2 => AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.96f, 0.58f, 0.25f), Reflectivity = 0.08f, MaterialKind = MaterialSbtData.Solid }),
                        _ => AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.98f, 0.98f, 0.98f), Reflectivity = 0.90f, MaterialKind = MaterialSbtData.Solid }),
                    };
                    AddSphere(c, 0.32f, (uint)m);
                }

                int glassMat = MakeGlass();
                int centerGlass = AddSphere(a + new Vec3(0.0f, 0.62f, 0.0f), 0.38f, (uint)glassMat);
                AddBobbing(centerGlass, 0.08f, 1.0f, 0.3f);
                int glowMat = AddMaterial(new MaterialSbtData { Emission = new Vec3(2.6f, 1.0f, 0.8f), MaterialKind = MaterialSbtData.Solid });
                AddSphere(a + new Vec3(0.0f, 0.62f, 0.0f), 0.12f, (uint)glowMat);

                int l0 = AddLight(a + new Vec3(0.0f, 2.4f, 0.0f), new Vec3(1.0f, 0.98f, 0.95f), 70f);
                int l1 = AddLight(a + new Vec3(1.6f, 1.8f, 0.0f), new Vec3(0.65f, 0.95f, 1.25f), 55f);
                int l2 = AddLight(a + new Vec3(-1.6f, 1.8f, 0.0f), new Vec3(1.20f, 0.75f, 0.70f), 55f);
                AddPulsing(l0, 0.20f, 0.0f);
                AddPulsing(l1, 0.18f, 0.6f);
                AddPulsing(l2, 0.18f, 1.2f);
            }

            void AddCheckerTerrace(Vec3 a)
            {
                const float w = 3.2f, d = 0.9f;
                int whiteMat = AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.88f, 0.88f, 0.90f), MaterialKind = MaterialSbtData.Solid });
                for (var i = 0; i < 4; i++)
                {
                    float y = a.y + (i * 0.3f);
                    xzRects.Add(new RectData { A0 = a.x - w, A1 = a.x + w, B0 = a.z - (d * (i + 1)), B1 = a.z + (d * (i + 1)), C = y });
                    xzRectMaterialIds.Add((uint)whiteMat);

                    float z0 = a.z - (d * (i + 1));
                    float z1 = a.z + (d * (i + 1));
                    xzRects.Add(new RectData { A0 = a.x - w, A1 = a.x + w, B0 = z0, B1 = z0 + 0.08f, C = y + 0.001f }); xzRectMaterialIds.Add((uint)emissiveWarm);
                    xzRects.Add(new RectData { A0 = a.x - w, A1 = a.x + w, B0 = z1 - 0.08f, B1 = z1, C = y + 0.001f }); xzRectMaterialIds.Add((uint)emissiveCool);
                }
                int topMat = RngBool(0.5f)
                    ? AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.98f, 0.98f, 0.98f), Reflectivity = 0.92f, MaterialKind = MaterialSbtData.Solid })
                    : MakeGlass();
                int top = AddSphere(a + new Vec3(0.0f, (0.3f * 4) + 0.35f, 0.0f), 0.35f, (uint)topMat);
                AddBobbing(top, 0.07f, 1.1f, 0.0f);

                int l = AddLight(a + new Vec3(0.0f, 2.0f, 0.0f), new Vec3(0.95f, 1.0f, 0.95f), 55f);
                AddPulsing(l, 0.22f, 0.4f);
            }

            void AddDiffuseStacks(Vec3 a)
            {
                const int stacks = 3;
                for (var i = 0; i < stacks; i++)
                {
                    Vec3 p = a + new Vec3((i - 1) * 1.4f, 0.0f, (i - 1) * 0.3f);
                    float h = RngRange(0.9f, 1.6f);
                    float r = RngRange(0.25f, 0.35f);
                    int bodyMat = AddMaterial(new MaterialSbtData { Albedo = RandSaturated(0.10f), Reflectivity = RngRange(0.02f, 0.06f), MaterialKind = MaterialSbtData.Solid });
                    cylindersY.Add(new CylinderYData { Center = p, Radius = r, YMin = 0f, YMax = h, Capped = 1 }); cylinderYMaterialIds.Add((uint)bodyMat);
                    AddSphere(p + new Vec3(0.0f, h + (r * 0.8f), 0.0f), r * 0.8f, (uint)bodyMat);
                    xyRects.Add(new RectData { A0 = p.x - 0.25f, A1 = p.x + 0.25f, B0 = 1.65f, B1 = 1.75f, C = p.z - 0.30f });
                    xyRectMaterialIds.Add((uint)emissiveCool);
                }
                int fill = AddLight(a + new Vec3(0.0f, 2.2f, 0.0f), new Vec3(1.00f, 0.98f, 0.95f), 60f);
                AddPulsing(fill, 0.20f, 0.0f);
            }

            float startAngle = -15.0f;
            float stepDeg = 360.0f / ringCount;

            AddMiniCornell(AnchorOnRing(startAngle + (stepDeg * 0) + RngRange(-5f, 5f)));
            AddGlassGarden(AnchorOnRing(startAngle + (stepDeg * 1) + RngRange(-5f, 5f)));
            AddMirrorCorridor(AnchorOnRing(startAngle + (stepDeg * 2) + RngRange(-5f, 5f)));
            AddVoxelGrotto(AnchorOnRing(startAngle + (stepDeg * 3) + RngRange(-5f, 5f)));
            AddPedestalPlaza(AnchorOnRing(startAngle + (stepDeg * 4) + RngRange(-5f, 5f)));
            AddTriangleField(AnchorOnRing(startAngle + (stepDeg * 5) + RngRange(-5f, 5f)));
            // Ring position 6 ("textured panel") intentionally left empty, matching the
            // reference's own commented-out AddTexturedPanel call.
            AddMeshShowcase(AnchorOnRing(startAngle + (stepDeg * 7) + RngRange(-5f, 5f)));
            AddLightStage(AnchorOnRing(startAngle + (stepDeg * 8) + RngRange(-5f, 5f)));
            AddMetalRing(AnchorOnRing(startAngle + (stepDeg * 9) + RngRange(-5f, 5f)));
            AddCheckerTerrace(AnchorOnRing(startAngle + (stepDeg * 10) + RngRange(-5f, 5f)));
            AddDiffuseStacks(AnchorOnRing(startAngle + (stepDeg * 11) + RngRange(-5f, 5f)));

            int g0 = AddLight(center + new Vec3(0.0f, 18.0f, 14.0f), new Vec3(1.0f, 0.98f, 0.95f), 330f);
            int g1 = AddLight(center + new Vec3(-22.0f, 14.0f, -18.0f), new Vec3(0.9f, 0.95f, 1.0f), 240f);
            int g2 = AddLight(center + new Vec3(22.0f, 15.0f, -6.0f), new Vec3(0.95f, 1.0f, 0.95f), 220f);
            AddPulsing(g0, 0.20f, 0.35f);
            AddPulsing(g1, 0.18f, 0.50f);
            AddPulsing(g2, 0.15f, 0.65f);

            return new SceneData
            {
                Name = "Radial Museum",
                Vertices = vertices.ToArray(),
                Normals = normals.ToArray(),
                TexCoords = texCoords.ToArray(),
                Indices = indices.ToArray(),
                TriangleMaterialIds = triangleMaterialIds.ToArray(),
                MeshRanges = FillMeshRangeGaps(meshRanges, indices.Count),
                Materials = materials.ToArray(),
                MaterialTexturePaths = materialTexturePaths.ToArray(),
                Spheres = spheres.ToArray(),
                SphereMaterialIds = sphereMaterialIds.ToArray(),
                Boxes = boxes.ToArray(),
                BoxMaterialIds = boxMaterialIds.ToArray(),
                CylindersY = cylindersY.ToArray(),
                CylinderYMaterialIds = cylinderYMaterialIds.ToArray(),
                Disks = disks.ToArray(),
                DiskMaterialIds = diskMaterialIds.ToArray(),
                XYRects = xyRects.ToArray(),
                XYRectMaterialIds = xyRectMaterialIds.ToArray(),
                XZRects = xzRects.ToArray(),
                XZRectMaterialIds = xzRectMaterialIds.ToArray(),
                YZRects = yzRects.ToArray(),
                YZRectMaterialIds = yzRectMaterialIds.ToArray(),
                VoxelMaterialIds = gridVoxelIds,
                VolumeGridMin = gridVoxelMin,
                VolumeVoxelSize = gridVoxelSize,
                VolumeDims = gridVoxelDims,
                Lights = lights.ToArray(),
                OrbitingLights = orbitingLights.ToArray(),
                PulsingLights = pulsingLights.ToArray(),
                BobbingSpheres = bobbingSpheres.ToArray(),
                AmbientColor = new Vec3(1f, 1f, 1f),
                AmbientIntensity = 0.055f,
                BackgroundTop = new Vec3(0.05f, 0.07f, 0.10f),
                BackgroundBottom = new Vec3(0.01f, 0.01f, 0.02f),
                CameraOrigin = new Vec3(0.0f, 2.4f, 10.5f),
                CameraLookAt = new Vec3(0.0f, 2.0f, -15f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 50f,
                CameraWorldScale = 25f,
            };
        }

        public static Func<SceneData>[] BuildSceneTable() => new Func<SceneData>[]
        {
            BuildDebugOrenNayarScene,
            BuildTextureTestScene,
            BuildSimpleTestScene,
            BuildDemoScene,
            BuildCornellBox,
            BuildMirrorSpheresOnChecker,
            BuildCylindersDisksAndTriangles,
            BuildBoxesShowcase,
            BuildVolumeGridTestScene,
            BuildAllMeshesScene,
            BuildBunnyScene,
            BuildTeapotScene,
            BuildCowScene,
            BuildDragonScene,
            BuildMuseumScene,
            BuildRadialMuseumScene,
        };
    }
}
