using System;

namespace Sample13
{
    /// <summary>
    /// The "Museum" scene: several widely-spaced dioramas along a corridor.
    /// </summary>
    public static class MuseumScene
    {
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
        // BuildMuseumVolumeGrids) - visually equivalent, just one shared array instead of
        // two independent ones.
        public static SceneData BuildMuseumScene()
        {
            var b = new SceneBuilder
            {
                Name = "Museum",
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

            // Shared base materials.
            int mirror = b.AddMaterial(MaterialPresets.Mirror(0.90f));
            int red = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.95f, 0.15f, 0.15f), 0.08f));
            int green = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.15f, 0.95f, 0.20f), 0.06f));
            int blue = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.15f, 0.25f, 0.95f), 0.06f));
            int gold = b.AddMaterial(MaterialPresets.Solid(new Vec3(1.00f, 0.85f, 0.57f), 0.25f));
            int brass = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.78f, 0.60f, 0.20f), 0.18f));
            int pedestal = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.85f, 0.85f, 0.85f)));
            int glassClear = b.AddMaterial(MaterialPresets.ClearGlass());
            int glassBlue = b.AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.9f, 0.95f, 1.0f), Transparency = 1f, IndexOfRefraction = 1.52f, TransmissionColor = new Vec3(0.9f, 0.95f, 1.0f), MaterialKind = MaterialSbtData.Solid });

            // Global floor and distant backdrop - one large XZRect/XYRect standing in for
            // the reference's infinite Planes (the established
            // Plane -> finite-rect substitution), sized to cover the whole corridor
            // (anchors range from Z=-12 out to Z=-88).
            int floorMat = b.AddMaterial(MaterialPresets.Checker(new Vec3(0.82f, 0.82f, 0.85f), new Vec3(0.12f, 0.12f, 0.12f), 0.8f));
            b.AddXZRect(-16f, 16f, -100f, 16f, 0f, (uint)floorMat);

            int backdropMat = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.02f, 0.02f, 0.03f)));
            b.AddXYRect(-20f, 20f, 0f, 20f, -99f, (uint)backdropMat);

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

                int white = b.AddMaterial(MaterialPresets.Solid(whiteColor));
                int leftWall = b.AddMaterial(MaterialPresets.Solid(leftColor));
                int rightWall = b.AddMaterial(MaterialPresets.Solid(rightColor));
                int emissive = b.AddMaterial(MaterialPresets.Emissive(new Vec3(4f, 4f, 4f)));

                b.AddYZRect(yB, yT, zB, zF, xL, (uint)leftWall);
                b.AddYZRect(yB, yT, zB, zF, xR, (uint)rightWall);
                b.AddXZRect(xL, xR, zB, zF, yB, (uint)white);
                b.AddXZRect(xL, xR, zB, zF, yT, (uint)white);
                b.AddXYRect(xL, xR, yB, yT, zB, (uint)white);

                float lx0 = xL + (0.20f * width);
                float lx1 = xR - (0.20f * width);
                float lz0 = zB + (0.35f * width);
                float lz1 = zB + (0.55f * width);
                float ly = yT - 0.01f;
                b.AddXZRect(lx0, lx1, lz0, lz1, ly, (uint)emissive);
                b.AddLight(new Vec3((lx0 + lx1) * 0.5f, yT - 0.2f, (lz0 + lz1) * 0.5f), new Vec3(1f, 0.98f, 0.95f), lightPower);

                float cx = (xL + xR) * 0.5f;
                float cz = (zB + zF) * 0.5f;

                int ped = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.88f, 0.88f, 0.88f)));
                int objA = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.90f, 0.20f, 0.20f), 0.08f));
                int objB = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.20f, 0.80f, 0.95f), 0.10f));
                int mirrorish = b.AddMaterial(MaterialPresets.Mirror(0.85f));
                int glassish = b.AddMaterial(MaterialPresets.ClearGlass());

                var stand0 = new Vec3(cx - 0.8f, yB, cz + 0.2f);
                var stand1 = new Vec3(cx + 0.8f, yB, cz - 0.2f);

                b.AddDisk(new Vec3(cx, yB + 0.01f, cz), new Vec3(0f, 1f, 0f), width * 0.32f,
                    (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.90f, 0.90f, 0.92f))));

                b.AddCylinderY(stand0, 0.28f, 0f, 0.9f, (uint)ped);
                b.AddSphere(stand0 + new Vec3(0f, 0.9f + 0.35f, 0f), 0.35f, (uint)mirrorish);

                b.AddCylinderY(stand1, 0.28f, 0f, 0.9f, (uint)ped);
                b.AddSphere(stand1 + new Vec3(0f, 0.9f + 0.32f, 0f), 0.32f, (uint)glassish);

                b.AddSphere(new Vec3(cx, yB + 0.35f, cz - 0.9f), 0.35f, (uint)objA);
                b.AddSphere(new Vec3(cx, yB + 0.22f, cz + 0.9f), 0.22f, (uint)objB);
            }

            AddCornellRoom(cornellAnchorA, 6f, 5f, new Vec3(0.80f, 0.10f, 0.10f), new Vec3(0.10f, 0.80f, 0.10f), new Vec3(0.82f, 0.82f, 0.82f), 65f);
            AddCornellRoom(cornellAnchorB, 6f, 5f, new Vec3(0.70f, 0.10f, 0.70f), new Vec3(0.10f, 0.70f, 0.70f), new Vec3(0.82f, 0.82f, 0.82f), 75f);
            AddCornellRoom(cornellAnchorC, 6f, 5f, new Vec3(0.80f, 0.20f, 0.10f), new Vec3(0.20f, 0.80f, 0.10f), new Vec3(0.90f, 0.90f, 0.90f), 90f);

            // Mesh gallery (uniform scale=1 per-mesh literal multiplier, matching the
            // reference's own scale values for these exact OBJ assets).
            b.AddMeshAutoGround("cow.obj", (uint)gold, 3.0f, meshGalleryAnchor + new Vec3(-2.6f, 1.0f, -0.4f));
            b.AddMeshAutoGround("stanford-bunny.obj", (uint)green, 3.0f, meshGalleryAnchor + new Vec3(0.0f, 1.0f, 0.0f));
            b.AddMeshAutoGround("teapot.obj", (uint)red, 2.0f, meshGalleryAnchor + new Vec3(2.6f, 1.0f, 0.4f));
            b.AddMeshAutoGround("xyzrgb_dragon.obj", (uint)mirror, 8.0f, meshGalleryAnchor + new Vec3(5.2f, 2.0f, -0.8f));
            int yellowDisk = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.85f, 0.85f, 0.1f)));
            b.AddDisk(meshGalleryAnchor + new Vec3(2.6f, 0.01f, 0.4f), new Vec3(0f, 1f, 0f), 0.9f, (uint)yellowDisk);

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
                    b.AddCylinderY(pts[i], 0.32f, 0f, pedH, (uint)pedestal);
                    b.AddSphere(pts[i] + new Vec3(0f, pedH + sphR, 0f), sphR, (uint)topMats[i]);
                }
            }

            // Triangle showcase.
            b.AddTriangle(
                new Vec3(2.2f, 0.0f, -6.0f),
                new Vec3(2.8f, 1.2f, -6.4f),
                new Vec3(1.6f, 0.7f, -6.8f),
                (uint)brass);

            // Textured demo sphere + textured endcap wall, using a bundled copy of the
            // reference's own RGBW-8k.png (only ~105KB despite the name, so copying it
            // in directly - see Sample13.csproj - was simpler than substituting a
            // different already-bundled texture).
            {
                int textured = b.AddMaterial(new MaterialSbtData { Albedo = new Vec3(1f, 1f, 1f), TextureWeight = 1f, UVScale = 1.5f, MaterialKind = MaterialSbtData.Solid }, "models/museum/RGBW-8k.png");
                b.AddSphere(new Vec3(-1.6f, 0.6f, -10.0f), 0.6f, (uint)textured);

                int texturedWall = b.AddMaterial(new MaterialSbtData { Albedo = new Vec3(1f, 1f, 1f), TextureWeight = 1f, UVScale = 0.35f, MaterialKind = MaterialSbtData.Solid }, "models/museum/RGBW-8k.png");
                b.AddXYRect(-10f, 10f, 0f, 8f, -98f, (uint)texturedWall);

                // Video-textured cube demo.
                int videoBox = b.AddMaterial(new MaterialSbtData { Albedo = new Vec3(1f, 1f, 1f), TextureWeight = 1f, UVScale = 1f, MaterialKind = MaterialSbtData.Solid }, "models/museum/RTConsole.mp4");
                Vec3 bMin = new Vec3(-2.2f, 0.4f, -7.8f);
                b.AddBox(bMin, bMin + new Vec3(1.2f, 1.2f, 1.2f), (uint)videoBox);
            }

            BuildMuseumVolumeGrids(b, volumeAnchorA, volumeAnchorB, red, green, blue, mirror, glassClear, pedestal, gold);

            // Video-cube diorama (reflection/refraction tests).
            {
                Vec3 basePos = new Vec3(9.0f, 0.0f, -60.0f);
                int platform = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.90f, 0.90f, 0.92f)));
                b.AddDisk(basePos + new Vec3(0f, 0.01f, 0f), new Vec3(0f, 1f, 0f), 1.7f, (uint)platform);

                int videoMat = b.AddMaterial(new MaterialSbtData { Albedo = new Vec3(1f, 1f, 1f), TextureWeight = 1f, UVScale = 1f, MaterialKind = MaterialSbtData.Solid }, "models/museum/RTConsole.mp4");
                Vec3 cubeC = basePos + new Vec3(0f, 0.65f, 0f);
                Vec3 half = new Vec3(0.40f, 0.40f, 0.40f);
                b.AddBox(cubeC - half, cubeC + half, (uint)videoMat);

                b.AddSphere(basePos + new Vec3(-0.95f, 0.65f + 0.35f, 0.65f), 0.35f, (uint)mirror);
                b.AddSphere(basePos + new Vec3(0.95f, 0.65f + 0.32f, -0.65f), 0.32f, (uint)glassClear);

                Vec3 standA = basePos + new Vec3(-0.9f, 0.0f, -0.7f);
                Vec3 standB = basePos + new Vec3(0.9f, 0.0f, 0.7f);
                b.AddCylinderY(standA, 0.22f, 0f, 0.9f, (uint)pedestal);
                b.AddCylinderY(standB, 0.22f, 0f, 0.9f, (uint)pedestal);

                int meshMirror = b.AddMaterial(MaterialPresets.Mirror(0.85f));
                b.AddMeshAutoGround("teapot.obj", (uint)meshMirror, 0.9f, standA + new Vec3(0f, 1.0f, 0f));

                int meshGlass = b.AddMaterial(MaterialPresets.ClearGlass());
                b.AddMeshAutoGround("stanford-bunny.obj", (uint)meshGlass, 0.9f, standB + new Vec3(0f, 1.0f, 0f));

                b.AddLight(basePos + new Vec3(0f, 2.0f, 0f), new Vec3(1f, 0.98f, 0.95f), 140f);
            }

            b.AddLight(new Vec3(0.0f, 12.0f, -50.0f), new Vec3(1f, 1f, 1f), 900f);

            return b.Build();
        }

        // Builds the museum's two volume-grid dioramas as one combined grid (see
        // BuildMuseumScene's comment on why - SceneData only supports one volume-grid
        // region per scene). Diorama A (16x8x16 @ 0.5) and diorama B (rescaled to
        // 13x7x13 @ 0.5, from the reference's 14x7x14 @ 0.45 - a proportionally-
        // equivalent footprint at the shared voxel size) are placed in non-overlapping
        // sub-regions of one combined array spanning both anchors.
        private static void BuildMuseumVolumeGrids(
            SceneBuilder b, Vec3 anchorA, Vec3 anchorB,
            int red, int green, int blue, int mirror, int glassClear, int pedestal, int gold)
        {
            Vec3 voxelSize = new Vec3(0.5f, 0.5f, 0.5f);
            Vec3 voxelMin = new Vec3(Math.Min(anchorA.x, anchorB.x), 0f, Math.Min(anchorA.z, anchorB.z));

            const int nxA = 16, nyA = 8, nzA = 16;
            const int nxB = 13, nyB = 7, nzB = 13;

            int combinedNx = (int)Math.Round((Math.Max(anchorA.x + (nxA * 0.5f), anchorB.x + (nxB * 0.5f)) - voxelMin.x) / voxelSize.x) + 1;
            int combinedNy = Math.Max(nyA, nyB);
            int combinedNz = (int)Math.Round((Math.Max(anchorA.z + (nzA * 0.5f), anchorB.z + (nzB * 0.5f)) - voxelMin.z) / voxelSize.z) + 1;
            var voxelDims = new Vec3i(combinedNx, combinedNy, combinedNz);

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
            uint vStone = (uint)(b.AddMaterial(MaterialPresets.Solid(new Vec3(0.82f, 0.82f, 0.85f))) + 1);
            uint vRed = (uint)(red + 1);
            uint vGreen = (uint)(green + 1);
            uint vBlue = (uint)(blue + 1);
            uint vMirror = (uint)(mirror + 1);
            uint vDark = (uint)(b.AddMaterial(MaterialPresets.Solid(new Vec3(0.15f, 0.15f, 0.18f))) + 1);

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
            b.AddCylinderY(pedC, 0.35f, 0f, 1.4f, (uint)pedestal);
            b.AddSphere(pedC + new Vec3(0f, 1.4f + 0.45f, 0f), 0.45f, (uint)glassClear);
            b.AddLight(anchorA + new Vec3(4.0f, 3.0f, 1.0f), new Vec3(0.9f, 0.95f, 1.0f), 110f);

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
            b.AddCylinderY(stand, 0.30f, 0f, 1.1f, (uint)pedestal);
            b.AddMeshAutoGround("teapot.obj", (uint)gold, 1.0f, stand + new Vec3(0f, 1.12f, 0f));
            b.AddLight(anchorB + new Vec3(2.5f, 2.8f, 7.0f), new Vec3(1f, 0.95f, 0.9f), 85f);

            b.SetVolumeGrid(voxels, voxelMin, voxelSize, voxelDims);
        }
    }
}
