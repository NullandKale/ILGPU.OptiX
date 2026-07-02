using ILGPU.Algorithms;
using System;

namespace Sample13
{
    /// <summary>
    /// The "Radial Museum" scene: a giant refractive dragon centerpiece surrounded by a
    /// ring of themed vignettes, with animated lights and bobbing spheres throughout.
    /// </summary>
    public static class RadialMuseumScene
    {
        // Direct port of the reference's TestScenesRandom.Build() ("Radial Museum") - a
        // giant refractive dragon centerpiece surrounded by a ring of 12 independently
        // themed vignettes (11 real + 1 intentionally empty slot, matching the
        // reference's own commented-out AddTexturedPanel call), plus animated
        // orbiting/pulsing point lights and bobbing spheres throughout. The reference's
        // OrbitingLightEntity/PulsingLightEntity/BobbingSphereEntity update objects are
        // expressed as SceneData.OrbitingLights/PulsingLights/BobbingSpheres
        // descriptors instead (see SceneData.cs's class comment) - applied every frame
        // by the renderer's animation step, not stored as live objects here. Uses a
        // fixed seed (matching the reference's own default Build(int seed = 1337))
        // rather than the reference's seed==-1 option, for a reproducible layout.
        //
        // NOTE: the section builders below are deliberately kept as local functions
        // sharing ONE Random - the scene layout is derived from the exact draw sequence,
        // so their bodies and call order must not be reordered.
        public static SceneData BuildRadialMuseumScene()
        {
            var rng = new Random(1337);
            float RngRange(float a, float b2) => a + ((float)rng.NextDouble() * (b2 - a));
            int RngInt(int a, int bInclusive) => a + rng.Next(bInclusive - a + 1);
            bool RngBool(float pTrue = 0.5f) => rng.NextDouble() < pTrue;
            Vec3 RandSaturated(float desat = 0f) => MaterialPresets.HsvToRgb((float)rng.NextDouble() * 360f, 0.75f - desat, 0.95f);

            var b = new SceneBuilder
            {
                Name = "Radial Museum",
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

            int MakeGlass()
            {
                float ior = RngRange(1.30f, 1.70f);
                var tint = new Vec3(RngRange(0.85f, 1.00f), RngRange(0.85f, 1.00f), RngRange(0.85f, 1.00f));
                return b.AddMaterial(MaterialPresets.Glass(ior, tint));
            }

            int matMirror = b.AddMaterial(MaterialPresets.Mirror(0.90f));
            int emissiveWarm = b.AddMaterial(MaterialPresets.Emissive(new Vec3(3.2f, 3.0f, 2.8f)));
            int emissiveCool = b.AddMaterial(MaterialPresets.Emissive(new Vec3(2.9f, 3.2f, 3.4f)));

            var center = new Vec3(0.0f, 0.0f, -26.0f);
            const float ringRadius = 22.0f;
            const int ringCount = 12;

            // Global floor + distant backdrop (see Museum's identical Plane -> rect
            // substitution note in BuildMuseumScene).
            int floorMat = b.AddMaterial(MaterialPresets.Checker(new Vec3(0.82f, 0.82f, 0.85f), new Vec3(0.12f, 0.12f, 0.12f), 0.9f));
            b.AddXZRect(-34f, 34f, -60f, 12f, 0f, (uint)floorMat);
            int backdropMat = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.02f, 0.02f, 0.03f)));
            b.AddXYRect(-34f, 34f, 0f, 24f, -80f, (uint)backdropMat);

            Vec3 AnchorOnRing(float angleDeg)
            {
                float t = angleDeg * (XMath.PI / 180.0f);
                return center + new Vec3(XMath.Cos(t) * ringRadius, 0.0f, XMath.Sin(t) * ringRadius);
            }

            // Center dragon.
            {
                int plinth = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.96f, 0.96f, 0.98f), 0.88f));
                b.AddDisk(center + new Vec3(0f, 0.015f, 0f), new Vec3(0f, 1f, 0f), 6.2f, (uint)plinth);

                int giantGlass = MakeGlass();
                b.AddMeshAutoGround("xyzrgb_dragon.obj", (uint)giantGlass, 20.0f, center + new Vec3(0.0f, -5.5f, 0.0f));

                int la = b.AddLight(center + new Vec3(7.0f, 6.5f, 6.0f), new Vec3(0.98f, 0.96f, 1.06f), 260f);
                int lb = b.AddLight(center + new Vec3(-8.0f, 6.0f, -6.5f), new Vec3(1.06f, 0.96f, 0.92f), 230f);
                int lc = b.AddLight(center + new Vec3(0.0f, 3.2f, 0.0f), new Vec3(0.65f, 0.80f, 1.30f), 140f);
                int ld = b.AddLight(center + new Vec3(0.0f, 3.2f, 2.2f), new Vec3(1.20f, 0.75f, 0.70f), 120f);
                b.AddOrbitingLight(la, center, 8.0f, 6.5f, 0.25f, 0.0f);
                b.AddOrbitingLight(lb, center, 8.0f, 6.0f, -0.22f, 1.8f);
                b.AddPulsingLight(lc, 0.25f, 0.9f);
                b.AddPulsingLight(ld, 0.22f, 1.4f);
            }

            void AddMiniCornell(Vec3 a)
            {
                float w = 5.2f, h = 4.4f;
                float xL = a.x - (w * 0.5f), xR = a.x + (w * 0.5f);
                float yB = a.y, yT = a.y + h;
                float zB = a.z - (w * 0.5f), zF = a.z + (w * 0.5f);

                int left = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.95f, 0.20f, 0.20f)));
                int right = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.20f, 0.28f, 0.95f)));
                int white = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.82f, 0.82f, 0.82f)));

                b.AddYZRect(yB, yT, zB, zF, xL, (uint)left);
                b.AddYZRect(yB, yT, zB, zF, xR, (uint)right);
                b.AddXZRect(xL, xR, zB, zF, yB, (uint)white);
                b.AddXZRect(xL, xR, zB, zF, yT, (uint)white);
                b.AddXYRect(xL, xR, yB, yT, zB, (uint)white);

                float lx0 = xL + (0.18f * w), lx1 = xR - (0.18f * w);
                float lz0 = zB + (0.38f * w), lz1 = zB + (0.58f * w);
                float ly = yT - 0.01f;
                int ceiling = RngBool(0.5f) ? emissiveWarm : emissiveCool;
                b.AddXZRect(lx0, lx1, lz0, lz1, ly, (uint)ceiling);

                Vec3 stand0 = new Vec3(a.x - 0.85f, yB, a.z + 0.25f);
                Vec3 stand1 = new Vec3(a.x + 0.85f, yB, a.z - 0.25f);
                int ped = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.90f, 0.90f, 0.92f)));

                b.AddCylinderY(stand0, 0.26f, 0f, 0.9f, (uint)ped);
                int sp0Mat = b.AddMaterial(MaterialPresets.Mirror(0.85f));
                int sp0 = b.AddSphere(stand0 + new Vec3(0f, 1.25f, 0f), 0.34f, (uint)sp0Mat);
                b.AddBobbingSphere(sp0, 0.06f, 1.6f, 0.0f);

                b.AddCylinderY(stand1, 0.26f, 0f, 0.9f, (uint)ped);
                int sp1Mat = MakeGlass();
                int sp1 = b.AddSphere(stand1 + new Vec3(0f, 1.18f, 0f), 0.30f, (uint)sp1Mat);
                b.AddBobbingSphere(sp1, 0.05f, 1.2f, 1.1f);

                b.AddXYRect(a.x - 0.55f, a.x + 0.55f, a.y + 1.9f, a.y + 2.0f, a.z + 0.2f, (uint)emissiveWarm);
                b.AddXYRect(a.x - 0.55f, a.x + 0.55f, a.y + 1.9f, a.y + 2.0f, a.z - 0.2f, (uint)emissiveCool);

                int l0 = b.AddLight(new Vec3(a.x - 0.8f, yT - 0.35f, a.z + 0.6f), new Vec3(1.2f, 0.7f, 0.6f), 55f);
                int l1 = b.AddLight(new Vec3(a.x + 0.8f, yT - 0.35f, a.z - 0.6f), new Vec3(0.7f, 0.9f, 1.2f), 55f);
                b.AddPulsingLight(l0, 0.30f, 0.0f);
                b.AddPulsingLight(l1, 0.30f, XMath.PI);
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
                            b.AddCylinderY(a + new Vec3(rx * spacing, 0.0f, rz * spacing), rr * 0.75f, 0f, rr * 2.2f, (uint)MakeGlass());
                        }
                        else
                        {
                            Vec3 c = a + new Vec3(rx * spacing, rr, rz * spacing);
                            int gsMat = MakeGlass();
                            int gs = b.AddSphere(c, rr, (uint)gsMat);
                            b.AddBobbingSphere(gs, 0.04f, 1.2f + RngRange(-0.2f, 0.2f), (rx * 1.1f) + (rz * 0.7f));
                        }
                    }
                }

                b.AddDisk(a + new Vec3(0f, 0.01f, 0f), new Vec3(0f, 1f, 0f), 2.2f,
                    (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.96f, 0.96f, 0.98f), 0.88f)));

                int l0 = b.AddLight(a + new Vec3(-1.2f, 2.4f, 1.0f), new Vec3(1.15f, 0.65f, 0.55f), 80f);
                int l1 = b.AddLight(a + new Vec3(1.2f, 2.6f, -1.0f), new Vec3(0.55f, 0.95f, 1.25f), 80f);
                int l2 = b.AddLight(a + new Vec3(0.0f, 3.0f, 0.0f), new Vec3(0.95f, 0.98f, 1.05f), 60f);
                b.AddOrbitingLight(l0, a, 2.0f, 2.4f, 0.6f, 0.0f);
                b.AddOrbitingLight(l1, a, 2.0f, 2.6f, -0.6f, 0.5f);
                b.AddPulsingLight(l2, 0.25f, 0.8f);
            }

            void AddMirrorCorridor(Vec3 a)
            {
                const int n = 6;
                const float step = 1.8f;
                int mirrorish = b.AddMaterial(MaterialPresets.Mirror(0.92f));
                for (var i = -n; i <= n; i++)
                {
                    Vec3 pL = a + new Vec3(-1.6f, 0.0f, i * step);
                    Vec3 pR = a + new Vec3(1.6f, 0.0f, i * step);
                    b.AddCylinderY(pL, 0.22f, 0f, 2.1f, (uint)mirrorish);
                    b.AddCylinderY(pR, 0.22f, 0f, 2.1f, (uint)mirrorish);
                    Vec3 lp = new Vec3(a.x, a.y + 2.9f, a.z + (i * step));
                    int pl = ((i & 1) == 0)
                        ? b.AddLight(lp, new Vec3(1.15f, 0.75f, 0.65f), 38f)
                        : b.AddLight(lp, new Vec3(0.65f, 0.95f, 1.20f), 38f);
                    b.AddPulsingLight(pl, 0.35f, i * 0.7f);
                }
                int floor = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.96f, 0.96f, 0.98f), 0.85f));
                b.AddXZRect(a.x - 2.2f, a.x + 2.2f, a.z - (n * step) - 0.4f, a.z + (n * step) + 0.4f, a.y + 0.01f, (uint)floor);
            }

            void AddVoxelGrotto(Vec3 a)
            {
                const int nx = 12, ny = 6, nz = 12;
                var localVoxels = new uint[nx * ny * nz];
                uint LocalIndex(int x, int y, int z) => (uint)((x * ny * nz) + (y * nz) + z);

                int stoneMat = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.80f, 0.80f, 0.82f)));
                int redMat = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.90f, 0.20f, 0.20f), 0.10f));
                int greenMat = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.20f, 0.85f, 0.20f), 0.08f));
                int blueMat = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.20f, 0.30f, 0.95f), 0.08f));
                int mirrorMat = b.AddMaterial(MaterialPresets.Mirror(0.88f));
                int checkerMat = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.85f, 0.85f, 0.88f)));
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
                b.SetVolumeGrid(localVoxels, a + new Vec3(-2.6f, 0.0f, -2.6f), new Vec3(0.45f, 0.45f, 0.45f), new Vec3i(nx, ny, nz));

                for (var i = 0; i < 10; i++)
                {
                    Vec3 p = a + new Vec3(RngRange(-2.2f, 2.2f), RngRange(0.6f, 2.2f), RngRange(-2.2f, 2.2f));
                    Vec3 e = RandSaturated(0.25f) * 2.0f;
                    int glowMat = b.AddMaterial(MaterialPresets.Emissive(e));
                    b.AddSphere(p, 0.08f, (uint)glowMat);
                }

                int pl = b.AddLight(a + new Vec3(0.0f, 2.4f, 0.0f), new Vec3(0.85f, 1.05f, 1.10f), 85f);
                b.AddOrbitingLight(pl, a, 2.2f, 2.4f, 0.7f, 0.0f);
            }

            void AddPedestalPlaza(Vec3 a)
            {
                const int count = 5;
                const float pedH = 1.1f;
                int pedMat = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.88f, 0.88f, 0.88f)));
                for (var i = 0; i < count; i++)
                {
                    float ang = i * (2f * XMath.PI) / count;
                    Vec3 p = a + new Vec3(XMath.Cos(ang) * 1.6f, 0.0f, XMath.Sin(ang) * 1.6f);
                    b.AddCylinderY(p, 0.30f, 0f, pedH, (uint)pedMat);

                    int topMat = RngInt(0, 4) switch
                    {
                        0 => b.AddMaterial(MaterialPresets.Mirror(0.90f)),
                        1 => MakeGlass(),
                        2 => b.AddMaterial(MaterialPresets.Solid(new Vec3(1.00f, 0.85f, 0.57f), 0.10f)),
                        3 => b.AddMaterial(MaterialPresets.Solid(new Vec3(0.78f, 0.60f, 0.20f), 0.08f)),
                        _ => b.AddMaterial(MaterialPresets.Solid(new Vec3(0.20f, 0.85f, 0.20f), 0.08f)),
                    };

                    int orb = b.AddSphere(p + new Vec3(0.0f, pedH + 0.38f, 0.0f), 0.36f, (uint)topMat);
                    b.AddBobbingSphere(orb, 0.07f, 1.3f, ang);
                }

                b.AddDisk(a + new Vec3(0.0f, 0.015f, 0.0f), new Vec3(0f, 1f, 0f), 2.0f,
                    (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.96f, 0.96f, 0.98f), 0.88f)));

                int l0 = b.AddLight(a + new Vec3(0.0f, 3.0f, 0.0f), new Vec3(1.00f, 0.92f, 0.80f), 55f);
                int l1 = b.AddLight(a + new Vec3(1.6f, 2.4f, 1.2f), new Vec3(0.65f, 0.95f, 1.25f), 50f);
                int l2 = b.AddLight(a + new Vec3(-1.6f, 2.4f, -1.2f), new Vec3(1.20f, 0.70f, 0.70f), 50f);
                b.AddPulsingLight(l0, 0.25f, 0.0f);
                b.AddPulsingLight(l1, 0.20f, 0.7f);
                b.AddPulsingLight(l2, 0.20f, 1.9f);
            }

            void AddTriangleField(Vec3 a)
            {
                const int n = 14;
                for (var i = 0; i < n; i++)
                {
                    Vec3 p0 = a + new Vec3(RngRange(-2.0f, 2.0f), RngRange(0.0f, 0.8f), RngRange(-2.0f, 2.0f));
                    Vec3 p1 = a + new Vec3(RngRange(-2.0f, 2.0f), RngRange(0.0f, 1.0f), RngRange(-2.0f, 2.0f));
                    Vec3 p2 = a + new Vec3(RngRange(-2.0f, 2.0f), RngRange(0.0f, 1.2f), RngRange(-2.0f, 2.0f));
                    int mat = b.AddMaterial(MaterialPresets.Solid(RandSaturated(0.15f), RngRange(0.02f, 0.08f)));
                    b.AddTriangle(p0, p1, p2, (uint)mat);
                }

                b.AddDisk(a + new Vec3(0.0f, 0.012f, 0.0f), new Vec3(0f, 1f, 0f), 2.2f,
                    (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.90f, 0.90f, 0.94f), 0.82f)));

                int l0 = b.AddLight(a + new Vec3(0.0f, 2.6f, -1.2f), new Vec3(1.15f, 0.80f, 0.75f), 60f);
                int l1 = b.AddLight(a + new Vec3(0.0f, 2.6f, 1.2f), new Vec3(0.70f, 0.95f, 1.25f), 60f);
                b.AddOrbitingLight(l0, a, 2.4f, 2.6f, 0.5f, 0.0f);
                b.AddOrbitingLight(l1, a, 2.4f, 2.6f, -0.5f, 0.9f);
            }

            void AddMeshShowcase(Vec3 a)
            {
                int cowMat = b.AddMaterial(MaterialPresets.Solid(new Vec3(1.00f, 0.85f, 0.57f), 0.10f));
                int bunnyMat = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.20f, 0.85f, 0.20f), 0.08f));
                int teapotMat = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.95f, 0.15f, 0.15f), 0.10f));
                int dragonMat = b.AddMaterial(MaterialPresets.Mirror(0.90f));

                b.AddMeshAutoGround("cow.obj", (uint)cowMat, 1.0f, a + new Vec3(-2.4f, 0.0f, 0.2f));
                b.AddMeshAutoGround("stanford-bunny.obj", (uint)bunnyMat, 1.0f, a + new Vec3(0.0f, 0.0f, 0.0f));
                b.AddMeshAutoGround("teapot.obj", (uint)teapotMat, 1.0f, a + new Vec3(2.4f, 0.0f, -0.2f));
                b.AddMeshAutoGround("xyzrgb_dragon.obj", (uint)dragonMat, 1.0f, a + new Vec3(4.8f, 0.0f, -0.6f));

                b.AddDisk(a + new Vec3(0.0f, 0.012f, 0.0f), new Vec3(0f, 1f, 0f), 3.0f,
                    (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.90f, 0.90f, 0.94f), 0.82f)));

                int l0 = b.AddLight(a + new Vec3(-1.6f, 2.4f, 1.2f), new Vec3(1.25f, 0.65f, 0.60f), 60f);
                int l1 = b.AddLight(a + new Vec3(1.6f, 2.4f, -1.2f), new Vec3(0.60f, 0.95f, 1.25f), 60f);
                int l2 = b.AddLight(a + new Vec3(0.0f, 3.0f, 0.0f), new Vec3(0.95f, 1.00f, 0.98f), 70f);
                b.AddOrbitingLight(l2, a, 2.6f, 3.0f, 0.5f, 0.0f);
                b.AddPulsingLight(l0, 0.25f, 0.0f);
                b.AddPulsingLight(l1, 0.25f, 1.0f);
            }

            void AddLightStage(Vec3 a)
            {
                int stageMat = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.85f, 0.85f, 0.90f)));
                b.AddDisk(a + new Vec3(0.0f, 0.02f, 0.0f), new Vec3(0f, 1f, 0f), 1.8f, (uint)stageMat);

                int canopyMat = b.AddMaterial(MaterialPresets.Emissive(new Vec3(3.0f, 3.0f, 3.2f)));
                b.AddDisk(a + new Vec3(0.0f, 2.4f, 0.0f), new Vec3(0f, -1f, 0f), 0.7f, (uint)canopyMat);

                int focusMat = MakeGlass();
                int focus = b.AddSphere(a + new Vec3(0.0f, 0.6f, 0.0f), 0.45f, (uint)focusMat);
                b.AddBobbingSphere(focus, 0.10f, 1.8f, 0.0f);

                int glow0 = b.AddMaterial(MaterialPresets.Emissive(new Vec3(2.8f, 0.9f, 0.7f)));
                int glow1 = b.AddMaterial(MaterialPresets.Emissive(new Vec3(0.7f, 2.6f, 3.0f)));
                int glow2 = b.AddMaterial(MaterialPresets.Emissive(new Vec3(2.6f, 2.6f, 0.9f)));
                b.AddSphere(a + new Vec3(0.9f, 0.12f, 0.0f), 0.10f, (uint)glow0);
                b.AddSphere(a + new Vec3(-0.9f, 0.12f, 0.0f), 0.10f, (uint)glow1);
                b.AddSphere(a + new Vec3(0.0f, 0.12f, 0.9f), 0.10f, (uint)glow2);

                int a0 = b.AddLight(a + new Vec3(1.2f, 2.0f, 0.8f), new Vec3(1.0f, 0.95f, 0.9f), 55f);
                int a1 = b.AddLight(a + new Vec3(-1.2f, 2.0f, -0.8f), new Vec3(0.9f, 0.95f, 1.0f), 55f);
                b.AddOrbitingLight(a0, a, 1.6f, 2.0f, 0.9f, 0.0f);
                b.AddOrbitingLight(a1, a, 1.6f, 2.0f, -0.9f, 0.6f);
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
                        0 => b.AddMaterial(MaterialPresets.Solid(new Vec3(1.00f, 0.85f, 0.57f), 0.10f)),
                        1 => b.AddMaterial(MaterialPresets.Solid(new Vec3(0.78f, 0.60f, 0.20f), 0.08f)),
                        2 => b.AddMaterial(MaterialPresets.Solid(new Vec3(0.96f, 0.58f, 0.25f), 0.08f)),
                        _ => b.AddMaterial(MaterialPresets.Mirror(0.90f)),
                    };
                    b.AddSphere(c, 0.32f, (uint)m);
                }

                int glassMat = MakeGlass();
                int centerGlass = b.AddSphere(a + new Vec3(0.0f, 0.62f, 0.0f), 0.38f, (uint)glassMat);
                b.AddBobbingSphere(centerGlass, 0.08f, 1.0f, 0.3f);
                int glowMat = b.AddMaterial(MaterialPresets.Emissive(new Vec3(2.6f, 1.0f, 0.8f)));
                b.AddSphere(a + new Vec3(0.0f, 0.62f, 0.0f), 0.12f, (uint)glowMat);

                int l0 = b.AddLight(a + new Vec3(0.0f, 2.4f, 0.0f), new Vec3(1.0f, 0.98f, 0.95f), 70f);
                int l1 = b.AddLight(a + new Vec3(1.6f, 1.8f, 0.0f), new Vec3(0.65f, 0.95f, 1.25f), 55f);
                int l2 = b.AddLight(a + new Vec3(-1.6f, 1.8f, 0.0f), new Vec3(1.20f, 0.75f, 0.70f), 55f);
                b.AddPulsingLight(l0, 0.20f, 0.0f);
                b.AddPulsingLight(l1, 0.18f, 0.6f);
                b.AddPulsingLight(l2, 0.18f, 1.2f);
            }

            void AddCheckerTerrace(Vec3 a)
            {
                const float w = 3.2f, d = 0.9f;
                int whiteMat = b.AddMaterial(MaterialPresets.Solid(new Vec3(0.88f, 0.88f, 0.90f)));
                for (var i = 0; i < 4; i++)
                {
                    float y = a.y + (i * 0.3f);
                    b.AddXZRect(a.x - w, a.x + w, a.z - (d * (i + 1)), a.z + (d * (i + 1)), y, (uint)whiteMat);

                    float z0 = a.z - (d * (i + 1));
                    float z1 = a.z + (d * (i + 1));
                    b.AddXZRect(a.x - w, a.x + w, z0, z0 + 0.08f, y + 0.001f, (uint)emissiveWarm);
                    b.AddXZRect(a.x - w, a.x + w, z1 - 0.08f, z1, y + 0.001f, (uint)emissiveCool);
                }
                int topMat = RngBool(0.5f)
                    ? b.AddMaterial(MaterialPresets.Mirror(0.92f))
                    : MakeGlass();
                int top = b.AddSphere(a + new Vec3(0.0f, (0.3f * 4) + 0.35f, 0.0f), 0.35f, (uint)topMat);
                b.AddBobbingSphere(top, 0.07f, 1.1f, 0.0f);

                int l = b.AddLight(a + new Vec3(0.0f, 2.0f, 0.0f), new Vec3(0.95f, 1.0f, 0.95f), 55f);
                b.AddPulsingLight(l, 0.22f, 0.4f);
            }

            void AddDiffuseStacks(Vec3 a)
            {
                const int stacks = 3;
                for (var i = 0; i < stacks; i++)
                {
                    Vec3 p = a + new Vec3((i - 1) * 1.4f, 0.0f, (i - 1) * 0.3f);
                    float h = RngRange(0.9f, 1.6f);
                    float r = RngRange(0.25f, 0.35f);
                    int bodyMat = b.AddMaterial(MaterialPresets.Solid(RandSaturated(0.10f), RngRange(0.02f, 0.06f)));
                    b.AddCylinderY(p, r, 0f, h, (uint)bodyMat);
                    b.AddSphere(p + new Vec3(0.0f, h + (r * 0.8f), 0.0f), r * 0.8f, (uint)bodyMat);
                    b.AddXYRect(p.x - 0.25f, p.x + 0.25f, 1.65f, 1.75f, p.z - 0.30f, (uint)emissiveCool);
                }
                int fill = b.AddLight(a + new Vec3(0.0f, 2.2f, 0.0f), new Vec3(1.00f, 0.98f, 0.95f), 60f);
                b.AddPulsingLight(fill, 0.20f, 0.0f);
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

            int g0 = b.AddLight(center + new Vec3(0.0f, 18.0f, 14.0f), new Vec3(1.0f, 0.98f, 0.95f), 330f);
            int g1 = b.AddLight(center + new Vec3(-22.0f, 14.0f, -18.0f), new Vec3(0.9f, 0.95f, 1.0f), 240f);
            int g2 = b.AddLight(center + new Vec3(22.0f, 15.0f, -6.0f), new Vec3(0.95f, 1.0f, 0.95f), 220f);
            b.AddPulsingLight(g0, 0.20f, 0.35f);
            b.AddPulsingLight(g1, 0.18f, 0.50f);
            b.AddPulsingLight(g2, 0.15f, 0.65f);

            return b.Build();
        }
    }
}
