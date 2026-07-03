namespace Sample14
{
    /// <summary>
    /// Volume-grid (voxel DDA) test scene.
    /// </summary>
    public static class VolumeScenes
    {
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
            var b = new SceneBuilder
            {
                Name = "Volume grid room (DDA)",
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
            uint stone = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.82f, 0.82f, 0.85f)));  // voxel 1: stone/white
            uint red = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.95f, 0.15f, 0.15f)));    // voxel 2: red
            uint green = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.15f, 0.95f, 0.20f)));  // voxel 3: green
            uint blue = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.15f, 0.25f, 0.95f)));   // voxel 4: blue
            uint mirror = (uint)b.AddMaterial(MaterialPresets.Mirror(0.9f));                          // voxel 5: mirror
            // Extra materials for the pedestals/spheres placed around the grid, below.
            uint pedestal = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.85f, 0.85f, 0.85f)));
            uint glass = (uint)b.AddMaterial(MaterialPresets.ClearGlass());

            b.SetVolumeGrid(voxels, new Vec3(-4f, 0f, -6f), new Vec3(0.5f, 0.5f, 0.5f), new Vec3i(nx, ny, nz));

            const float pedR = 0.25f, pedH = 1.2f, sphR = 0.35f;
            Vec3 posL = new Vec3(-1.6f, 0f, -2.0f);
            Vec3 posR = new Vec3(1.6f, 0f, -2.0f);
            Vec3 posF = new Vec3(0f, 0f, -0.8f);
            Vec3 posB = new Vec3(0f, 0f, -3.2f);

            b.AddCylinderY(posL, pedR, 0f, pedH, pedestal);
            b.AddCylinderY(posR, pedR, 0f, pedH, pedestal);
            b.AddCylinderY(posF, pedR, 0f, pedH, pedestal);
            b.AddCylinderY(posB, pedR, 0f, pedH, pedestal);

            b.AddSphere(posL + new Vec3(0f, pedH + sphR, 0f), sphR, mirror);
            b.AddSphere(posR + new Vec3(0f, pedH + sphR, 0f), sphR, red);
            b.AddSphere(posF + new Vec3(0f, pedH + sphR, 0f), sphR, blue);
            b.AddSphere(posB + new Vec3(0f, pedH + sphR, 0f), sphR, green);
            b.AddSphere(new Vec3(0f, 2f, -2.0f), 0.5f, glass);

            b.AddLight(new Vec3(0f, 5f, -3f), new Vec3(1f, 1f, 1f), 220f);
            b.AddLight(new Vec3(-2.5f, 3f, -1.8f), new Vec3(1f, 0.95f, 0.9f), 90f);

            return b.Build();
        }
    }
}
