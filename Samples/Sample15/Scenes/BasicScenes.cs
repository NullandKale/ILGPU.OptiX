using ILGPU.OptiX.Cuda;
namespace Sample15
{
    /// <summary>
    /// Small hand-authored debug/smoke-test scenes.
    /// </summary>
    public static class BasicScenes
    {
        public static SceneData BuildDebugMaterialsScene()
        {
            var b = new SceneBuilder
            {
                Name = "Debug: Materials + multi-light",
                AmbientColor = new Vec3(0.5f, 0.6f, 0.8f),
                AmbientIntensity = 0.1f,
                BackgroundTop = new Vec3(0.4f, 0.55f, 0.8f),
                BackgroundBottom = new Vec3(0.05f, 0.05f, 0.08f),
                CameraOrigin = new Vec3(0f, 3f, 9f),
                CameraLookAt = new Vec3(0f, 1.2f, -1f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 55f,
                CameraWorldScale = 10f,
            };

            // 0: checkered floor (always diffuse, matching the reference's Checker
            // closures - Metallic/Transmission stay 0).
            uint floor = (uint)b.AddMaterial(new MaterialSbtData
            {
                BaseColor = new Vec3(0.85f, 0.85f, 0.85f),
                Roughness = 1f,
                MaterialKind = MaterialSbtData.Checker,
                CheckerColorB = new Vec3(0.1f, 0.1f, 0.1f),
                CheckerScale = 1f,
                TextureWeight = 1f,
                UVScale = 1f,
            });
            // 1: solid red diffuse back wall.
            uint redWall = (uint)b.AddMaterial(new MaterialSbtData
            {
                BaseColor = new Vec3(0.75f, 0.15f, 0.12f),
                Roughness = 1f,
                MaterialKind = MaterialSbtData.Solid,
                TextureWeight = 1f,
                UVScale = 1f,
            });
            // 2: solid green diffuse floating triangle.
            uint greenTri = (uint)b.AddMaterial(new MaterialSbtData
            {
                BaseColor = new Vec3(0.15f, 0.6f, 0.2f),
                Roughness = 1f,
                MaterialKind = MaterialSbtData.Solid,
                TextureWeight = 1f,
                UVScale = 1f,
            });
            // 3: mirror panel (Roughness defaults to 0, below Bsdf.DeltaRoughnessThreshold,
            // so ShadeSurface takes the perfect-mirror fast path regardless of Metallic).
            uint mirror = (uint)b.AddMaterial(new MaterialSbtData
            {
                BaseColor = new Vec3(0.95f, 0.95f, 0.95f),
                Metallic = 0.95f,
                MaterialKind = MaterialSbtData.Solid,
            });
            // 4: glass panel (Transmission > 0 dispatches to the dielectric branch) -
            // exercises the M2 bounce loop and multi-hit shadow transmittance.
            uint glass = (uint)b.AddMaterial(MaterialPresets.Glass(1.5f, new Vec3(0.9f, 0.97f, 0.95f)));
            // Frosted glass - same IOR/tint as the clear glass sphere above, just
            // with TransmissionRoughness > 0, so the two can be compared side by
            // side for plausible frosted transmission.
            uint frostedGlass = (uint)b.AddMaterial(new MaterialSbtData
            {
                BaseColor = new Vec3(1f, 1f, 1f),
                Transmission = 1f,
                TransmissionRoughness = 0.35f,
                IOR = 1.5f,
                TransmissionColor = new Vec3(0.9f, 0.97f, 0.95f),
                MaterialKind = MaterialSbtData.Solid,
            });
            // 5: blue sphere.
            uint blue = (uint)b.AddMaterial(new MaterialSbtData
            {
                BaseColor = new Vec3(0.2f, 0.35f, 0.85f),
                Roughness = 1f,
                MaterialKind = MaterialSbtData.Solid,
                TextureWeight = 1f,
                UVScale = 1f,
            });
            uint yellow = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.9f, 0.8f, 0.1f)));   // 6: cylinder
            uint orange = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.9f, 0.45f, 0.1f)));  // 7: box
            uint purple = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.55f, 0.2f, 0.75f))); // 8: disk
            uint cyan = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.15f, 0.75f, 0.75f)));  // 9: XYRect
            uint magenta = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.8f, 0.2f, 0.6f)));  // 10: XZRect
            uint teal = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.1f, 0.55f, 0.45f)));   // 11: YZRect

            // Three spheres near the floor, between the mirror/glass panels - exercises
            // the material set against the mesh-sphere geometry (mirror, glass, diffuse).
            b.AddSphereMesh(new Vec3(-1.2f, 0.8f, 1f), 0.8f, mirror);
            b.AddSphereMesh(new Vec3(1.2f, 0.8f, 1f), 0.8f, glass);
            b.AddSphereMesh(new Vec3(2.9f, 0.6f, 1.4f), 0.6f, frostedGlass);
            b.AddSphereMesh(new Vec3(0f, 0.6f, 2.5f), 0.6f, blue);

            // One of each mesh-primitive kind, scattered near the existing geometry for
            // visual verification.
            b.AddBoxMesh(new Vec3(-5.5f, 0f, 2.5f), new Vec3(-4.5f, 1f, 3.5f), orange);
            b.AddCylinderMesh(new Vec3(5f, 0f, 3f), 0.5f, 0f, 1.5f, yellow);
            b.AddDiskMesh(new Vec3(0f, 0.02f, 6f), new Vec3(0f, 1f, 0f), 1f, purple);
            // XYRect(-6,-4, 1,2, c=-3): X in [-6,-4], Y in [1,2], Z = -3, facing +Z.
            b.AddQuad(new Vec3(-6f, 1f, -3f), new Vec3(-4f, 1f, -3f), new Vec3(-4f, 2f, -3f), new Vec3(-6f, 2f, -3f), new Vec3(0f, 0f, 1f), cyan);
            // XZRect(4,6, 1,3, c=2.5): X in [4,6], Z in [1,3], Y = 2.5, facing +Y.
            b.AddQuad(new Vec3(4f, 2.5f, 3f), new Vec3(6f, 2.5f, 3f), new Vec3(6f, 2.5f, 1f), new Vec3(4f, 2.5f, 1f), new Vec3(0f, 1f, 0f), magenta);
            // YZRect(0,1, 3,5, c=6.5): Y in [0,1], Z in [3,5], X = 6.5, facing +X.
            b.AddQuad(new Vec3(6.5f, 0f, 3f), new Vec3(6.5f, 1f, 3f), new Vec3(6.5f, 1f, 5f), new Vec3(6.5f, 0f, 5f), new Vec3(1f, 0f, 0f), teal);

            // Floor: X in [-8, 8], Z in [-8, 8], Y = 0, facing up.
            b.AddQuad(
                new Vec3(-8f, 0f, 8f), new Vec3(8f, 0f, 8f),
                new Vec3(8f, 0f, -8f), new Vec3(-8f, 0f, -8f),
                new Vec3(0f, 1f, 0f), floor);

            // Back wall: X in [-4, 4], Y in [0, 4], Z = -4, facing +Z (toward the camera).
            b.AddQuad(
                new Vec3(-4f, 0f, -4f), new Vec3(4f, 0f, -4f),
                new Vec3(4f, 4f, -4f), new Vec3(-4f, 4f, -4f),
                new Vec3(0f, 0f, 1f), redWall);

            // Small floating triangle for shading variety.
            b.AddTriangle(
                new Vec3(2f, 1.5f, -1f), new Vec3(3f, 1.5f, -1f), new Vec3(2.5f, 2.5f, -1f),
                new Vec3(0f, 0f, 1f), greenTri);

            // Mirror panel, facing +X (toward scene center) - exercises the mirror
            // bounce branch.
            b.AddQuad(
                new Vec3(-3f, 0f, -3f), new Vec3(-3f, 0f, 1f),
                new Vec3(-3f, 3f, 1f), new Vec3(-3f, 3f, -3f),
                new Vec3(1f, 0f, 0f), mirror);

            // Glass panel, facing -X (toward scene center) - exercises the dielectric
            // reflect/refract branch and multi-hit shadow transmittance.
            b.AddQuad(
                new Vec3(3f, 0f, -3f), new Vec3(3f, 0f, 1f),
                new Vec3(3f, 3f, 1f), new Vec3(3f, 3f, -3f),
                new Vec3(-1f, 0f, 0f), glass);

            b.AddLight(new Vec3(-3f, 4f, 3f), new Vec3(1f, 0.85f, 0.6f), 40f);
            b.AddLight(new Vec3(4f, 5f, -2f), new Vec3(0.6f, 0.75f, 1f), 40f);

            return b.Build();
        }

        // Direct port of the reference's Scenes.BuildTestScene() - four spheres (three
        // diffuse, one mirror) over a near-black background, lit by two point lights and
        // a low fill ambient. The simplest possible smoke test for the sphere primitive.
        public static SceneData BuildSimpleTestScene()
        {
            var b = new SceneBuilder
            {
                Name = "Simple test (4 spheres)",
                AmbientColor = new Vec3(1f, 1f, 1f),
                AmbientIntensity = 0.1f,
                BackgroundTop = new Vec3(0.05f, 0.05f, 0.05f),
                BackgroundBottom = new Vec3(0.05f, 0.05f, 0.05f),
                CameraOrigin = new Vec3(0f, 1f, 0f),
                CameraLookAt = new Vec3(0f, 1f, -10f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 45f,
                CameraWorldScale = 6f,
            };

            uint red = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(1f, 0f, 0f)));
            uint green = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0f, 1f, 0f)));
            uint blue = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0f, 0f, 1f)));
            uint mirror = (uint)b.AddMaterial(MaterialPresets.Mirror(0.9f));

            b.AddSphereMesh(new Vec3(-1.2f, 0.9f, -2.2f), 0.9f, red);
            b.AddSphereMesh(new Vec3(1.2f, 0.9f, -2.2f), 0.9f, green);
            b.AddSphereMesh(new Vec3(-1.2f, 0.9f, -3.6f), 0.9f, blue);
            b.AddSphereMesh(new Vec3(1.2f, 0.9f, -3.6f), 0.9f, mirror);

            b.AddLight(new Vec3(0f, 3.2f, -2.9f), new Vec3(1f, 1f, 1f), 140f);
            b.AddLight(new Vec3(-2.2f, 2.0f, -2.4f), new Vec3(1f, 1f, 1f), 60f);

            return b.Build();
        }

        // Ambient-only lit, single textured quad - reuses Sample08's CudaTextureObject/
        // CudaTex2D pattern (trivial) via one texture already bundled for Sponza,
        // rather than fetching a new asset.
        public static SceneData BuildTextureTestScene()
        {
            var b = new SceneBuilder
            {
                Name = "Debug: Texture (ambient-only)",
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

            uint textured = (uint)b.AddMaterial(new MaterialSbtData
            {
                BaseColor = new Vec3(1f, 1f, 1f),
                Roughness = 1f,
                MaterialKind = MaterialSbtData.Solid,
                TextureWeight = 1f,
                UVScale = 1f,
            }, "models/sponza/textures/spnza_bricks_a_diff.png");

            b.AddQuad(
                new Vec3(-3f, 0f, 3f), new Vec3(3f, 0f, 3f),
                new Vec3(3f, 3f, 3f), new Vec3(-3f, 3f, 3f),
                new Vec3(0f, 0f, 1f), textured);

            return b.Build();
        }

        // GGX validation scene: a grid of spheres
        // sweeping Roughness across columns (0 = mirror-like at the left edge, 1 = fully
        // diffuse-like at the right) and Metallic across rows (dielectric red on top,
        // conductor gold on the bottom), lit only by point lights (no NEE/env light of
        // any other kind exists yet at M2) - isolates whether the GGX lobe itself is
        // energy-conserving and free of fireflies/NaNs at grazing angles before NEE+MIS
        // (M4) or HDRI lighting (M5) are layered on top. Each (row, column) material
        // also gets a Stanford Bunny instance behind its sphere, so the same material
        // grid is validated against actual triangle/vertex-normal shading too, not
        // just custom-primitive analytic normals.
        public static SceneData BuildGgxRoughnessSweepScene()
        {
            var b = new SceneBuilder
            {
                Name = "Debug: GGX roughness x metallic sweep",
                AmbientColor = new Vec3(1f, 1f, 1f),
                AmbientIntensity = 0.1f,
                BackgroundTop = new Vec3(0.1f, 0.1f, 0.12f),
                BackgroundBottom = new Vec3(0.02f, 0.02f, 0.03f),
                // Widened/pulled back from the original two-row framing - a third
                // (glass) row and the per-material Bunny instances roughly triple
                // this scene's vertical and depth extent.
                CameraOrigin = new Vec3(0f, 4.5f, 15f),
                CameraLookAt = new Vec3(0f, 2.5f, -3f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 50f,
                CameraWorldScale = 8f,
            };

            uint floor = (uint)b.AddMaterial(MaterialPresets.Checker(new Vec3(0.5f, 0.5f, 0.5f), new Vec3(0.08f, 0.08f, 0.08f), 0.8f));
            // XZRect(-10,10, -10,6, c=0): X in [-10,10], Z in [-10,6], Y = 0, facing +Y.
            b.AddQuad(new Vec3(-10f, 0f, 6f), new Vec3(10f, 0f, 6f), new Vec3(10f, 0f, -10f), new Vec3(-10f, 0f, -10f), new Vec3(0f, 1f, 0f), floor);

            const int columns = 7;
            const int rows = 2;
            Vec3[] rowBaseColor = { new Vec3(0.8f, 0.15f, 0.12f), new Vec3(1.0f, 0.78f, 0.35f) }; // dielectric red, conductor gold
            float[] rowMetallic = { 0f, 1f };

            // Each material-sweep row gets both a sphere and a real-mesh (Stanford
            // Bunny) instance of the *same* material per column - one material grid
            // validated against both the analytic custom-primitive normal (sphere)
            // and actual triangle/vertex-normal shading (bunny), not two separate
            // color palettes. Bunnies sit on the floor behind their matching sphere
            // row (offset in Z, not Y, since AddMesh always ground-places a mesh).
            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < columns; col++)
                {
                    float roughness = col / (float)(columns - 1);
                    float x = -6f + (col * 2f);
                    float sphereY = 1f + (row * 2.2f);
                    float meshZ = -3f - (row * 3f);

                    uint mat = (uint)b.AddMaterial(new MaterialSbtData
                    {
                        BaseColor = rowBaseColor[row],
                        Metallic = rowMetallic[row],
                        Roughness = roughness,
                        MaterialKind = MaterialSbtData.Solid,
                    });
                    b.AddSphereMesh(new Vec3(x, sphereY, 0f), 0.9f, mat);
                    b.AddMesh("stanford-bunny.obj", new Vec3(x, 0f, meshZ), mat, targetSize: 1.8f);
                }
            }

            // Third row: rough dielectric (glass) sweep - same column layout,
            // sweeping TransmissionRoughness
            // instead of the opaque Roughness/Metallic pair (0 = clear glass at the
            // left edge, 1 = fully frosted at the right), each column again getting
            // both a sphere and a matching Bunny instance of the same material.
            const int glassRow = rows;
            for (int col = 0; col < columns; col++)
            {
                float transmissionRoughness = col / (float)(columns - 1);
                float x = -6f + (col * 2f);
                float sphereY = 1f + (glassRow * 2.2f);
                float meshZ = -3f - (glassRow * 3f);

                uint mat = (uint)b.AddMaterial(new MaterialSbtData
                {
                    BaseColor = new Vec3(1f, 1f, 1f),
                    Transmission = 1f,
                    TransmissionRoughness = transmissionRoughness,
                    IOR = 1.5f,
                    TransmissionColor = new Vec3(0.9f, 0.95f, 1f),
                    MaterialKind = MaterialSbtData.Solid,
                });
                b.AddSphereMesh(new Vec3(x, sphereY, 0f), 0.9f, mat);
                b.AddMesh("stanford-bunny.obj", new Vec3(x, 0f, meshZ), mat, targetSize: 1.8f);
            }

            b.AddLight(new Vec3(-4f, 6f, 6f), new Vec3(1f, 0.97f, 0.92f), 220f);
            b.AddLight(new Vec3(5f, 4f, 4f), new Vec3(0.9f, 0.94f, 1f), 140f);

            return b.Build();
        }

        // Pure point-light demo - AmbientIntensity=0, no env map, near-black background,
        // so ShadeSurface's flat unshadowed ambient term (see MaterialShading.cs's own
        // comment on it being a pre-GGX hack with no occlusion) can't wash out the NEE
        // shadowing/falloff this scene exists to show off cleanly.
        public static SceneData BuildNoAmbientLightDemoScene()
        {
            var b = new SceneBuilder
            {
                Name = "Debug: Point lights only (no ambient)",
                AmbientColor = new Vec3(0f, 0f, 0f),
                AmbientIntensity = 0f,
                BackgroundTop = new Vec3(0f, 0f, 0f),
                BackgroundBottom = new Vec3(0f, 0f, 0f),
                CameraOrigin = new Vec3(0f, 2.5f, 8f),
                CameraLookAt = new Vec3(0f, 1f, 0f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 50f,
                CameraWorldScale = 8f,
            };

            uint floor = (uint)b.AddMaterial(MaterialPresets.Checker(new Vec3(0.6f, 0.6f, 0.6f), new Vec3(0.05f, 0.05f, 0.05f), 1f));
            b.AddQuad(
                new Vec3(-8f, 0f, 6f), new Vec3(8f, 0f, 6f),
                new Vec3(8f, 0f, -6f), new Vec3(-8f, 0f, -6f),
                new Vec3(0f, 1f, 0f), floor);

            uint white = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.85f, 0.85f, 0.85f)));
            uint red = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.85f, 0.15f, 0.15f)));
            uint gold = (uint)b.AddMaterial(new MaterialSbtData { BaseColor = new Vec3(1f, 0.85f, 0.55f), Metallic = 1f, Roughness = 0.2f, MaterialKind = MaterialSbtData.Solid });

            b.AddSphereMesh(new Vec3(-2.2f, 1f, 0f), 1f, white);
            b.AddSphereMesh(new Vec3(0.2f, 0.8f, -1.5f), 0.8f, red);
            b.AddBoxMesh(new Vec3(1.6f, 0f, -0.5f), new Vec3(3.2f, 1.6f, 1.1f), gold);

            // Three colored point lights, no ambient/env-map competitor - each object's
            // shadow and falloff should read entirely from these.
            b.AddLight(new Vec3(-4f, 4f, 3f), new Vec3(1f, 0.4f, 0.3f), 60f);
            b.AddLight(new Vec3(3f, 5f, -2f), new Vec3(0.3f, 0.5f, 1f), 70f);
            b.AddLight(new Vec3(0f, 3f, 5f), new Vec3(0.4f, 1f, 0.5f), 45f);

            return b.Build();
        }
    }
}
