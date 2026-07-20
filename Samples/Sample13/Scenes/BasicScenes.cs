using ILGPU.OptiX.Cuda;
namespace Sample13
{
    /// <summary>
    /// Small hand-authored debug/smoke-test scenes.
    /// </summary>
    public static class BasicScenes
    {
        public static SceneData BuildDebugOrenNayarScene()
        {
            var b = new SceneBuilder
            {
                Name = "Debug: Oren-Nayar + multi-light",
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

            // 0: checkered floor (always diffuse, matching the reference's Checker
            // closures - Reflectivity/Transparency stay 0).
            uint floor = (uint)b.AddMaterial(new MaterialSbtData
            {
                Albedo = new Vec3(0.85f, 0.85f, 0.85f),
                MaterialKind = MaterialSbtData.Checker,
                CheckerColorB = new Vec3(0.1f, 0.1f, 0.1f),
                CheckerScale = 1f,
                TextureWeight = 1f,
                UVScale = 1f,
            });
            // 1: solid red diffuse back wall.
            uint redWall = (uint)b.AddMaterial(new MaterialSbtData
            {
                Albedo = new Vec3(0.75f, 0.15f, 0.12f),
                MaterialKind = MaterialSbtData.Solid,
                TextureWeight = 1f,
                UVScale = 1f,
            });
            // 2: solid green diffuse floating triangle.
            uint greenTri = (uint)b.AddMaterial(new MaterialSbtData
            {
                Albedo = new Vec3(0.15f, 0.6f, 0.2f),
                MaterialKind = MaterialSbtData.Solid,
                TextureWeight = 1f,
                UVScale = 1f,
            });
            // 3: mirror panel (Reflectivity >= 0.9 dispatches to the mirror branch in
            // __closest__radiance).
            uint mirror = (uint)b.AddMaterial(new MaterialSbtData
            {
                Albedo = new Vec3(0.95f, 0.95f, 0.95f),
                Reflectivity = 0.95f,
                MaterialKind = MaterialSbtData.Solid,
            });
            // 4: glass panel (Transparency > 0 dispatches to the dielectric branch) -
            // exercises the bounce loop and multi-hit shadow transmittance.
            uint glass = (uint)b.AddMaterial(MaterialPresets.Glass(1.5f, new Vec3(0.9f, 0.97f, 0.95f)));
            // 5: blue sphere.
            uint blue = (uint)b.AddMaterial(new MaterialSbtData
            {
                Albedo = new Vec3(0.2f, 0.35f, 0.85f),
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
            // the custom-primitive pipeline against materials already proven on
            // triangles (mirror, glass, diffuse).
            b.AddSphere(new Vec3(-1.2f, 0.8f, 1f), 0.8f, mirror);
            b.AddSphere(new Vec3(1.2f, 0.8f, 1f), 0.8f, glass);
            b.AddSphere(new Vec3(0f, 0.6f, 2.5f), 0.6f, blue);

            // One of each remaining custom-primitive kind, scattered near the existing
            // geometry for visual verification.
            b.AddBox(new Vec3(-5.5f, 0f, 2.5f), new Vec3(-4.5f, 1f, 3.5f), orange);
            b.AddCylinderY(new Vec3(5f, 0f, 3f), 0.5f, 0f, 1.5f, yellow);
            b.AddDisk(new Vec3(0f, 0.02f, 6f), new Vec3(0f, 1f, 0f), 1f, purple);
            b.AddXYRect(-6f, -4f, 1f, 2f, -3f, cyan);
            b.AddXZRect(4f, 6f, 1f, 3f, 2.5f, magenta);
            b.AddYZRect(0f, 1f, 3f, 5f, 6.5f, teal);

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
        // almost no ambient. The simplest possible smoke test for the sphere primitive.
        public static SceneData BuildSimpleTestScene()
        {
            var b = new SceneBuilder
            {
                Name = "Simple test (4 spheres)",
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

            uint red = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(1f, 0f, 0f)));
            uint green = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0f, 1f, 0f)));
            uint blue = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0f, 0f, 1f)));
            uint mirror = (uint)b.AddMaterial(MaterialPresets.Mirror(0.9f));

            b.AddSphere(new Vec3(-1.2f, 0.9f, -2.2f), 0.9f, red);
            b.AddSphere(new Vec3(1.2f, 0.9f, -2.2f), 0.9f, green);
            b.AddSphere(new Vec3(-1.2f, 0.9f, -3.6f), 0.9f, blue);
            b.AddSphere(new Vec3(1.2f, 0.9f, -3.6f), 0.9f, mirror);

            b.AddLight(new Vec3(0f, 3.2f, -2.9f), new Vec3(1f, 1f, 1f), 140f);
            b.AddLight(new Vec3(-2.2f, 2.0f, -2.4f), new Vec3(1f, 1f, 1f), 60f);

            return b.Build();
        }

        // Ambient-only lit, single textured quad - a trivial CudaTextureObject/
        // CudaTex2D usage via one texture already bundled for Sponza, rather than
        // fetching a new asset.
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
                Albedo = new Vec3(1f, 1f, 1f),
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
    }
}
