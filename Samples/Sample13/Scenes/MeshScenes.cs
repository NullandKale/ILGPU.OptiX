using System;
using System.IO;

namespace Sample13
{
    /// <summary>
    /// OBJ mesh scenes: one scene per bundled mesh plus the combined "all meshes" scene.
    /// </summary>
    public static class MeshScenes
    {
        // Shared by every single-mesh scene (M6) - reuses the existing triangle GAS
        // pipeline as-is (meshes are pure triangle geometry, no new subsystem needed),
        // auto-fits the camera/lights to each mesh's own bounding box since these 4 OBJs
        // (cow, stanford-bunny, teapot, xyzrgb_dragon) are at
        // very different real-world scales. Keeps the mesh's own vertex arrays directly
        // (no SceneBuilder re-accumulation needed for a single unmodified mesh).
        private static SceneData BuildMeshScene(string name, string objFileName, MaterialSbtData material, Vec3 lightColorA, Vec3 lightColorB)
        {
            string objPath = Path.Combine(AppContext.BaseDirectory, "models", "meshes", objFileName);
            var mesh = OBJModel.Load(objPath);

            SceneBuilder.ComputeBounds(mesh.Vertices, out Vec3 min, out Vec3 max);
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
            MaterialPresets.Solid(new Vec3(0.83f, 0.68f, 0.21f)),
            new Vec3(1f, 0.95f, 0.85f), new Vec3(0.6f, 0.7f, 1f));

        public static SceneData BuildBunnyScene() => BuildMeshScene(
            "Mesh: Stanford Bunny",
            "stanford-bunny.obj",
            MaterialPresets.Solid(new Vec3(0.2f, 0.6f, 0.35f)),
            new Vec3(1f, 0.95f, 0.85f), new Vec3(0.6f, 0.7f, 1f));

        public static SceneData BuildTeapotScene() => BuildMeshScene(
            "Mesh: Teapot",
            "teapot.obj",
            MaterialPresets.Solid(new Vec3(0.6f, 0.05f, 0.08f)),
            new Vec3(1f, 0.95f, 0.85f), new Vec3(0.6f, 0.7f, 1f));

        // Mirror (Reflectivity >= 0.9) rather than diffuse - exercises the M2 bounce
        // loop against a real high-poly mesh, matching the reference's own
        // sapphire-mirror dragon.
        public static SceneData BuildDragonScene() => BuildMeshScene(
            "Mesh: XYZRGB Dragon (mirror)",
            "xyzrgb_dragon.obj",
            MaterialPresets.Solid(new Vec3(0.85f, 0.9f, 0.95f), 0.95f),
            new Vec3(1f, 0.95f, 0.85f), new Vec3(0.6f, 0.7f, 1f));

        // Direct port of the reference's MeshScenes.BuildAllMeshesScene() - all four
        // meshes normalized to a common size and placed side-by-side over one floor.
        // The reference recenters each mesh at its largest-connected-component's
        // triangle-centroid purely to compute placement (not to filter the actual
        // rendered triangles) - approximated here with a plain bounding-box center,
        // which places these well-formed meshes equivalently for rendering purposes.
        public static SceneData BuildAllMeshesScene()
        {
            var b = new SceneBuilder
            {
                Name = "All meshes (cow/bunny/teapot/dragon)",
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

            uint cow = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.80f, 0.45f, 0.25f)));           // copper
            uint bunny = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.0f, 0.5f, 0.5f)));            // jade
            uint teapot = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.9f, 0.9f, 0.0f), 0.06f));    // gold
            uint dragon = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.88f, 0.0f, 0.88f), 0.65f));  // amethyst (below the 0.9 mirror threshold - renders diffuse)
            uint floor = (uint)b.AddMaterial(MaterialPresets.Checker(new Vec3(0.82f, 0.82f, 0.82f), new Vec3(0.15f, 0.15f, 0.15f), 0.7f));

            b.AddMesh("cow.obj", new Vec3(-3.2f, 0f, -4.0f), cow);
            b.AddMesh("stanford-bunny.obj", new Vec3(-1.0f, 0f, -3.0f), bunny);
            b.AddMesh("teapot.obj", new Vec3(1.6f, 0f, -3.2f), teapot);
            b.AddMesh("xyzrgb_dragon.obj", new Vec3(3.2f, 0f, -4.6f), dragon);

            b.AddXZRect(-10f, 10f, -10f, 4f, 0f, floor);

            b.AddLight(new Vec3(0f, 6f, 0f), new Vec3(1f, 0.95f, 0.88f), 65f);
            b.AddLight(new Vec3(-3f, 3f, 2f), new Vec3(0.85f, 0.90f, 1.0f), 35f);

            return b.Build();
        }

        public static SceneData BuildSponzaScene()
        {
            string objPath = Path.Combine(AppContext.BaseDirectory, "models", "sponza", "sponza.obj");
            Console.WriteLine($"[Sponza] BuildSponzaScene starting, objPath={objPath}");
            var mesh = OBJModel.Load(objPath);

            SceneBuilder.ComputeBounds(mesh.Vertices, out Vec3 min, out Vec3 max);
            Vec3 center = (min + max) / 2f;
            float radius = (max - min).length() * 0.5f;

            // The bounding-box diagonal (radius above) is much larger than any single
            // room/hall dimension, so using it directly to offset the camera places it
            // far outside the building. Instead position the camera inside the atrium,
            // near floor height, looking down the long axis (X, per the actual OBJ
            // bounds) toward the far end - CameraWorldScale still uses the diagonal
            // radius so WASD/orbit speed scales correctly with the model's real size.
            float sizeX = max.x - min.x;
            float sizeY = max.y - min.y;
            float eyeHeight = min.y + (sizeY * 0.12f);

            var b = new SceneBuilder
            {
                Name = "Sponza",
                // Lower than the other mesh scenes' ambient - Sponza is an enclosed
                // building (most of the scene is indoors/shadowed), so a bright uniform
                // ambient floor washes out the shadowed interior instead of just filling
                // in outdoor sky-facing surfaces the way it does for a single object on
                // an open floor.
                AmbientColor = new Vec3(0.3f, 0.3f, 0.35f),
                AmbientIntensity = 0.025f,
                BackgroundTop = new Vec3(0.15f, 0.15f, 0.2f),
                BackgroundBottom = new Vec3(0.05f, 0.05f, 0.08f),
                CameraOrigin = new Vec3(min.x + (sizeX * 0.15f), eyeHeight, center.z),
                CameraLookAt = new Vec3(max.x - (sizeX * 0.15f), eyeHeight, center.z),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 60f,
                CameraWorldScale = radius,
            };

            Console.WriteLine($"[Sponza] Processing {mesh.Materials.Length} materials");

            // Add materials from the OBJ file, using their textures
            var materialIdMap = new uint[mesh.Materials.Length];
            string modelDir = Path.GetDirectoryName(objPath) ?? ".";
            string baseDir = AppContext.BaseDirectory;

            for (int i = 0; i < mesh.Materials.Length; i++)
            {
                var objMat = mesh.Materials[i];
                string relativePath = null;

                // If material has a texture, compute relative path for TextureCache.GetOrLoad()
                if (objMat.DiffuseTexturePath != null)
                {
                    string fullTexturePath = Path.Combine(modelDir, objMat.DiffuseTexturePath);
                    Console.WriteLine($"[Sponza]   Material {i} '{objMat.Name}': checking texture {fullTexturePath}");
                    if (File.Exists(fullTexturePath))
                    {
                        // TextureCache expects paths relative to AppContext.BaseDirectory
                        // Manually compute relative path for .NET Framework compatibility
                        fullTexturePath = Path.GetFullPath(fullTexturePath);
                        baseDir = Path.GetFullPath(baseDir);
                        if (fullTexturePath.StartsWith(baseDir, StringComparison.OrdinalIgnoreCase))
                        {
                            relativePath = fullTexturePath.Substring(baseDir.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                            Console.WriteLine($"[Sponza]     FOUND! relativePath={relativePath}");
                        }
                        else
                        {
                            Console.WriteLine($"[Sponza]     Path not under baseDir, skipping");
                        }
                    }
                    else
                    {
                        Console.WriteLine($"[Sponza]     NOT FOUND");
                    }
                }
                else
                {
                    Console.WriteLine($"[Sponza]   Material {i} '{objMat.Name}': no texture path");
                }

                // These three materials are alpha-cutout in the original asset (MTL
                // "d 1.000000" + a separate map_d mask) - the mask files themselves
                // aren't bundled, but their diffuse PNGs already carry the same cutout
                // shape baked into their own alpha channel, so AlphaCutoff alone is
                // enough (see __anyhit__radiance/__anyhit__shadow). A no-op for
                // Material__57/chain here since their textures aren't bundled and
                // TextureObject stays 0.
                bool isAlphaCutout = objMat.Name == "leaf" || objMat.Name == "Material__57" || objMat.Name == "chain";

                materialIdMap[i] = (uint)b.AddMaterial(
                    new MaterialSbtData
                    {
                        Albedo = objMat.Diffuse,
                        MaterialKind = MaterialSbtData.Solid,
                        TextureWeight = relativePath != null ? 1f : 0f,
                        UVScale = 1f,
                        AlphaCutoff = isAlphaCutout ? 0.5f : 0f,
                    },
                    relativePath);
            }

            Console.WriteLine($"[Sponza] Adding mesh with {mesh.Indices.Length} triangles");
            // Add the mesh with per-triangle material assignments
            b.AddMeshWithPerTriangleMaterials(mesh, materialIdMap);

            // The bounding-box diagonal (radius) is a whole-building-scale number - using
            // it directly for light offset/intensity (as this used to) puts lights
            // hundreds of units above the actual roof (maxY) and, for intensity, several
            // million units bright (radius^2), wildly overexposing anything nearby. Use
            // the vertical extent (sizeY, the "room-scale" dimension) instead, and place
            // lights inside the hall below the roofline, spread along the long (X) axis
            // since Sponza's atrium is one long corridor rather than a single room.
            float lightHeight = min.y + (sizeY * 0.75f);
            float lightIntensity = sizeY * sizeY * 0.4f;
            b.AddLight(new Vec3(min.x + (sizeX * 0.2f), lightHeight, center.z), new Vec3(1f, 0.95f, 0.88f), lightIntensity);
            b.AddLight(new Vec3(min.x + (sizeX * 0.5f), lightHeight, center.z), new Vec3(1f, 0.95f, 0.88f), lightIntensity);
            b.AddLight(new Vec3(min.x + (sizeX * 0.8f), lightHeight, center.z), new Vec3(0.9f, 0.93f, 1.0f), lightIntensity);

            Console.WriteLine($"[Sponza] BuildSponzaScene complete");
            return b.Build();
        }
    }
}
