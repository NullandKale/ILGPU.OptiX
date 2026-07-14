using System;

namespace Sample13
{
    /// <summary>
    /// Direct ports of the reference renderer's primitive-showcase scenes: random
    /// spheres, Cornell box, mirror spheres, cylinders/disks, and boxes.
    /// </summary>
    public static class ShowcaseScenes
    {
        // Direct port of the reference's Scenes.BuildDemoScene() - a checkered floor, an
        // emissive "sun" sphere, 3 hand-placed base spheres (red/blue/mirror), and 100
        // randomly scattered small spheres (unseeded System.Random + HSV color
        // randomization + rejection sampling against previously-placed spheres, exactly
        // matching the reference so it looks different every run). The reference's
        // position-range code comments are stale; the literal X in [-9,0), Z in
        // [-9.8,-5.2) bounds below are what the reference code actually executes.
        public static SceneData BuildDemoScene()
        {
            var b = new SceneBuilder
            {
                Name = "Demo scene (random spheres)",
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

            uint red = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.9f, 0.2f, 0.2f), 0.2f));
            uint blue = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.2f, 0.2f, 0.9f), 0.5f));
            uint mirror = (uint)b.AddMaterial(new MaterialSbtData { Albedo = new Vec3(0.95f, 0.95f, 0.95f), Reflectivity = 0.9f, MaterialKind = MaterialSbtData.Solid });
            uint sun = (uint)b.AddMaterial(new MaterialSbtData { Albedo = new Vec3(1f, 1f, 1f), Emission = new Vec3(8f, 8f, 8f), MaterialKind = MaterialSbtData.Solid });

            uint floor = (uint)b.AddMaterial(MaterialPresets.Checker(new Vec3(0.8f, 0.8f, 0.8f), new Vec3(0.1f, 0.1f, 0.1f), 0.5f));
            b.AddXZRect(-10f, 10f, -11f, 2f, 0f, floor);

            b.AddSphere(new Vec3(-1.2f, 1f, 0f), 1f, red);
            b.AddSphere(new Vec3(1.2f, 1f, -0.5f), 1f, blue);
            b.AddSphere(new Vec3(0f, 0.5f, -2.5f), 0.5f, mirror);
            b.AddSphere(new Vec3(0f, 5f, 2f), 0.5f, sun);

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
                    foreach (var existing in b.Spheres)
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
                    Vec3 color = MaterialPresets.HsvToRgb(hue, 0.65f, 0.9f);
                    bool isReflective = rng.NextDouble() < 0.2;
                    uint material = (uint)b.AddMaterial(MaterialPresets.Solid(color, isReflective ? 0.6f : 0.05f));
                    b.AddSphere(center, radius, material);
                    break;
                }
            }

            b.AddLight(new Vec3(-2f, 4f, 3f), new Vec3(1f, 0.9f, 0.8f), 60f);
            b.AddLight(new Vec3(2.5f, 3.5f, -1.5f), new Vec3(0.8f, 0.9f, 1f), 40f);

            return b.Build();
        }

        // Direct port of the reference's Scenes.BuildCornellBox() - the classic
        // red/green/white box with a ceiling light panel and two boxes, using our
        // XYRect/XZRect/YZRect custom primitives in place of the reference's infinite
        // Planes (finite rects, sized to the box).
        public static SceneData BuildCornellBox()
        {
            var b = new SceneBuilder
            {
                Name = "Cornell box",
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

            uint white = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.82f, 0.82f, 0.82f)));
            uint red = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.8f, 0.1f, 0.1f)));
            uint green = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.1f, 0.8f, 0.1f)));
            uint lightPanel = (uint)b.AddMaterial(new MaterialSbtData { Albedo = new Vec3(0f, 0f, 0f), Emission = new Vec3(0.6f, 0.6f, 0.6f), MaterialKind = MaterialSbtData.Solid });

            b.AddYZRect(0f, 5f, -5f, 0f, -3f, red);   // left wall
            b.AddYZRect(0f, 5f, -5f, 0f, 3f, green);  // right wall
            b.AddXZRect(-3f, 3f, -5f, 0f, 0f, white); // floor
            b.AddXZRect(-3f, 3f, -5f, 0f, 5f, white); // ceiling
            b.AddXZRect(-0.9f, 0.9f, -3.2f, -2.2f, 4.99f, lightPanel); // light panel
            b.AddXYRect(-3f, 3f, 0f, 5f, -5f, white); // back wall

            b.AddBox(new Vec3(-2.2f, 0f, -4.0f), new Vec3(-0.8f, 1.0f, -2.8f), white);
            b.AddBox(new Vec3(0.6f, 0f, -3.3f), new Vec3(2.0f, 1.8f, -2.1f), white);

            b.AddLight(new Vec3(0f, 4.6f, -2.7f), new Vec3(1f, 1f, 1f), 20f);

            return b.Build();
        }

        // Direct port of the reference's Scenes.BuildMirrorSpheresOnChecker() - gold,
        // "glassy" (reflectivity 0.6, below the 0.9 mirror-dispatch threshold in both
        // engines - it renders as ordinary diffuse, matching the reference's own
        // behavior), and true-mirror spheres over a checkered floor.
        public static SceneData BuildMirrorSpheresOnChecker()
        {
            var b = new SceneBuilder
            {
                Name = "Mirror spheres on checker",
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

            uint floor = (uint)b.AddMaterial(MaterialPresets.Checker(new Vec3(0.8f, 0.8f, 0.8f), new Vec3(0.15f, 0.15f, 0.15f), 0.6f));
            uint gold = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(1.0f, 0.85f, 0.57f), 0.1f));
            uint glassy = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.9f, 0.95f, 1.0f), 0.6f));
            uint mirror = (uint)b.AddMaterial(MaterialPresets.Mirror(0.85f));

            b.AddXZRect(-8f, 8f, -8f, 4f, 0f, floor);

            b.AddSphere(new Vec3(-1.2f, 1f, -2.0f), 1f, gold);
            b.AddSphere(new Vec3(1.3f, 1f, -2.6f), 1f, glassy);
            b.AddSphere(new Vec3(0f, 0.5f, -4.2f), 0.5f, mirror);

            b.AddLight(new Vec3(-2.5f, 3.5f, -1.5f), new Vec3(1f, 0.95f, 0.9f), 90f);
            b.AddLight(new Vec3(2.0f, 2.8f, -3.8f), new Vec3(0.9f, 0.95f, 1.0f), 70f);

            return b.Build();
        }

        // Direct port of the reference's Scenes.BuildCylindersDisksAndTriangles() -
        // exercises a capped Y-cylinder, a disk, and a raw triangle together with the
        // custom-primitive GAS, over a checkered floor (mixed triangle+custom-primitive
        // GAS, same combination already proven by BuildDebugOrenNayarScene).
        public static SceneData BuildCylindersDisksAndTriangles()
        {
            var b = new SceneBuilder
            {
                Name = "Cylinders, disks and triangles",
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

            uint floor = (uint)b.AddMaterial(MaterialPresets.Checker(new Vec3(0.75f, 0.75f, 0.75f), new Vec3(0.2f, 0.2f, 0.2f), 0.8f));
            uint blue = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.2f, 0.35f, 0.9f)));   // matte blue (cylinder)
            uint matteRed = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.9f, 0.25f, 0.25f))); // matte red (triangle)
            uint yellow = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.8f, 0.8f, 0.1f)));  // yellow (disk)

            b.AddXZRect(-10f, 10f, -10f, 4f, 0f, floor);
            b.AddCylinderY(new Vec3(-1.2f, 0f, -3.0f), 0.6f, 0f, 1.6f, blue);
            b.AddDisk(new Vec3(1.6f, 0.01f, -2.2f), new Vec3(0f, 1f, 0f), 0.9f, yellow);
            b.AddTriangle(new Vec3(0.2f, 0f, -3.6f), new Vec3(1.3f, 1.4f, -3.0f), new Vec3(-0.7f, 0.7f, -2.8f), matteRed);

            b.AddLight(new Vec3(-2.2f, 3.2f, -2.0f), new Vec3(1f, 0.95f, 0.9f), 70f);
            b.AddLight(new Vec3(2.4f, 2.2f, -4.4f), new Vec3(0.9f, 0.95f, 1.0f), 60f);

            return b.Build();
        }

        // Direct port of the reference's Scenes.BuildBoxesShowcase() - three plain white
        // diffuse boxes over a checkered floor. The reference's box constructors pass
        // trailing reflectivity args that its own Solid() closure quirk always ignores -
        // all 3 boxes render as diffuse white, matching the
        // reference's actual behavior rather than its misleading constructor arguments.
        public static SceneData BuildBoxesShowcase()
        {
            var b = new SceneBuilder
            {
                Name = "Boxes showcase",
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

            uint floor = (uint)b.AddMaterial(MaterialPresets.Checker(new Vec3(0.85f, 0.85f, 0.85f), new Vec3(0.15f, 0.15f, 0.15f), 0.7f));
            uint white = (uint)b.AddMaterial(MaterialPresets.Solid(new Vec3(0.86f, 0.86f, 0.86f)));

            b.AddXZRect(-10f, 10f, -10f, 4f, 0f, floor);

            b.AddBox(new Vec3(-2.2f, 0f, -3.6f), new Vec3(-1.0f, 1.2f, -2.4f), white);
            b.AddBox(new Vec3(-0.6f, 0f, -4.2f), new Vec3(0.6f, 0.6f, -3.0f), white);
            b.AddBox(new Vec3(1.0f, 0f, -3.0f), new Vec3(2.4f, 2.0f, -1.8f), white);

            b.AddLight(new Vec3(-2.0f, 3.0f, -2.0f), new Vec3(1f, 0.95f, 0.9f), 70f);
            b.AddLight(new Vec3(2.0f, 2.5f, -4.2f), new Vec3(0.9f, 0.95f, 1.0f), 50f);

            return b.Build();
        }
    }
}
