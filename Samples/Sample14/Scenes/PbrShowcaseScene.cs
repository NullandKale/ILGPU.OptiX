// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: PbrShowcaseScene.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace Sample14
{
    // A dedicated validation scene - the one place every PBR subsystem gets
    // exercised together in isolation from Sponza's asset limitations: a
    // roughness x metallic sphere grid, HDRI-only lighting (no point lights, no flat
    // ambient - AmbientIntensity is 0), a texture-mapped normal/ORM surface (the only
    // one in this whole sample - Sponza has no real ORM/normal data), and a
    // rough-glass object exercising the BTDF.
    public static class PbrShowcaseScene
    {
        public static SceneData BuildPbrShowcaseScene()
        {
            var b = new SceneBuilder
            {
                Name = "PBR showcase (HDRI-lit)",
                AmbientColor = new Vec3(0f, 0f, 0f),
                AmbientIntensity = 0f,
                // Never actually sampled - EnvMapPath below means __miss__radiance
                // always takes the environment-map branch - kept only because
                // SceneData.BackgroundTop/Bottom have no "unset" sentinel.
                BackgroundTop = new Vec3(0.05f, 0.05f, 0.07f),
                BackgroundBottom = new Vec3(0.02f, 0.02f, 0.03f),
                EnvMapPath = "models/hdri/venice_sunset_1k.hdr",
                CameraOrigin = new Vec3(0f, 3.2f, 11f),
                CameraLookAt = new Vec3(0f, 1.2f, 0f),
                CameraUp = new Vec3(0f, 1f, 0f),
                CameraFovDeg = 45f,
                CameraWorldScale = 9f,
            };

            // Textured floor - the sample's only real normal/ORM-mapped surface.
            // Roughness/Metallic scalars are pure multipliers against arm.png's G/B
            // channels (the glTF convention - see MaterialShading.ShadeSurface's ORM
            // block), so both stay at 1 here to reproduce the texture's own
            // per-texel values unscaled.
            uint texturedFloor = (uint)b.AddMaterial(
                new MaterialSbtData
                {
                    BaseColor = new Vec3(1f, 1f, 1f),
                    Roughness = 1f,
                    Metallic = 1f,
                    TextureWeight = 1f,
                    UVScale = 4f,
                    NormalStrength = 1f,
                    MaterialKind = MaterialSbtData.Solid,
                },
                "models/textures/rusty_metal_02/diffuse.png",
                "models/textures/rusty_metal_02/normal.png",
                "models/textures/rusty_metal_02/arm.png");
            b.AddQuad(
                new Vec3(-6f, 0f, 6f), new Vec3(6f, 0f, 6f), new Vec3(6f, 0f, -6f), new Vec3(-6f, 0f, -6f),
                new Vec3(0f, 1f, 0f), texturedFloor);

            // A standing panel of the same material, facing the camera - normal-map
            // detail reads very differently at a grazing floor angle vs. head-on, so
            // both are worth showing.
            b.AddQuad(
                new Vec3(-3f, 0f, -5.9f), new Vec3(3f, 0f, -5.9f), new Vec3(3f, 3.5f, -5.9f), new Vec3(-3f, 3.5f, -5.9f),
                new Vec3(0f, 0f, 1f), texturedFloor);

            // Roughness x metallic sphere grid (dielectric front row, conductor back
            // row) - same sweep shape as BasicScenes.BuildGgxRoughnessSweepScene, but
            // HDRI-lit only here instead of point-lit, isolating how the GGX lobe
            // responds to real environment lighting/importance sampling.
            const int columns = 5;
            Vec3 dielectricColor = new Vec3(0.75f, 0.15f, 0.15f);
            Vec3 conductorColor = new Vec3(1f, 0.86f, 0.6f); // gold
            for (int col = 0; col < columns; col++)
            {
                float roughness = col / (float)(columns - 1);
                float x = -4f + (col * 2f);

                uint dielectric = (uint)b.AddMaterial(new MaterialSbtData
                {
                    BaseColor = dielectricColor,
                    Metallic = 0f,
                    Roughness = roughness,
                    MaterialKind = MaterialSbtData.Solid,
                });
                b.AddSphereMesh(new Vec3(x, 0.8f, 2f), 0.8f, dielectric);

                uint conductor = (uint)b.AddMaterial(new MaterialSbtData
                {
                    BaseColor = conductorColor,
                    Metallic = 1f,
                    Roughness = roughness,
                    MaterialKind = MaterialSbtData.Solid,
                });
                b.AddSphereMesh(new Vec3(x, 0.8f, -1f), 0.8f, conductor);
            }

            // Rough-glass object - front and center, where the HDRI's own sun disc
            // gives it something distinctive to refract/frost.
            uint roughGlass = (uint)b.AddMaterial(new MaterialSbtData
            {
                BaseColor = new Vec3(1f, 1f, 1f),
                Transmission = 1f,
                TransmissionRoughness = 0.25f,
                IOR = 1.5f,
                TransmissionColor = new Vec3(0.95f, 0.97f, 1f),
                MaterialKind = MaterialSbtData.Solid,
            });
            b.AddSphereMesh(new Vec3(0f, 1f, 4.2f), 1f, roughGlass);

            return b.Build();
        }
    }
}
