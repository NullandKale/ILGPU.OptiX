// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: LaunchParams.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace Sample11
{
    // Unlike Sample10, ColorBuffer is the HDR accumulation target the denoiser reads
    // from (float4/Vec4, not packed RGBA8) - raygen blends each frame's average sample
    // into it (see devicePrograms.cs), and SampleRenderer denoises + tonemaps it to
    // bytes for display after each optixLaunch. Per-material data (diffuse color,
    // texture handle) is looked up via OptixGetSbtDataPointer against one hitgroup
    // record per material per ray type (see MaterialSbtData.cs/SampleRenderer.cs's
    // buildAccel/constructor) - see Sample07's LaunchParams.cs for why.
    public unsafe struct LaunchParams
    {
        public int NumPixelSamples;
        public int FrameID;
        public Vec4* ColorBuffer;
        public Camera camera;
        public ulong traversable;

        public Vec3* Vertices;
        public Vec3* Normals;
        public Vec2* TexCoords;
        public Vec3i* Indices;

        public Vec3 LightOrigin;
        public Vec3 LightDu;
        public Vec3 LightDv;
        public Vec3 LightPower;
    }
}
