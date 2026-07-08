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

using ILGPU.OptiX.Device;

namespace Sample12
{
    // Adds AlbedoBuffer/NormalBuffer to Sample11's LaunchParams - AOV guide images the
    // denoiser uses to better distinguish real geometric detail from noise (see
    // devicePrograms.cs and SampleRenderer.cs's OptixDenoiserGuideLayer setup). Unlike
    // ColorBuffer, these are NOT blended across frames - each frame overwrites them
    // fresh, matching example12_denoiseSeparateChannels/devicePrograms.cu. Per-material
    // data (diffuse color, texture handle) is looked up via OptixGetSbtDataPointer
    // against one hitgroup record per material per ray type (see MaterialSbtData.cs/
    // SampleRenderer.cs's buildAccel/constructor) - see Sample07's LaunchParams.cs for
    // why.
    public struct LaunchParams
    {
        public int NumPixelSamples;
        public int FrameID;
        public OptixDeviceView<Vec4> ColorBuffer;
        public OptixDeviceView<Vec4> AlbedoBuffer;
        public OptixDeviceView<Vec4> NormalBuffer;
        public Camera camera;
        public ulong traversable;

        public OptixDeviceView<Vec3> Vertices;
        public OptixDeviceView<Vec3> Normals;
        public OptixDeviceView<Vec2> TexCoords;
        public OptixDeviceView<Vec3i> Indices;

        public Vec3 LightOrigin;
        public Vec3 LightDu;
        public Vec3 LightDv;
        public Vec3 LightPower;
    }
}
