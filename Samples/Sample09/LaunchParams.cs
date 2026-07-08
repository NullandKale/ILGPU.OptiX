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

namespace Sample09
{
    // Per-material data (diffuse color, texture handle) is looked up via
    // OptixGetSbtDataPointer against one hitgroup record per material per ray type
    // (see MaterialSbtData.cs/SampleRenderer.cs's buildAccel/constructor) - see
    // Sample07's LaunchParams.cs for why. Vertices/Normals/TexCoords/Indices remain
    // global merged-buffer arrays shared by the whole model.
    public struct LaunchParams
    {
        public int FrameID;
        public OptixDeviceView<uint> ColorBuffer;
        public Camera camera;
        public ulong traversable;

        public OptixDeviceView<Vec3> Vertices;
        public OptixDeviceView<Vec3> Normals;
        public OptixDeviceView<Vec2> TexCoords;
        public OptixDeviceView<Vec3i> Indices;
    }
}
