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

namespace Sample07
{
    // The whole model still shares one merged vertex/index buffer (one GAS build
    // input) rather than one buffer set per mesh - that part was never blocked by any
    // ILGPU limitation, it's just simpler than the reference's per-material buffer
    // splitting. What DOES vary per material is looked up via OptixGetSbtDataPointer
    // (see devicePrograms.cs/HitgroupRecord in SampleRenderer.cs): the build input's
    // SbtIndexOffsetBuffer is Model's per-triangle TriangleMaterialIds array, so OptiX
    // itself selects the right hitgroup record's custom data per triangle - no flat
    // MaterialColors[] lookup array needed in LaunchParams.
    public unsafe struct LaunchParams
    {
        public int FrameID;
        public uint* ColorBuffer;
        public Camera camera;
        public ulong traversable;

        public Vec3* Vertices;
        public Vec3* Normals;
        public Vec2* TexCoords;
        public Vec3i* Indices;
    }
}
