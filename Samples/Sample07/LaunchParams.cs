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
    // All device pointers below are single-level (Vec3*/Vec3i*/uint*, never
    // pointer-to-pointer) and index a single merged model buffer - see Model.cs for why:
    // ILGPU.OptiX doesn't wrap optixGetSbtDataPointer, and ILGPU's
    // TypeInformationManager infinite-recurses on pointer-to-pointer LaunchParams
    // fields, so per-mesh/per-material data can't be an array-of-pointers indexed by
    // mesh or material ID. Instead the whole model is one GAS build input/one hitgroup
    // record, and per-triangle material lookup goes through MaterialIds[primId], then
    // MaterialColors[materialId].
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

        public uint* MaterialIds;
        public Vec3* MaterialColors;
    }
}
