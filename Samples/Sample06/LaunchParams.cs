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

namespace Sample06
{
    public struct LaunchParams
    {
        public int FrameID;
        public OptixDeviceView<uint> ColorBuffer;
        public Camera camera;
        public ulong traversable;

        // Per-mesh data, selected by OptixGetSbtGASIndex.Value (i.e. which build input
        // was hit). ILGPU.OptiX doesn't wrap optixGetSbtDataPointer (see
        // devicePrograms.cs), so per-object data that would normally live in each
        // mesh's own SBT record is instead exposed as named fields here.
        //
        // A Vec3**/Vec3i** "array of per-mesh pointers" indexed by mesh ID was tried
        // first, but ILGPU's kernel-compile-time type resolution
        // (TypeInformationManager) infinite-recurses on pointer-to-pointer *fields*
        // specifically (isolated `typeof(T**)` reflection works fine - this appears to
        // be a narrower quirk in how field types get resolved) - so this sample is
        // capped at two meshes with directly-named fields instead, keeping everything
        // at the single-view level that's already proven to work (Sample05).
        public OptixDeviceView<Vec3> mesh0Vertices;
        public OptixDeviceView<Vec3i> mesh0Indices;
        public Vec3 mesh0Color;

        public OptixDeviceView<Vec3> mesh1Vertices;
        public OptixDeviceView<Vec3i> mesh1Indices;
        public Vec3 mesh1Color;
    }
}
