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

namespace Sample13
{
    // ILGPU compiles device kernels against one concrete unmanaged struct layout, so this
    // must stay a single superset shape across every scene the runtime scene-switcher can
    // select (docs/SAMPLE13_PLAN.md, "LaunchParams superset design"). Unused buffers on a
    // given scene are left null/zero-length rather than changing the struct itself.
    //
    // Triangle geometry + multi-point-light Oren-Nayar/ambient shading (M1), mirror/
    // dielectric materials (M2), custom primitives - Sphere/Box/CylinderY/Disk/rects/
    // volume grid (M3-M5), mesh scenes (M6, reusing the triangle buffers as-is), and
    // AOV/denoiser buffers (M7) are all wired up.
    public unsafe struct LaunchParams
    {
        public int NumPixelSamples;
        public int FrameID;
        public Vec4* ColorBuffer;
        // AOV guide buffers for the denoiser (see SampleRenderer.cs's OptixDenoiser
        // wiring) - overwritten fresh every frame from the primary ray's own hit, not
        // blended across frames like ColorBuffer (see devicePrograms.cs's raygen).
        public Vec4* NormalBuffer;
        public Vec4* AlbedoBuffer;
        public Camera camera;
        public ulong traversable;

        public Vec3* Vertices;
        public Vec3* Normals;
        public Vec2* TexCoords;
        public Vec3i* Indices;

        // Custom-primitive parameter buffers - always allocated (possibly zero-length)
        // so this struct's layout never changes across scene switches.
        public SphereData* Spheres;
        public BoxData* Boxes;
        public CylinderYData* CylindersY;
        public DiskData* Disks;
        public RectData* XYRects;
        public RectData* XZRects;
        public RectData* YZRects;

        // Volume grid - only one grid is ever active at a time (unlike the arrays
        // above), so its parameters are plain scalar fields rather than a per-primitive
        // buffer. VoxelMaterialIds is a flat row-major (x*dims.y*dims.z + y*dims.z + z)
        // array; 0 means empty/air, N means launchParams.Materials[N-1] (see
        // devicePrograms.cs's ShadeVolumeGrid - this is the one place in the sample
        // where material lookup bypasses the per-primitive SBT convention, since a
        // single GAS primitive's material can't otherwise depend on which voxel the
        // intersection program's DDA loop found).
        public uint* VoxelMaterialIds;
        public Vec3 VolumeGridMin;
        public Vec3 VolumeVoxelSize;
        public Vec3i VolumeDims;

        // Device-side copy of the active scene's Materials[] palette, indexed directly
        // (not through an SBT record) - needed only for the volume grid's per-voxel
        // material lookup above.
        public MaterialSbtData* Materials;

        public PointLightGpu* PointLights;
        public int NumPointLights;
        public Vec3 AmbientColor;
        public float AmbientIntensity;

        public Vec3 BackgroundTop;
        public Vec3 BackgroundBottom;
    }
}
