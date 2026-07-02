// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: MaterialSbtData.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace Sample12
{
    // Device-side view of a hitgroup record's custom data (the bytes after the header)
    // - what OptixGetSbtDataPointer.Value points at inside __closest__radiance. Field
    // layout must match HitgroupRecord in SampleRenderer.cs starting right after its
    // fixed Header array. Only the radiance hitgroup record's data is ever read (the
    // shadow ray's closest-hit is an empty stub, matching the reference); TextureObject
    // is a cudaTextureObject_t handle (an opaque ulong, not a pointer), 0 = no texture.
    public struct MaterialSbtData
    {
        public Vec3 Color;
        public ulong TextureObject;
    }
}
