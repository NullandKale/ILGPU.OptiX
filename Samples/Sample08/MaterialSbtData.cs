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

namespace Sample08
{
    // Device-side view of a hitgroup record's custom data (the bytes after the header)
    // - what OptixGetSbtDataPointer.Value points at inside __closest__radiance. Field
    // layout must match HitgroupRecord in SampleRenderer.cs starting right after its
    // fixed Header array. TextureObject is a cudaTextureObject_t handle (an opaque
    // ulong, not a pointer); 0 means "no diffuse texture".
    public struct MaterialSbtData
    {
        public Vec3 Color;
        public ulong TextureObject;
    }
}
