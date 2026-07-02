// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixAabb.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace ILGPU.OptiX
{
    /// <summary>
    /// Mirrors OptiX's OptixAabb (optix_types.h) - the per-primitive bounding box layout
    /// OptixBuildInputCustomPrimitiveArray.AabbBuffers expects, one instance per custom
    /// primitive.
    /// </summary>
    public struct OptixAabb
    {
        public float MinX;
        public float MinY;
        public float MinZ;
        public float MaxX;
        public float MaxY;
        public float MaxZ;
    }
}
