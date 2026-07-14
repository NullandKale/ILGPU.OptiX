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

using System.Numerics;
using ILGPU.OptiX.DeviceApi;

namespace ILGPU.OptiX.AccelStructures
{
    /// <summary>
    /// Mirrors OptiX's OptixAabb (optix_types.h) - the per-primitive bounding box layout
    /// OptixBuildInputCustomPrimitiveArray.AabbBuffers expects, one instance per custom
    /// primitive. The 6 individual float fields are the layout OptiX's native struct
    /// requires; <see cref="OptixAabb(Vector3, Vector3)"/> and the <see cref="Min"/>/
    /// <see cref="Max"/> properties are a <c>Vector3</c>-based convenience on top -
    /// this library's standard vector type (see <see cref="OptixTrace.Trace{T}"/>) -
    /// without changing the field layout the AABB buffer upload depends on.
    /// </summary>
    public struct OptixAabb
    {
        public float MinX;
        public float MinY;
        public float MinZ;
        public float MaxX;
        public float MaxY;
        public float MaxZ;

        public OptixAabb(Vector3 min, Vector3 max)
        {
            MinX = min.X;
            MinY = min.Y;
            MinZ = min.Z;
            MaxX = max.X;
            MaxY = max.Y;
            MaxZ = max.Z;
        }

        public Vector3 Min
        {
            readonly get => new Vector3(MinX, MinY, MinZ);
            set { MinX = value.X; MinY = value.Y; MinZ = value.Z; }
        }

        public Vector3 Max
        {
            readonly get => new Vector3(MaxX, MaxY, MaxZ);
            set { MaxX = value.X; MaxY = value.Y; MaxZ = value.Z; }
        }
    }
}
