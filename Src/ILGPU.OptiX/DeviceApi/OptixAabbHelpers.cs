// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixAabbHelpers.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using ILGPU.OptiX.AccelStructures;
using System;
using System.Numerics;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Helpers for Axis-Aligned Bounding Box (AABB) construction and custom-primitive
    /// hit reporting. Used by samples with custom primitives.
    /// Works with System.Numerics.Vector3 for standardized geometry operations.
    /// </summary>
    public static class OptixAabbHelpers
    {
        /// <summary>
        /// Creates an AABB for a sphere given center and radius.
        /// </summary>
        public static OptixAabb MakeSphere(Vector3 center, float radius)
        {
            return new OptixAabb
            {
                MinX = center.X - radius,
                MinY = center.Y - radius,
                MinZ = center.Z - radius,
                MaxX = center.X + radius,
                MaxY = center.Y + radius,
                MaxZ = center.Z + radius
            };
        }

        /// <summary>
        /// Creates an AABB for a cube (axis-aligned rectangular prism) given min and max corners.
        /// </summary>
        public static OptixAabb MakeCube(Vector3 min, Vector3 max)
        {
            return new OptixAabb
            {
                MinX = min.X,
                MinY = min.Y,
                MinZ = min.Z,
                MaxX = max.X,
                MaxY = max.Y,
                MaxZ = max.Z
            };
        }

        /// <summary>
        /// Reports an intersection hit for a custom primitive.
        /// Called from an intersection program to record a valid hit at the given ray parameter (rayTmin).
        /// The hitKind parameter identifies which surface was hit (e.g., front face, back face, etc.).
        /// </summary>
        public static void ReportHitRecord(uint hitKind, float rayTmin)
        {
            OptixReportIntersection.Report(rayTmin, hitKind);
        }
    }
}
