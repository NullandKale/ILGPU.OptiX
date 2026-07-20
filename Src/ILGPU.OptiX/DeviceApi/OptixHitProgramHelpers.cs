// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixHitProgramHelpers.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;
using System.Numerics;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Helpers for common triangle hit program operations: barycentric interpolation,
    /// normal computation, and back-face handling, using <see cref="Vector3"/>.
    /// </summary>
    public static class OptixHitProgramHelpers
    {
        /// <summary>
        /// Extracts barycentric weights from OptixGetTriangleBarycentrics.
        /// Returns (bw, bu, bv) for convenience; OptixGetTriangleBarycentrics provides (bu, bv),
        /// and bw is computed as 1 - bu - bv.
        /// </summary>
        public static (float bw, float bu, float bv) GetTriangleBarycentrics()
        {
            var (bu, bv) = OptixGetTriangleBarycentrics.Value;
            float bw = 1f - bu - bv;
            return (bw, bu, bv);
        }

        /// <summary>
        /// Interpolates a Vector3 attribute across a triangle using barycentric weights.
        /// Common usage: interpolate normals, texture coordinates, or other per-vertex attributes.
        /// </summary>
        public static Vector3 InterpolateAttribute(Vector3 a, Vector3 b, Vector3 c, float bw, float bu, float bv)
            => (bw * a) + (bu * b) + (bv * c);

        /// <summary>
        /// Computes the geometric normal from three triangle vertices using the cross product.
        /// Result is normalized. Suitable for flat shading or when per-vertex normals are unavailable.
        /// Uses System.Numerics.Vector3 for standard cross product and normalize operations.
        /// </summary>
        public static Vector3 GetGeometricNormal(Vector3 a, Vector3 b, Vector3 c)
        {
            Vector3 edge1 = b - a;
            Vector3 edge2 = c - a;
            Vector3 normal = Vector3.Cross(edge1, edge2);
            return Vector3.Normalize(normal);
        }

        /// <summary>
        /// Orients a normal to face the ray origin (back-face handling).
        /// If the normal points away from the ray direction, it is flipped.
        /// Used to ensure consistent shading even if normals are authored inconsistently.
        /// </summary>
        public static Vector3 OrientGeometricNormal(Vector3 geometricNormal, Vector3 rayDir)
        {
            if (Vector3.Dot(geometricNormal, rayDir) > 0f)
                return -geometricNormal;
            return geometricNormal;
        }
    }
}
