// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixPrimitiveTypeFlags.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Diagnostics.CodeAnalysis;

#pragma warning disable CA1707 // Identifiers should not contain underscores
#pragma warning disable CA1711 // Identifiers should not have incorrect suffix

namespace ILGPU.OptiX.AccelStructures
{
    [Flags]
    [SuppressMessage("Usage", "CA2217:Do not mark enums with FlagsAttribute")]
    public enum OptixPrimitiveTypeFlags
    {
        /// <summary>
        /// Custom primitive.
        /// </summary>
        Custom = 1 << 0,

        /// <summary>
        /// B-spline curve of degree 2 with circular cross-section.
        /// </summary>
        RoundQuadraticBSpline = 1 << 1,

        /// <summary>
        /// B-spline curve of degree 3 with circular cross-section.
        /// </summary>
        RoundCubicBSpline = 1 << 2,

        /// <summary>
        /// Piecewise linear curve with circular cross-section.
        /// </summary>
        RoundLinear = 1 << 3,

        /// <summary>
        /// CatmullRom curve with circular cross-section.
        /// </summary>
        RoundCatmullRom = 1 << 4,

        /// <summary>
        /// B-spline curve of degree 2 with oriented flat cross-section (ribbon).
        /// </summary>
        FlatQuadraticBSpline = 1 << 5,

        /// <summary>
        /// Sphere.
        /// </summary>
        Sphere = 1 << 6,

        /// <summary>
        /// Bezier curve of degree 3 with circular cross-section.
        /// </summary>
        RoundCubicBezier = 1 << 7,

        /// <summary>
        /// B-spline curve of degree 2 with circular cross-section, with rounded
        /// endcaps (rocaps) at the ends of each segment.
        /// </summary>
        RoundQuadraticBSplineRocaps = 1 << 8,

        /// <summary>
        /// B-spline curve of degree 3 with circular cross-section, with rounded
        /// endcaps (rocaps) at the ends of each segment.
        /// </summary>
        RoundCubicBSplineRocaps = 1 << 9,

        /// <summary>
        /// CatmullRom curve with circular cross-section, with rounded endcaps
        /// (rocaps) at the ends of each segment.
        /// </summary>
        RoundCatmullRomRocaps = 1 << 10,

        /// <summary>
        /// Bezier curve of degree 3 with circular cross-section, with rounded
        /// endcaps (rocaps) at the ends of each segment.
        /// </summary>
        RoundCubicBezierRocaps = 1 << 11,

        /// <summary>
        /// Triangle.
        /// </summary>
        Triangle = 1 << 31,
    }
}

#pragma warning restore CA1707 // Identifiers should not contain underscores
#pragma warning restore CA1711 // Identifiers should not have incorrect suffix
