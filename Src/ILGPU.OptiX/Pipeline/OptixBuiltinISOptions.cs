// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixBuiltinISOptions.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using ILGPU.OptiX.AccelStructures;

#pragma warning disable CA1051 // Do not declare visible instance fields
#pragma warning disable CA1815 // Override equals and operator equals on value types

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Mirrors OptixBuiltinISOptions (optix_types.h) - options for
    /// <c>optixBuiltinISModuleGet</c>, which retrieves OptiX's own built-in
    /// intersection program for a non-custom primitive type (curves, spheres) - user
    /// intersection programs are not possible for these types, unlike custom
    /// primitives.
    /// </summary>
    [CLSCompliant(false)]
    public struct OptixBuiltinISOptions
    {
        /// <summary>Must not be <see cref="OptixPrimitiveType.Custom"/>.</summary>
        public OptixPrimitiveType BuiltinISModuleType;

        /// <summary>Whether vertex motion blur is used (not motion transform blur).</summary>
        public int UsesMotionBlur;

        /// <summary>See <see cref="AccelStructures.OptixBuildFlags"/>.</summary>
        public uint BuildFlags;

        /// <summary>See <see cref="OptixCurveEndcapFlags"/> - 0 for non-curve types.</summary>
        public uint CurveEndcapFlags;
    }
}

#pragma warning restore CA1051 // Do not declare visible instance fields
#pragma warning restore CA1815 // Override equals and operator equals on value types
