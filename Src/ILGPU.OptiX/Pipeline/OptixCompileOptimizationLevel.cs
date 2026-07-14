// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixCompileOptimizationLevel.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

#pragma warning disable CA1707 // Identifiers should not contain underscores

namespace ILGPU.OptiX.Pipeline
{
    public enum OptixCompileOptimizationLevel
    {
        /// <summary>
        /// Default is to run all optimizations.
        /// </summary>
        Default = 0,

        /// <summary>
        /// No optimizations.
        /// </summary>
        Level0 = 0x2340,

        /// <summary>
        /// Some optimizations.
        /// </summary>
        Level1 = 0x2341,

        /// <summary>
        /// Most optimizations.
        /// </summary>
        Level2 = 0x2342,

        /// <summary>
        /// All optimizations.
        /// </summary>
        Level3 = 0x2343,
    }
}

#pragma warning restore CA1707 // Identifiers should not contain underscores
