// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixCompileDebugLevel.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

#pragma warning disable CA1707 // Identifiers should not contain underscores

namespace ILGPU.OptiX.Pipeline
{
    public enum OptixCompileDebugLevel
    {
        /// <summary>
        /// Default currently is to add line info.
        /// </summary>
        Default = 0,

        /// <summary>
        /// No debug information.
        /// </summary>
        None = 0x2350,

        /// <summary>
        /// Generate lineinfo only.
        /// </summary>
        LineInfo = 0x2351,

        /// <summary>
        /// Generate dwarf debug information.
        /// </summary>
        Full = 0x2352,
    }
}

#pragma warning restore CA1707 // Identifiers should not contain underscores
