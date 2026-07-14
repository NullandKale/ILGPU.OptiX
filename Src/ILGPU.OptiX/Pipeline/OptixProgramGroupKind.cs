// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixProgramGroupKind.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

#pragma warning disable CA1008 // Enums should have zero value
#pragma warning disable CA1707 // Identifiers should not contain underscores

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Distinguishes different kinds of program groups.
    /// </summary>
    public enum OptixProgramGroupKind
    {
        /// <summary>
        /// Program group containing a raygen (RG) program.
        /// </summary>
        Raygen = 0x2421,

        /// <summary>
        /// Program group containing a miss (MS) program.
        /// </summary>
        Miss = 0x2422,

        /// <summary>
        /// Program group containing an exception (EX) program.
        /// </summary>
        Exception = 0x2423,

        /// <summary>
        /// Program group containing an intersection (IS), any hit (AH), and/or closest
        /// hit (CH) program.
        /// </summary>
        Hitgroup = 0x2424,

        /// <summary>
        /// Program group containing a direct (DC) or continuation (CC) callable program
        /// \see OptixProgramGroupCallables, #OptixProgramGroupDesc::callables
        /// </summary>
        Callables = 0x2425
    }
}

#pragma warning restore CA1008 // Enums should have zero value
#pragma warning restore CA1707 // Identifiers should not contain underscores
