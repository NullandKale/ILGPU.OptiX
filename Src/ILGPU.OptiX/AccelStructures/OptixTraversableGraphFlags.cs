// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixTraversableGraphFlags.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;

#pragma warning disable CA1008 // Enums should have zero value
#pragma warning disable CA1707 // Identifiers should not contain underscores
#pragma warning disable CA1711 // Identifiers should not have incorrect suffix

namespace ILGPU.OptiX.AccelStructures
{
    [Flags]
    public enum OptixTraversableGraphFlags
    {
        /// <summary>
        ///  Used to signal that any traversable graphs is valid.
        ///  This flag is mutually exclusive with all other flags.
        /// </summary>
        AllowAny = 0,

        /// <summary>
        ///  Used to signal that a traversable graph of a single Geometry Acceleration
        ///  Structure (GAS) without any transforms is valid. This flag may be combined
        ///  with other flags except for AllowAny.
        /// </summary>
        AllowSingleGas = 1 << 0,

        /// <summary>
        ///  Used to signal that a traversable graph of a single Instance Acceleration
        ///  Structure (IAS) directly connected to Geometry Acceleration Structure (GAS)
        ///  traversables without transform traversables in between is valid.  This flag
        ///  may be combined with other flags except for
        ///  AllowAny.
        /// </summary>
        AllowSingleLevelInstancing = 1 << 1,
    }
}

#pragma warning restore CA1008 // Enums should have zero value
#pragma warning restore CA1707 // Identifiers should not contain underscores
#pragma warning restore CA1711 // Identifiers should not have incorrect suffix
