// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixDeviceContextValidationMode.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;

#pragma warning disable CA1028 // Enum Storage should be Int32
#pragma warning disable CA1707 // Identifiers should not contain underscores

namespace ILGPU.OptiX.Pipeline
{
    [CLSCompliant(false)]
    public enum OptixDeviceContextValidationMode : uint
    {
        Off = 0,
        All = 0xFFFFFFFF
    }
}

#pragma warning restore CA1028 // Enum Storage should be Int32
#pragma warning restore CA1707 // Identifiers should not contain underscores
