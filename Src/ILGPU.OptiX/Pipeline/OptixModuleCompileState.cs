// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixModuleCompileState.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;

#pragma warning disable CA1028 // Enum Storage should be Int32

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Mirrors OptixModuleCompileState (optix_types.h) - the state of a module created
    /// via optixModuleCreateWithTasks, queried via optixModuleGetCompilationState.
    /// </summary>
    [CLSCompliant(false)]
    public enum OptixModuleCompileState : uint
    {
        NotStarted = 0x2360,
        Started = 0x2361,
        ImpendingFailure = 0x2362,
        Failed = 0x2363,
        Completed = 0x2364,
    }
}

#pragma warning restore CA1028 // Enum Storage should be Int32
