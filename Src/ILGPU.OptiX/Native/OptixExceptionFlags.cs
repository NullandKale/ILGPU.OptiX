// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixExceptionFlags.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;

#pragma warning disable CA1008 // Enums should have zero value
#pragma warning disable CA1707 // Identifiers should not contain underscores
#pragma warning disable CA1711 // Identifiers should not have incorrect suffix

namespace ILGPU.OptiX.Native
{
    [Flags]
    public enum OptixExceptionFlags
    {
        /// <summary>
        /// No exception are enabled.
        /// </summary>
        None = 0,

        /// <summary>
        /// Enables exceptions check related to the continuation stack.
        /// </summary>
        StackOverflow = 1 << 0,

        /// <summary>
        /// Enables exceptions check related to trace depth.
        /// </summary>
        TraceDepth = 1 << 1,

        /// <summary>
        /// Enables user exceptions via optixThrowException(). This flag must be specified
        /// for all modules in a pipeline if any module calls optixThrowException().
        /// </summary>
        User = 1 << 2,

        /// <summary>
        /// Enables various exceptions check related to traversal.
        /// </summary>
        Debug = 1 << 3
    }
}

#pragma warning restore CA1008 // Enums should have zero value
#pragma warning restore CA1707 // Identifiers should not contain underscores
#pragma warning restore CA1711 // Identifiers should not have incorrect suffix
