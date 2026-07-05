// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixProgramGroupOptions.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;

#pragma warning disable CA1051 // Do not declare visible instance fields
#pragma warning disable CA1815 // Override equals and operator equals on value types

namespace ILGPU.OptiX
{
    public struct OptixProgramGroupOptions
    {
        /// <summary>
        /// Specifies the payload type of this program group. All programs in the
        /// group must support the payload type. If left IntPtr.Zero, a unique type
        /// is deduced (this is the correct default when not using custom payload
        /// types).
        /// </summary>
        public IntPtr PayloadType;
    }
}

#pragma warning restore CA1051 // Do not declare visible instance fields
#pragma warning restore CA1815 // Override equals and operator equals on value types
