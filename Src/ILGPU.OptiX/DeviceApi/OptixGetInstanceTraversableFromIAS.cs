// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetInstanceTraversableFromIAS.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetInstanceTraversableFromIAS built-in
    /// function - looks up the traversable handle referenced by the instance at
    /// <paramref name="instanceIndex"/> within the given IAS. Callable from any
    /// program type.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetInstanceTraversableFromIAS
    {
        public static ulong Get(ulong ias, uint instanceIndex)
        {
            Input<ulong> _ias = ias;
            Input<uint> _instanceIndex = instanceIndex;
            Output<ulong> _handle = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_get_instance_traversable_from_ias, (%1, %2);",
                ref _handle, ref _ias, ref _instanceIndex);
            return _handle.Value;
        }
    }
}
