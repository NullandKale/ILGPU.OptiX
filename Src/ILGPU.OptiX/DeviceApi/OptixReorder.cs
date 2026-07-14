// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixReorder.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of optixReorder - the second step of the Shader
    /// Execution Reordering (SER) 3-step pattern:
    /// call after <see cref="OptixHitObject.Traverse"/> and before the matching
    /// <see cref="OptixHitObject.Invoke"/> overload, so the driver can batch threads
    /// with coherent hits together before running their hit/miss programs. Purely a
    /// scheduling hint - correctness is identical whether or not this is called
    /// (that's exactly what makes it safe to add to an existing traverse/invoke pair
    /// without changing shading results, only potentially performance).
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixReorder
    {
        /// <summary>Reorders using only the driver's own default hit coherence (no extra application hint bits).</summary>
        public static void Invoke()
        {
            Input<uint> _hint = 0u;
            Input<uint> _bits = 0u;
            CudaAsm.EmitRef(
                "call (), _optix_hitobject_reorder, (%0, %1);",
                ref _hint,
                ref _bits);
        }

        /// <summary>
        /// Reorders using the driver's default hit coherence plus an application-supplied
        /// coherence hint (e.g. a material ID) packed into the low
        /// <paramref name="numCoherenceHintBits"/> bits of <paramref name="coherenceHint"/>.
        /// </summary>
        public static void Invoke(uint coherenceHint, uint numCoherenceHintBits)
        {
            Input<uint> _hint = coherenceHint;
            Input<uint> _bits = numCoherenceHintBits;
            CudaAsm.EmitRef(
                "call (), _optix_hitobject_reorder, (%0, %1);",
                ref _hint,
                ref _bits);
        }
    }
}
