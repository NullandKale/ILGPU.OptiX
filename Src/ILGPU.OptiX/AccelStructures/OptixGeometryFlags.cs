// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGeometryFlags.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;

namespace ILGPU.OptiX.AccelStructures
{
    /// <summary>
    /// Per-SBT-record geometry flags (OptixGeometryFlags in optix_types.h) - supplied
    /// per build input via <see cref="OptixAccelBuilder.AddTriangleMesh"/>'s
    /// perSbtRecordFlags parameter, one entry per SBT record (i.e. per material when a
    /// per-primitive SBT index buffer is used). These configure the acceleration
    /// structure itself and apply to every ray traced against it, unlike the per-trace
    /// <see cref="DeviceApi.OptixRayFlags"/> (which can still override them, e.g.
    /// EnforceAnyHit).
    /// </summary>
    [Flags]
    public enum OptixGeometryFlags : uint
    {
        /// <summary>Default behavior - anyhit programs run for every intersection candidate.</summary>
        None = 0,

        /// <summary>
        /// Never invoke anyhit programs for primitives mapping to this SBT record -
        /// lets the traversal treat these hits as fully opaque in fixed-function
        /// hardware, skipping the round trip into software anyhit entirely. The
        /// standard, large traversal win for materials with no alpha cutout and no
        /// transparency; the trace call's own logic must not RELY on an anyhit running
        /// for these primitives (e.g. an occlusion ray whose only payload write lives
        /// in anyhit needs a closesthit/miss fallback path instead).
        /// </summary>
        DisableAnyHit = 1u << 0,

        /// <summary>
        /// Guarantee at most one anyhit invocation per primitive per ray (OptiX may
        /// otherwise invoke anyhit more than once for the same primitive) - needed for
        /// non-idempotent anyhit logic (e.g. accumulation), at some traversal cost.
        /// </summary>
        RequireSingleAnyHitCall = 1u << 1,
    }
}
