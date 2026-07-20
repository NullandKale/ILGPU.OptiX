// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetClusterId.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetClusterId built-in function - the
    /// user-defined cluster id of the hit cluster (see
    /// OptixClusterAccelBuildInputTrianglesArgs.ClusterId), or
    /// <see cref="Invalid"/> when the hit is not on cluster geometry. Valid in
    /// AH/CH programs.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetClusterId
    {
        /// <summary>
        /// Mirrors OPTIX_CLUSTER_ID_INVALID (optix_types.h) - returned for
        /// non-cluster hits.
        /// </summary>
        public const uint Invalid = 0xFFFFFFFF;

        public static uint Value
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_get_cluster_id, ();", out uint id);
                return id;
            }
        }
    }

    public static partial class OptixHitObject
    {
        /// <summary>
        /// Provides the functionality of optixHitObjectGetClusterId - the cluster id
        /// of the recorded hit, or <see cref="OptixGetClusterId.Invalid"/> for
        /// non-cluster hits.
        /// </summary>
        public static uint ClusterId
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_hitobject_get_cluster_id, ();", out uint id);
                return id;
            }
        }
    }
}
