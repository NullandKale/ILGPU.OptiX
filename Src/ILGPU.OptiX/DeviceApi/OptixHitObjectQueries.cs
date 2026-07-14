// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixHitObjectQueries.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Query methods on the current thread's implicit "hit object" state, valid after
    /// <see cref="Traverse"/> - mirrors the
    /// optixHitObjectIsHit/IsMiss/GetInstanceId/GetPrimitiveIndex/GetHitKind/GetRayTmax
    /// built-in functions (internal/optix_device_impl.h), same
    /// single-output-no-input CudaAsm.Emit idiom as OptixGetHitKind.cs and friends.
    /// </summary>
    [CLSCompliant(false)]
    public static partial class OptixHitObject
    {
        /// <summary>True if the traversed hit object represents an actual (non-miss, non-nop) hit.</summary>
        public static bool IsHit
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_hitobject_is_hit, ();", out uint result);
                return result != 0;
            }
        }

        /// <summary>True if the traversed hit object represents a miss.</summary>
        public static bool IsMiss
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_hitobject_is_miss, ();", out uint result);
                return result != 0;
            }
        }

        /// <summary>True if the hit object is a "no-op" (e.g. from <c>optixMakeNopHitObject</c>) - Invoke() is a no-op for it.</summary>
        public static bool IsNop
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_hitobject_is_nop, ();", out uint result);
                return result != 0;
            }
        }

        /// <summary>The hit instance's user-assigned <c>OptixInstance.InstanceId</c> - only valid when <see cref="IsHit"/>.</summary>
        public static uint InstanceId
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_hitobject_get_instance_id, ();", out uint result);
                return result;
            }
        }

        /// <summary>The hit primitive's index within its build input - only valid when <see cref="IsHit"/>.</summary>
        public static uint PrimitiveIndex
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_hitobject_get_primitive_idx, ();", out uint result);
                return result;
            }
        }

        /// <summary>See <see cref="OptixGetHitKind"/> - only valid when <see cref="IsHit"/>.</summary>
        public static uint HitKind
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_hitobject_get_hitkind, ();", out uint result);
                return result;
            }
        }

        /// <summary>The hit distance along the ray - only valid when <see cref="IsHit"/>.</summary>
        public static float RayTmax
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_hitobject_get_ray_tmax, ();", out float result);
                return result;
            }
        }
    }
}
