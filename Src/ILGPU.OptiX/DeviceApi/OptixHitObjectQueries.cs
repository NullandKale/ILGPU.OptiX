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
using System.Numerics;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// The SER hit-object transform-stack accessor (see
    /// <see cref="IOptixHitStateAccessor"/>), backed by
    /// _optix_hitobject_get_transform_list_* and _optix_hitobject_get_ray_time.
    /// </summary>
    internal readonly struct OptixHitObjectAccessor : IOptixHitStateAccessor
    {
        public uint TransformListSize => OptixHitObject.TransformListSize;

        public ulong GetTransformListHandle(uint index) =>
            OptixHitObject.GetTransformListHandle(index);

        public float RayTime => OptixHitObject.RayTime;
    }

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

        /// <summary>The hit instance's zero-based index within its IAS build input - only valid when <see cref="IsHit"/>.</summary>
        public static uint InstanceIndex
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_hitobject_get_instance_idx, ();", out uint result);
                return result;
            }
        }

        /// <summary>See <see cref="OptixGetSbtGASIndex"/> - only valid when <see cref="IsHit"/>.</summary>
        public static uint SbtGASIndex
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_hitobject_get_sbt_gas_idx, ();", out uint result);
                return result;
            }
        }

        /// <summary>The tmin of the recorded ray.</summary>
        public static float RayTmin
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_hitobject_get_ray_tmin, ();", out float result);
                return result;
            }
        }

        /// <summary>The motion time of the recorded ray.</summary>
        public static float RayTime
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_hitobject_get_ray_time, ();", out float result);
                return result;
            }
        }

        /// <summary>The <see cref="OptixRayFlags"/> of the recorded ray.</summary>
        public static OptixRayFlags RayFlags
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_hitobject_get_ray_flags, ();", out uint result);
                return (OptixRayFlags)result;
            }
        }

        /// <summary>
        /// The SBT record index the hit object resolves to on
        /// <see cref="Invoke()"/> - settable via <see cref="SetSbtRecordIndex"/> to
        /// redirect which program runs.
        /// </summary>
        public static uint SbtRecordIndex
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_hitobject_get_sbt_record_index, ();", out uint result);
                return result;
            }
        }

        /// <summary>
        /// Overrides the SBT record index the hit object resolves to on
        /// <see cref="Invoke()"/> (optixHitObjectSetSbtRecordIndex).
        /// </summary>
        public static void SetSbtRecordIndex(uint sbtRecordIndex)
        {
            Input<uint> _sbtRecordIndex = sbtRecordIndex;
            CudaAsm.EmitRef(
                "call _optix_hitobject_set_sbt_record_index, (%0);",
                ref _sbtRecordIndex);
        }

        /// <summary>
        /// The device address of the SBT record data of the resolved program - the
        /// hit-object counterpart of <see cref="OptixGetSbtDataPointer"/>.
        /// </summary>
        public static ulong SbtDataPointer
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_hitobject_get_sbt_data_pointer, ();", out ulong result);
                return result;
            }
        }

        /// <summary>The GAS traversable handle of the recorded hit - only valid when <see cref="IsHit"/>.</summary>
        public static ulong GASTraversableHandle
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_hitobject_get_gas_traversable_handle, ();", out ulong result);
                return result;
            }
        }

        /// <summary>The world-space origin of the recorded ray.</summary>
        public static Vector3 WorldRayOrigin
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_hitobject_get_world_ray_origin_x, ();", out float x);
                CudaAsm.Emit("call (%0), _optix_hitobject_get_world_ray_origin_y, ();", out float y);
                CudaAsm.Emit("call (%0), _optix_hitobject_get_world_ray_origin_z, ();", out float z);
                return new Vector3(x, y, z);
            }
        }

        /// <summary>The world-space direction of the recorded ray.</summary>
        public static Vector3 WorldRayDirection
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_hitobject_get_world_ray_direction_x, ();", out float x);
                CudaAsm.Emit("call (%0), _optix_hitobject_get_world_ray_direction_y, ();", out float y);
                CudaAsm.Emit("call (%0), _optix_hitobject_get_world_ray_direction_z, ();", out float z);
                return new Vector3(x, y, z);
            }
        }

        /// <summary>The triangle barycentrics of the recorded hit - only valid for triangle hits.</summary>
        public static (float U, float V) TriangleBarycentrics
        {
            get
            {
                Output<float> u = default;
                Output<float> v = default;
                CudaAsm.EmitRef(
                    "call (%0, %1), _optix_hitobject_get_triangle_barycentrics, ();",
                    ref u, ref v);
                return (u.Value, v.Value);
            }
        }

        /// <summary>
        /// Intersection attribute register <paramref name="index"/> (0..7) of the
        /// recorded hit (optixHitObjectGetAttribute_0..7 - a single pseudo-call
        /// taking the index as an operand, unlike the current-hit spelling).
        /// </summary>
        public static uint GetAttribute(uint index)
        {
            Input<uint> _index = index;
            Output<uint> _value = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_hitobject_get_attribute, (%1);",
                ref _value, ref _index);
            return _value.Value;
        }

        /// <summary>The recorded hit's transform-stack size - the hit-object counterpart of <see cref="OptixTransformList.Size"/>.</summary>
        public static uint TransformListSize
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_hitobject_get_transform_list_size, ();", out uint result);
                return result;
            }
        }

        /// <summary>The recorded hit's transform-stack handle at <paramref name="index"/>.</summary>
        public static ulong GetTransformListHandle(uint index)
        {
            Input<uint> _index = index;
            Output<ulong> _handle = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_hitobject_get_transform_list_handle, (%1);",
                ref _handle, ref _index);
            return _handle.Value;
        }

        /// <summary>
        /// The composed world-to-object matrix of the recorded hit's transform stack
        /// - the hit-object counterpart of
        /// <see cref="OptixTransforms.GetWorldToObjectTransformMatrix(out Vector4, out Vector4, out Vector4)"/>.
        /// </summary>
        public static void GetWorldToObjectTransformMatrix(
            out Vector4 m0,
            out Vector4 m1,
            out Vector4 m2) =>
            OptixTransforms.GetWorldToObjectTransformMatrix(
                default(OptixHitObjectAccessor), out m0, out m1, out m2);

        /// <summary>
        /// The composed object-to-world matrix of the recorded hit's transform stack.
        /// </summary>
        public static void GetObjectToWorldTransformMatrix(
            out Vector4 m0,
            out Vector4 m1,
            out Vector4 m2) =>
            OptixTransforms.GetObjectToWorldTransformMatrix(
                default(OptixHitObjectAccessor), out m0, out m1, out m2);

        /// <summary>Provides the functionality of optixHitObjectTransformPointFromWorldToObjectSpace.</summary>
        public static Vector3 TransformPointFromWorldToObjectSpace(Vector3 point)
        {
            if (TransformListSize == 0)
                return point;
            GetWorldToObjectTransformMatrix(out var m0, out var m1, out var m2);
            return OptixTransforms.TransformPoint(m0, m1, m2, point);
        }

        /// <summary>Provides the functionality of optixHitObjectTransformVectorFromWorldToObjectSpace.</summary>
        public static Vector3 TransformVectorFromWorldToObjectSpace(Vector3 vector)
        {
            if (TransformListSize == 0)
                return vector;
            GetWorldToObjectTransformMatrix(out var m0, out var m1, out var m2);
            return OptixTransforms.TransformVector(m0, m1, m2, vector);
        }

        /// <summary>
        /// Provides the functionality of
        /// optixHitObjectTransformNormalFromWorldToObjectSpace (transpose-multiplies
        /// with the object-to-world matrix, per the SDK contract).
        /// </summary>
        public static Vector3 TransformNormalFromWorldToObjectSpace(Vector3 normal)
        {
            if (TransformListSize == 0)
                return normal;
            GetObjectToWorldTransformMatrix(out var m0, out var m1, out var m2);
            return OptixTransforms.TransformNormal(m0, m1, m2, normal);
        }

        /// <summary>Provides the functionality of optixHitObjectTransformPointFromObjectToWorldSpace.</summary>
        public static Vector3 TransformPointFromObjectToWorldSpace(Vector3 point)
        {
            if (TransformListSize == 0)
                return point;
            GetObjectToWorldTransformMatrix(out var m0, out var m1, out var m2);
            return OptixTransforms.TransformPoint(m0, m1, m2, point);
        }

        /// <summary>Provides the functionality of optixHitObjectTransformVectorFromObjectToWorldSpace.</summary>
        public static Vector3 TransformVectorFromObjectToWorldSpace(Vector3 vector)
        {
            if (TransformListSize == 0)
                return vector;
            GetObjectToWorldTransformMatrix(out var m0, out var m1, out var m2);
            return OptixTransforms.TransformVector(m0, m1, m2, vector);
        }

        /// <summary>
        /// Provides the functionality of
        /// optixHitObjectTransformNormalFromObjectToWorldSpace (transpose-multiplies
        /// with the world-to-object matrix, per the SDK contract).
        /// </summary>
        public static Vector3 TransformNormalFromObjectToWorldSpace(Vector3 normal)
        {
            if (TransformListSize == 0)
                return normal;
            GetWorldToObjectTransformMatrix(out var m0, out var m1, out var m2);
            return OptixTransforms.TransformNormal(m0, m1, m2, normal);
        }
    }
}
