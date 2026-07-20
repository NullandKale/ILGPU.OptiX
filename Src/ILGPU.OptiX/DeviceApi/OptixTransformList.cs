// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixTransformList.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the transform-list built-in functions
    /// (optixGetTransformListSize, optixGetTransformListHandle,
    /// optixGetTransformTypeFromHandle and the FromHandle pointer getters) - the raw
    /// building blocks the SDK composes world/object transforms from. Most code
    /// should use <see cref="OptixTransforms"/> instead; this class is the low-level
    /// surface for custom traversal of the transform stack. Valid in IS/AH/CH
    /// programs (the FromHandle getters are callable anywhere a handle is available).
    /// </summary>
    /// <remarks>
    /// The pointer getters return raw device addresses of the corresponding
    /// optix_types.h structs (OptixStaticTransform, OptixSRTMotionTransform,
    /// OptixMatrixMotionTransform, or the instance's 3x4 row-major transform). Read
    /// them with <see cref="OptixTransforms.LoadFloat4"/>-style 16-byte-aligned
    /// global loads, never by dereferencing managed pointers.
    /// </remarks>
    [CLSCompliant(false)]
    public static class OptixTransformList
    {
        /// <summary>
        /// Provides the functionality of optixGetTransformListSize - the number of
        /// entries on the current hit's transform stack.
        /// </summary>
        public static uint Size
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_transform_list_size, ();",
                    out uint size);
                return size;
            }
        }

        /// <summary>
        /// Provides the functionality of optixGetTransformListHandle - the
        /// traversable handle at <paramref name="index"/> on the current hit's
        /// transform stack.
        /// </summary>
        public static ulong GetHandle(uint index)
        {
            Input<uint> _index = index;
            Output<ulong> _handle = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_get_transform_list_handle, (%1);",
                ref _handle, ref _index);
            return _handle.Value;
        }

        /// <summary>
        /// Provides the functionality of optixGetTransformTypeFromHandle.
        /// </summary>
        public static OptixTransformType GetTransformType(ulong handle)
        {
            Input<ulong> _handle = handle;
            Output<uint> _type = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_get_transform_type_from_handle, (%1);",
                ref _type, ref _handle);
            return (OptixTransformType)_type.Value;
        }

        /// <summary>
        /// Provides the functionality of optixGetStaticTransformFromHandle - the
        /// device address of the OptixStaticTransform struct.
        /// </summary>
        public static ulong GetStaticTransformPointer(ulong handle)
        {
            Input<ulong> _handle = handle;
            Output<ulong> _pointer = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_get_static_transform_from_handle, (%1);",
                ref _pointer, ref _handle);
            return _pointer.Value;
        }

        /// <summary>
        /// Provides the functionality of optixGetSRTMotionTransformFromHandle - the
        /// device address of the OptixSRTMotionTransform struct.
        /// </summary>
        public static ulong GetSrtMotionTransformPointer(ulong handle)
        {
            Input<ulong> _handle = handle;
            Output<ulong> _pointer = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_get_srt_motion_transform_from_handle, (%1);",
                ref _pointer, ref _handle);
            return _pointer.Value;
        }

        /// <summary>
        /// Provides the functionality of optixGetMatrixMotionTransformFromHandle -
        /// the device address of the OptixMatrixMotionTransform struct.
        /// </summary>
        public static ulong GetMatrixMotionTransformPointer(ulong handle)
        {
            Input<ulong> _handle = handle;
            Output<ulong> _pointer = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_get_matrix_motion_transform_from_handle, (%1);",
                ref _pointer, ref _handle);
            return _pointer.Value;
        }

        /// <summary>
        /// Provides the functionality of optixGetInstanceTransformFromHandle - the
        /// device address of the instance's object-to-world 3x4 row-major transform
        /// (three 16-byte rows).
        /// </summary>
        public static ulong GetInstanceTransformPointer(ulong handle)
        {
            Input<ulong> _handle = handle;
            Output<ulong> _pointer = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_get_instance_transform_from_handle, (%1);",
                ref _pointer, ref _handle);
            return _pointer.Value;
        }

        /// <summary>
        /// Provides the functionality of optixGetInstanceInverseTransformFromHandle -
        /// the device address of the instance's world-to-object 3x4 row-major
        /// transform (three 16-byte rows).
        /// </summary>
        public static ulong GetInstanceInverseTransformPointer(ulong handle)
        {
            Input<ulong> _handle = handle;
            Output<ulong> _pointer = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_get_instance_inverse_transform_from_handle, (%1);",
                ref _pointer, ref _handle);
            return _pointer.Value;
        }

        /// <summary>
        /// Provides the functionality of optixGetInstanceIdFromHandle - the
        /// user-supplied instance id of an instance transform-list entry.
        /// </summary>
        public static uint GetInstanceIdFromHandle(ulong handle)
        {
            Input<ulong> _handle = handle;
            Output<uint> _id = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_get_instance_id_from_handle, (%1);",
                ref _id, ref _handle);
            return _id.Value;
        }

        /// <summary>
        /// Provides the functionality of optixGetInstanceChildFromHandle - the
        /// traversable handle the instance refers to.
        /// </summary>
        public static ulong GetInstanceChildFromHandle(ulong handle)
        {
            Input<ulong> _handle = handle;
            Output<ulong> _child = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_get_instance_child_from_handle, (%1);",
                ref _child, ref _handle);
            return _child.Value;
        }
    }
}
