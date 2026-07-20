// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixMakeHitObject.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;
using System.Numerics;
using System.Runtime.InteropServices;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Mirrors OptixTraverseData (optix_types.h) - the 20-word opaque payload of a
    /// SER hit object, captured via <see cref="OptixHitObject.GetTraverseData"/> and
    /// replayed via
    /// <see cref="OptixHitObject.Make(ulong, Vector3, Vector3, float, float, OptixRayFlags, in OptixTraverseData, ulong, uint)"/>.
    /// Declared as 20 individual words (not a fixed buffer) so it stays blittable
    /// and fully usable inside ILGPU device code.
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixTraverseData
    {
        public uint D0;
        public uint D1;
        public uint D2;
        public uint D3;
        public uint D4;
        public uint D5;
        public uint D6;
        public uint D7;
        public uint D8;
        public uint D9;
        public uint D10;
        public uint D11;
        public uint D12;
        public uint D13;
        public uint D14;
        public uint D15;
        public uint D16;
        public uint D17;
        public uint D18;
        public uint D19;
    }

    public static partial class OptixHitObject
    {
        /// <summary>
        /// Provides the functionality of optixMakeNopHitObject - replaces the
        /// current hit object with a no-op one (<see cref="IsNop"/> becomes true and
        /// <see cref="Invoke()"/> does nothing). Valid in RG/CH/MS programs.
        /// </summary>
        public static void MakeNop() =>
            CudaAsm.Emit("call (), _optix_hitobject_make_nop, ();");

        /// <summary>
        /// Provides the functionality of optixMakeMissHitObject - replaces the
        /// current hit object with a miss recording the given ray;
        /// <see cref="Invoke()"/> then runs the miss program at
        /// <paramref name="missSbtIndex"/>. Valid in RG/CH/MS programs.
        /// </summary>
        [CLSCompliant(false)]
        public static void MakeMiss(
            uint missSbtIndex,
            Vector3 rayOrigin,
            Vector3 rayDirection,
            float tmin,
            float tmax,
            float rayTime,
            OptixRayFlags rayFlags)
        {
            Input<uint> _missSbtIndex = missSbtIndex;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _rayFlags = (uint)rayFlags;
            CudaAsm.EmitRef(
                "call (), _optix_hitobject_make_miss_v2, " +
                "(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10);",
                ref _missSbtIndex,
                ref _ox, ref _oy, ref _oz,
                ref _dx, ref _dy, ref _dz,
                ref _tmin, ref _tmax, ref _rayTime, ref _rayFlags);
        }

        /// <summary>
        /// Provides the functionality of optixMakeHitObject (the traverse-data
        /// variant) - reconstructs a hit object from a
        /// <see cref="GetTraverseData"/> capture plus the recorded ray, e.g. to
        /// re-invoke a hit found on an earlier traversal. The transform list is
        /// supplied as a device pointer to <paramref name="numTransforms"/>
        /// traversable handles. Valid in RG/CH/MS programs.
        /// </summary>
        [CLSCompliant(false)]
        public static void Make(
            ulong gasHandle,
            Vector3 rayOrigin,
            Vector3 rayDirection,
            float tmin,
            float rayTime,
            OptixRayFlags rayFlags,
            in OptixTraverseData traverseData,
            ulong transformsPointer,
            uint numTransforms)
        {
            Input<ulong> _handle = gasHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _rayTime = rayTime;
            Input<uint> _rayFlags = (uint)rayFlags;
            Input<uint> _t0 = traverseData.D0;
            Input<uint> _t1 = traverseData.D1;
            Input<uint> _t2 = traverseData.D2;
            Input<uint> _t3 = traverseData.D3;
            Input<uint> _t4 = traverseData.D4;
            Input<uint> _t5 = traverseData.D5;
            Input<uint> _t6 = traverseData.D6;
            Input<uint> _t7 = traverseData.D7;
            Input<uint> _t8 = traverseData.D8;
            Input<uint> _t9 = traverseData.D9;
            Input<uint> _t10 = traverseData.D10;
            Input<uint> _t11 = traverseData.D11;
            Input<uint> _t12 = traverseData.D12;
            Input<uint> _t13 = traverseData.D13;
            Input<uint> _t14 = traverseData.D14;
            Input<uint> _t15 = traverseData.D15;
            Input<uint> _t16 = traverseData.D16;
            Input<uint> _t17 = traverseData.D17;
            Input<uint> _t18 = traverseData.D18;
            Input<uint> _t19 = traverseData.D19;
            Input<ulong> _transforms = transformsPointer;
            Input<uint> _numTransforms = numTransforms;
            CudaAsm.EmitRef(
                "call (), _optix_hitobject_make_with_traverse_data_v2, " +
                "(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, " +
                "%10, %11, %12, %13, %14, %15, %16, %17, %18, %19, " +
                "%20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31);",
                ref _handle,
                ref _ox, ref _oy, ref _oz,
                ref _dx, ref _dy, ref _dz,
                ref _tmin, ref _rayTime, ref _rayFlags,
                ref _t0, ref _t1, ref _t2, ref _t3, ref _t4,
                ref _t5, ref _t6, ref _t7, ref _t8, ref _t9,
                ref _t10, ref _t11, ref _t12, ref _t13, ref _t14,
                ref _t15, ref _t16, ref _t17, ref _t18, ref _t19,
                ref _transforms, ref _numTransforms);
        }

        /// <summary>
        /// Provides the functionality of optixHitObjectGetTraverseData - captures
        /// the current hit object's opaque 20-word traverse data for later
        /// reconstruction via
        /// <see cref="Make(ulong, Vector3, Vector3, float, float, OptixRayFlags, in OptixTraverseData, ulong, uint)"/>.
        /// </summary>
        [CLSCompliant(false)]
        public static void GetTraverseData(out OptixTraverseData data)
        {
            Output<uint> _t0 = default;
            Output<uint> _t1 = default;
            Output<uint> _t2 = default;
            Output<uint> _t3 = default;
            Output<uint> _t4 = default;
            Output<uint> _t5 = default;
            Output<uint> _t6 = default;
            Output<uint> _t7 = default;
            Output<uint> _t8 = default;
            Output<uint> _t9 = default;
            Output<uint> _t10 = default;
            Output<uint> _t11 = default;
            Output<uint> _t12 = default;
            Output<uint> _t13 = default;
            Output<uint> _t14 = default;
            Output<uint> _t15 = default;
            Output<uint> _t16 = default;
            Output<uint> _t17 = default;
            Output<uint> _t18 = default;
            Output<uint> _t19 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, " +
                "%10, %11, %12, %13, %14, %15, %16, %17, %18, %19), " +
                "_optix_hitobject_get_traverse_data, ();",
                ref _t0, ref _t1, ref _t2, ref _t3, ref _t4,
                ref _t5, ref _t6, ref _t7, ref _t8, ref _t9,
                ref _t10, ref _t11, ref _t12, ref _t13, ref _t14,
                ref _t15, ref _t16, ref _t17, ref _t18, ref _t19);
            data = new OptixTraverseData
            {
                D0 = _t0.Value,
                D1 = _t1.Value,
                D2 = _t2.Value,
                D3 = _t3.Value,
                D4 = _t4.Value,
                D5 = _t5.Value,
                D6 = _t6.Value,
                D7 = _t7.Value,
                D8 = _t8.Value,
                D9 = _t9.Value,
                D10 = _t10.Value,
                D11 = _t11.Value,
                D12 = _t12.Value,
                D13 = _t13.Value,
                D14 = _t14.Value,
                D15 = _t15.Value,
                D16 = _t16.Value,
                D17 = _t17.Value,
                D18 = _t18.Value,
                D19 = _t19.Value,
            };
        }
    }
}
