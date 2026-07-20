// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                           Copyright (c) 2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixVertexData.tt/OptixVertexData.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------


using ILGPU.Runtime.Cuda;
using System;
using System.Numerics;

// Generated vertex-data getters for every built-in primitive type - ports of the
// optixGet*VertexData / optixGetSphereData family (optix_device.h). Each primitive
// type gets three spellings, matching the SDK:
//   FromHandle(gas, primIndex, sbtGasIndex, time, out ...) - fetch by GAS handle
//     (requires the GAS to be built with OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS);
//   CurrentHit(out ...) - the primitive of the current AH/CH invocation;
//   OptixHitObject.Get* - the primitive recorded in the current SER hit object.
// Curve control points and sphere data come back as float4s (xyz = position,
// w = radius); triangles as three float3 vertices in object space.

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetTriangleVertexData built-in function
    /// family. The three object-space vertices of a triangle primitive.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetTriangleVertexData
    {
        /// <summary>
        /// Fetches by GAS traversable handle (optixGetTriangleVertexDataFromHandle). The
        /// GAS must be built with OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
        /// </summary>
        public static void FromHandle(
            ulong gas,
            uint primitiveIndex,
            uint sbtGasIndex,
            float time,
            out Vector3 v0,
            out Vector3 v1,
            out Vector3 v2)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Input<ulong> _gas = gas;
            Input<uint> _primitiveIndex = primitiveIndex;
            Input<uint> _sbtGasIndex = sbtGasIndex;
            Input<float> _time = time;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8), " +
                "_optix_get_triangle_vertex_data_from_handle, " +
                "(%9, %10, %11, %12);"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _gas, ref _primitiveIndex, ref _sbtGasIndex, ref _time);
            v0 = new Vector3(_f0.Value, _f1.Value, _f2.Value);
            v1 = new Vector3(_f3.Value, _f4.Value, _f5.Value);
            v2 = new Vector3(_f6.Value, _f7.Value, _f8.Value);
        }

        /// <summary>
        /// Fetches the primitive of the current AH/CH invocation
        /// (the parameterless optixGetTriangleVertexData).
        /// </summary>
        public static void CurrentHit(out Vector3 v0,
            out Vector3 v1,
            out Vector3 v2)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8), " +
                "_optix_get_triangle_vertex_data_current_hit, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8);
            v0 = new Vector3(_f0.Value, _f1.Value, _f2.Value);
            v1 = new Vector3(_f3.Value, _f4.Value, _f5.Value);
            v2 = new Vector3(_f6.Value, _f7.Value, _f8.Value);
        }
    }

    /// <summary>
    /// Provides the functionality of the optixGetSphereData built-in function
    /// family. The center (xyz) and radius (w) of a sphere primitive.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetSphereData
    {
        /// <summary>
        /// Fetches by GAS traversable handle (optixGetSphereDataFromHandle). The
        /// GAS must be built with OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
        /// </summary>
        public static void FromHandle(
            ulong gas,
            uint primitiveIndex,
            uint sbtGasIndex,
            float time,
            out Vector4 data)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Input<ulong> _gas = gas;
            Input<uint> _primitiveIndex = primitiveIndex;
            Input<uint> _sbtGasIndex = sbtGasIndex;
            Input<float> _time = time;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3), " +
                "_optix_get_sphere_data_from_handle, " +
                "(%4, %5, %6, %7);"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _gas, ref _primitiveIndex, ref _sbtGasIndex, ref _time);
            data = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
        }

        /// <summary>
        /// Fetches the primitive of the current AH/CH invocation
        /// (the parameterless optixGetSphereData).
        /// </summary>
        public static void CurrentHit(out Vector4 data)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3), " +
                "_optix_get_sphere_data_current_hit, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3);
            data = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
        }
    }

    /// <summary>
    /// Provides the functionality of the optixGetLinearCurveVertexData built-in function
    /// family. The two control points (xyz = position, w = radius) of a round linear curve segment.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetLinearCurveVertexData
    {
        /// <summary>
        /// Fetches by GAS traversable handle (optixGetLinearCurveVertexDataFromHandle). The
        /// GAS must be built with OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
        /// </summary>
        public static void FromHandle(
            ulong gas,
            uint primitiveIndex,
            uint sbtGasIndex,
            float time,
            out Vector4 v0,
            out Vector4 v1)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Input<ulong> _gas = gas;
            Input<uint> _primitiveIndex = primitiveIndex;
            Input<uint> _sbtGasIndex = sbtGasIndex;
            Input<float> _time = time;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7), " +
                "_optix_get_linear_curve_vertex_data_from_handle, " +
                "(%8, %9, %10, %11);"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _gas, ref _primitiveIndex, ref _sbtGasIndex, ref _time);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
        }

        /// <summary>
        /// Fetches the primitive of the current AH/CH invocation
        /// (the parameterless optixGetLinearCurveVertexData).
        /// </summary>
        public static void CurrentHit(out Vector4 v0,
            out Vector4 v1)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7), " +
                "_optix_get_linear_curve_vertex_data_current_hit, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
        }
    }

    /// <summary>
    /// Provides the functionality of the optixGetQuadraticBSplineVertexData built-in function
    /// family. The three control points (xyz = position, w = radius) of a round quadratic B-spline segment.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetQuadraticBSplineVertexData
    {
        /// <summary>
        /// Fetches by GAS traversable handle (optixGetQuadraticBSplineVertexDataFromHandle). The
        /// GAS must be built with OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
        /// </summary>
        public static void FromHandle(
            ulong gas,
            uint primitiveIndex,
            uint sbtGasIndex,
            float time,
            out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Input<ulong> _gas = gas;
            Input<uint> _primitiveIndex = primitiveIndex;
            Input<uint> _sbtGasIndex = sbtGasIndex;
            Input<float> _time = time;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11), " +
                "_optix_get_quadratic_bspline_vertex_data_from_handle, " +
                "(%12, %13, %14, %15);"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _gas, ref _primitiveIndex, ref _sbtGasIndex, ref _time);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
        }

        /// <summary>
        /// Fetches the primitive of the current AH/CH invocation
        /// (the parameterless optixGetQuadraticBSplineVertexData).
        /// </summary>
        public static void CurrentHit(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11), " +
                "_optix_get_quadratic_bspline_vertex_data_current_hit, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
        }
    }

    /// <summary>
    /// Provides the functionality of the optixGetQuadraticBSplineRocapsVertexData built-in function
    /// family. The three control points (xyz = position, w = radius) of a round quadratic B-spline segment with rocaps.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetQuadraticBSplineRocapsVertexData
    {
        /// <summary>
        /// Fetches by GAS traversable handle (optixGetQuadraticBSplineRocapsVertexDataFromHandle). The
        /// GAS must be built with OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
        /// </summary>
        public static void FromHandle(
            ulong gas,
            uint primitiveIndex,
            uint sbtGasIndex,
            float time,
            out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Input<ulong> _gas = gas;
            Input<uint> _primitiveIndex = primitiveIndex;
            Input<uint> _sbtGasIndex = sbtGasIndex;
            Input<float> _time = time;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11), " +
                "_optix_get_quadratic_bspline_rocaps_vertex_data_from_handle, " +
                "(%12, %13, %14, %15);"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _gas, ref _primitiveIndex, ref _sbtGasIndex, ref _time);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
        }

        /// <summary>
        /// Fetches the primitive of the current AH/CH invocation
        /// (the parameterless optixGetQuadraticBSplineRocapsVertexData).
        /// </summary>
        public static void CurrentHit(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11), " +
                "_optix_get_quadratic_bspline_rocaps_vertex_data_current_hit, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
        }
    }

    /// <summary>
    /// Provides the functionality of the optixGetCubicBSplineVertexData built-in function
    /// family. The four control points (xyz = position, w = radius) of a round cubic B-spline segment.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetCubicBSplineVertexData
    {
        /// <summary>
        /// Fetches by GAS traversable handle (optixGetCubicBSplineVertexDataFromHandle). The
        /// GAS must be built with OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
        /// </summary>
        public static void FromHandle(
            ulong gas,
            uint primitiveIndex,
            uint sbtGasIndex,
            float time,
            out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            Input<ulong> _gas = gas;
            Input<uint> _primitiveIndex = primitiveIndex;
            Input<uint> _sbtGasIndex = sbtGasIndex;
            Input<float> _time = time;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_get_cubic_bspline_vertex_data_from_handle, " +
                "(%16, %17, %18, %19);"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15,
                ref _gas, ref _primitiveIndex, ref _sbtGasIndex, ref _time);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }

        /// <summary>
        /// Fetches the primitive of the current AH/CH invocation
        /// (the parameterless optixGetCubicBSplineVertexData).
        /// </summary>
        public static void CurrentHit(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_get_cubic_bspline_vertex_data_current_hit, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }
    }

    /// <summary>
    /// Provides the functionality of the optixGetCubicBSplineRocapsVertexData built-in function
    /// family. The four control points (xyz = position, w = radius) of a round cubic B-spline segment with rocaps.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetCubicBSplineRocapsVertexData
    {
        /// <summary>
        /// Fetches by GAS traversable handle (optixGetCubicBSplineRocapsVertexDataFromHandle). The
        /// GAS must be built with OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
        /// </summary>
        public static void FromHandle(
            ulong gas,
            uint primitiveIndex,
            uint sbtGasIndex,
            float time,
            out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            Input<ulong> _gas = gas;
            Input<uint> _primitiveIndex = primitiveIndex;
            Input<uint> _sbtGasIndex = sbtGasIndex;
            Input<float> _time = time;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_get_cubic_bspline_rocaps_vertex_data_from_handle, " +
                "(%16, %17, %18, %19);"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15,
                ref _gas, ref _primitiveIndex, ref _sbtGasIndex, ref _time);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }

        /// <summary>
        /// Fetches the primitive of the current AH/CH invocation
        /// (the parameterless optixGetCubicBSplineRocapsVertexData).
        /// </summary>
        public static void CurrentHit(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_get_cubic_bspline_rocaps_vertex_data_current_hit, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }
    }

    /// <summary>
    /// Provides the functionality of the optixGetCatmullRomVertexData built-in function
    /// family. The four control points (xyz = position, w = radius) of a round CatmullRom segment.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetCatmullRomVertexData
    {
        /// <summary>
        /// Fetches by GAS traversable handle (optixGetCatmullRomVertexDataFromHandle). The
        /// GAS must be built with OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
        /// </summary>
        public static void FromHandle(
            ulong gas,
            uint primitiveIndex,
            uint sbtGasIndex,
            float time,
            out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            Input<ulong> _gas = gas;
            Input<uint> _primitiveIndex = primitiveIndex;
            Input<uint> _sbtGasIndex = sbtGasIndex;
            Input<float> _time = time;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_get_catmullrom_vertex_data_from_handle, " +
                "(%16, %17, %18, %19);"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15,
                ref _gas, ref _primitiveIndex, ref _sbtGasIndex, ref _time);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }

        /// <summary>
        /// Fetches the primitive of the current AH/CH invocation
        /// (the parameterless optixGetCatmullRomVertexData).
        /// </summary>
        public static void CurrentHit(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_get_catmullrom_vertex_data_current_hit, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }
    }

    /// <summary>
    /// Provides the functionality of the optixGetCatmullRomRocapsVertexData built-in function
    /// family. The four control points (xyz = position, w = radius) of a round CatmullRom segment with rocaps.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetCatmullRomRocapsVertexData
    {
        /// <summary>
        /// Fetches by GAS traversable handle (optixGetCatmullRomRocapsVertexDataFromHandle). The
        /// GAS must be built with OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
        /// </summary>
        public static void FromHandle(
            ulong gas,
            uint primitiveIndex,
            uint sbtGasIndex,
            float time,
            out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            Input<ulong> _gas = gas;
            Input<uint> _primitiveIndex = primitiveIndex;
            Input<uint> _sbtGasIndex = sbtGasIndex;
            Input<float> _time = time;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_get_catmullrom_rocaps_vertex_data_from_handle, " +
                "(%16, %17, %18, %19);"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15,
                ref _gas, ref _primitiveIndex, ref _sbtGasIndex, ref _time);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }

        /// <summary>
        /// Fetches the primitive of the current AH/CH invocation
        /// (the parameterless optixGetCatmullRomRocapsVertexData).
        /// </summary>
        public static void CurrentHit(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_get_catmullrom_rocaps_vertex_data_current_hit, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }
    }

    /// <summary>
    /// Provides the functionality of the optixGetCubicBezierVertexData built-in function
    /// family. The four control points (xyz = position, w = radius) of a round cubic Bezier segment.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetCubicBezierVertexData
    {
        /// <summary>
        /// Fetches by GAS traversable handle (optixGetCubicBezierVertexDataFromHandle). The
        /// GAS must be built with OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
        /// </summary>
        public static void FromHandle(
            ulong gas,
            uint primitiveIndex,
            uint sbtGasIndex,
            float time,
            out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            Input<ulong> _gas = gas;
            Input<uint> _primitiveIndex = primitiveIndex;
            Input<uint> _sbtGasIndex = sbtGasIndex;
            Input<float> _time = time;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_get_cubic_bezier_vertex_data_from_handle, " +
                "(%16, %17, %18, %19);"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15,
                ref _gas, ref _primitiveIndex, ref _sbtGasIndex, ref _time);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }

        /// <summary>
        /// Fetches the primitive of the current AH/CH invocation
        /// (the parameterless optixGetCubicBezierVertexData).
        /// </summary>
        public static void CurrentHit(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_get_cubic_bezier_vertex_data_current_hit, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }
    }

    /// <summary>
    /// Provides the functionality of the optixGetCubicBezierRocapsVertexData built-in function
    /// family. The four control points (xyz = position, w = radius) of a round cubic Bezier segment with rocaps.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetCubicBezierRocapsVertexData
    {
        /// <summary>
        /// Fetches by GAS traversable handle (optixGetCubicBezierRocapsVertexDataFromHandle). The
        /// GAS must be built with OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
        /// </summary>
        public static void FromHandle(
            ulong gas,
            uint primitiveIndex,
            uint sbtGasIndex,
            float time,
            out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            Input<ulong> _gas = gas;
            Input<uint> _primitiveIndex = primitiveIndex;
            Input<uint> _sbtGasIndex = sbtGasIndex;
            Input<float> _time = time;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_get_cubic_bezier_rocaps_vertex_data_from_handle, " +
                "(%16, %17, %18, %19);"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15,
                ref _gas, ref _primitiveIndex, ref _sbtGasIndex, ref _time);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }

        /// <summary>
        /// Fetches the primitive of the current AH/CH invocation
        /// (the parameterless optixGetCubicBezierRocapsVertexData).
        /// </summary>
        public static void CurrentHit(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_get_cubic_bezier_rocaps_vertex_data_current_hit, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }
    }

    /// <summary>
    /// Provides the functionality of the optixGetRibbonVertexData built-in function
    /// family. The three control points (xyz = position, w = width) of a flat quadratic B-spline (ribbon) segment.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetRibbonVertexData
    {
        /// <summary>
        /// Fetches by GAS traversable handle (optixGetRibbonVertexDataFromHandle). The
        /// GAS must be built with OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
        /// </summary>
        public static void FromHandle(
            ulong gas,
            uint primitiveIndex,
            uint sbtGasIndex,
            float time,
            out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Input<ulong> _gas = gas;
            Input<uint> _primitiveIndex = primitiveIndex;
            Input<uint> _sbtGasIndex = sbtGasIndex;
            Input<float> _time = time;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11), " +
                "_optix_get_ribbon_vertex_data_from_handle, " +
                "(%12, %13, %14, %15);"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _gas, ref _primitiveIndex, ref _sbtGasIndex, ref _time);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
        }

        /// <summary>
        /// Fetches the primitive of the current AH/CH invocation
        /// (the parameterless optixGetRibbonVertexData).
        /// </summary>
        public static void CurrentHit(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11), " +
                "_optix_get_ribbon_vertex_data_current_hit, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
        }
    }

    public static partial class OptixHitObject
    {
        /// <summary>
        /// Provides the functionality of optixHitObjectGetTriangleVertexData -
        /// fetches the primitive recorded in the current SER hit object.
        /// The three object-space vertices of a triangle primitive.
        /// </summary>
        [CLSCompliant(false)]
        public static void GetTriangleVertexData(out Vector3 v0,
            out Vector3 v1,
            out Vector3 v2)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8), " +
                "_optix_hitobject_get_triangle_vertex_data, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8);
            v0 = new Vector3(_f0.Value, _f1.Value, _f2.Value);
            v1 = new Vector3(_f3.Value, _f4.Value, _f5.Value);
            v2 = new Vector3(_f6.Value, _f7.Value, _f8.Value);
        }

        /// <summary>
        /// Provides the functionality of optixHitObjectGetSphereData -
        /// fetches the primitive recorded in the current SER hit object.
        /// The center (xyz) and radius (w) of a sphere primitive.
        /// </summary>
        [CLSCompliant(false)]
        public static void GetSphereData(out Vector4 data)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3), " +
                "_optix_hitobject_get_sphere_data, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3);
            data = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
        }

        /// <summary>
        /// Provides the functionality of optixHitObjectGetLinearCurveVertexData -
        /// fetches the primitive recorded in the current SER hit object.
        /// The two control points (xyz = position, w = radius) of a round linear curve segment.
        /// </summary>
        [CLSCompliant(false)]
        public static void GetLinearCurveVertexData(out Vector4 v0,
            out Vector4 v1)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7), " +
                "_optix_hitobject_get_linear_curve_vertex_data, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
        }

        /// <summary>
        /// Provides the functionality of optixHitObjectGetQuadraticBSplineVertexData -
        /// fetches the primitive recorded in the current SER hit object.
        /// The three control points (xyz = position, w = radius) of a round quadratic B-spline segment.
        /// </summary>
        [CLSCompliant(false)]
        public static void GetQuadraticBSplineVertexData(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11), " +
                "_optix_hitobject_get_quadratic_bspline_vertex_data, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
        }

        /// <summary>
        /// Provides the functionality of optixHitObjectGetQuadraticBSplineRocapsVertexData -
        /// fetches the primitive recorded in the current SER hit object.
        /// The three control points (xyz = position, w = radius) of a round quadratic B-spline segment with rocaps.
        /// </summary>
        [CLSCompliant(false)]
        public static void GetQuadraticBSplineRocapsVertexData(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11), " +
                "_optix_hitobject_get_quadratic_bspline_rocaps_vertex_data, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
        }

        /// <summary>
        /// Provides the functionality of optixHitObjectGetCubicBSplineVertexData -
        /// fetches the primitive recorded in the current SER hit object.
        /// The four control points (xyz = position, w = radius) of a round cubic B-spline segment.
        /// </summary>
        [CLSCompliant(false)]
        public static void GetCubicBSplineVertexData(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_hitobject_get_cubic_bspline_vertex_data, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }

        /// <summary>
        /// Provides the functionality of optixHitObjectGetCubicBSplineRocapsVertexData -
        /// fetches the primitive recorded in the current SER hit object.
        /// The four control points (xyz = position, w = radius) of a round cubic B-spline segment with rocaps.
        /// </summary>
        [CLSCompliant(false)]
        public static void GetCubicBSplineRocapsVertexData(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_hitobject_get_cubic_bspline_rocaps_vertex_data, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }

        /// <summary>
        /// Provides the functionality of optixHitObjectGetCatmullRomVertexData -
        /// fetches the primitive recorded in the current SER hit object.
        /// The four control points (xyz = position, w = radius) of a round CatmullRom segment.
        /// </summary>
        [CLSCompliant(false)]
        public static void GetCatmullRomVertexData(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_hitobject_get_catmullrom_vertex_data, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }

        /// <summary>
        /// Provides the functionality of optixHitObjectGetCatmullRomRocapsVertexData -
        /// fetches the primitive recorded in the current SER hit object.
        /// The four control points (xyz = position, w = radius) of a round CatmullRom segment with rocaps.
        /// </summary>
        [CLSCompliant(false)]
        public static void GetCatmullRomRocapsVertexData(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_hitobject_get_catmullrom_rocaps_vertex_data, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }

        /// <summary>
        /// Provides the functionality of optixHitObjectGetCubicBezierVertexData -
        /// fetches the primitive recorded in the current SER hit object.
        /// The four control points (xyz = position, w = radius) of a round cubic Bezier segment.
        /// </summary>
        [CLSCompliant(false)]
        public static void GetCubicBezierVertexData(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_hitobject_get_cubic_bezier_vertex_data, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }

        /// <summary>
        /// Provides the functionality of optixHitObjectGetCubicBezierRocapsVertexData -
        /// fetches the primitive recorded in the current SER hit object.
        /// The four control points (xyz = position, w = radius) of a round cubic Bezier segment with rocaps.
        /// </summary>
        [CLSCompliant(false)]
        public static void GetCubicBezierRocapsVertexData(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2,
            out Vector4 v3)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            Output<float> _f12 = default;
            Output<float> _f13 = default;
            Output<float> _f14 = default;
            Output<float> _f15 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15), " +
                "_optix_hitobject_get_cubic_bezier_rocaps_vertex_data, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11,
                ref _f12,
                ref _f13,
                ref _f14,
                ref _f15);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
            v3 = new Vector4(_f12.Value, _f13.Value, _f14.Value, _f15.Value);
        }

        /// <summary>
        /// Provides the functionality of optixHitObjectGetRibbonVertexData -
        /// fetches the primitive recorded in the current SER hit object.
        /// The three control points (xyz = position, w = width) of a flat quadratic B-spline (ribbon) segment.
        /// </summary>
        [CLSCompliant(false)]
        public static void GetRibbonVertexData(out Vector4 v0,
            out Vector4 v1,
            out Vector4 v2)
        {
            Output<float> _f0 = default;
            Output<float> _f1 = default;
            Output<float> _f2 = default;
            Output<float> _f3 = default;
            Output<float> _f4 = default;
            Output<float> _f5 = default;
            Output<float> _f6 = default;
            Output<float> _f7 = default;
            Output<float> _f8 = default;
            Output<float> _f9 = default;
            Output<float> _f10 = default;
            Output<float> _f11 = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11), " +
                "_optix_hitobject_get_ribbon_vertex_data, ();"
,
                ref _f0,
                ref _f1,
                ref _f2,
                ref _f3,
                ref _f4,
                ref _f5,
                ref _f6,
                ref _f7,
                ref _f8,
                ref _f9,
                ref _f10,
                ref _f11);
            v0 = new Vector4(_f0.Value, _f1.Value, _f2.Value, _f3.Value);
            v1 = new Vector4(_f4.Value, _f5.Value, _f6.Value, _f7.Value);
            v2 = new Vector4(_f8.Value, _f9.Value, _f10.Value, _f11.Value);
        }

    }
}
