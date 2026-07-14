// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                           Copyright (c) 2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixHitObjectTraverse.tt/OptixHitObjectTraverse.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------


using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of optixTraverse - the first step of the
    /// Shader Execution Reordering (SER) 3-step pattern: traverse-then-reorder-then-invoke instead of a single optixTrace call,
    /// letting the driver batch coherent hits together (via <see cref="OptixReorder"/>)
    /// before running any hit/miss program (<see cref="Invoke"/>). Traverse() alone
    /// does NOT run the hit/miss program - it only finds the closest hit and leaves the
    /// result in an implicit per-thread "hit object" state, same underlying registers
    /// optixTrace itself uses (confirmed via internal/optix_device_impl.h:
    /// _optix_hitobject_traverse has the exact same pseudo-call argument shape as
    /// _optix_trace_typed_32 - this template mirrors OptixTrace.tt's generation
    /// approach directly for that reason).
    /// </summary>
    public static partial class OptixHitObject
    {
        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying no payload registers -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            )
        {
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %1, 0; mov.u32 %2, 0; call (%0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0), _optix_hitobject_traverse, (%1, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %2, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0);"
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 1 payload register (p0) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %2, 0; mov.u32 %3, 1; call (%0, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1), _optix_hitobject_traverse, (%2, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %3, %0, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1);"
                , ref _p0
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 2 payload registers (p0..p1) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %3, 0; mov.u32 %4, 2; call (%0, %1, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2), _optix_hitobject_traverse, (%3, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %4, %0, %1, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2, %2);"
                , ref _p0
                , ref _p1
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 3 payload registers (p0..p2) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %4, 0; mov.u32 %5, 3; call (%0, %1, %2, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3), _optix_hitobject_traverse, (%4, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %5, %0, %1, %2, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3, %3);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 4 payload registers (p0..p3) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %5, 0; mov.u32 %6, 4; call (%0, %1, %2, %3, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4), _optix_hitobject_traverse, (%5, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %6, %0, %1, %2, %3, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4, %4);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 5 payload registers (p0..p4) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %6, 0; mov.u32 %7, 5; call (%0, %1, %2, %3, %4, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5), _optix_hitobject_traverse, (%6, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %7, %0, %1, %2, %3, %4, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5, %5);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 6 payload registers (p0..p5) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %7, 0; mov.u32 %8, 6; call (%0, %1, %2, %3, %4, %5, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6), _optix_hitobject_traverse, (%7, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %8, %0, %1, %2, %3, %4, %5, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6, %6);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 7 payload registers (p0..p6) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %8, 0; mov.u32 %9, 7; call (%0, %1, %2, %3, %4, %5, %6, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7), _optix_hitobject_traverse, (%8, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %9, %0, %1, %2, %3, %4, %5, %6, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7, %7);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 8 payload registers (p0..p7) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %9, 0; mov.u32 %10, 8; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8), _optix_hitobject_traverse, (%9, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %10, %0, %1, %2, %3, %4, %5, %6, %7, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8, %8);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 9 payload registers (p0..p8) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %10, 0; mov.u32 %11, 9; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9), _optix_hitobject_traverse, (%10, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %11, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9, %9);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 10 payload registers (p0..p9) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %11, 0; mov.u32 %12, 10; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10), _optix_hitobject_traverse, (%11, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %12, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10, %10);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 11 payload registers (p0..p10) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            , ref uint p10
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %12, 0; mov.u32 %13, 11; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11), _optix_hitobject_traverse, (%12, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %13, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11, %11);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 12 payload registers (p0..p11) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            , ref uint p10
            , ref uint p11
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %13, 0; mov.u32 %14, 12; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12), _optix_hitobject_traverse, (%13, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %14, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12, %12);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 13 payload registers (p0..p12) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            , ref uint p10
            , ref uint p11
            , ref uint p12
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %14, 0; mov.u32 %15, 13; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13), _optix_hitobject_traverse, (%14, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %15, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13, %13);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 14 payload registers (p0..p13) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            , ref uint p10
            , ref uint p11
            , ref uint p12
            , ref uint p13
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %15, 0; mov.u32 %16, 14; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14), _optix_hitobject_traverse, (%15, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %16, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14, %14);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 15 payload registers (p0..p14) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            , ref uint p10
            , ref uint p11
            , ref uint p12
            , ref uint p13
            , ref uint p14
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %16, 0; mov.u32 %17, 15; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15), _optix_hitobject_traverse, (%16, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %17, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15, %15);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 16 payload registers (p0..p15) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            , ref uint p10
            , ref uint p11
            , ref uint p12
            , ref uint p13
            , ref uint p14
            , ref uint p15
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %17, 0; mov.u32 %18, 16; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16), _optix_hitobject_traverse, (%17, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %18, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16, %16);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 17 payload registers (p0..p16) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            , ref uint p10
            , ref uint p11
            , ref uint p12
            , ref uint p13
            , ref uint p14
            , ref uint p15
            , ref uint p16
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %18, 0; mov.u32 %19, 17; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17), _optix_hitobject_traverse, (%18, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %19, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17, %17);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 18 payload registers (p0..p17) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            , ref uint p10
            , ref uint p11
            , ref uint p12
            , ref uint p13
            , ref uint p14
            , ref uint p15
            , ref uint p16
            , ref uint p17
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %19, 0; mov.u32 %20, 18; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18), _optix_hitobject_traverse, (%19, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %20, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18, %18);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 19 payload registers (p0..p18) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            , ref uint p10
            , ref uint p11
            , ref uint p12
            , ref uint p13
            , ref uint p14
            , ref uint p15
            , ref uint p16
            , ref uint p17
            , ref uint p18
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _p18 = p18;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %20, 0; mov.u32 %21, 19; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %19, %19, %19, %19, %19, %19, %19, %19, %19, %19, %19, %19), _optix_hitobject_traverse, (%20, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %21, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %19, %19, %19, %19, %19, %19, %19, %19, %19, %19, %19, %19);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _p18
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
            p18 = _p18.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 20 payload registers (p0..p19) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            , ref uint p10
            , ref uint p11
            , ref uint p12
            , ref uint p13
            , ref uint p14
            , ref uint p15
            , ref uint p16
            , ref uint p17
            , ref uint p18
            , ref uint p19
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _p18 = p18;
            Ref<uint> _p19 = p19;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %21, 0; mov.u32 %22, 20; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %20, %20, %20, %20, %20, %20, %20, %20, %20, %20, %20), _optix_hitobject_traverse, (%21, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %22, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %20, %20, %20, %20, %20, %20, %20, %20, %20, %20, %20);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _p18
                , ref _p19
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
            p18 = _p18.Value;
            p19 = _p19.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 21 payload registers (p0..p20) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            , ref uint p10
            , ref uint p11
            , ref uint p12
            , ref uint p13
            , ref uint p14
            , ref uint p15
            , ref uint p16
            , ref uint p17
            , ref uint p18
            , ref uint p19
            , ref uint p20
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _p18 = p18;
            Ref<uint> _p19 = p19;
            Ref<uint> _p20 = p20;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %22, 0; mov.u32 %23, 21; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %21, %21, %21, %21, %21, %21, %21, %21, %21, %21), _optix_hitobject_traverse, (%22, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %23, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %21, %21, %21, %21, %21, %21, %21, %21, %21, %21);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _p18
                , ref _p19
                , ref _p20
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
            p18 = _p18.Value;
            p19 = _p19.Value;
            p20 = _p20.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 22 payload registers (p0..p21) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            , ref uint p10
            , ref uint p11
            , ref uint p12
            , ref uint p13
            , ref uint p14
            , ref uint p15
            , ref uint p16
            , ref uint p17
            , ref uint p18
            , ref uint p19
            , ref uint p20
            , ref uint p21
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _p18 = p18;
            Ref<uint> _p19 = p19;
            Ref<uint> _p20 = p20;
            Ref<uint> _p21 = p21;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %23, 0; mov.u32 %24, 22; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %22, %22, %22, %22, %22, %22, %22, %22, %22), _optix_hitobject_traverse, (%23, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %24, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %22, %22, %22, %22, %22, %22, %22, %22, %22);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _p18
                , ref _p19
                , ref _p20
                , ref _p21
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
            p18 = _p18.Value;
            p19 = _p19.Value;
            p20 = _p20.Value;
            p21 = _p21.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 23 payload registers (p0..p22) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            , ref uint p10
            , ref uint p11
            , ref uint p12
            , ref uint p13
            , ref uint p14
            , ref uint p15
            , ref uint p16
            , ref uint p17
            , ref uint p18
            , ref uint p19
            , ref uint p20
            , ref uint p21
            , ref uint p22
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _p18 = p18;
            Ref<uint> _p19 = p19;
            Ref<uint> _p20 = p20;
            Ref<uint> _p21 = p21;
            Ref<uint> _p22 = p22;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %24, 0; mov.u32 %25, 23; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %23, %23, %23, %23, %23, %23, %23, %23), _optix_hitobject_traverse, (%24, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %25, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %23, %23, %23, %23, %23, %23, %23, %23);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _p18
                , ref _p19
                , ref _p20
                , ref _p21
                , ref _p22
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
            p18 = _p18.Value;
            p19 = _p19.Value;
            p20 = _p20.Value;
            p21 = _p21.Value;
            p22 = _p22.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 24 payload registers (p0..p23) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            , ref uint p10
            , ref uint p11
            , ref uint p12
            , ref uint p13
            , ref uint p14
            , ref uint p15
            , ref uint p16
            , ref uint p17
            , ref uint p18
            , ref uint p19
            , ref uint p20
            , ref uint p21
            , ref uint p22
            , ref uint p23
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _p18 = p18;
            Ref<uint> _p19 = p19;
            Ref<uint> _p20 = p20;
            Ref<uint> _p21 = p21;
            Ref<uint> _p22 = p22;
            Ref<uint> _p23 = p23;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %25, 0; mov.u32 %26, 24; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %24, %24, %24, %24, %24, %24, %24), _optix_hitobject_traverse, (%25, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %26, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %24, %24, %24, %24, %24, %24, %24);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _p18
                , ref _p19
                , ref _p20
                , ref _p21
                , ref _p22
                , ref _p23
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
            p18 = _p18.Value;
            p19 = _p19.Value;
            p20 = _p20.Value;
            p21 = _p21.Value;
            p22 = _p22.Value;
            p23 = _p23.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 25 payload registers (p0..p24) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            , ref uint p10
            , ref uint p11
            , ref uint p12
            , ref uint p13
            , ref uint p14
            , ref uint p15
            , ref uint p16
            , ref uint p17
            , ref uint p18
            , ref uint p19
            , ref uint p20
            , ref uint p21
            , ref uint p22
            , ref uint p23
            , ref uint p24
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _p18 = p18;
            Ref<uint> _p19 = p19;
            Ref<uint> _p20 = p20;
            Ref<uint> _p21 = p21;
            Ref<uint> _p22 = p22;
            Ref<uint> _p23 = p23;
            Ref<uint> _p24 = p24;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %26, 0; mov.u32 %27, 25; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %25, %25, %25, %25, %25, %25), _optix_hitobject_traverse, (%26, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %27, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %25, %25, %25, %25, %25, %25);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _p18
                , ref _p19
                , ref _p20
                , ref _p21
                , ref _p22
                , ref _p23
                , ref _p24
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
            p18 = _p18.Value;
            p19 = _p19.Value;
            p20 = _p20.Value;
            p21 = _p21.Value;
            p22 = _p22.Value;
            p23 = _p23.Value;
            p24 = _p24.Value;
        }

        /// <summary>
        /// Traverses (but does not yet shade) a ray, carrying 26 payload registers (p0..p25) -
        /// call <see cref="OptixReorder"/> then the matching <c>Invoke(...)</c>
        /// overload (same payload count) to actually run the resulting hit/miss
        /// program.
        /// </summary>
        public static void Traverse(
            ulong traversableHandle
            , (float X, float Y, float Z) rayOrigin
            , (float X, float Y, float Z) rayDirection
            , float tmin
            , float tmax
            , float rayTime
            , uint visibilityMask
            , OptixRayFlags rayFlags
            , uint SbtOffset
            , uint SbtStride
            , uint missSbtIndex
            , ref uint p0
            , ref uint p1
            , ref uint p2
            , ref uint p3
            , ref uint p4
            , ref uint p5
            , ref uint p6
            , ref uint p7
            , ref uint p8
            , ref uint p9
            , ref uint p10
            , ref uint p11
            , ref uint p12
            , ref uint p13
            , ref uint p14
            , ref uint p15
            , ref uint p16
            , ref uint p17
            , ref uint p18
            , ref uint p19
            , ref uint p20
            , ref uint p21
            , ref uint p22
            , ref uint p23
            , ref uint p24
            , ref uint p25
            )
        {
            Ref<uint> _p0 = p0;
            Ref<uint> _p1 = p1;
            Ref<uint> _p2 = p2;
            Ref<uint> _p3 = p3;
            Ref<uint> _p4 = p4;
            Ref<uint> _p5 = p5;
            Ref<uint> _p6 = p6;
            Ref<uint> _p7 = p7;
            Ref<uint> _p8 = p8;
            Ref<uint> _p9 = p9;
            Ref<uint> _p10 = p10;
            Ref<uint> _p11 = p11;
            Ref<uint> _p12 = p12;
            Ref<uint> _p13 = p13;
            Ref<uint> _p14 = p14;
            Ref<uint> _p15 = p15;
            Ref<uint> _p16 = p16;
            Ref<uint> _p17 = p17;
            Ref<uint> _p18 = p18;
            Ref<uint> _p19 = p19;
            Ref<uint> _p20 = p20;
            Ref<uint> _p21 = p21;
            Ref<uint> _p22 = p22;
            Ref<uint> _p23 = p23;
            Ref<uint> _p24 = p24;
            Ref<uint> _p25 = p25;
            Ref<uint> _pad = 0u;
            Output<uint> _type = default;
            Output<uint> _count = default;
            Input<ulong> _traversableHandle = traversableHandle;
            Input<float> _ox = rayOrigin.X;
            Input<float> _oy = rayOrigin.Y;
            Input<float> _oz = rayOrigin.Z;
            Input<float> _dx = rayDirection.X;
            Input<float> _dy = rayDirection.Y;
            Input<float> _dz = rayDirection.Z;
            Input<float> _tmin = tmin;
            Input<float> _tmax = tmax;
            Input<float> _rayTime = rayTime;
            Input<uint> _visibilityMask = visibilityMask;
            Input<OptixRayFlags> _rayFlags = rayFlags;
            Input<uint> _SbtOffset = SbtOffset;
            Input<uint> _SbtStride = SbtStride;
            Input<uint> _missSbtIndex = missSbtIndex;

            CudaAsm.EmitRef(
                "mov.u32 %27, 0; mov.u32 %28, 26; call (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %26, %26, %26, %26, %26), _optix_hitobject_traverse, (%27, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %28, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %26, %26, %26, %26, %26);"
                , ref _p0
                , ref _p1
                , ref _p2
                , ref _p3
                , ref _p4
                , ref _p5
                , ref _p6
                , ref _p7
                , ref _p8
                , ref _p9
                , ref _p10
                , ref _p11
                , ref _p12
                , ref _p13
                , ref _p14
                , ref _p15
                , ref _p16
                , ref _p17
                , ref _p18
                , ref _p19
                , ref _p20
                , ref _p21
                , ref _p22
                , ref _p23
                , ref _p24
                , ref _p25
                , ref _pad
                , ref _type
                , ref _count
                , ref _traversableHandle
                , ref _ox
                , ref _oy
                , ref _oz
                , ref _dx
                , ref _dy
                , ref _dz
                , ref _tmin
                , ref _tmax
                , ref _rayTime
                , ref _visibilityMask
                , ref _rayFlags
                , ref _SbtOffset
                , ref _SbtStride
                , ref _missSbtIndex
                );

            p0 = _p0.Value;
            p1 = _p1.Value;
            p2 = _p2.Value;
            p3 = _p3.Value;
            p4 = _p4.Value;
            p5 = _p5.Value;
            p6 = _p6.Value;
            p7 = _p7.Value;
            p8 = _p8.Value;
            p9 = _p9.Value;
            p10 = _p10.Value;
            p11 = _p11.Value;
            p12 = _p12.Value;
            p13 = _p13.Value;
            p14 = _p14.Value;
            p15 = _p15.Value;
            p16 = _p16.Value;
            p17 = _p17.Value;
            p18 = _p18.Value;
            p19 = _p19.Value;
            p20 = _p20.Value;
            p21 = _p21.Value;
            p22 = _p22.Value;
            p23 = _p23.Value;
            p24 = _p24.Value;
            p25 = _p25.Value;
        }

    }
}
