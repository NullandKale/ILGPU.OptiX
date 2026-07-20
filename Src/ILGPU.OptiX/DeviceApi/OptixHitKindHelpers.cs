// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixHitKindHelpers.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX.AccelStructures;
using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the hit-kind interpretation built-in functions
    /// (optixGetPrimitiveType, optixIsFrontFaceHit, optixIsBackFaceHit,
    /// optixIsTriangleHit and friends). Valid in AH/CH programs, where a hit kind is
    /// available via <see cref="OptixGetHitKind"/>.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixHitKindHelpers
    {
        /// <summary>
        /// Mirrors OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE (optix_types.h).
        /// </summary>
        public const uint TriangleFrontFace = 0xFE;

        /// <summary>
        /// Mirrors OPTIX_HIT_KIND_TRIANGLE_BACK_FACE (optix_types.h).
        /// </summary>
        public const uint TriangleBackFace = 0xFF;

        /// <summary>
        /// Provides the functionality of optixGetPrimitiveType(hitKind) - decodes the
        /// built-in primitive type from a hit kind.
        /// </summary>
        public static OptixPrimitiveType GetPrimitiveType(uint hitKind)
        {
            Input<uint> _hitKind = hitKind;
            Output<uint> _type = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_get_primitive_type_from_hit_kind, (%1);",
                ref _type, ref _hitKind);
            return (OptixPrimitiveType)_type.Value;
        }

        /// <summary>
        /// Provides the functionality of the parameterless optixGetPrimitiveType() -
        /// the primitive type of the current hit.
        /// </summary>
        public static OptixPrimitiveType GetPrimitiveType() =>
            GetPrimitiveType(OptixGetHitKind.Value);

        /// <summary>
        /// Provides the functionality of optixIsBackFaceHit(hitKind).
        /// </summary>
        public static bool IsBackFaceHit(uint hitKind)
        {
            Input<uint> _hitKind = hitKind;
            Output<uint> _backface = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_get_backface_from_hit_kind, (%1);",
                ref _backface, ref _hitKind);
            return _backface.Value == 1;
        }

        /// <summary>
        /// Provides the functionality of the parameterless optixIsBackFaceHit().
        /// </summary>
        public static bool IsBackFaceHit() => IsBackFaceHit(OptixGetHitKind.Value);

        /// <summary>
        /// Provides the functionality of optixIsFrontFaceHit(hitKind).
        /// </summary>
        public static bool IsFrontFaceHit(uint hitKind) => !IsBackFaceHit(hitKind);

        /// <summary>
        /// Provides the functionality of the parameterless optixIsFrontFaceHit().
        /// </summary>
        public static bool IsFrontFaceHit() => IsFrontFaceHit(OptixGetHitKind.Value);

        /// <summary>
        /// Provides the functionality of optixIsTriangleFrontFaceHit().
        /// </summary>
        public static bool IsTriangleFrontFaceHit() =>
            OptixGetHitKind.Value == TriangleFrontFace;

        /// <summary>
        /// Provides the functionality of optixIsTriangleBackFaceHit().
        /// </summary>
        public static bool IsTriangleBackFaceHit() =>
            OptixGetHitKind.Value == TriangleBackFace;

        /// <summary>
        /// Provides the functionality of optixIsTriangleHit().
        /// </summary>
        public static bool IsTriangleHit() =>
            IsTriangleFrontFaceHit() || IsTriangleBackFaceHit();
    }
}
