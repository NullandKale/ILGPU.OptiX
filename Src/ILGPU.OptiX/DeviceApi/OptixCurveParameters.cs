// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixCurveParameters.cs
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
    /// Provides the functionality of the optixGetCurveParameter built-in function -
    /// the curve parameter u in [0,1] of the current hit on a round curve segment.
    /// Valid in AH/CH programs for built-in curve primitives.
    /// </summary>
    public static class OptixGetCurveParameter
    {
        public static float Value
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_curve_parameter, ();",
                    out float u);
                return u;
            }
        }
    }

    /// <summary>
    /// Provides the functionality of the optixGetRibbonParameters built-in function -
    /// the (u, v) patch parameters of the current hit on a flat quadratic B-spline
    /// (ribbon) segment. Valid in AH/CH programs for ribbon primitives.
    /// </summary>
    public static class OptixGetRibbonParameters
    {
        public static (float U, float V) Value
        {
            get
            {
                Output<float> u = default;
                Output<float> v = default;
                CudaAsm.EmitRef(
                    "call (%0, %1), _optix_get_ribbon_parameters, ();",
                    ref u, ref v);
                return (u.Value, v.Value);
            }
        }

        /// <summary>
        /// The ribbon parameters as a <see cref="Vector2"/>.
        /// </summary>
        public static Vector2 AsVector2
        {
            get
            {
                var (u, v) = Value;
                return new Vector2(u, v);
            }
        }
    }

    /// <summary>
    /// Provides the functionality of the optixGetRibbonNormal built-in function
    /// family - the ribbon normal at given ribbon parameters, for flat quadratic
    /// B-spline (ribbon) primitives.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetRibbonNormal
    {
        /// <summary>
        /// The ribbon normal of the current hit at the given ribbon parameters
        /// (typically <see cref="OptixGetRibbonParameters"/>). Valid in AH/CH
        /// programs.
        /// </summary>
        public static Vector3 CurrentHit(Vector2 ribbonParameters)
        {
            Input<float> _u = ribbonParameters.X;
            Input<float> _v = ribbonParameters.Y;
            Output<float> _x = default;
            Output<float> _y = default;
            Output<float> _z = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2), _optix_get_ribbon_normal_current_hit, (%3, %4);",
                ref _x, ref _y, ref _z, ref _u, ref _v);
            return new Vector3(_x.Value, _y.Value, _z.Value);
        }

        /// <summary>
        /// Fetches by GAS traversable handle (optixGetRibbonNormalFromHandle). The
        /// GAS must be built with OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
        /// </summary>
        public static Vector3 FromHandle(
            ulong gas,
            uint primitiveIndex,
            uint sbtGasIndex,
            float time,
            Vector2 ribbonParameters)
        {
            Input<ulong> _gas = gas;
            Input<uint> _primitiveIndex = primitiveIndex;
            Input<uint> _sbtGasIndex = sbtGasIndex;
            Input<float> _time = time;
            Input<float> _u = ribbonParameters.X;
            Input<float> _v = ribbonParameters.Y;
            Output<float> _x = default;
            Output<float> _y = default;
            Output<float> _z = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2), _optix_get_ribbon_normal_from_handle, " +
                "(%3, %4, %5, %6, %7, %8);",
                ref _x, ref _y, ref _z,
                ref _gas, ref _primitiveIndex, ref _sbtGasIndex, ref _time,
                ref _u, ref _v);
            return new Vector3(_x.Value, _y.Value, _z.Value);
        }
    }

    public static partial class OptixHitObject
    {
        /// <summary>
        /// Provides the functionality of optixHitObjectGetCurveParameter - the curve
        /// parameter of the hit recorded in the current SER hit object.
        /// </summary>
        public static float CurveParameter
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_hitobject_get_curve_parameter, ();",
                    out float u);
                return u;
            }
        }

        /// <summary>
        /// Provides the functionality of optixHitObjectGetRibbonParameters - the
        /// (u, v) ribbon parameters of the hit recorded in the current SER hit
        /// object.
        /// </summary>
        public static (float U, float V) RibbonParameters
        {
            get
            {
                Output<float> u = default;
                Output<float> v = default;
                CudaAsm.EmitRef(
                    "call (%0, %1), _optix_hitobject_get_ribbon_parameters, ();",
                    ref u, ref v);
                return (u.Value, v.Value);
            }
        }

        /// <summary>
        /// Provides the functionality of optixHitObjectGetRibbonNormal - the ribbon
        /// normal of the hit recorded in the current SER hit object at the given
        /// ribbon parameters.
        /// </summary>
        public static Vector3 GetRibbonNormal(Vector2 ribbonParameters)
        {
            Input<float> _u = ribbonParameters.X;
            Input<float> _v = ribbonParameters.Y;
            Output<float> _x = default;
            Output<float> _y = default;
            Output<float> _z = default;
            CudaAsm.EmitRef(
                "call (%0, %1, %2), _optix_hitobject_get_ribbon_normal, (%3, %4);",
                ref _x, ref _y, ref _z, ref _u, ref _v);
            return new Vector3(_x.Value, _y.Value, _z.Value);
        }
    }
}
