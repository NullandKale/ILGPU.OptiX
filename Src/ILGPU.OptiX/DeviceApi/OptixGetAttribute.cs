// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetAttribute.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetAttribute_0..7 built-in functions -
    /// the intersection attribute registers reported by
    /// <see cref="OptixReportIntersection.Report(float, uint)"/> (custom primitives)
    /// or by the built-in intersection programs (e.g. barycentrics for triangles,
    /// curve parameter for curves). Valid in AH/CH programs. Float-valued attributes
    /// round-trip via <see cref="ILGPU.Interop.FloatAsInt(float)"/> /
    /// <see cref="ILGPU.Interop.IntAsFloat(uint)"/>.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetAttribute
    {
        public static uint Attribute0
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_get_attribute_0, ();", out uint value);
                return value;
            }
        }

        public static uint Attribute1
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_get_attribute_1, ();", out uint value);
                return value;
            }
        }

        public static uint Attribute2
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_get_attribute_2, ();", out uint value);
                return value;
            }
        }

        public static uint Attribute3
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_get_attribute_3, ();", out uint value);
                return value;
            }
        }

        public static uint Attribute4
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_get_attribute_4, ();", out uint value);
                return value;
            }
        }

        public static uint Attribute5
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_get_attribute_5, ();", out uint value);
                return value;
            }
        }

        public static uint Attribute6
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_get_attribute_6, ();", out uint value);
                return value;
            }
        }

        public static uint Attribute7
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_get_attribute_7, ();", out uint value);
                return value;
            }
        }

        /// <summary>
        /// Attribute register <paramref name="index"/> reinterpreted as float - the
        /// common case for geometric attributes.
        /// </summary>
        public static float GetFloat(int index) =>
            ILGPU.Interop.IntAsFloat(Get(index));

        /// <summary>
        /// Attribute register by runtime index (0..7). Prefer the per-index
        /// properties when the index is a compile-time constant.
        /// </summary>
        public static uint Get(int index) =>
            index switch
            {
                0 => Attribute0,
                1 => Attribute1,
                2 => Attribute2,
                3 => Attribute3,
                4 => Attribute4,
                5 => Attribute5,
                6 => Attribute6,
                _ => Attribute7,
            };
    }
}
