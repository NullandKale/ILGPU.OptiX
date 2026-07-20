// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixGetExceptionInfo.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixGetExceptionCode/Detail_0..7/LineInfo
    /// built-in functions - only valid inside an exception program (see
    /// <c>RayTracingPipelineBuilder.WithExceptionProgram</c>). Built-in exception
    /// codes are negative (e.g. OPTIX_EXCEPTION_CODE_STACK_OVERFLOW = -1,
    /// TRACE_DEPTH_EXCEEDED = -2); user codes thrown via
    /// <see cref="OptixThrowException"/> are non-negative.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixGetExceptionInfo
    {
        /// <summary>
        /// The exception code (optixGetExceptionCode).
        /// </summary>
        public static int Code
        {
            get
            {
                CudaAsm.Emit("call (%0), _optix_get_exception_code, ();", out int code);
                return code;
            }
        }

        public static uint Detail0
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_exception_detail_0, ();", out uint value);
                return value;
            }
        }

        public static uint Detail1
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_exception_detail_1, ();", out uint value);
                return value;
            }
        }

        public static uint Detail2
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_exception_detail_2, ();", out uint value);
                return value;
            }
        }

        public static uint Detail3
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_exception_detail_3, ();", out uint value);
                return value;
            }
        }

        public static uint Detail4
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_exception_detail_4, ();", out uint value);
                return value;
            }
        }

        public static uint Detail5
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_exception_detail_5, ();", out uint value);
                return value;
            }
        }

        public static uint Detail6
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_exception_detail_6, ();", out uint value);
                return value;
            }
        }

        public static uint Detail7
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_exception_detail_7, ();", out uint value);
                return value;
            }
        }

        /// <summary>
        /// The device address of a null-terminated line-info string for the
        /// exception site (optixGetExceptionLineInfo), or 0 when line info is
        /// unavailable. Only populated when the module was compiled with debug
        /// line information.
        /// </summary>
        public static ulong LineInfoPointer
        {
            get
            {
                CudaAsm.Emit(
                    "call (%0), _optix_get_exception_line_info, ();", out ulong pointer);
                return pointer;
            }
        }

        /// <summary>
        /// Exception detail register by runtime index (0..7). Prefer the per-index
        /// properties when the index is a compile-time constant.
        /// </summary>
        public static uint GetDetail(int index) =>
            index switch
            {
                0 => Detail0,
                1 => Detail1,
                2 => Detail2,
                3 => Detail3,
                4 => Detail4,
                5 => Detail5,
                6 => Detail6,
                _ => Detail7,
            };
    }
}
