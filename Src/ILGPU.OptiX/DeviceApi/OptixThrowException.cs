// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixThrowException.tt/OptixThrowException.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------


using ILGPU.Runtime.Cuda;
using System;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixThrowException built-in function
    /// overloads (_optix_throw_exception_0..8) - aborts the current launch index and
    /// invokes the pipeline's exception program with a user exception code and up to
    /// 8 detail values (readable there via <see cref="OptixGetExceptionInfo"/>).
    /// Requires <see cref="Native.OptixExceptionFlags.User"/> in the pipeline's
    /// exception flags; the code must be in [0, 2^30). Valid in RG/IS/AH/CH/MS
    /// programs.
    /// </summary>
    [CLSCompliant(false)]
    public static class OptixThrowException
    {
        /// <summary>
        /// Throws a user exception carrying no detail values.
        /// </summary>
        public static void Throw(
            int exceptionCode)
        {
            Input<int> _exceptionCode = exceptionCode;
            CudaAsm.EmitRef(
                "call _optix_throw_exception_0, " +
                "(%0);",
                ref _exceptionCode);
        }

        /// <summary>
        /// Throws a user exception carrying 1 detail value (d0).
        /// </summary>
        public static void Throw(
            int exceptionCode,
            uint d0)
        {
            Input<int> _exceptionCode = exceptionCode;
            Input<uint> _d0 = d0;
            CudaAsm.EmitRef(
                "call _optix_throw_exception_1, " +
                "(%0, %1);",
                ref _exceptionCode,
                ref _d0);
        }

        /// <summary>
        /// Throws a user exception carrying 2 detail values (d0..d1).
        /// </summary>
        public static void Throw(
            int exceptionCode,
            uint d0,
            uint d1)
        {
            Input<int> _exceptionCode = exceptionCode;
            Input<uint> _d0 = d0;
            Input<uint> _d1 = d1;
            CudaAsm.EmitRef(
                "call _optix_throw_exception_2, " +
                "(%0, %1, %2);",
                ref _exceptionCode,
                ref _d0,
                ref _d1);
        }

        /// <summary>
        /// Throws a user exception carrying 3 detail values (d0..d2).
        /// </summary>
        public static void Throw(
            int exceptionCode,
            uint d0,
            uint d1,
            uint d2)
        {
            Input<int> _exceptionCode = exceptionCode;
            Input<uint> _d0 = d0;
            Input<uint> _d1 = d1;
            Input<uint> _d2 = d2;
            CudaAsm.EmitRef(
                "call _optix_throw_exception_3, " +
                "(%0, %1, %2, %3);",
                ref _exceptionCode,
                ref _d0,
                ref _d1,
                ref _d2);
        }

        /// <summary>
        /// Throws a user exception carrying 4 detail values (d0..d3).
        /// </summary>
        public static void Throw(
            int exceptionCode,
            uint d0,
            uint d1,
            uint d2,
            uint d3)
        {
            Input<int> _exceptionCode = exceptionCode;
            Input<uint> _d0 = d0;
            Input<uint> _d1 = d1;
            Input<uint> _d2 = d2;
            Input<uint> _d3 = d3;
            CudaAsm.EmitRef(
                "call _optix_throw_exception_4, " +
                "(%0, %1, %2, %3, %4);",
                ref _exceptionCode,
                ref _d0,
                ref _d1,
                ref _d2,
                ref _d3);
        }

        /// <summary>
        /// Throws a user exception carrying 5 detail values (d0..d4).
        /// </summary>
        public static void Throw(
            int exceptionCode,
            uint d0,
            uint d1,
            uint d2,
            uint d3,
            uint d4)
        {
            Input<int> _exceptionCode = exceptionCode;
            Input<uint> _d0 = d0;
            Input<uint> _d1 = d1;
            Input<uint> _d2 = d2;
            Input<uint> _d3 = d3;
            Input<uint> _d4 = d4;
            CudaAsm.EmitRef(
                "call _optix_throw_exception_5, " +
                "(%0, %1, %2, %3, %4, %5);",
                ref _exceptionCode,
                ref _d0,
                ref _d1,
                ref _d2,
                ref _d3,
                ref _d4);
        }

        /// <summary>
        /// Throws a user exception carrying 6 detail values (d0..d5).
        /// </summary>
        public static void Throw(
            int exceptionCode,
            uint d0,
            uint d1,
            uint d2,
            uint d3,
            uint d4,
            uint d5)
        {
            Input<int> _exceptionCode = exceptionCode;
            Input<uint> _d0 = d0;
            Input<uint> _d1 = d1;
            Input<uint> _d2 = d2;
            Input<uint> _d3 = d3;
            Input<uint> _d4 = d4;
            Input<uint> _d5 = d5;
            CudaAsm.EmitRef(
                "call _optix_throw_exception_6, " +
                "(%0, %1, %2, %3, %4, %5, %6);",
                ref _exceptionCode,
                ref _d0,
                ref _d1,
                ref _d2,
                ref _d3,
                ref _d4,
                ref _d5);
        }

        /// <summary>
        /// Throws a user exception carrying 7 detail values (d0..d6).
        /// </summary>
        public static void Throw(
            int exceptionCode,
            uint d0,
            uint d1,
            uint d2,
            uint d3,
            uint d4,
            uint d5,
            uint d6)
        {
            Input<int> _exceptionCode = exceptionCode;
            Input<uint> _d0 = d0;
            Input<uint> _d1 = d1;
            Input<uint> _d2 = d2;
            Input<uint> _d3 = d3;
            Input<uint> _d4 = d4;
            Input<uint> _d5 = d5;
            Input<uint> _d6 = d6;
            CudaAsm.EmitRef(
                "call _optix_throw_exception_7, " +
                "(%0, %1, %2, %3, %4, %5, %6, %7);",
                ref _exceptionCode,
                ref _d0,
                ref _d1,
                ref _d2,
                ref _d3,
                ref _d4,
                ref _d5,
                ref _d6);
        }

        /// <summary>
        /// Throws a user exception carrying 8 detail values (d0..d7).
        /// </summary>
        public static void Throw(
            int exceptionCode,
            uint d0,
            uint d1,
            uint d2,
            uint d3,
            uint d4,
            uint d5,
            uint d6,
            uint d7)
        {
            Input<int> _exceptionCode = exceptionCode;
            Input<uint> _d0 = d0;
            Input<uint> _d1 = d1;
            Input<uint> _d2 = d2;
            Input<uint> _d3 = d3;
            Input<uint> _d4 = d4;
            Input<uint> _d5 = d5;
            Input<uint> _d6 = d6;
            Input<uint> _d7 = d7;
            CudaAsm.EmitRef(
                "call _optix_throw_exception_8, " +
                "(%0, %1, %2, %3, %4, %5, %6, %7, %8);",
                ref _exceptionCode,
                ref _d0,
                ref _d1,
                ref _d2,
                ref _d3,
                ref _d4,
                ref _d5,
                ref _d6,
                ref _d7);
        }

    }
}
