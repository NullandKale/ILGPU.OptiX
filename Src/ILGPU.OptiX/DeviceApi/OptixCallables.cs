// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixCallables.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.Cuda;
using System;
using System.Runtime.CompilerServices;

namespace ILGPU.OptiX.DeviceApi
{
    /// <summary>
    /// Provides the functionality of the optixDirectCall and optixContinuationCall
    /// built-in functions - invokes the direct/continuation callable program at the
    /// given index in the SBT's callables table (see
    /// <c>OptixSbtBuilder.SetCallableRecords</c> /
    /// <c>RayTracingPipelineBuilder.WithDirectCallable</c>).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The SDK's optixDirectCall is a variadic template: the intrinsic returns a raw
    /// function pointer which C++ then calls with the real CUDA ABI. That parameter
    /// passing cannot be expressed as ordinary ILGPU code, so this binding currently
    /// supports the <b>zero-argument, void-return</b> prototype only - exchange data
    /// through device buffers reachable from the launch params, keyed by
    /// <paramref name="sbtIndex"/> if multiple callables share state. Word-sized
    /// argument passing is a planned follow-up (needs nvcc-reference PTX for the
    /// .callprototype ABI, same methodology as OptixTrace's mov.u32 discovery).
    /// </para>
    /// <para>
    /// The methods are deliberately [MethodImpl(NoInlining)]: the indirect-call
    /// sequence declares a PTX label (the .callprototype), and keeping each sequence
    /// inside its own non-inlined function guarantees the label appears exactly once
    /// per PTX function no matter how many call sites exist.
    /// </para>
    /// </remarks>
    [CLSCompliant(false)]
    public static class OptixCallables
    {
        /// <summary>
        /// Provides the functionality of optixDirectCall&lt;void&gt;(sbtIndex) -
        /// invokes the zero-argument direct callable at
        /// <paramref name="sbtIndex"/> in the SBT callables table. Valid in
        /// RG/IS/AH/CH/MS/DC/CC programs.
        /// </summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void DirectCall(uint sbtIndex)
        {
            Input<uint> _sbtIndex = sbtIndex;
            Output<ulong> _function = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_call_direct_callable, (%1);",
                ref _function, ref _sbtIndex);

            Input<ulong> _callee = _function.Value;
            CudaAsm.EmitRef(
                "{ optix_dc_proto: .callprototype ()_ (); " +
                "call %0, (), optix_dc_proto; }",
                ref _callee);
        }

        /// <summary>
        /// Provides the functionality of optixContinuationCall&lt;void&gt;(sbtIndex) -
        /// invokes the zero-argument continuation callable at
        /// <paramref name="sbtIndex"/> in the SBT callables table. Valid in
        /// RG/CH/MS/CC programs. Continuation callables consume continuation stack -
        /// account for them in the pipeline stack sizes.
        /// </summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void ContinuationCall(uint sbtIndex)
        {
            Input<uint> _sbtIndex = sbtIndex;
            Output<ulong> _function = default;
            CudaAsm.EmitRef(
                "call (%0), _optix_call_continuation_callable, (%1);",
                ref _function, ref _sbtIndex);

            Input<ulong> _callee = _function.Value;
            CudaAsm.EmitRef(
                "{ optix_cc_proto: .callprototype ()_ (); " +
                "call %0, (), optix_cc_proto; }",
                ref _callee);
        }
    }
}
