// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixStackSizes.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ILGPU.OptiX.Native;

#pragma warning disable CA1051 // Do not declare visible instance fields
#pragma warning disable CA1815 // Override equals and operator equals on value types

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Mirrors OptiX SDK's <c>OptixStackSizes</c> (optix_types.h). Per-program-group
    /// stack requirements as reported by <c>optixProgramGroupGetStackSize</c>.
    /// </summary>
    [CLSCompliant(false)]
    [StructLayout(LayoutKind.Sequential)]
    public struct OptixStackSizes
    {
        /// <summary>Continuation stack size of RG programs, in bytes.</summary>
        public uint CssRG;

        /// <summary>Continuation stack size of MS programs, in bytes.</summary>
        public uint CssMS;

        /// <summary>Continuation stack size of CH programs, in bytes.</summary>
        public uint CssCH;

        /// <summary>Continuation stack size of AH programs, in bytes.</summary>
        public uint CssAH;

        /// <summary>Continuation stack size of IS programs, in bytes.</summary>
        public uint CssIS;

        /// <summary>Continuation stack size of CC programs, in bytes.</summary>
        public uint CssCC;

        /// <summary>Direct stack size of DC programs, in bytes.</summary>
        public uint DssDC;

        /// <summary>
        /// Accumulates the upper bound of this and <paramref name="other"/> field by
        /// field, matching <c>optixUtilAccumulateStackSizes</c>.
        /// </summary>
        public void AccumulateMax(in OptixStackSizes other)
        {
            CssRG = Math.Max(CssRG, other.CssRG);
            CssMS = Math.Max(CssMS, other.CssMS);
            CssCH = Math.Max(CssCH, other.CssCH);
            CssAH = Math.Max(CssAH, other.CssAH);
            CssIS = Math.Max(CssIS, other.CssIS);
            CssCC = Math.Max(CssCC, other.CssCC);
            DssDC = Math.Max(DssDC, other.DssDC);
        }
    }

    /// <summary>
    /// The three values <see cref="OptixPipeline.SetStackSize"/> needs, computed by
    /// <see cref="OptixStackSizeUtil.ComputeStackSizes"/>.
    /// </summary>
    [CLSCompliant(false)]
    public readonly struct OptixComputedStackSize
    {
        public OptixComputedStackSize(
            uint directCallableStackSizeFromTraversal,
            uint directCallableStackSizeFromState,
            uint continuationStackSize)
        {
            DirectCallableStackSizeFromTraversal = directCallableStackSizeFromTraversal;
            DirectCallableStackSizeFromState = directCallableStackSizeFromState;
            ContinuationStackSize = continuationStackSize;
        }

        public uint DirectCallableStackSizeFromTraversal { get; }
        public uint DirectCallableStackSizeFromState { get; }
        public uint ContinuationStackSize { get; }
    }

    /// <summary>
    /// Ports the pure host-side math from the OptiX SDK's <c>optix_stack_size.h</c>
    /// (<c>optixUtilAccumulateStackSizes</c> / <c>optixUtilComputeStackSizes</c>) so
    /// consumers never have to guess pipeline stack sizes (the "SetStackSize(2048,
    /// 2048, 2048, 1)" magic numbers every sample used to cargo-cult).
    /// </summary>
    public static class OptixStackSizeUtil
    {
        /// <summary>
        /// Queries <paramref name="programGroup"/>'s stack requirements (optionally
        /// against a specific <paramref name="pipeline"/>, to account for external
        /// function calls) and folds the per-field maximum into
        /// <paramref name="accumulated"/>.
        /// </summary>
        [CLSCompliant(false)]
        public static void AccumulateStackSizes(
            OptixProgramGroup programGroup,
            ref OptixStackSizes accumulated,
            OptixPipeline? pipeline = null)
        {
            if (programGroup == null)
                throw new ArgumentNullException(nameof(programGroup));

            OptixException.ThrowIfFailed(
                OptixAPI.Current.ProgramGroupGetStackSize(
                    programGroup.ProgramGroupPtr,
                    out var local,
                    pipeline?.PipelinePtr ?? IntPtr.Zero));

            accumulated.AccumulateMax(local);
        }

        /// <summary>
        /// Computes the three stack-size values a pipeline needs from the accumulated
        /// stack sizes of every program group in its call graph, matching
        /// <c>optixUtilComputeStackSizes</c>. <paramref name="maxCCDepth"/> and
        /// <paramref name="maxDCDepth"/> default to 0 (no continuation/direct
        /// callables), which is correct for every sample in this repository today.
        /// </summary>
        public static OptixComputedStackSize ComputeStackSizes(
            in OptixStackSizes stackSizes,
            uint maxTraceDepth,
            uint maxCCDepth = 0,
            uint maxDCDepth = 0)
        {
            uint directCallableStackSizeFromTraversal = maxDCDepth * stackSizes.DssDC;
            uint directCallableStackSizeFromState = maxDCDepth * stackSizes.DssDC;

            // Upper bound on continuation stack used by call trees of continuation
            // callables.
            uint cssCCTree = maxCCDepth * stackSizes.CssCC;

            // Upper bound on continuation stack used by CH or MS programs including the
            // call tree of continuation callables.
            uint cssCHOrMSPlusCCTree = Math.Max(stackSizes.CssCH, stackSizes.CssMS) + cssCCTree;

            uint continuationStackSize =
                stackSizes.CssRG + cssCCTree
                + (Math.Max(maxTraceDepth, 1u) - 1) * cssCHOrMSPlusCCTree
                + Math.Min(maxTraceDepth, 1u) * Math.Max(cssCHOrMSPlusCCTree, stackSizes.CssIS + stackSizes.CssAH);

            return new OptixComputedStackSize(
                directCallableStackSizeFromTraversal,
                directCallableStackSizeFromState,
                continuationStackSize);
        }

        /// <summary>
        /// Convenience one-shot: accumulates stack sizes across every kernel's program
        /// group, computes the pipeline's required stack sizes, and applies them via
        /// <see cref="OptixPipeline.SetStackSize"/> - the entire replacement for a
        /// hand-picked <c>SetStackSize(2048, 2048, 2048, 1)</c> call.
        /// </summary>
        /// <param name="pipeline">The pipeline to configure.</param>
        /// <param name="kernels">Every kernel (program group) linked into the pipeline.</param>
        /// <param name="maxTraceDepth">The maximum depth of <c>optixTrace()</c> calls.</param>
        /// <param name="maxTraversableGraphDepth">
        /// The maximum depth of the traversable graph passed to trace; 1 for a single
        /// GAS, 2 when using instancing.
        /// </param>
        [CLSCompliant(false)]
        public static OptixComputedStackSize ComputeAndApply(
            OptixPipeline pipeline,
            IEnumerable<OptixKernel> kernels,
            uint maxTraceDepth,
            uint maxTraversableGraphDepth = 1)
        {
            if (pipeline == null)
                throw new ArgumentNullException(nameof(pipeline));
            if (kernels == null)
                throw new ArgumentNullException(nameof(kernels));

            var accumulated = default(OptixStackSizes);
            foreach (var kernel in kernels)
                AccumulateStackSizes(kernel.ProgramGroup, ref accumulated, pipeline);

            var computed = ComputeStackSizes(accumulated, maxTraceDepth);
            pipeline.SetStackSize(
                computed.DirectCallableStackSizeFromTraversal,
                computed.DirectCallableStackSizeFromState,
                computed.ContinuationStackSize,
                maxTraversableGraphDepth);
            return computed;
        }
    }
}

#pragma warning restore CA1051 // Do not declare visible instance fields
#pragma warning restore CA1815 // Override equals and operator equals on value types
