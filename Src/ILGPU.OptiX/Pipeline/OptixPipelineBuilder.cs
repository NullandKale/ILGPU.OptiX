// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixPipelineBuilder.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.Linq;
using ILGPU.Util;

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Manages kernels and pipeline creation/disposal.
    /// </summary>
    public sealed class OptixPipelineBuilder : DisposeBase
    {
        private readonly List<OptixKernel> kernels = new List<OptixKernel>();

        /// <summary>
        /// Adds a kernel to the pipeline.
        /// </summary>
        public OptixPipelineBuilder AddKernel(OptixKernel kernel)
        {
            if (kernel == null)
                throw new ArgumentNullException(nameof(kernel));

            kernels.Add(kernel);
            return this;
        }

        /// <summary>
        /// Adds multiple kernels to the pipeline.
        /// </summary>
        public OptixPipelineBuilder AddKernels(params OptixKernel[] kernelArray)
        {
            if (kernelArray == null)
                throw new ArgumentNullException(nameof(kernelArray));

            foreach (var k in kernelArray)
            {
                if (k != null)
                    kernels.Add(k);
            }
            return this;
        }

        /// <summary>
        /// Builds the pipeline with the added kernels.
        /// </summary>
        public OptixPipeline Build(OptixDeviceContext context, OptixPipelineCompileOptions compileOptions,
            OptixPipelineLinkOptions linkOptions)
        {
            if (context == null)
                throw new ArgumentNullException(nameof(context));
            if (kernels.Count == 0)
                throw new InvalidOperationException("At least one kernel must be added.");

            var programGroups = kernels.Select(k => k.ProgramGroup).ToArray();
            return context.CreatePipeline(compileOptions, linkOptions, programGroups);
        }

        /// <summary>
        /// Builds the pipeline with the added kernels using compile options stored in the device context.
        /// </summary>
        public OptixPipeline Build(OptixDeviceContext context)
        {
            if (context == null)
                throw new ArgumentNullException(nameof(context));
            if (context.PipelineCompileOptions == null)
                throw new InvalidOperationException("Pipeline compile options not configured in device context. Call WithPipelineCompileOptions() first.");
            if (context.PipelineLinkOptions == null)
                throw new InvalidOperationException("Pipeline link options not configured in device context. Call WithPipelineLinkOptions() first.");
            if (kernels.Count == 0)
                throw new InvalidOperationException("At least one kernel must be added.");

            var programGroups = kernels.Select(k => k.ProgramGroup).ToArray();
            return context.CreatePipeline(context.PipelineCompileOptions.Value, context.PipelineLinkOptions.Value, programGroups);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                // The builder does not own the pipeline or the kernels it was given -
                // kernels are caller-supplied, and the pipeline was already handed back
                // to the caller from Build(). Matches the "builders own nothing after
                // Build()" policy every other builder in this library follows
                // (OptixSbtBuilder, OptixAccelBuilder, OptixDenoiserBuilder).
                kernels.Clear();
            }
            base.Dispose(disposing);
        }
    }
}
