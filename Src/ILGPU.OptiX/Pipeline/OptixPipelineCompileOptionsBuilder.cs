// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: OptixPipelineCompileOptionsBuilder.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using System;

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Fluent builder for pipeline compile options with validation and the defaults
    /// every consumer needs (launch-params variable name, triangle attribute count,
    /// single-GAS traversable graph).
    /// </summary>
    public sealed class OptixPipelineCompileOptionsBuilder
    {
        private int numPayloadValues;
        private int numAttributeValues = 2;
        private OptixExceptionFlags exceptionFlags = OptixExceptionFlags.OPTIX_EXCEPTION_FLAG_NONE;
        private OptixTraversableGraphFlags traversableGraphFlags =
            OptixTraversableGraphFlags.OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        private string pipelineLaunchParamsVariableName = OptixLaunchParams.VariableName;

        /// <summary>
        /// Sets the number of payload values. Payload values and attribute values are
        /// unrelated OptiX concepts (payload registers vs. intersection attributes) and
        /// may both be set freely.
        /// </summary>
        /// <param name="count">Payload count; must be 0 to 26.</param>
        public OptixPipelineCompileOptionsBuilder WithNumPayloadValues(int count)
        {
            if (count < 0 || count > 26)
                throw new ArgumentOutOfRangeException(
                    nameof(count),
                    $"Payload count must be between 0 and 26 (OptiX ceiling from CudaAsm.EmitRef's 44-ref cap), got {count}.");

            numPayloadValues = count;
            return this;
        }

        /// <summary>
        /// Sets the number of attribute values. Defaults to 2, which is correct for
        /// triangle geometry.
        /// </summary>
        /// <param name="count">Attribute count; must be 2 to 32.</param>
        public OptixPipelineCompileOptionsBuilder WithNumAttributeValues(int count)
        {
            if (count < 2 || count > 32)
                throw new ArgumentOutOfRangeException(
                    nameof(count),
                    $"Attribute count must be between 2 and 32 (OptiX documented limit), got {count}.");

            numAttributeValues = count;
            return this;
        }

        /// <summary>
        /// Sets exception flags.
        /// </summary>
        public OptixPipelineCompileOptionsBuilder WithExceptionFlags(OptixExceptionFlags flags)
        {
            exceptionFlags = flags;
            return this;
        }

        /// <summary>
        /// Sets the traversable graph flags. Defaults to single-GAS; pass
        /// <see cref="OptixTraversableGraphFlags.OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING"/>
        /// (optionally combined with single-GAS) when launching against an
        /// instance-acceleration-structure traversable.
        /// </summary>
        public OptixPipelineCompileOptionsBuilder WithTraversableGraphFlags(
            OptixTraversableGraphFlags flags)
        {
            traversableGraphFlags = flags;
            return this;
        }

        /// <summary>
        /// Overrides the pipeline launch-params variable name. Every sample uses
        /// <see cref="OptixLaunchParams.VariableName"/> (the default); this only exists
        /// for consumers with an unusual reason to name their launch-params global
        /// something else.
        /// </summary>
        public OptixPipelineCompileOptionsBuilder WithPipelineLaunchParamsVariableName(
            string variableName)
        {
            pipelineLaunchParamsVariableName = variableName
                ?? throw new ArgumentNullException(nameof(variableName));
            return this;
        }

        /// <summary>
        /// Builds the compile options.
        /// </summary>
        public OptixPipelineCompileOptions Build() =>
            new OptixPipelineCompileOptions
            {
                NumPayloadValues = numPayloadValues,
                NumAttributeValues = numAttributeValues,
                ExceptionFlags = exceptionFlags,
                TraversableGraphFlags = traversableGraphFlags,
                PipelineLaunchParamsVariableName = pipelineLaunchParamsVariableName,
            };
    }
}
