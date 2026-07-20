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
using ILGPU.OptiX.Native;
using ILGPU.OptiX.AccelStructures;

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
        private OptixExceptionFlags exceptionFlags = OptixExceptionFlags.None;
        private OptixTraversableGraphFlags traversableGraphFlags =
            OptixTraversableGraphFlags.AllowSingleGas;
        private string pipelineLaunchParamsVariableName = OptixLaunchParams.VariableName;
        private bool usesMotionBlur;
        private OptixPrimitiveTypeFlags primitiveTypeFlags;
        private bool allowOpacityMicromaps;
        private bool allowClusteredGeometry;

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
        /// <see cref="OptixTraversableGraphFlags.AllowSingleLevelInstancing"/>
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
        /// Sets whether the traversable graph this pipeline launches against may
        /// contain motion transforms - required
        /// whenever any <c>OptixMatrixMotionTransform</c>/<c>OptixSRTMotionTransform</c>
        /// or a GAS with more than one motion key appears anywhere in the scene.
        /// Defaults to false.
        /// </summary>
        public OptixPipelineCompileOptionsBuilder WithUsesMotionBlur(bool usesMotionBlur)
        {
            this.usesMotionBlur = usesMotionBlur;
            return this;
        }

        /// <summary>
        /// Sets which non-default primitive types the traversable graph may contain,
        /// e.g. <see cref="OptixPrimitiveTypeFlags.RoundCubicBSpline"/> for curves.
        /// Required whenever any <c>GetBuiltinISModule</c>/curve geometry is used.
        /// Defaults to 0, which means Custom + Triangle only - setting any non-zero
        /// value REPLACES that implicit default rather than adding to it, so include
        /// <see cref="OptixPrimitiveTypeFlags.Triangle"/> explicitly if the pipeline
        /// also uses triangles alongside curves.
        /// </summary>
        public OptixPipelineCompileOptionsBuilder WithUsesPrimitiveTypeFlags(OptixPrimitiveTypeFlags flags)
        {
            primitiveTypeFlags = flags;
            return this;
        }

        /// <summary>
        /// Sets whether the traversable graph may contain opacity micromaps.
        /// Defaults to false.
        /// </summary>
        public OptixPipelineCompileOptionsBuilder WithAllowOpacityMicromaps(bool allowOpacityMicromaps)
        {
            this.allowOpacityMicromaps = allowOpacityMicromaps;
            return this;
        }

        /// <summary>
        /// Sets whether the traversable graph may contain cluster acceleration
        /// structures (see OptixClusterAccelBuild). Defaults to false. Requires
        /// driver support - query
        /// <c>OptixDeviceProperty.ClusterAccel</c> before enabling.
        /// </summary>
        public OptixPipelineCompileOptionsBuilder WithAllowClusteredGeometry(bool allowClusteredGeometry)
        {
            this.allowClusteredGeometry = allowClusteredGeometry;
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
                UsesMotionBlur = usesMotionBlur ? 1 : 0,
                UsesPrimitiveTypeFlags = primitiveTypeFlags,
                AllowOpacityMicromaps = allowOpacityMicromaps ? 1 : 0,
                AllowClusteredGeometry = allowClusteredGeometry ? 1 : 0,
            };
    }
}
