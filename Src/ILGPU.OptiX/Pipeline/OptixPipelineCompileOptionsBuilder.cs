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
    /// Fluent builder for pipeline compile options with validation.
    /// </summary>
    public sealed class OptixPipelineCompileOptionsBuilder
    {
        private int? numPayloadValues;
        private int? numAttributeValues;
        private OptixExceptionFlags exceptionFlags;

        /// <summary>
        /// Sets the number of payload values.
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
        /// Sets the number of attribute values.
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
        /// Builds the compile options.
        /// </summary>
        public OptixPipelineCompileOptions Build()
        {
            if (numPayloadValues.HasValue && numAttributeValues.HasValue &&
                numPayloadValues.Value != 0 && numAttributeValues.Value != 0)
            {
                throw new InvalidOperationException(
                    "Cannot set both num_payload_values and num_attribute_values; they are mutually exclusive.");
            }

            return new OptixPipelineCompileOptions
            {
                NumPayloadValues = numPayloadValues ?? 0,
                NumAttributeValues = numAttributeValues ?? 0,
                ExceptionFlags = exceptionFlags,
            };
        }
    }
}
