// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: NrcPipelines.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU.OptiX.Pipeline;
using ILGPU.Runtime.Cuda;
using Sample21.Device;
using Sample21.Device.Network;
using System;
using System.Collections.Generic;

namespace Sample21.Rendering
{
    /// <summary>
    /// Owns the three raygen-only OptiX pipelines the NRC training loop needs
    /// (<see cref="SelfTrainingTargetProgram"/>, <see cref="GradientProgram"/>,
    /// <see cref="CompositeProgram"/>) - built via the low-level
    /// <c>OptixPipelineBuilder</c>/<c>OptixSbtBuilder</c>/<c>OptixLaunchExtensions</c>
    /// path, since <see cref="RayTracingPipelineBuilder{TLaunchParams}"/> forbids a
    /// raygen-only pipeline.
    /// </summary>
    public sealed class NrcPipelines : IDisposable
    {
        private readonly OptixPipeline selfTrainingPipeline;
        private readonly BuiltSbt selfTrainingSbt;
        private readonly OptixLauncher<SelfTrainingTargetLaunchParams> selfTrainingLauncher;
        private readonly OptixPipeline gradientPipeline;
        private readonly BuiltSbt gradientSbt;
        private readonly OptixLauncher<GradientLaunchParams> gradientLauncher;
        private readonly OptixPipeline compositePipeline;
        private readonly BuiltSbt compositeSbt;
        private readonly OptixLauncher<CompositeLaunchParams> compositeLauncher;

        // Width is a genuine PTX compile-time constant baked into named OptixCoopVec
        // calls (see INrcNetworkOps's own doc comment) - this sample only ever runs the
        // network at NrcConstants.LayerWidth (64, see SampleRenderer's nrcLayerWidth
        // field), so this class hardcodes NetworkOpsW64 rather than carrying a
        // multi-width switch (NetworkOpsW32/W128) that nothing ever selects.
        public NrcPipelines(OptixDeviceContext deviceContext, CudaAccelerator accelerator)
        {
            (selfTrainingPipeline, selfTrainingSbt) = BuildRaygenOnly<SelfTrainingTargetLaunchParams>(
                deviceContext, accelerator, SelfTrainingTargetProgram.ResolveTarget<NetworkOpsW64>);
            (gradientPipeline, gradientSbt) = BuildRaygenOnly<GradientLaunchParams>(
                deviceContext, accelerator, GradientProgram.ComputeGradient<NetworkOpsW64>);
            (compositePipeline, compositeSbt) = BuildRaygenOnly<CompositeLaunchParams>(
                deviceContext, accelerator, CompositeProgram.Composite<NetworkOpsW64>);

            // Persistent per-pipeline launchers - the raw
            // accelerator.OptixLaunch(...) extension these replace allocates and
            // frees a launch-params device buffer plus a native SBT copy on EVERY
            // call (the device free forcing an implicit sync), and the composite
            // launch runs every frame.
            selfTrainingLauncher = new OptixLauncher<SelfTrainingTargetLaunchParams>(
                accelerator, selfTrainingPipeline, selfTrainingSbt.Sbt);
            gradientLauncher = new OptixLauncher<GradientLaunchParams>(
                accelerator, gradientPipeline, gradientSbt.Sbt);
            compositeLauncher = new OptixLauncher<CompositeLaunchParams>(
                accelerator, compositePipeline, compositeSbt.Sbt);
        }

        private static (OptixPipeline, BuiltSbt) BuildRaygenOnly<T>(
            OptixDeviceContext deviceContext, CudaAccelerator accelerator, Action<T> raygen)
            where T : unmanaged
        {
            var pipelineOptions = new OptixPipelineCompileOptionsBuilder()
                .WithNumPayloadValues(0)
                .WithNumAttributeValues(2)
                .Build();

            var kernel = deviceContext.CreateRaygenKernel(raygen, OptixCompilePresets.Release, pipelineOptions);
            var linkOptions = new OptixPipelineLinkOptions { MaxTraceDepth = 0 };
            var pipeline = deviceContext.CreatePipeline(pipelineOptions, linkOptions, kernel.ProgramGroup);
            OptixStackSizeUtil.ComputeAndApply(pipeline, new[] { kernel }, maxTraceDepth: 0, maxTraversableGraphDepth: 1);

            var packedRaygen = OptixSbtRecords.Pack<byte>(new List<OptixKernel> { kernel }, new byte[] { 0 });
            var sbt = new OptixSbtBuilder()
                .WithAccelerator(accelerator)
                .SetRaygenRecords<byte>(packedRaygen)
                .Build();

            return (pipeline, sbt);
        }

        public void RunSelfTrainingTarget(CudaAccelerator accelerator, SelfTrainingTargetLaunchParams p, uint recordCount) =>
            selfTrainingLauncher.Launch(accelerator.DefaultStream, p, recordCount, 1);

        public void RunGradient(CudaAccelerator accelerator, GradientLaunchParams p, uint recordCount) =>
            gradientLauncher.Launch(accelerator.DefaultStream, p, recordCount, 1);

        public void RunComposite(CudaAccelerator accelerator, CompositeLaunchParams p, uint pixelCount) =>
            compositeLauncher.Launch(accelerator.DefaultStream, p, pixelCount, 1);

        public void Dispose()
        {
            selfTrainingLauncher?.Dispose();
            gradientLauncher?.Dispose();
            compositeLauncher?.Dispose();
            selfTrainingSbt?.Dispose();
            selfTrainingPipeline?.Dispose();
            gradientSbt?.Dispose();
            gradientPipeline?.Dispose();
            compositeSbt?.Dispose();
            compositePipeline?.Dispose();
        }
    }
}
