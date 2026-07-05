// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: SampleRenderer.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.Interop;
using ILGPU.OptiX.Pipeline;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows;

namespace Sample03
{
    [StructLayout(LayoutKind.Sequential, Pack = OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT, Size = 48)]
    public unsafe struct RaygenRecord
    {
        public fixed byte Header[OptixAPI.OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        public int ObjectID;
    }

    [StructLayout(LayoutKind.Sequential, Pack = OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT, Size = 48)]
    public unsafe struct MissRecord
    {
        public fixed byte Header[OptixAPI.OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        public int ObjectID;
    }

    [StructLayout(LayoutKind.Sequential, Pack = OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT, Size = 48)]
    public unsafe struct HitgroupRecord
    {
        public fixed byte Header[OptixAPI.OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        public int ObjectID;
    }

    public class SampleRenderer
    {
        int width;
        int height;
        MainWindow window;

        Context context;
        CudaAccelerator accelerator;
        OptixDeviceContext deviceContext;

        OptixKernel raygenKernel;
        OptixKernel missKernel;
        OptixKernel hitgroupKernel;

        OptixKernel[] raygenKernels;
        OptixKernel[] missKernels;
        OptixKernel[] hitgroupKernels;

        OptixPipeline pipeline;

        RaygenRecord[] raygenRecordsArray;
        MissRecord[] missRecordsArray;
        HitgroupRecord[] hitgroupRecordsArray;

        BuiltSbt? builtSbt;
        OptixShaderBindingTable sbt;

        MemoryBuffer1D<byte, Stride1D.Dense> colorBuffer0;
        MemoryBuffer1D<byte, Stride1D.Dense> colorBuffer1;

        byte[] colorArray;

        LaunchParams launchParams;

        Action<Index1D, int, int, ArrayView<byte>, ArrayView<byte>> flipBitmap;

        public unsafe SampleRenderer(int width, int height, MainWindow window)
        {
            this.window = window;

            //init optix
            context = Context.Create(b => b.Cuda().InitOptiX());
            accelerator = context.CreateCudaAccelerator(0);
            deviceContext = accelerator.CreateDeviceContext()
                .WithModuleCompileOptions(new OptixModuleCompileOptions()
                {
                    MaxRegisterCount = 50,
                    OptimizationLevel = OptixCompileOptimizationLevel.OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
                    DebugLevel = OptixCompileDebugLevel.OPTIX_COMPILE_DEBUG_LEVEL_NONE
                })
                .WithPipelineCompileOptions(new OptixPipelineCompileOptions()
                {
                    TraversableGraphFlags = OptixTraversableGraphFlags.OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
                    NumPayloadValues = 2,
                    NumAttributeValues = 2,
                    ExceptionFlags = OptixExceptionFlags.OPTIX_EXCEPTION_FLAG_NONE,
                    PipelineLaunchParamsVariableName = OptixLaunchParams.VariableName
                })
                .WithPipelineLinkOptions(new OptixPipelineLinkOptions()
                {
                    MaxTraceDepth = 2
                });

            raygenKernel = deviceContext.CreateRaygenKernel<LaunchParams>(
                devicePrograms.__raygen__renderFrame);

            missKernel = deviceContext.CreateMissKernel<LaunchParams>(
                devicePrograms.__miss__radiance);

            hitgroupKernel = deviceContext.CreateHitgroupKernel<LaunchParams>(
                devicePrograms.__closest__radiance,
                devicePrograms.__anyhit__radiance,
                null);

            raygenKernels = new[] { raygenKernel };
            missKernels = new[] { missKernel };
            hitgroupKernels = new[] { hitgroupKernel };

            // Build pipeline using builder
            var pipelineBuilder = new OptixPipelineBuilder();
            pipelineBuilder.AddKernels(raygenKernels);
            pipelineBuilder.AddKernels(missKernels);
            pipelineBuilder.AddKernels(hitgroupKernels);
            pipeline = pipelineBuilder.Build(deviceContext);

            pipeline.SetStackSize(
                2 * 1024,
                2 * 1024,
                2 * 1024,
                1);


            // Setup SBT using builder
            raygenRecordsArray = OptixSbt.PackRecords<RaygenRecord>(raygenKernels);
            missRecordsArray = OptixSbt.PackRecords<MissRecord>(missKernels);
            hitgroupRecordsArray = OptixSbt.PackRecords<HitgroupRecord>(hitgroupKernels);

            var sbtBuilder = new OptixSbtBuilder();
            sbtBuilder.WithAccelerator(accelerator);
            sbtBuilder.SetRaygenRecords(raygenRecordsArray);
            sbtBuilder.SetMissRecords(missRecordsArray);
            sbtBuilder.AddHitgroupRecords(hitgroupRecordsArray);
            builtSbt = sbtBuilder.Build();
            sbt = builtSbt.Sbt;

            resize(width, height);
            flipBitmap = accelerator.LoadAutoGroupedStreamKernel<Index1D, int, int, ArrayView<byte>, ArrayView<byte>>(devicePrograms.flipBitmap);
        }

        public void Dispose()
        {
            builtSbt?.Dispose();

            pipeline.Dispose();

            hitgroupKernel.Dispose();
            missKernel.Dispose();
            raygenKernel.Dispose();

            deviceContext.Dispose();
            accelerator.Dispose();
            context.Dispose();
        }

        public unsafe void resize(int width, int height)
        {
            if (width != 0 && height != 0)
            {
                this.width = width;
                this.height = height;

                colorBuffer0 = accelerator.Allocate1D<byte>(width * height * sizeof(uint));
                colorBuffer0.MemSetToZero();
                colorBuffer1 = accelerator.Allocate1D<byte>(width * height * sizeof(uint));
                colorBuffer1.MemSetToZero();

                colorArray = new byte[colorBuffer0.Length];
                launchParams = new LaunchParams()
                {
                    ColorBuffer = (uint*)colorBuffer0.NativePtr,
                    FbSizeX = width,
                    FbSizeY = height
                };
            }
        }

        public void render()
        {
            // Launch pipeline.

            launchParams.FrameID++;

            accelerator.OptixLaunch(
                accelerator.DefaultStream,
                pipeline,
                launchParams,
                sbt,
                (uint)launchParams.FbSizeX,
                (uint)launchParams.FbSizeY,
                1);

            //need to flip bitmap because of wpf weirdness
            flipBitmap(new Index1D(width * height), width, height, colorBuffer0.View, colorBuffer1.View);
            accelerator.Synchronize();

            // Write output
            colorBuffer1.CopyToCPU(colorArray);

            //draws colorArray to window and waits for completion avoiding locking
            Application.Current.Dispatcher.Invoke(() => { window.draw(ref colorArray); });

            Thread.Sleep(10);
        }
    }
}
