// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: Program.cs
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
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace Sample02
{
    public class Program
    {
        public unsafe static void __raygen__renderFrame(LaunchParams launchParams)
        {
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;

            uint r = (ix % 256);
            uint g = (iy % 256);
            uint b = ((ix + iy) % 256);

            // convert to 32-bit rgba value (we explicitly set alpha to 0xff
            // to make stb_image_write happy ...
            uint rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

            // and write to frame buffer ...
            long fbIndex = ix + iy * launchParams.FbSizeX;
            launchParams.ColorBuffer[fbIndex] = rgba;
        }

        public static void __miss__radiance(LaunchParams launchParams)
        { }

        public static void __closest__radiance(LaunchParams launchParams)
        { }

        public static void __anyhit__radiance(LaunchParams launchParams)
        { }

        unsafe static void Main()
        {
            try
            {
                Run();
            }
            catch (Exception ex)
            {
                PrintException(ex);
                Environment.Exit(1);
            }
        }

        static void PrintException(Exception ex)
        {
            Console.Error.WriteLine("Unhandled exception:");
            for (var current = ex; current != null; current = current.InnerException)
            {
                Console.Error.WriteLine($"--- {current.GetType().FullName}: {current.Message}");
                if (current is OptixException optixEx)
                    Console.Error.WriteLine($"    OptixResult: {optixEx.OptixResult}");
                Console.Error.WriteLine(current.StackTrace);
            }
        }

        unsafe static void Run()
        {
            Console.WriteLine("Initializing CUDA + OptiX...");
            using var context = Context.Create(b => b.Cuda().InitOptiX());
            using var accelerator = context.CreateCudaAccelerator(0);
            using var deviceContext = accelerator.CreateDeviceContext()
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

            using var raygenKernel = deviceContext.CreateRaygenKernel<LaunchParams>(
                __raygen__renderFrame);

            using var missKernel = deviceContext.CreateMissKernel<LaunchParams>(
                __miss__radiance);

            using var hitgroupKernel = deviceContext.CreateHitgroupKernel<LaunchParams>(
                __closest__radiance,
                __anyhit__radiance,
                null);

            var raygenKernels = new[] { raygenKernel };
            var missKernels = new[] { missKernel };
            var hitgroupKernels = new[] { hitgroupKernel };

            // Build pipeline using builder
            using var pipelineBuilder = new OptixPipelineBuilder();
            pipelineBuilder.AddKernels(raygenKernels);
            pipelineBuilder.AddKernels(missKernels);
            pipelineBuilder.AddKernels(hitgroupKernels);
            using var pipeline = pipelineBuilder.Build(deviceContext);

            pipeline.SetStackSize(
                2 * 1024,
                2 * 1024,
                2 * 1024,
                1);

            // Build SBT using builder
            var raygenRecordsArray = OptixSbt.PackRecords<RaygenRecord>(raygenKernels);
            var missRecordsArray = OptixSbt.PackRecords<MissRecord>(missKernels);
            var hitgroupRecordsArray = OptixSbt.PackRecords<HitgroupRecord>(hitgroupKernels);

            using var sbtBuilder = new OptixSbtBuilder();
            sbtBuilder.WithAccelerator(accelerator);
            sbtBuilder.SetRaygenRecords(raygenRecordsArray);
            sbtBuilder.SetMissRecords(missRecordsArray);
            sbtBuilder.AddHitgroupRecords(hitgroupRecordsArray);
            using var builtSbt = sbtBuilder.Build();
            var sbt = builtSbt.Sbt;

            // Setup launch parameters.
            var sizeX = 1200;
            var sizeY = 1024;
            using var colorBuffer = accelerator.Allocate1D<byte>(sizeX * sizeY * sizeof(uint));
            colorBuffer.MemSetToZero();

            var launchParams =
                new LaunchParams()
                {
                    ColorBuffer = (uint*)colorBuffer.NativePtr,
                    FbSizeX = sizeX,
                    FbSizeY = sizeY
                };

            // Launch pipeline.
            Console.WriteLine($"Rendering a {sizeX}x{sizeY} gradient...");
            accelerator.OptixLaunch(
                accelerator.DefaultStream,
                pipeline,
                launchParams,
                sbt,
                (uint)launchParams.FbSizeX,
                (uint)launchParams.FbSizeY,
                1);
            accelerator.Synchronize();

            // Write output.
            var outputArray = colorBuffer.GetAsArray1D();
            string outputPath = Path.GetFullPath("sample02.png");
            using var pngStream = File.OpenWrite(outputPath);
            var writer = new StbImageWriteSharp.ImageWriter();
            writer.WritePng(
                outputArray,
                launchParams.FbSizeX,
                launchParams.FbSizeY,
                StbImageWriteSharp.ColorComponents.RedGreenBlueAlpha,
                pngStream);
            Console.WriteLine($"Success: wrote {outputPath}");
        }
    }
}
