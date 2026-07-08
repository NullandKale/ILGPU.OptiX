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
using ILGPU.OptiX.Device;
using ILGPU.OptiX.Pipeline;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.IO;
using System.Runtime.InteropServices;

namespace Sample02
{
    public struct LaunchParams
    {
        public int FrameID;
        public OptixDeviceView<uint> ColorBuffer;
        public int Width;
        public int Height;
    }

    public class Program
    {
        public static void RenderFrame(LaunchParams launchParams)
        {
            // Get the pixel position
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;

            // generate a gradient based on the pixel position
            uint r = (ix % 256);
            uint g = (iy % 256);
            uint b = ((ix + iy) % 256);

            // convert to 32-bit rgba value (we explicitly set alpha to 0xff
            // to make stb_image_write happy ...
            uint rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

            // and write to frame buffer ...
            long pixel = ix + iy * launchParams.Width;
            launchParams.ColorBuffer[pixel] = rgba;
        }

        public static void MissRadiance(LaunchParams launchParams) { }
        public static void ClosestHitRadiance(LaunchParams launchParams) { }
        public static void AnyHitRadiance(LaunchParams launchParams) { }

        static void Main()
        {
            Console.WriteLine("Initializing CUDA + OptiX...");
            using var context = Context.Create(b => b.Cuda());
            using var accelerator = context.CreateCudaAccelerator(0);
            using var rt = OptixRayTracer.Create(accelerator);

            // No module/pipeline/link compile options, no SBT record structs, no
            // manually chosen stack sizes - see docs/API_USABILITY_PLAN.md. This
            // sample never calls OptixTrace.Trace(...), so its single ray type needs
            // no Payload<T>() and its hitgroup needs no real per-material data - byte
            // is just a 1-byte placeholder record.
            using var pipeline = rt.CreatePipeline<LaunchParams>(b => b
                .Raygen(RenderFrame)
                .RayType("radiance", r => r
                    .Miss(MissRadiance)
                    .HitGroup<byte>(ClosestHitRadiance, AnyHitRadiance))
                .MaxTraceDepth(2));

            pipeline.SetHitRecords<byte>(new byte[] { 0 });

            // Setup launch parameters.
            var sizeX = 1200;
            var sizeY = 1024;
            using var colorBuffer = accelerator.Allocate1D<uint>(sizeX * sizeY);
            colorBuffer.MemSetToZero();

            var launchParams =
                new LaunchParams()
                {
                    ColorBuffer = OptixDeviceView<uint>.From(colorBuffer),
                    Width = sizeX,
                    Height = sizeY
                };

            // Launch pipeline.
            Console.WriteLine($"Rendering a {sizeX}x{sizeY} gradient...");
            pipeline.Launch(launchParams, launchParams.Width, launchParams.Height);
            accelerator.Synchronize();

            // Write output - stb_image_write wants raw RGBA bytes, so reinterpret the
            // uint pixels (already packed 0xAABBGGRR by RenderFrame above) as bytes.
            var outputArray = MemoryMarshal.AsBytes(colorBuffer.GetAsArray1D().AsSpan()).ToArray();
            string outputPath = Path.GetFullPath("sample02.png");
            using var pngStream = File.OpenWrite(outputPath);
            var writer = new StbImageWriteSharp.ImageWriter();
            writer.WritePng(
                outputArray,
                launchParams.Width,
                launchParams.Height,
                StbImageWriteSharp.ColorComponents.RedGreenBlueAlpha,
                pngStream);
            Console.WriteLine($"Success: wrote {outputPath}");
        }
    }
}
