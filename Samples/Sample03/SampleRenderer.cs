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
using ILGPU.OptiX.Device;
using ILGPU.OptiX.Interop;
using ILGPU.OptiX.Pipeline;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.OptiX.DeviceApi;
using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows;

namespace Sample03
{
    public class SampleRenderer
    {
        int width;
        int height;
        MainWindow window;

        Context context;
        CudaAccelerator accelerator;
        OptixRayTracer rayTracer;

        RayTracingPipeline<LaunchParams> pipeline;

        MemoryBuffer1D<byte, Stride1D.Dense> colorBuffer0;
        MemoryBuffer1D<byte, Stride1D.Dense> colorBuffer1;

        byte[] colorArray;

        LaunchParams launchParams;

        Action<Index1D, int, int, ArrayView<byte>, ArrayView<byte>> flipBitmap;

        public SampleRenderer(int width, int height, MainWindow window)
        {
            this.window = window;

            //init optix
            context = Context.Create(b => b.Cuda());
            accelerator = context.CreateCudaAccelerator(0);
            rayTracer = OptixRayTracer.Create(accelerator);

            // No module/pipeline/link compile options, no SBT record structs, no
            // manually chosen stack sizes - these are all computed automatically. This
            // sample never calls OptixTrace.Trace(...), so its single ray type needs
            // no Payload<T>() and its hitgroup needs no real per-material data - byte
            // is just a 1-byte placeholder record.
            pipeline = rayTracer.CreatePipeline<LaunchParams>(b => b
                .Raygen(devicePrograms.__raygen__renderFrame)
                .RayType("radiance", r => r
                    .Miss(devicePrograms.__miss__radiance)
                    .HitGroup<byte>(devicePrograms.__closest__radiance, devicePrograms.__anyhit__radiance))
                .MaxTraceDepth(2));

            pipeline.SetHitRecords<byte>(new byte[] { 0 });

            resize(width, height);
            flipBitmap = accelerator.LoadAutoGroupedStreamKernel<Index1D, int, int, ArrayView<byte>, ArrayView<byte>>(devicePrograms.flipBitmap);
        }

        public void Dispose()
        {
            pipeline.Dispose();

            rayTracer.Dispose();
            accelerator.Dispose();
            context.Dispose();
        }

        public void resize(int width, int height)
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
                    ColorBuffer = OptixDeviceView<uint>.FromBytes(colorBuffer0),
                    FbSizeX = width,
                    FbSizeY = height
                };
            }
        }

        public void render()
        {
            // Launch pipeline.

            launchParams.FrameID++;

            pipeline.Launch(launchParams, launchParams.FbSizeX, launchParams.FbSizeY);

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
