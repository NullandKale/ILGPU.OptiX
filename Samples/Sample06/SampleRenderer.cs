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
using ILGPU.OptiX.AccelStructures;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.OptiX.DeviceApi;
using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows;

namespace Sample06
{
    struct RadiancePayload
    {
        public uint P0, P1, P2;
    }

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

        //world data
        public Camera camera { get; private set; }
        TriangleMesh[] meshes;
        BuiltAccelStructure builtAccel;
        IntPtr traversable;

        public SampleRenderer(int width, int height, MainWindow window)
        {
            this.window = window;

            //init optix
            context = Context.Create(b => b.Cuda());
            accelerator = context.CreateCudaAccelerator(0);
            rayTracer = OptixRayTracer.Create(accelerator);

            // No module/pipeline/link compile options, no SBT record structs, no
            // manually chosen stack sizes - these are all computed automatically. Every
            // build input below declares NumSbtRecords=1 and the launch always uses
            // SBToffset=0/SBTstride=1, so all meshes route to this single shared
            // hitgroup record regardless of which one was hit - per-mesh data comes
            // from the launchParams pointer arrays instead (see devicePrograms.cs),
            // not from separate SBT records per mesh, so byte is just a 1-byte
            // placeholder record. The payload struct's 3 uint fields preserve the
            // old NumPayloadValues = 3 (matches the ref p0/p1/p2 args in
            // devicePrograms.cs's OptixTrace.Trace(...) call).
            pipeline = rayTracer.CreatePipeline<LaunchParams>(b => b
                .Raygen(devicePrograms.__raygen__renderFrame)
                .RayType("radiance", r => r
                    .Payload<RadiancePayload>()
                    .Miss(devicePrograms.__miss__radiance)
                    .HitGroup<byte>(devicePrograms.__closest__radiance, devicePrograms.__anyhit__radiance))
                .MaxTraceDepth(2));

            pipeline.SetHitRecords<byte>(new byte[] { 0 });

            var floor = new TriangleMesh(accelerator, new Vec3(0.6f, 0.6f, 0.6f));
            floor.addUnitCube(new Vec3(0, -1.5f, 0), new Vec3(10, 0.1f, 10));

            var cube = new TriangleMesh(accelerator, new Vec3(0.9f, 0.2f, 0.2f));
            cube.addUnitCube(new Vec3(0, 0, 0), new Vec3(2, 2, 2));

            meshes = new[] { floor, cube };

            //                      from                   at                 up                                no hit color  vfov  scale
            camera = new Camera(new Vec3(-10, 2, -12), new Vec3(0, 0, 0), new Vec3(0, 1, 0), width, height, new Vec3(1, 1, 1), 40f, 10f);

            var accelBuilder = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .AllowCompaction();
            foreach (var mesh in meshes)
            {
                accelBuilder.AddTriangleMesh(mesh.d_vertexBuffer as MemoryBuffer, mesh.d_triangleIndexBuffer as MemoryBuffer);
            }
            builtAccel = accelBuilder.Build();
            traversable = builtAccel.TraversableHandle;

            resize(width, height);
            flipBitmap = accelerator.LoadAutoGroupedStreamKernel<Index1D, int, int, ArrayView<byte>, ArrayView<byte>>(devicePrograms.flipBitmap);
        }

        public void Dispose()
        {
            builtAccel?.Dispose();

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
                    camera = this.camera,
                    traversable = unchecked((ulong)this.traversable.ToInt64()),
                    mesh0Vertices = OptixDeviceView<Vec3>.From(meshes[0].d_vertexBuffer),
                    mesh0Indices = OptixDeviceView<Vec3i>.From(meshes[0].d_triangleIndexBuffer),
                    mesh0Color = meshes[0].color,
                    mesh1Vertices = OptixDeviceView<Vec3>.From(meshes[1].d_vertexBuffer),
                    mesh1Indices = OptixDeviceView<Vec3i>.From(meshes[1].d_triangleIndexBuffer),
                    mesh1Color = meshes[1].color
                };
            }
        }

        public void setCamera(Camera camera)
        {
            this.camera = camera;
            launchParams.camera = new Camera(camera, width, height);
        }

        public void render()
        {
            // Launch pipeline.

            launchParams.FrameID++;

            pipeline.Launch(launchParams, width, height);

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
