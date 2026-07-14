using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.AccelStructures;
using ILGPU.OptiX.Device;
using ILGPU.OptiX.Interop;
using ILGPU.OptiX.Pipeline;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;

namespace Sample05
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
        TriangleMesh model;
        BuiltAccelStructure builtAccel;
        IntPtr traversable;

        public SampleRenderer(int width, int height, MainWindow window)
        {
            this.window = window;

            initOptixAndOptixContext();

            pipeline = rayTracer.CreatePipeline<LaunchParams>(b => b
                .Raygen(devicePrograms.__raygen__renderFrame)
                .RayType("radiance", r => r
                    .Payload<RadiancePayload>()
                    .Miss(devicePrograms.__miss__radiance)
                    .HitGroup<byte>(
                        devicePrograms.__closest__radiance,
                        devicePrograms.__anyhit__radiance))
                .MaxTraceDepth(2));

            pipeline.SetHitRecords<byte>(new byte[] { 0 });

            model = new TriangleMesh(accelerator);
            model.AddCube(new Affine3f(new Vec3(0, -1.5f, 0), new Vec3(10, 0.1f, 10)), new Vec3(0.6f, 0.6f, 0.6f));
            model.AddCube(new Affine3f(new Vec3(0, 0, 0), new Vec3(2, 2, 2)), new Vec3(0.9f, 0.2f, 0.2f));

            //                      from                   at                 up                                no hit color  vfov  scale
            camera = new Camera(new Vec3(-10, 2, -12), new Vec3(0, 0, 0), new Vec3(0, 1, 0), width, height, new Vec3(1, 1, 1), 40f, 10f);

            var accelBuilder = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .AddTriangleMesh(model.d_vertexBuffer, model.d_triangleIndexBuffer)
                .AllowCompaction();
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
                    vertices = OptixDeviceView<Vec3>.From(model.d_vertexBuffer),
                    indices = OptixDeviceView<Vec3i>.From(model.d_triangleIndexBuffer),
                    primitiveColors = OptixDeviceView<Vec3>.From(model.d_triangleColorBuffer)
                };
            }
        }

        public void initOptixAndOptixContext()
        {
            Trace.WriteLine("Init Optix");
            context = Context.Create(b => b.Cuda());
            accelerator = context.CreateCudaAccelerator(0);
            rayTracer = OptixRayTracer.Create(accelerator);
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
