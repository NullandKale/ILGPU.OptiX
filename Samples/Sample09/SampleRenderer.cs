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
using ILGPU.OptiX.Cuda;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Windows;

namespace Sample09
{
    // Shared placeholder payload for both ray types - the payload count is a shared
    // pipeline-wide register count, so both ray types declare the same 3-value struct
    // even though __closest__radiance's SetPRD writes all 3 slots (p0/p1/p2 color
    // channels) while the shadow ray type only ever uses slot 0
    // (OptixPayloadInterop.SetFloat(0, ...) in __miss__shadow).
    public struct RadiancePayload
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
        OBJModel model;
        MemoryBuffer1D<Vec3, Stride1D.Dense> d_vertices;
        MemoryBuffer1D<Vec3, Stride1D.Dense> d_normals;
        MemoryBuffer1D<Vec2, Stride1D.Dense> d_texCoords;
        MemoryBuffer1D<Vec3i, Stride1D.Dense> d_indices;

        // Also the GAS build input's SbtIndexOffsetBuffer - see buildAccel - so OptiX
        // selects the right per-material hitgroup record pair (built in the
        // constructor) per triangle natively, instead of an explicit LaunchParams
        // lookup array.
        MemoryBuffer1D<uint, Stride1D.Dense> d_materialIds;

        List<CudaTextureObject> textureObjects = new List<CudaTextureObject>();

        BuiltAccelStructure builtAccel;
        IntPtr traversable;

        public SampleRenderer(int width, int height, MainWindow window)
        {
            this.window = window;

            context = Context.Create(b => b.Cuda());
            accelerator = context.CreateCudaAccelerator(0);
            rayTracer = OptixRayTracer.Create(accelerator);

            pipeline = rayTracer.CreatePipeline<LaunchParams>(b => b
                .Raygen(devicePrograms.__raygen__renderFrame)
                .RayType("radiance", r => r
                    .Payload<RadiancePayload>()
                    .Miss(devicePrograms.__miss__radiance)
                    .HitGroup<MaterialSbtData>(devicePrograms.__closest__radiance, devicePrograms.__anyhit__radiance))
                .RayType("shadow", r => r
                    .Payload<RadiancePayload>()
                    .Miss(devicePrograms.__miss__shadow)
                    .HitGroup<MaterialSbtData>(devicePrograms.__closesthit__shadow, devicePrograms.__anyhit__shadow))
                .MaxTraceDepth(2));

            // Per-material hitgroup data - only the radiance ray type's hit program
            // (__closest__radiance) ever reads its record via OptixGetSbtDataPointer;
            // the shadow ray type's records stay zeroed. See
            // RayTracingPipeline<T>.SetHitRecords(TMaterial[][], int) for how these two
            // arrays get interleaved into [mat0-radiance, mat0-shadow, mat1-radiance,
            // ...] to match OptiX's sbtGASIndex*sbtStride + sbtOffset addressing
            // (sbtGASIndex = the material ID from d_materialIds/SbtIndexOffsetBuffer,
            // sbtStride = RAY_TYPE_COUNT, sbtOffset = the ray type passed to
            // OptixTrace.Trace in devicePrograms.cs).
            var modelPath = Path.Combine(AppContext.BaseDirectory, "models", "sponza.obj");
            Console.WriteLine($"Loading model: {modelPath}...");
            model = OBJModel.Load(modelPath);
            Console.WriteLine($"Loaded {model.Indices.Length} triangles, {model.Vertices.Length} vertices, {model.Materials.Length} materials.");
            var materialTextures = LoadMaterialTextures(model, Path.GetDirectoryName(modelPath));
            Console.WriteLine($"Loaded {textureObjects.Count} unique texture(s) for {materialTextures.Count(h => h != 0)} of {model.Materials.Length} materials.");

            var radianceMaterials = new MaterialSbtData[model.Materials.Length];
            var shadowMaterials = new MaterialSbtData[model.Materials.Length];
            for (var i = 0; i < model.Materials.Length; i++)
            {
                radianceMaterials[i].Color = model.Materials[i].Diffuse;
                radianceMaterials[i].TextureObject = materialTextures[i];
            }
            pipeline.SetHitRecords(new[] { radianceMaterials, shadowMaterials });

            d_vertices = accelerator.Allocate1D(model.Vertices);
            d_normals = accelerator.Allocate1D(model.Normals);
            d_texCoords = accelerator.Allocate1D(model.TexCoords);
            d_indices = accelerator.Allocate1D(model.Indices);
            d_materialIds = accelerator.Allocate1D(model.TriangleMaterialIds);

            Console.WriteLine("Camera positioned at a fixed viewpoint chosen to show shadows under Sponza's colonnade.");
            camera = FitCameraToModel(model, width, height);

            var accelBuilder = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .AddTriangleMesh(d_vertices as MemoryBuffer, d_indices as MemoryBuffer, d_materialIds as MemoryBuffer, (uint)model.Materials.Length)
                .AllowCompaction();
            builtAccel = accelBuilder.Build();
            traversable = builtAccel.TraversableHandle;

            resize(width, height);
            flipBitmap = accelerator.LoadAutoGroupedStreamKernel<Index1D, int, int, ArrayView<byte>, ArrayView<byte>>(devicePrograms.flipBitmap);
        }

        private ulong[] LoadMaterialTextures(OBJModel model, string baseDir)
        {
            var cache = new Dictionary<string, ulong>(StringComparer.OrdinalIgnoreCase);
            var result = new List<ulong>(model.Materials.Length);

            foreach (var material in model.Materials)
            {
                if (material.DiffuseTexturePath == null)
                {
                    result.Add(0);
                    continue;
                }

                var pngRelativePath = Path.ChangeExtension(material.DiffuseTexturePath, ".png");
                var texturePath = Path.Combine(baseDir, pngRelativePath);

                if (!cache.TryGetValue(texturePath, out var handle))
                {
                    if (File.Exists(texturePath))
                    {
                        var pixels = TextureLoader.LoadRgba8(texturePath, out var texWidth, out var texHeight);
                        var textureObject = new CudaTextureObject(pixels, texWidth, texHeight);
                        textureObjects.Add(textureObject);
                        handle = textureObject.TextureObject;
                    }
                    else
                    {
                        Console.WriteLine($"[Warning] Texture not found, material will use its flat diffuse color instead: {texturePath}");
                        handle = 0;
                    }
                    cache[texturePath] = handle;
                }
                result.Add(handle);
            }
            return result.ToArray();
        }

        // Fixed camera.from/cosFovy values valid for this sample's bundled,
        // untransformed sponza.obj; only camera.at depends on the loaded model's own
        // bounds.
        private static Camera FitCameraToModel(OBJModel model, int width, int height)
        {
            Vec3 min = new Vec3(float.MaxValue, float.MaxValue, float.MaxValue);
            Vec3 max = new Vec3(float.MinValue, float.MinValue, float.MinValue);
            foreach (var v in model.Vertices)
            {
                min = new Vec3(Math.Min(v.x, min.x), Math.Min(v.y, min.y), Math.Min(v.z, min.z));
                max = new Vec3(Math.Max(v.x, max.x), Math.Max(v.y, max.y), Math.Max(v.z, max.z));
            }
            Vec3 center = (min + max) / 2f;
            Vec3 size = max - min;

            Vec3 origin = new Vec3(-1293.07f, 154.681f, -0.7304f);
            Vec3 lookAt = center - new Vec3(0, 400, 0);
            Vec3 up = new Vec3(0, 1, 0);

            float verticalFov = 2f * (float)Math.Atan(0.5f * 0.66f) * (180f / (float)Math.PI);
            float worldScale = size.length();

            return new Camera(origin, lookAt, up, width, height, new Vec3(1f, 1f, 1f), verticalFov, worldScale);
        }

        public void Dispose()
        {
            foreach (var textureObject in textureObjects)
                textureObject.Dispose();

            d_materialIds.Dispose();
            d_indices.Dispose();
            d_texCoords.Dispose();
            d_normals.Dispose();
            d_vertices.Dispose();

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
                    Vertices = OptixDeviceView<Vec3>.From(d_vertices),
                    Normals = OptixDeviceView<Vec3>.From(d_normals),
                    TexCoords = OptixDeviceView<Vec2>.From(d_texCoords),
                    Indices = OptixDeviceView<Vec3i>.From(d_indices)
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
            launchParams.FrameID++;

            pipeline.Launch(launchParams, width, height);

            //need to flip bitmap because of wpf weirdness
            flipBitmap(new Index1D(width * height), width, height, colorBuffer0.View, colorBuffer1.View);
            accelerator.Synchronize();

            colorBuffer1.CopyToCPU(colorArray);

            //draws colorArray to window and waits for completion avoiding locking
            Application.Current.Dispatcher.Invoke(() => { window.draw(ref colorArray); });

            Thread.Sleep(10);
        }
    }
}
