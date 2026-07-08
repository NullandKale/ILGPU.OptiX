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
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows;

namespace Sample08
{
    [StructLayout(LayoutKind.Sequential, Pack = OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT, Size = 48)]
    public unsafe struct RaygenRecord
    {
        public fixed byte Header[OptixAPI.OPTIX_SBT_RECORD_HEADER_SIZE];
        public int ObjectID;
    }

    [StructLayout(LayoutKind.Sequential, Pack = OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT, Size = 48)]
    public unsafe struct MissRecord
    {
        public fixed byte Header[OptixAPI.OPTIX_SBT_RECORD_HEADER_SIZE];
        public int ObjectID;
    }

    // One record per material - custom data layout must match MaterialSbtData.cs,
    // which is what __closest__radiance actually reads via OptixGetSbtDataPointer
    // (that pointer starts right after Header, not at the start of this struct).
    [StructLayout(LayoutKind.Sequential, Pack = OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT, Size = 64)]
    public unsafe struct HitgroupRecord
    {
        public fixed byte Header[OptixAPI.OPTIX_SBT_RECORD_HEADER_SIZE];
        public Vec3 Color;
        public ulong TextureObject;
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

        //world data
        public Camera camera { get; private set; }
        OBJModel model;
        MemoryBuffer1D<Vec3, Stride1D.Dense> d_vertices;
        MemoryBuffer1D<Vec3, Stride1D.Dense> d_normals;
        MemoryBuffer1D<Vec2, Stride1D.Dense> d_texCoords;
        MemoryBuffer1D<Vec3i, Stride1D.Dense> d_indices;

        // Also the GAS build input's SbtIndexOffsetBuffer - see buildAccel - so OptiX
        // selects the right per-material hitgroup record (built in the constructor)
        // per triangle natively, instead of an explicit LaunchParams lookup array.
        MemoryBuffer1D<uint, Stride1D.Dense> d_materialIds;

        // Kept alive (and disposed) for the renderer's lifetime - one per unique
        // diffuse-texture file referenced by the model's materials.
        List<CudaTextureObject> textureObjects = new List<CudaTextureObject>();

        BuiltAccelStructure builtAccel;
        IntPtr traversable;

        public SampleRenderer(int width, int height, MainWindow window)
        {
            this.window = window;

            context = Context.Create(b => b.Cuda());
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
                    NumPayloadValues = 3,
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
            // One hitgroup record per material, all sharing the same hitgroupKernel
            // program group - PackRecords below packs an identical header into each
            // (SbtRecordPackHeader only depends on the program group), and the
            // per-material Color/TextureObject are filled in afterward.
            var modelPath = Path.Combine(AppContext.BaseDirectory, "models", "sponza.obj");
            Console.WriteLine($"Loading model: {modelPath}...");
            model = OBJModel.Load(modelPath);
            Console.WriteLine($"Loaded {model.Indices.Length} triangles, {model.Vertices.Length} vertices, {model.Materials.Length} materials.");
            var materialTextures = LoadMaterialTextures(model, Path.GetDirectoryName(modelPath));
            Console.WriteLine($"Loaded {textureObjects.Count} unique texture(s) for {materialTextures.Count(h => h != 0)} of {model.Materials.Length} materials.");
            hitgroupKernels = Enumerable.Repeat(hitgroupKernel, model.Materials.Length).ToArray();

            // Build pipeline using builder
            var pipelineBuilder = new OptixPipelineBuilder();
            pipelineBuilder.AddKernels(raygenKernels);
            pipelineBuilder.AddKernels(missKernels);
            pipelineBuilder.AddKernels(new[] { hitgroupKernel });
            pipeline = pipelineBuilder.Build(deviceContext);

            pipeline.SetStackSize(
                2 * 1024,
                2 * 1024,
                2 * 1024,
                1);

            raygenRecordsArray = OptixSbt.PackRecords<RaygenRecord>(raygenKernels);
            missRecordsArray = OptixSbt.PackRecords<MissRecord>(missKernels);
            hitgroupRecordsArray = OptixSbt.PackRecords<HitgroupRecord>(hitgroupKernels);
            for (var i = 0; i < hitgroupRecordsArray.Length; i++)
            {
                hitgroupRecordsArray[i].Color = model.Materials[i].Diffuse;
                hitgroupRecordsArray[i].TextureObject = materialTextures[i];
            }

            var sbtBuilder = new OptixSbtBuilder();
            sbtBuilder.WithAccelerator(accelerator);
            sbtBuilder.SetRaygenRecords(raygenRecordsArray);
            sbtBuilder.SetMissRecords(missRecordsArray);
            sbtBuilder.AddHitgroupRecords(hitgroupRecordsArray);
            builtSbt = sbtBuilder.Build();
            sbt = builtSbt.Sbt;

            d_vertices = accelerator.Allocate1D(model.Vertices);
            d_normals = accelerator.Allocate1D(model.Normals);
            d_texCoords = accelerator.Allocate1D(model.TexCoords);
            d_indices = accelerator.Allocate1D(model.Indices);
            d_materialIds = accelerator.Allocate1D(model.TriangleMaterialIds);

            camera = FitCameraToModel(model, width, height);

            var accelBuilder = new OptixAccelBuilder()
                .WithDeviceContext(deviceContext)
                .WithAccelerator(accelerator)
                .AddTriangleMesh(d_vertices as MemoryBuffer, d_indices as MemoryBuffer, d_materialIds as MemoryBuffer, (uint)model.Materials.Length)
                .AllowCompaction();
            builtAccel = accelBuilder.Build();
            traversable = builtAccel.TraversableHandle;

            resize(width, height);
            flipBitmap = accelerator.LoadAutoGroupedStreamKernel<Index1D, int, int, ArrayView<byte>, ArrayView<byte>>(devicePrograms.flipBitmap);
        }

        // Creates one CudaTextureObject per unique diffuse-texture file referenced by
        // the model's materials (materials sharing a texture file share the CUDA
        // texture object), and returns a flat array - one entry per material,
        // parallel to model.Materials, later copied into each material's hitgroup
        // record (see the constructor) - of texture handles (0 = no texture). The
        // bundled Sponza assets were pre-converted from the reference's .tga textures
        // to .png (see docs/OPTIX_ROADMAP.md), so ".tga" paths recorded by the OBJ/MTL
        // parser are remapped to ".png" here.
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

        // Places the camera outside the model's bounding box looking at its center,
        // rather than hardcoding a Sponza-specific eye position - keeps this sample
        // working reasonably if the bundled model is ever swapped out.
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
            float radius = Math.Max(size.x, Math.Max(size.y, size.z)) * 0.5f;
            if (radius <= 0f)
                radius = 1f;

            Vec3 eye = center + new Vec3(radius * 0.9f, radius * 0.35f, radius * 0.9f);
            return new Camera(eye, center, new Vec3(0, 1, 0), width, height, new Vec3(0.6f, 0.7f, 0.9f), 60f, radius / 10f);
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

            builtSbt?.Dispose();

            pipeline.Dispose();

            hitgroupKernel.Dispose();
            missKernel.Dispose();
            raygenKernel.Dispose();

            deviceContext.Dispose();
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

            accelerator.OptixLaunch(
                accelerator.DefaultStream,
                pipeline,
                launchParams,
                sbt,
                (uint)width,
                (uint)height,
                1);

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
