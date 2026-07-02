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
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows;

namespace Sample09
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

    // Two records per material (radiance then shadow, see the constructor) - custom
    // data layout must match MaterialSbtData.cs, which is what __closest__radiance
    // actually reads via OptixGetSbtDataPointer (that pointer starts right after
    // Header, not at the start of this struct). Only radiance records ever have their
    // custom data read; shadow records' Color/TextureObject stay zeroed.
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

        OptixModuleCompileOptions moduleCompileOptions;
        OptixPipelineCompileOptions pipelineCompileOptions;
        OptixPipelineLinkOptions pipelineLinkOptions;

        OptixKernel raygenKernel;
        OptixKernel radianceMissKernel;
        OptixKernel shadowMissKernel;
        OptixKernel radianceHitgroupKernel;
        OptixKernel shadowHitgroupKernel;

        OptixKernel[] raygenKernels;
        OptixKernel[] missKernels;
        OptixKernel[] hitgroupKernels;
        OptixKernel[] allKernels;

        OptixPipeline pipeline;

        RaygenRecord[] raygenRecordsArray;
        MissRecord[] missRecordsArray;
        HitgroupRecord[] hitgroupRecordsArray;

        MemoryBuffer1D<RaygenRecord, Stride1D.Dense> raygenRecordsBuffer;
        MemoryBuffer1D<MissRecord, Stride1D.Dense> missRecordsBuffer;
        MemoryBuffer1D<HitgroupRecord, Stride1D.Dense> hitgroupRecordsBuffer;

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
        // selects the right per-material hitgroup record pair (built in the
        // constructor) per triangle natively, instead of an explicit LaunchParams
        // lookup array.
        MemoryBuffer1D<uint, Stride1D.Dense> d_materialIds;

        List<CudaTextureObject> textureObjects = new List<CudaTextureObject>();

        MemoryBuffer1D<byte, Stride1D.Dense> asBuffer;
        IntPtr traversable;

        public unsafe SampleRenderer(int width, int height, MainWindow window)
        {
            this.window = window;

            context = Context.Create(b => b.Cuda().InitOptiX());
            accelerator = context.CreateCudaAccelerator(0);
            deviceContext = accelerator.CreateDeviceContext();

            moduleCompileOptions = new OptixModuleCompileOptions()
            {
                MaxRegisterCount = 50,
                OptimizationLevel = OptixCompileOptimizationLevel.OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
                DebugLevel = OptixCompileDebugLevel.OPTIX_COMPILE_DEBUG_LEVEL_NONE
            };

            pipelineCompileOptions = new OptixPipelineCompileOptions()
            {
                TraversableGraphFlags = OptixTraversableGraphFlags.OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
                NumPayloadValues = 3,
                NumAttributeValues = 2,
                ExceptionFlags = OptixExceptionFlags.OPTIX_EXCEPTION_FLAG_NONE,
                PipelineLaunchParamsVariableName = OptixLaunchParams.VariableName
            };

            pipelineLinkOptions = new OptixPipelineLinkOptions()
            {
                MaxTraceDepth = 2
            };

            raygenKernel = deviceContext.CreateRaygenKernel<LaunchParams>(
                devicePrograms.__raygen__renderFrame,
                moduleCompileOptions,
                pipelineCompileOptions);

            radianceMissKernel = deviceContext.CreateMissKernel<LaunchParams>(
                devicePrograms.__miss__radiance,
                moduleCompileOptions,
                pipelineCompileOptions);

            shadowMissKernel = deviceContext.CreateMissKernel<LaunchParams>(
                devicePrograms.__miss__shadow,
                moduleCompileOptions,
                pipelineCompileOptions);

            radianceHitgroupKernel = deviceContext.CreateHitgroupKernel<LaunchParams>(
                devicePrograms.__closest__radiance,
                devicePrograms.__anyhit__radiance,
                null,
                moduleCompileOptions,
                pipelineCompileOptions);

            shadowHitgroupKernel = deviceContext.CreateHitgroupKernel<LaunchParams>(
                devicePrograms.__closesthit__shadow,
                devicePrograms.__anyhit__shadow,
                null,
                moduleCompileOptions,
                pipelineCompileOptions);

            raygenKernels = new[] { raygenKernel };
            missKernels = new[] { radianceMissKernel, shadowMissKernel };

            // Two hitgroup records per material - [mat0 radiance, mat0 shadow, mat1
            // radiance, mat1 shadow, ...] - matching how OptiX resolves a hit record
            // index as sbtGASIndex*sbtStride + sbtOffset (sbtGASIndex = the material ID
            // from d_materialIds/SbtIndexOffsetBuffer, sbtStride = RAY_TYPE_COUNT,
            // sbtOffset = the ray type passed to OptixTrace.Trace in
            // devicePrograms.cs). PackRecords below packs each record's header from its
            // corresponding kernel's program group; per-material Color/TextureObject
            // are filled in afterward, radiance records only.
            var modelPath = Path.Combine(AppContext.BaseDirectory, "models", "sponza.obj");
            model = OBJModel.Load(modelPath);
            var materialTextures = LoadMaterialTextures(model, Path.GetDirectoryName(modelPath));
            var hitgroupKernelsList = new List<OptixKernel>(model.Materials.Length * 2);
            for (var i = 0; i < model.Materials.Length; i++)
            {
                hitgroupKernelsList.Add(radianceHitgroupKernel);
                hitgroupKernelsList.Add(shadowHitgroupKernel);
            }
            hitgroupKernels = hitgroupKernelsList.ToArray();
            allKernels = (raygenKernels.Concat(missKernels).Concat(new[] { radianceHitgroupKernel, shadowHitgroupKernel })).ToArray();

            pipeline = deviceContext.CreatePipeline(
                pipelineCompileOptions,
                pipelineLinkOptions,
                allKernels.Select(x => x.ProgramGroup).ToArray());

            pipeline.SetStackSize(
                2 * 1024,
                2 * 1024,
                2 * 1024,
                1);

            raygenRecordsArray = OptixSbt.PackRecords<RaygenRecord>(raygenKernels);
            missRecordsArray = OptixSbt.PackRecords<MissRecord>(missKernels);
            hitgroupRecordsArray = OptixSbt.PackRecords<HitgroupRecord>(hitgroupKernels);
            for (var i = 0; i < model.Materials.Length; i++)
            {
                hitgroupRecordsArray[i * 2].Color = model.Materials[i].Diffuse;
                hitgroupRecordsArray[i * 2].TextureObject = materialTextures[i];
            }

            raygenRecordsBuffer = accelerator.Allocate1D(raygenRecordsArray);
            missRecordsBuffer = accelerator.Allocate1D(missRecordsArray);
            hitgroupRecordsBuffer = accelerator.Allocate1D(hitgroupRecordsArray);

            sbt = new OptixShaderBindingTable()
            {
                RaygenRecord = raygenRecordsBuffer.NativePtr,
                MissRecordBase = missRecordsBuffer.NativePtr,
                MissRecordStrideInBytes = (uint)Marshal.SizeOf<MissRecord>(),
                MissRecordCount = (uint)missRecordsBuffer.Length,
                HitgroupRecordBase = hitgroupRecordsBuffer.NativePtr,
                HitgroupRecordStrideInBytes = (uint)Marshal.SizeOf<HitgroupRecord>(),
                HitgroupRecordCount = (uint)hitgroupRecordsBuffer.Length
            };

            d_vertices = accelerator.Allocate1D(model.Vertices);
            d_normals = accelerator.Allocate1D(model.Normals);
            d_texCoords = accelerator.Allocate1D(model.TexCoords);
            d_indices = accelerator.Allocate1D(model.Indices);
            d_materialIds = accelerator.Allocate1D(model.TriangleMaterialIds);

            camera = FitCameraToModel(model, width, height);

            traversable = buildAccel(model);

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
                        handle = 0;
                    }
                    cache[texturePath] = handle;
                }
                result.Add(handle);
            }
            return result.ToArray();
        }

        // Matches example09_shadowRays/main.cpp exactly (same camera.from/cosFovy as
        // Sample10) - valid because this sample bundles the identical, untransformed
        // sponza.obj. Only camera.at depends on the loaded model's own bounds.
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

            hitgroupRecordsBuffer.Dispose();
            missRecordsBuffer.Dispose();
            raygenRecordsBuffer.Dispose();

            pipeline.Dispose();

            shadowHitgroupKernel.Dispose();
            radianceHitgroupKernel.Dispose();
            shadowMissKernel.Dispose();
            radianceMissKernel.Dispose();
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
                    camera = this.camera,
                    traversable = unchecked((ulong)this.traversable.ToInt64()),
                    Vertices = (Vec3*)d_vertices.NativePtr,
                    Normals = (Vec3*)d_normals.NativePtr,
                    TexCoords = (Vec2*)d_texCoords.NativePtr,
                    Indices = (Vec3i*)d_indices.NativePtr
                };
            }
        }

        public unsafe IntPtr buildAccel(OBJModel model)
        {
            OptixBuildInput triangleInput = new OptixBuildInput()
            {
                Type = OptixBuildInputType.OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
            };

            var vertexBuffers = stackalloc IntPtr[1];
            vertexBuffers[0] = d_vertices.NativePtr;

            triangleInput.TriangleArray.VertexFormat = OptixVertexFormat.OPTIX_VERTEX_FORMAT_FLOAT3;
            triangleInput.TriangleArray.VertexStrideInBytes = (uint)sizeof(Vec3);
            triangleInput.TriangleArray.NumVerticies = (uint)model.Vertices.Length;
            triangleInput.TriangleArray.VertexBuffers = new IntPtr(vertexBuffers);

            triangleInput.TriangleArray.IndexFormat = OptixIndicesFormat.OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangleInput.TriangleArray.IndexStrideInBytes = (uint)sizeof(Vec3i);
            triangleInput.TriangleArray.NumIndexTriplets = (uint)model.Indices.Length;
            triangleInput.TriangleArray.IndexBuffer = d_indices.NativePtr;

            // One SBT-GAS-index per material (NOT per ray type - OptiX multiplies by
            // sbtStride/RAY_TYPE_COUNT at trace time, see the constructor) - OptiX
            // looks up d_materialIds[triangleIndex] (Model's TriangleMaterialIds, one
            // uint per triangle) to select which material's hitgroup records apply, so
            // every material needs its own flag entry here too.
            var triangleInputFlags = stackalloc uint[model.Materials.Length];
            triangleInput.TriangleArray.Flags = triangleInputFlags;
            triangleInput.TriangleArray.NumSbtRecords = (uint)model.Materials.Length;
            triangleInput.TriangleArray.SbtIndexOffsetBuffer = d_materialIds.NativePtr;
            triangleInput.TriangleArray.SbtIndexOffsetSizeInBytes = sizeof(uint);
            triangleInput.TriangleArray.SbtIndexOffsetStrideInBytes = 0;

            OptixAccelBuildOptions accelOptions = new OptixAccelBuildOptions()
            {
                BuildFlags = OptixBuildFlags.OPTIX_BUILD_FLAG_NONE | OptixBuildFlags.OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
                Operation = OptixBuildOperation.OPTIX_BUILD_OPERATION_BUILD
            };
            accelOptions.MotionOptions.NumKeys = 1;

            OptixAccelBufferSizes blasBufferSizes = deviceContext.AccelComputeMemoryUsage(accelOptions, triangleInput);

            using MemoryBuffer1D<ulong, Stride1D.Dense> compactedSizeBuffer = accelerator.Allocate1D<ulong>(1);

            OptixAccelEmitDesc[] emitDesc = {
                new OptixAccelEmitDesc()
                {
                    Type = OptixAccelPropertyType.OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
                    Result = compactedSizeBuffer.NativePtr
                }
            };

            OptixBuildInput[] buildInputs = { triangleInput };

            using MemoryBuffer1D<byte, Stride1D.Dense> tempBuffer = accelerator.Allocate1D<byte>((long)blasBufferSizes.TempSizeInBytes);

            asBuffer = accelerator.Allocate1D<byte>((long)blasBufferSizes.OutputSizeInBytes);

            return deviceContext.AccelBuild(accelerator.DefaultStream, accelOptions, buildInputs, tempBuffer, asBuffer, emitDesc);
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
