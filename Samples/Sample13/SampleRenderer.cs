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
using ILGPU.Algorithms;
using ILGPU.OptiX;
using ILGPU.OptiX.Interop;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;

namespace Sample13
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

    // Two records per material (radiance then shadow) - custom data layout must match
    // MaterialSbtData.cs exactly (field-for-field, same order), which is what
    // __closest__radiance actually reads via OptixGetSbtDataPointer (that pointer starts
    // right after Header, not at the start of this struct). Only radiance records ever
    // have their custom data read; shadow records' fields stay zeroed. Size=128 is the
    // natural (no unnecessary padding, per .NET default sequential-layout alignment
    // rules) 116-byte tail + 32-byte header, rounded up to the next 16-byte multiple
    // OptixSbt.PackRecords requires.
    [StructLayout(LayoutKind.Sequential, Pack = OptixAPI.OPTIX_SBT_RECORD_ALIGNMENT, Size = 128)]
    public unsafe struct HitgroupRecord
    {
        public fixed byte Header[OptixAPI.OPTIX_SBT_RECORD_HEADER_SIZE];
        public Vec3 Albedo;
        public float Reflectivity;
        public Vec3 Emission;
        public float Transparency;
        public float IndexOfRefraction;
        public Vec3 TransmissionColor;
        public ulong TextureObject;
        public float TextureWeight;
        public float UVScale;
        public int MaterialKind;
        public Vec3 CheckerColorB;
        public float CheckerScale;
    }

    public class SampleRenderer
    {
        int width;
        int height;
        MainWindow window;

        Context context;
        CudaAccelerator accelerator;
        OptixDeviceContext deviceContext;
        OptixLogCallback logCallback;

        OptixModuleCompileOptions moduleCompileOptions;
        OptixPipelineCompileOptions pipelineCompileOptions;
        OptixPipelineLinkOptions pipelineLinkOptions;

        OptixKernel raygenKernel;
        OptixKernel radianceMissKernel;
        OptixKernel shadowMissKernel;
        OptixKernel radianceHitgroupKernel;
        OptixKernel shadowHitgroupKernel;
        // Same closest-hit/any-hit entry points as the triangle hitgroup kernels above,
        // but with a per-custom-primitive-kind __intersection__ program attached -
        // CreateHitgroupKernel bundles module compilation with program-group creation,
        // so a distinct intersection module means a distinct OptixKernel/program group
        // even though the shading code is identical (docs/SAMPLE13_PLAN.md's
        // custom-primitive design). Indexed by IntersectionPrograms.HitKind* (0=Sphere,
        // 1=Box, 2=CylinderY, 3=Disk, 4=XYRect, 5=XZRect, 6=YZRect).
        OptixKernel[] radianceHitgroupKernelsCustom = new OptixKernel[7];
        OptixKernel[] shadowHitgroupKernelsCustom = new OptixKernel[7];
        // The volume grid is one GAS primitive whose hitKind range (7-12) encodes which
        // face was hit rather than "which kind" (see IntersectionPrograms.cs), so it
        // gets one dedicated kernel pair rather than a slot in the arrays above.
        OptixKernel radianceHitgroupKernelVolumeGrid;
        OptixKernel shadowHitgroupKernelVolumeGrid;

        OptixKernel[] raygenKernels;
        OptixKernel[] missKernels;
        OptixKernel[] allKernels;

        OptixPipeline pipeline;

        RaygenRecord[] raygenRecordsArray;
        MissRecord[] missRecordsArray;

        MemoryBuffer1D<RaygenRecord, Stride1D.Dense> raygenRecordsBuffer;
        MemoryBuffer1D<MissRecord, Stride1D.Dense> missRecordsBuffer;
        MemoryBuffer1D<HitgroupRecord, Stride1D.Dense> hitgroupRecordsBuffer;

        OptixShaderBindingTable sbt;

        MemoryBuffer1D<Vec4, Stride1D.Dense> hdrColorBuffer;
        MemoryBuffer1D<Vec4, Stride1D.Dense> albedoBuffer;
        MemoryBuffer1D<Vec4, Stride1D.Dense> normalBuffer;
        MemoryBuffer1D<Vec4, Stride1D.Dense> denoisedColorBuffer;
        MemoryBuffer1D<byte, Stride1D.Dense> displayBuffer;
        MemoryBuffer1D<byte, Stride1D.Dense> denoiserIntensity;

        // Double-buffered async present, DX12-swapchain style (2 buffers, max frame
        // latency 2) - render and present interleave: the render thread can start
        // producing the next frame into the other buffer as soon as it publishes one,
        // while the UI thread is still presenting the previous one, but the render
        // thread blocks (bufferFree[idx].Wait, see render()) once it's produced 2
        // frames the UI thread hasn't consumed yet. That backpressure is what couples
        // the two rates together - unlike a fully decoupled "always grab the latest"
        // scheme, frames are presented strictly in production order and the render
        // thread's own pace is throttled by how fast the UI thread actually consumes
        // them, the same way a real swap chain's CPU submission thread stalls on its
        // frame-latency-waitable object. Each buffer index has its own pair of
        // semaphores: bufferFree[i] (1,1) starts signaled - "render may write here" -
        // and bufferReady[i] (0,1) - "UI may read this, render just published it".
        // Only ever touched via SemaphoreSlim.Wait/Release, never a plain lock.
        byte[][] frameBuffers;
        SemaphoreSlim[] bufferFree;
        SemaphoreSlim[] bufferReady;
        int writeIndex;

        OptixDenoiser denoiser;
        MemoryBuffer1D<byte, Stride1D.Dense> denoiserState;
        MemoryBuffer1D<byte, Stride1D.Dense> denoiserScratch;

        public bool DenoiserOn { get; set; } = true;

        // accelerator.DefaultStream is declared as the base AcceleratorStream type; the
        // denoiser API needs the raw CUstream pointer, which only CudaStream exposes.
        IntPtr DefaultStreamPtr => ((CudaStream)accelerator.DefaultStream).StreamPtr;

        // Loaded lazily per scene switch (BuildHitgroupSbt), keyed by relative path -
        // reused if the same texture is referenced by multiple materials/scenes.
        Dictionary<string, ulong> textureCache = new Dictionary<string, ulong>(StringComparer.OrdinalIgnoreCase);
        List<CudaTextureObject> textureObjects = new List<CudaTextureObject>();

        // Video textures (paths ending in a known video extension - see
        // GetOrLoadTexture) get routed here instead: one VideoTexture per distinct
        // path, refreshed once per rendered frame (render()'s ApplyAnimation call)
        // rather than loaded once and left static. Disposed alongside the rest of the
        // scene's GPU state in DisposeSceneBuffers.
        static readonly string[] VideoExtensions = { ".mp4", ".avi", ".mov", ".mkv" };
        Dictionary<string, VideoTexture> videoTextureCache = new Dictionary<string, VideoTexture>(StringComparer.OrdinalIgnoreCase);
        List<VideoTexture> activeVideoTextures = new List<VideoTexture>();

        // Animation (museum/radial-museum scenes) - see ApplyAnimation. currentScene
        // and the two host-side arrays are private per-frame working copies, refreshed
        // at every SwitchToScene - never the same array instances as the cached
        // SceneData in sceneCache, since that same SceneData is reused verbatim if the
        // user cycles back to this scene later.
        SceneData currentScene;
        PointLightGpu[] animatedLightsHost = Array.Empty<PointLightGpu>();
        SphereData[] animatedSpheresHost = Array.Empty<SphereData>();
        readonly Stopwatch animationClock = new Stopwatch();

        // Persistent scratch temp buffer for per-frame OPTIX_BUILD_OPERATION_UPDATE
        // calls against the custom-primitives GAS (only allocated/used when the active
        // scene has SceneData.HasAnimatedGeometry) - sized once from
        // OptixAccelBufferSizes.TempUpdateSizeInBytes at the scene's initial BUILD, then
        // reused every frame (update temp-buffer requirements don't grow beyond what a
        // fixed topology needs, since only AABB/vertex contents change, never primitive
        // counts).
        MemoryBuffer1D<byte, Stride1D.Dense> customPrimitivesUpdateTempBuffer;

        LaunchParams launchParams;

        Action<Index1D, int, int, ArrayView<Vec4>, ArrayView<byte>> tonemapAndFlip;

        public Camera camera { get; private set; }

        MemoryBuffer1D<Vec3, Stride1D.Dense> d_vertices;
        MemoryBuffer1D<Vec3, Stride1D.Dense> d_normals;
        MemoryBuffer1D<Vec2, Stride1D.Dense> d_texCoords;
        MemoryBuffer1D<Vec3i, Stride1D.Dense> d_indices;
        MemoryBuffer1D<uint, Stride1D.Dense> d_materialIds;
        MemoryBuffer1D<PointLightGpu, Stride1D.Dense> d_lights;

        MemoryBuffer1D<SphereData, Stride1D.Dense> d_spheres;
        MemoryBuffer1D<uint, Stride1D.Dense> d_sphereMaterialIds;
        MemoryBuffer1D<OptixAabb, Stride1D.Dense> d_sphereAabbs;

        MemoryBuffer1D<BoxData, Stride1D.Dense> d_boxes;
        MemoryBuffer1D<uint, Stride1D.Dense> d_boxMaterialIds;
        MemoryBuffer1D<OptixAabb, Stride1D.Dense> d_boxAabbs;

        MemoryBuffer1D<CylinderYData, Stride1D.Dense> d_cylindersY;
        MemoryBuffer1D<uint, Stride1D.Dense> d_cylinderYMaterialIds;
        MemoryBuffer1D<OptixAabb, Stride1D.Dense> d_cylinderYAabbs;

        MemoryBuffer1D<DiskData, Stride1D.Dense> d_disks;
        MemoryBuffer1D<uint, Stride1D.Dense> d_diskMaterialIds;
        MemoryBuffer1D<OptixAabb, Stride1D.Dense> d_diskAabbs;

        MemoryBuffer1D<RectData, Stride1D.Dense> d_xyRects;
        MemoryBuffer1D<uint, Stride1D.Dense> d_xyRectMaterialIds;
        MemoryBuffer1D<OptixAabb, Stride1D.Dense> d_xyRectAabbs;

        MemoryBuffer1D<RectData, Stride1D.Dense> d_xzRects;
        MemoryBuffer1D<uint, Stride1D.Dense> d_xzRectMaterialIds;
        MemoryBuffer1D<OptixAabb, Stride1D.Dense> d_xzRectAabbs;

        MemoryBuffer1D<RectData, Stride1D.Dense> d_yzRects;
        MemoryBuffer1D<uint, Stride1D.Dense> d_yzRectMaterialIds;
        MemoryBuffer1D<OptixAabb, Stride1D.Dense> d_yzRectAabbs;

        MemoryBuffer1D<uint, Stride1D.Dense> d_voxelMaterialIds;
        MemoryBuffer1D<OptixAabb, Stride1D.Dense> d_volumeGridAabb;

        // Device-side copy of the active scene's Materials[] palette - see
        // LaunchParams.Materials.
        MemoryBuffer1D<MaterialSbtData, Stride1D.Dense> d_materials;

        // Triangles and custom primitives each get their own GAS (see buildAccel) -
        // combined via an IAS, since a single GAS cannot mix build-input types. All 7
        // custom-primitive kinds share ONE GAS (multiple build inputs of the SAME
        // OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES type, which - unlike mixing types -
        // is supported, the same way multiple triangle meshes already share one GAS).
        MemoryBuffer1D<byte, Stride1D.Dense> trianglesGasBuffer;
        MemoryBuffer1D<byte, Stride1D.Dense> customPrimitivesGasBuffer;
        MemoryBuffer1D<byte, Stride1D.Dense> iasBuffer;
        MemoryBuffer1D<OptixInstance, Stride1D.Dense> instancesBuffer;
        IntPtr traversable;

        public int NumPixelSamples { get; set; } = 1;
        public bool Accumulate { get; set; } = true;

        // Defaults to the original single-merged-build-input triangles GAS (fast) -
        // splitting into one build input per mesh (see GetTriangleMeshRanges) multiplies
        // SBT hitgroup record count by (build-input count * Materials.Length), which
        // gets very large for scenes like the radial museum that also scatter dozens of
        // one-off AddTriangle calls (each its own per-triangle material - see Scenes.cs's
        // AddTriangleField) between AddMeshAutoGround calls, turning into many small
        // "gap" build inputs. Toggled live with the M key (MainWindow_KeyDown), which
        // forces a scene rebuild since it changes the GAS/SBT shape.
        public bool UseMergedTrianglesGas { get; set; } = true;

        // Scene-switcher state - mirrors the reference's RaytraceEntity.BuildSceneTable
        // (lazily built, cached per index) - see docs/SAMPLE13_PLAN.md.
        Func<SceneData>[] sceneBuilders;
        Dictionary<int, SceneData> sceneCache = new Dictionary<int, SceneData>();
        int currentSceneIndex;

        public string CurrentSceneName { get; private set; } = "";

        // Scene-static HUD info - recomputed once per SwitchToScene (not per frame,
        // unlike LastStats below), since none of it changes between frames of the same
        // scene. Mirrors the geometry counts already known to BuildHitgroupSbt/
        // buildAccel, just cached on public properties so MainWindow's overlay can read
        // them without re-deriving anything from SceneData itself.
        public int TriangleCount { get; private set; }
        public int MaterialCount { get; private set; }
        public int SphereCount { get; private set; }
        public int BoxCount { get; private set; }
        public int CylinderYCount { get; private set; }
        public int DiskCount { get; private set; }
        public int XYRectCount { get; private set; }
        public int XZRectCount { get; private set; }
        public int YZRectCount { get; private set; }
        public bool HasVolumeGrid { get; private set; }
        public Vec3i VolumeGridDims { get; private set; }
        public string AccelStructureSummary { get; private set; } = "";

        // Per-frame HUD info - see FrameStats. Updated once at the very end of each
        // render() call; read by MainWindow's own CompositionTarget.Rendering handler
        // (a different thread, at a different rate), which is fine since it's just a
        // plain struct-copying property read/write with no torn-read hazard worse than
        // "occasionally see last frame's numbers instead of this one" - acceptable for
        // a HUD.
        public FrameStats LastStats { get; private set; }

        readonly Stopwatch stepStopwatch = new Stopwatch();
        long lastFrameStartTicks;
        bool hasRenderedFrame;
        double smoothedFrameMs = 16.0;

        public unsafe SampleRenderer(int width, int height, MainWindow window)
        {
            this.window = window;

            context = Context.Create(b => b.Cuda().InitOptiX().EnableAlgorithms());
            accelerator = context.CreateCudaAccelerator(0);

            // Validation mode ALL + a log callback surfaces OptiX's own descriptive
            // error/warning text (e.g. "these build input types cannot be combined in
            // one GAS") instead of a bare OptixResult code with no message - the accel
            // build entry points (unlike module/pipeline creation) never return a log
            // string of their own, so this is the only way to get that detail out of
            // the driver. logCallback is kept as a field (not a local) since the GC has
            // no visibility into the native function pointer OptiX holds onto.
            logCallback = (level, tag, message, _) =>
                Console.Error.WriteLine($"[OptiX][{level}][{Marshal.PtrToStringAnsi(tag)}] {Marshal.PtrToStringAnsi(message)}");
            deviceContext = accelerator.CreateDeviceContext(new OptixDeviceContextOptions
            {
                LogCallbackFunction = Marshal.GetFunctionPointerForDelegate(logCallback),
                LogCallbackLevel = 4,
                ValidationMode = OptixDeviceContextValidationMode.OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL,
            });

            moduleCompileOptions = new OptixModuleCompileOptions()
            {
                MaxRegisterCount = 50,
                OptimizationLevel = OptixCompileOptimizationLevel.OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
                DebugLevel = OptixCompileDebugLevel.OPTIX_COMPILE_DEBUG_LEVEL_NONE
            };

            pipelineCompileOptions = new OptixPipelineCompileOptions()
            {
                // Triangle and custom-primitive geometry cannot be combined as multiple
                // build inputs within a single GAS (confirmed against the OptiX SDK's
                // own optixSimpleMotionBlur sample, which builds one GAS per build-input
                // type and combines them via an IAS) - so this is a single level of
                // instancing (one IAS directly over two GASes), not a single GAS.
                TraversableGraphFlags = OptixTraversableGraphFlags.OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                // color(3) + flag(1) + new-ray-origin(3) + new-ray-direction(3) +
                // throughput-tint(3) + normal(3) + albedo(3) = 19 - see
                // devicePrograms.cs's SetContinuePayload/SetTerminalPayload/SetAovPayload
                // and docs/SAMPLE13_PLAN.md design (e). The shadow ray's own
                // transmittance(3)+hitCount(1) = 4 payloads fit within this.
                NumPayloadValues = 19,
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

            void CreateCustomHitgroupKernel(uint kind, Action<LaunchParams> intersectionKernel)
            {
                radianceHitgroupKernelsCustom[kind] = deviceContext.CreateHitgroupKernel<LaunchParams>(
                    devicePrograms.__closest__radiance,
                    devicePrograms.__anyhit__radiance,
                    intersectionKernel,
                    moduleCompileOptions,
                    pipelineCompileOptions);

                shadowHitgroupKernelsCustom[kind] = deviceContext.CreateHitgroupKernel<LaunchParams>(
                    devicePrograms.__closesthit__shadow,
                    devicePrograms.__anyhit__shadow,
                    intersectionKernel,
                    moduleCompileOptions,
                    pipelineCompileOptions);
            }

            CreateCustomHitgroupKernel(IntersectionPrograms.HitKindSphere, IntersectionPrograms.__intersection__sphere);
            CreateCustomHitgroupKernel(IntersectionPrograms.HitKindBox, IntersectionPrograms.__intersection__box);
            CreateCustomHitgroupKernel(IntersectionPrograms.HitKindCylinderY, IntersectionPrograms.__intersection__cylinderY);
            CreateCustomHitgroupKernel(IntersectionPrograms.HitKindDisk, IntersectionPrograms.__intersection__disk);
            CreateCustomHitgroupKernel(IntersectionPrograms.HitKindXYRect, IntersectionPrograms.__intersection__xyRect);
            CreateCustomHitgroupKernel(IntersectionPrograms.HitKindXZRect, IntersectionPrograms.__intersection__xzRect);
            CreateCustomHitgroupKernel(IntersectionPrograms.HitKindYZRect, IntersectionPrograms.__intersection__yzRect);

            radianceHitgroupKernelVolumeGrid = deviceContext.CreateHitgroupKernel<LaunchParams>(
                devicePrograms.__closest__radiance,
                devicePrograms.__anyhit__radiance,
                IntersectionPrograms.__intersection__volumeGrid,
                moduleCompileOptions,
                pipelineCompileOptions);

            shadowHitgroupKernelVolumeGrid = deviceContext.CreateHitgroupKernel<LaunchParams>(
                devicePrograms.__closesthit__shadow,
                devicePrograms.__anyhit__shadow,
                IntersectionPrograms.__intersection__volumeGrid,
                moduleCompileOptions,
                pipelineCompileOptions);

            raygenKernels = new[] { raygenKernel };
            missKernels = new[] { radianceMissKernel, shadowMissKernel };
            allKernels = raygenKernels.Concat(missKernels)
                .Concat(new[] { radianceHitgroupKernel, shadowHitgroupKernel, radianceHitgroupKernelVolumeGrid, shadowHitgroupKernelVolumeGrid })
                .Concat(radianceHitgroupKernelsCustom)
                .Concat(shadowHitgroupKernelsCustom)
                .ToArray();

            pipeline = deviceContext.CreatePipeline(
                pipelineCompileOptions,
                pipelineLinkOptions,
                allKernels.Select(x => x.ProgramGroup).ToArray());

            // maxTraversableGraphDepth = 2: our traversable graph is now one IAS
            // directly over GASes (triangles/spheres), not a single bare GAS - per
            // optix_host.h's optixPipelineSetStackSize docs, "for a simple IAS -> GAS
            // traversal graph, the maxTraversableGraphDepth is two".
            pipeline.SetStackSize(2 * 1024, 2 * 1024, 2 * 1024, 2);

            raygenRecordsArray = OptixSbt.PackRecords<RaygenRecord>(raygenKernels);
            missRecordsArray = OptixSbt.PackRecords<MissRecord>(missKernels);
            raygenRecordsBuffer = accelerator.Allocate1D(raygenRecordsArray);
            missRecordsBuffer = accelerator.Allocate1D(missRecordsArray);

            sceneBuilders = Scenes.BuildSceneTable();

            // Guide-layer denoiser (matches Sample12's OPTIX_DENOISER_MODEL_KIND_LDR +
            // GuideAlbedo/GuideNormal setup) - one denoiser instance works across every
            // scene switch since it only depends on width/height (set up in resize),
            // not on scene content.
            denoiser = deviceContext.CreateDenoiser(
                OptixDenoiserModelKind.OPTIX_DENOISER_MODEL_KIND_LDR,
                new OptixDenoiserOptions { GuideAlbedo = 1, GuideNormal = 1 });

            resize(width, height);
            tonemapAndFlip = accelerator.LoadAutoGroupedStreamKernel<Index1D, int, int, ArrayView<Vec4>, ArrayView<byte>>(devicePrograms.tonemapAndFlip);

            SwitchToScene(0);
        }

        static OptixAabb[] ComputeSphereAabbs(SphereData[] spheres)
        {
            var result = new OptixAabb[spheres.Length];
            for (var i = 0; i < spheres.Length; i++)
            {
                var s = spheres[i];
                result[i] = new OptixAabb
                {
                    MinX = s.Center.x - s.Radius, MinY = s.Center.y - s.Radius, MinZ = s.Center.z - s.Radius,
                    MaxX = s.Center.x + s.Radius, MaxY = s.Center.y + s.Radius, MaxZ = s.Center.z + s.Radius,
                };
            }
            return result;
        }

        static OptixAabb[] ComputeBoxAabbs(BoxData[] boxes)
        {
            var result = new OptixAabb[boxes.Length];
            for (var i = 0; i < boxes.Length; i++)
            {
                var b = boxes[i];
                result[i] = new OptixAabb
                {
                    MinX = b.Min.x, MinY = b.Min.y, MinZ = b.Min.z,
                    MaxX = b.Max.x, MaxY = b.Max.y, MaxZ = b.Max.z,
                };
            }
            return result;
        }

        static OptixAabb[] ComputeCylinderYAabbs(CylinderYData[] cylinders)
        {
            var result = new OptixAabb[cylinders.Length];
            for (var i = 0; i < cylinders.Length; i++)
            {
                var c = cylinders[i];
                result[i] = new OptixAabb
                {
                    MinX = c.Center.x - c.Radius, MinY = c.YMin, MinZ = c.Center.z - c.Radius,
                    MaxX = c.Center.x + c.Radius, MaxY = c.YMax, MaxZ = c.Center.z + c.Radius,
                };
            }
            return result;
        }

        // A disk's AABB depends on its (arbitrary) orientation - a spherical bound
        // around its center is a simple, always-correct (if occasionally loose) choice.
        static OptixAabb[] ComputeDiskAabbs(DiskData[] disks)
        {
            var result = new OptixAabb[disks.Length];
            for (var i = 0; i < disks.Length; i++)
            {
                var d = disks[i];
                result[i] = new OptixAabb
                {
                    MinX = d.Center.x - d.Radius, MinY = d.Center.y - d.Radius, MinZ = d.Center.z - d.Radius,
                    MaxX = d.Center.x + d.Radius, MaxY = d.Center.y + d.Radius, MaxZ = d.Center.z + d.Radius,
                };
            }
            return result;
        }

        const float RectAabbEpsilon = 1e-4f; // finite thickness along the fixed axis

        static OptixAabb[] ComputeXYRectAabbs(RectData[] rects)
        {
            var result = new OptixAabb[rects.Length];
            for (var i = 0; i < rects.Length; i++)
            {
                var r = rects[i];
                result[i] = new OptixAabb
                {
                    MinX = r.A0, MinY = r.B0, MinZ = r.C - RectAabbEpsilon,
                    MaxX = r.A1, MaxY = r.B1, MaxZ = r.C + RectAabbEpsilon,
                };
            }
            return result;
        }

        static OptixAabb[] ComputeXZRectAabbs(RectData[] rects)
        {
            var result = new OptixAabb[rects.Length];
            for (var i = 0; i < rects.Length; i++)
            {
                var r = rects[i];
                result[i] = new OptixAabb
                {
                    MinX = r.A0, MinY = r.C - RectAabbEpsilon, MinZ = r.B0,
                    MaxX = r.A1, MaxY = r.C + RectAabbEpsilon, MaxZ = r.B1,
                };
            }
            return result;
        }

        static OptixAabb[] ComputeYZRectAabbs(RectData[] rects)
        {
            var result = new OptixAabb[rects.Length];
            for (var i = 0; i < rects.Length; i++)
            {
                var r = rects[i];
                result[i] = new OptixAabb
                {
                    MinX = r.C - RectAabbEpsilon, MinY = r.A0, MinZ = r.B0,
                    MaxX = r.C + RectAabbEpsilon, MaxY = r.A1, MaxZ = r.B1,
                };
            }
            return result;
        }

        static OptixAabb[] ComputeVolumeGridAabb(SceneData scene)
        {
            if (scene.VoxelMaterialIds.Length == 0)
                return Array.Empty<OptixAabb>();

            var min = scene.VolumeGridMin;
            var size = scene.VolumeVoxelSize;
            var dims = scene.VolumeDims;
            return new[]
            {
                new OptixAabb
                {
                    MinX = min.x, MinY = min.y, MinZ = min.z,
                    MaxX = min.x + (dims.x * size.x), MaxY = min.y + (dims.y * size.y), MaxZ = min.z + (dims.z * size.z),
                },
            };
        }

        public void NextScene() => SwitchToScene((currentSceneIndex + 1) % sceneBuilders.Length);
        public void PreviousScene() => SwitchToScene((currentSceneIndex - 1 + sceneBuilders.Length) % sceneBuilders.Length);

        // Re-runs SwitchToScene against the same scene index - used after toggling
        // UseMergedTrianglesGas, which changes the triangles GAS/SBT shape and so
        // needs a full rebuild, not just a launchParams/camera tweak.
        public void RebuildCurrentScene() => SwitchToScene(currentSceneIndex);

        // A zero-length source array must NOT be passed to Allocate1D - ILGPU's
        // MemoryBuffer1D constructor throws a NullReferenceException for a zero-element
        // allocation (its underlying device pointer comes back null and the view
        // wrapper doesn't handle that), so a scene lacking a given primitive kind (e.g.
        // the volume-grid test scene has zero triangles, and the debug scene has no
        // volume grid) must leave that buffer field null instead of allocating.
        MemoryBuffer1D<T, Stride1D.Dense> AllocateOrNull<T>(T[] data) where T : unmanaged =>
            data.Length > 0 ? accelerator.Allocate1D(data) : null;

        static IntPtr NativePtrOrZero<T>(MemoryBuffer1D<T, Stride1D.Dense> buffer) where T : unmanaged =>
            buffer?.NativePtr ?? IntPtr.Zero;

        // HUD summary of the actual GAS/IAS shape built for this scene - mirrors
        // buildAccel's fixed BuildTrianglesGas -> BuildCustomPrimitivesGas ->
        // BuildInstanceAccel pipeline (see buildAccel below), reporting which of the two
        // GASes are actually present this scene (a scene can have either, both, or -
        // in principle - neither) and their per-kind primitive counts.
        string BuildAccelStructureSummary(SceneData scene, bool hasTriangles, bool hasCustomPrimitives)
        {
            var gasParts = new List<string>();
            if (hasTriangles)
            {
                int triangleMeshCount = GetTriangleMeshRanges(scene).Length;
                gasParts.Add($"GAS[Triangles: {triangleMeshCount} input(s), {scene.Indices.Length} tris]");
            }

            if (hasCustomPrimitives)
            {
                var kinds = new List<string>();
                if (scene.Spheres.Length > 0) kinds.Add($"{scene.Spheres.Length} sphere");
                if (scene.Boxes.Length > 0) kinds.Add($"{scene.Boxes.Length} box");
                if (scene.CylindersY.Length > 0) kinds.Add($"{scene.CylindersY.Length} cylinderY");
                if (scene.Disks.Length > 0) kinds.Add($"{scene.Disks.Length} disk");
                if (scene.XYRects.Length > 0) kinds.Add($"{scene.XYRects.Length} xyRect");
                if (scene.XZRects.Length > 0) kinds.Add($"{scene.XZRects.Length} xzRect");
                if (scene.YZRects.Length > 0) kinds.Add($"{scene.YZRects.Length} yzRect");
                if (scene.VoxelMaterialIds.Length > 0)
                    kinds.Add($"1 volumeGrid({scene.VolumeDims.x}x{scene.VolumeDims.y}x{scene.VolumeDims.z})");

                gasParts.Add($"GAS[CustomPrimitives: {kinds.Count} inputs]\n    " + string.Join("\n    ", kinds));
            }

            string header = $"{gasParts.Count} GAS -> 1 IAS:";
            return gasParts.Count > 0 ? header + "\n  " + string.Join("\n  ", gasParts) : header + " (none)";
        }

        public unsafe void SwitchToScene(int index)
        {
            lock (gpuLock)
            {
                if (!sceneCache.TryGetValue(index, out var scene))
                {
                    scene = sceneBuilders[index]();
                    sceneCache[index] = scene;
                }
                currentSceneIndex = index;
                CurrentSceneName = scene.Name;
                currentScene = scene;
                // Private working copies - ApplyAnimation mutates these every frame, never
                // the cached SceneData itself (sceneCache reuses the same instance if the
                // user cycles back to this scene later).
                animatedLightsHost = (PointLightGpu[])scene.Lights.Clone();
                animatedSpheresHost = (SphereData[])scene.Spheres.Clone();
                animationClock.Restart();

                DisposeSceneBuffers();

                d_vertices = AllocateOrNull(scene.Vertices);
                d_normals = AllocateOrNull(scene.Normals);
                d_texCoords = AllocateOrNull(scene.TexCoords);
                d_indices = AllocateOrNull(scene.Indices);
                d_materialIds = AllocateOrNull(scene.TriangleMaterialIds);
                d_lights = AllocateOrNull(scene.Lights);

                d_spheres = AllocateOrNull(scene.Spheres);
                d_sphereMaterialIds = AllocateOrNull(scene.SphereMaterialIds);
                d_sphereAabbs = AllocateOrNull(ComputeSphereAabbs(scene.Spheres));

                d_boxes = AllocateOrNull(scene.Boxes);
                d_boxMaterialIds = AllocateOrNull(scene.BoxMaterialIds);
                d_boxAabbs = AllocateOrNull(ComputeBoxAabbs(scene.Boxes));

                d_cylindersY = AllocateOrNull(scene.CylindersY);
                d_cylinderYMaterialIds = AllocateOrNull(scene.CylinderYMaterialIds);
                d_cylinderYAabbs = AllocateOrNull(ComputeCylinderYAabbs(scene.CylindersY));

                d_disks = AllocateOrNull(scene.Disks);
                d_diskMaterialIds = AllocateOrNull(scene.DiskMaterialIds);
                d_diskAabbs = AllocateOrNull(ComputeDiskAabbs(scene.Disks));

                d_xyRects = AllocateOrNull(scene.XYRects);
                d_xyRectMaterialIds = AllocateOrNull(scene.XYRectMaterialIds);
                d_xyRectAabbs = AllocateOrNull(ComputeXYRectAabbs(scene.XYRects));

                d_xzRects = AllocateOrNull(scene.XZRects);
                d_xzRectMaterialIds = AllocateOrNull(scene.XZRectMaterialIds);
                d_xzRectAabbs = AllocateOrNull(ComputeXZRectAabbs(scene.XZRects));

                d_yzRects = AllocateOrNull(scene.YZRects);
                d_yzRectMaterialIds = AllocateOrNull(scene.YZRectMaterialIds);
                d_yzRectAabbs = AllocateOrNull(ComputeYZRectAabbs(scene.YZRects));

                d_voxelMaterialIds = AllocateOrNull(scene.VoxelMaterialIds);
                d_volumeGridAabb = AllocateOrNull(ComputeVolumeGridAabb(scene));
                d_materials = AllocateOrNull(scene.Materials);

                BuildHitgroupSbt(scene);
                traversable = buildAccel(scene);

                camera = new Camera(
                    scene.CameraOrigin,
                    scene.CameraLookAt,
                    scene.CameraUp,
                    width,
                    height,
                    new Vec3(1f, 1f, 1f),
                    scene.CameraFovDeg,
                    scene.CameraWorldScale);

                launchParams.camera = camera;
                launchParams.traversable = unchecked((ulong)traversable.ToInt64());
                launchParams.Vertices = (Vec3*)NativePtrOrZero(d_vertices);
                launchParams.Normals = (Vec3*)NativePtrOrZero(d_normals);
                launchParams.TexCoords = (Vec2*)NativePtrOrZero(d_texCoords);
                launchParams.Indices = (Vec3i*)NativePtrOrZero(d_indices);
                launchParams.Spheres = (SphereData*)NativePtrOrZero(d_spheres);
                launchParams.Boxes = (BoxData*)NativePtrOrZero(d_boxes);
                launchParams.CylindersY = (CylinderYData*)NativePtrOrZero(d_cylindersY);
                launchParams.Disks = (DiskData*)NativePtrOrZero(d_disks);
                launchParams.XYRects = (RectData*)NativePtrOrZero(d_xyRects);
                launchParams.XZRects = (RectData*)NativePtrOrZero(d_xzRects);
                launchParams.YZRects = (RectData*)NativePtrOrZero(d_yzRects);
                launchParams.VoxelMaterialIds = (uint*)NativePtrOrZero(d_voxelMaterialIds);
                launchParams.VolumeGridMin = scene.VolumeGridMin;
                launchParams.VolumeVoxelSize = scene.VolumeVoxelSize;
                launchParams.VolumeDims = scene.VolumeDims;
                launchParams.Materials = (MaterialSbtData*)NativePtrOrZero(d_materials);
                launchParams.PointLights = (PointLightGpu*)NativePtrOrZero(d_lights);
                launchParams.NumPointLights = scene.Lights.Length;
                launchParams.AmbientColor = scene.AmbientColor;
                launchParams.AmbientIntensity = scene.AmbientIntensity;
                launchParams.BackgroundTop = scene.BackgroundTop;
                launchParams.BackgroundBottom = scene.BackgroundBottom;
                launchParams.FrameID = 0;

                hdrColorBuffer.MemSetToZero();

                bool hasTriangles = scene.Vertices.Length > 0 && scene.Indices.Length > 0;
                bool hasCustomPrimitives = scene.Spheres.Length > 0 || scene.Boxes.Length > 0 ||
                    scene.CylindersY.Length > 0 || scene.Disks.Length > 0 || scene.XYRects.Length > 0 ||
                    scene.XZRects.Length > 0 || scene.YZRects.Length > 0 || scene.VoxelMaterialIds.Length > 0;

                TriangleCount = scene.Indices.Length;
                MaterialCount = scene.Materials.Length;
                SphereCount = scene.Spheres.Length;
                BoxCount = scene.Boxes.Length;
                CylinderYCount = scene.CylindersY.Length;
                DiskCount = scene.Disks.Length;
                XYRectCount = scene.XYRects.Length;
                XZRectCount = scene.XZRects.Length;
                YZRectCount = scene.YZRects.Length;
                HasVolumeGrid = scene.VoxelMaterialIds.Length > 0;
                VolumeGridDims = scene.VolumeDims;
                AccelStructureSummary = BuildAccelStructureSummary(scene, hasTriangles, hasCustomPrimitives);
            }
        }

        // Hitgroup records are laid out as [triangle mat0-radiance, mat0-shadow, mat1-
        // radiance, mat1-shadow, ...] followed by the same per-material sequence again
        // for each custom-primitive kind actually present in the scene, in canonical
        // kind order (Sphere, Box, CylinderY, Disk, XYRect, XZRect, YZRect - matching
        // IntersectionPrograms.HitKind* and BuildCustomPrimitivesGas's build-input
        // order). This relies on OptiX automatically summing NumSbtRecords across build
        // inputs within a GAS to compute each build input's base SBT-GAS-index, so each
        // build input's own SbtIndexOffsetBuffer values stay local/0-based - see
        // docs/SAMPLE13_PLAN.md.
        // Loads (or reuses an already-loaded) CudaTextureObject for a relative path
        // under the output directory - same TextureLoader.LoadRgba8/CudaTextureObject
        // pattern as Sample08. Cached/disposed per scene switch (DisposeSceneBuffers),
        // not across the whole app lifetime, since only the active scene's textures
        // need to stay resident.
        ulong GetOrLoadTexture(string relativePath)
        {
            if (VideoExtensions.Contains(Path.GetExtension(relativePath), StringComparer.OrdinalIgnoreCase))
                return GetOrLoadVideoTexture(relativePath);

            if (textureCache.TryGetValue(relativePath, out var cachedHandle))
                return cachedHandle;

            string fullPath = Path.Combine(AppContext.BaseDirectory, relativePath);
            if (!File.Exists(fullPath))
            {
                textureCache[relativePath] = 0;
                return 0;
            }

            var pixels = TextureLoader.LoadRgba8(fullPath, out var texWidth, out var texHeight);
            var textureObject = new CudaTextureObject(pixels, texWidth, texHeight);
            textureObjects.Add(textureObject);
            var handle = textureObject.TextureObject;
            textureCache[relativePath] = handle;
            return handle;
        }

        // Loads (or reuses) a VideoTexture for a relative path under the output
        // directory - same cache-by-path convention as GetOrLoadTexture, but tracked
        // separately in activeVideoTextures so render() knows which textures need a
        // per-frame Refresh(). Requires ffmpeg.exe on PATH; if the video file itself is
        // missing, degrades the same way GetOrLoadTexture does for a missing image
        // (TextureObject stays 0 - untextured material).
        ulong GetOrLoadVideoTexture(string relativePath)
        {
            if (videoTextureCache.TryGetValue(relativePath, out var cached))
            {
                if (!activeVideoTextures.Contains(cached))
                    activeVideoTextures.Add(cached);
                return cached.TextureObject;
            }

            string fullPath = Path.Combine(AppContext.BaseDirectory, relativePath);
            if (!File.Exists(fullPath))
                return 0;

            VideoTexture videoTexture;
            try
            {
                videoTexture = new VideoTexture(fullPath);
            }
            catch (Exception ex)
            {
                // ffmpeg.exe missing from PATH, or OpenCvSharp couldn't probe the file -
                // degrade to "no texture" rather than crashing the whole scene switch.
                Console.Error.WriteLine($"[VideoTexture] Failed to open '{fullPath}': {ex.Message}");
                return 0;
            }

            videoTextureCache[relativePath] = videoTexture;
            activeVideoTextures.Add(videoTexture);
            return videoTexture.TextureObject;
        }

        unsafe void BuildHitgroupSbt(SceneData scene)
        {
            bool hasTriangles = scene.Vertices.Length > 0 && scene.Indices.Length > 0;
            // One full Materials.Length radiance+shadow block per triangle mesh build
            // input (see GetTriangleMeshRanges/BuildTrianglesGas), not just one for the
            // whole scene - mirrors the per-custom-primitive-kind blocks below.
            int triangleMeshCount = hasTriangles ? GetTriangleMeshRanges(scene).Length : 0;
            int[] customCounts =
            {
                scene.Spheres.Length, scene.Boxes.Length, scene.CylindersY.Length, scene.Disks.Length,
                scene.XYRects.Length, scene.XZRects.Length, scene.YZRects.Length,
            };

            var hitgroupKernelsList = new List<OptixKernel>();
            for (var m = 0; m < triangleMeshCount; m++)
            {
                for (var i = 0; i < scene.Materials.Length; i++)
                {
                    hitgroupKernelsList.Add(radianceHitgroupKernel);
                    hitgroupKernelsList.Add(shadowHitgroupKernel);
                }
            }
            for (var kind = 0; kind < customCounts.Length; kind++)
            {
                if (customCounts[kind] == 0)
                    continue;
                for (var i = 0; i < scene.Materials.Length; i++)
                {
                    hitgroupKernelsList.Add(radianceHitgroupKernelsCustom[kind]);
                    hitgroupKernelsList.Add(shadowHitgroupKernelsCustom[kind]);
                }
            }

            bool hasVolumeGrid = scene.VoxelMaterialIds.Length > 0;
            if (hasVolumeGrid)
            {
                // NumSbtRecords=1 for this build input (see BuildCustomPrimitivesGas) -
                // its record's own custom data is never read (ShadeVolumeGrid looks up
                // materials directly via LaunchParams.Materials instead), so only the
                // kernel entries (for header/program-group dispatch) matter here.
                hitgroupKernelsList.Add(radianceHitgroupKernelVolumeGrid);
                hitgroupKernelsList.Add(shadowHitgroupKernelVolumeGrid);
            }

            var hitgroupRecordsArray = OptixSbt.PackRecords<HitgroupRecord>(hitgroupKernelsList);

            // Resolved once per material index - any material without a texture path
            // (the overwhelming majority of Sample13's scenes) keeps TextureObject=0,
            // matching Sample08's "0 = no texture" convention.
            var textureHandles = new ulong[scene.Materials.Length];
            for (var i = 0; i < scene.Materials.Length; i++)
            {
                string path = i < scene.MaterialTexturePaths.Length ? scene.MaterialTexturePaths[i] : null;
                textureHandles[i] = string.IsNullOrEmpty(path) ? 0 : GetOrLoadTexture(path);
            }

            void FillMaterialRecords(int baseRecordIndex)
            {
                for (var i = 0; i < scene.Materials.Length; i++)
                {
                    var mat = scene.Materials[i];
                    ref var record = ref hitgroupRecordsArray[baseRecordIndex + (i * 2)];
                    record.Albedo = mat.Albedo;
                    record.Reflectivity = mat.Reflectivity;
                    record.Emission = mat.Emission;
                    record.Transparency = mat.Transparency;
                    record.IndexOfRefraction = mat.IndexOfRefraction;
                    record.TransmissionColor = mat.TransmissionColor;
                    record.TextureObject = textureHandles[i];
                    record.TextureWeight = mat.TextureWeight;
                    record.UVScale = mat.UVScale;
                    record.MaterialKind = mat.MaterialKind;
                    record.CheckerColorB = mat.CheckerColorB;
                    record.CheckerScale = mat.CheckerScale;
                }
            }

            var recordIndex = 0;
            for (var m = 0; m < triangleMeshCount; m++)
            {
                FillMaterialRecords(recordIndex);
                recordIndex += scene.Materials.Length * 2;
            }
            for (var kind = 0; kind < customCounts.Length; kind++)
            {
                if (customCounts[kind] == 0)
                    continue;
                FillMaterialRecords(recordIndex);
                recordIndex += scene.Materials.Length * 2;
            }

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
        }

        void DisposeSceneBuffers()
        {
            foreach (var textureObject in textureObjects)
                textureObject.Dispose();
            textureObjects.Clear();
            textureCache.Clear();

            foreach (var videoTexture in videoTextureCache.Values)
                videoTexture.Dispose();
            videoTextureCache.Clear();
            activeVideoTextures.Clear();

            customPrimitivesUpdateTempBuffer?.Dispose();
            customPrimitivesUpdateTempBuffer = null;

            instancesBuffer?.Dispose();
            iasBuffer?.Dispose();
            customPrimitivesGasBuffer?.Dispose();
            trianglesGasBuffer?.Dispose();
            hitgroupRecordsBuffer?.Dispose();

            d_materials?.Dispose();
            d_volumeGridAabb?.Dispose();
            d_voxelMaterialIds?.Dispose();
            d_yzRectAabbs?.Dispose();
            d_yzRectMaterialIds?.Dispose();
            d_yzRects?.Dispose();
            d_xzRectAabbs?.Dispose();
            d_xzRectMaterialIds?.Dispose();
            d_xzRects?.Dispose();
            d_xyRectAabbs?.Dispose();
            d_xyRectMaterialIds?.Dispose();
            d_xyRects?.Dispose();
            d_diskAabbs?.Dispose();
            d_diskMaterialIds?.Dispose();
            d_disks?.Dispose();
            d_cylinderYAabbs?.Dispose();
            d_cylinderYMaterialIds?.Dispose();
            d_cylindersY?.Dispose();
            d_boxAabbs?.Dispose();
            d_boxMaterialIds?.Dispose();
            d_boxes?.Dispose();
            d_sphereAabbs?.Dispose();
            d_sphereMaterialIds?.Dispose();
            d_spheres?.Dispose();

            d_lights?.Dispose();
            d_materialIds?.Dispose();
            d_indices?.Dispose();
            d_texCoords?.Dispose();
            d_normals?.Dispose();
            d_vertices?.Dispose();
        }

        public void Dispose()
        {
            DisposeSceneBuffers();

            bufferFree?[0]?.Dispose();
            bufferFree?[1]?.Dispose();
            bufferReady?[0]?.Dispose();
            bufferReady?[1]?.Dispose();

            denoiserIntensity?.Dispose();
            denoiserScratch?.Dispose();
            denoiserState?.Dispose();
            denoiser.Dispose();

            displayBuffer.Dispose();
            denoisedColorBuffer.Dispose();
            normalBuffer.Dispose();
            albedoBuffer.Dispose();
            hdrColorBuffer.Dispose();

            missRecordsBuffer.Dispose();
            raygenRecordsBuffer.Dispose();

            pipeline.Dispose();

            shadowHitgroupKernelVolumeGrid.Dispose();
            radianceHitgroupKernelVolumeGrid.Dispose();
            foreach (var kernel in radianceHitgroupKernelsCustom)
                kernel.Dispose();
            foreach (var kernel in shadowHitgroupKernelsCustom)
                kernel.Dispose();
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
            if (width == 0 || height == 0)
                return;

            this.width = width;
            this.height = height;

            hdrColorBuffer = accelerator.Allocate1D<Vec4>(width * height);
            hdrColorBuffer.MemSetToZero();
            albedoBuffer = accelerator.Allocate1D<Vec4>(width * height);
            albedoBuffer.MemSetToZero();
            normalBuffer = accelerator.Allocate1D<Vec4>(width * height);
            normalBuffer.MemSetToZero();
            denoisedColorBuffer = accelerator.Allocate1D<Vec4>(width * height);
            denoisedColorBuffer.MemSetToZero();
            displayBuffer = accelerator.Allocate1D<byte>(width * height * 4);
            displayBuffer.MemSetToZero();

            // Index 0 is MainWindow's initial read index (see TryPresentFrame and
            // MainWindow's presentIndex field) and also render's initial writeIndex -
            // fine, since render always claims bufferFree[writeIndex] before writing,
            // and both start signaled-free. Sample13's window is ResizeMode="NoResize"
            // so this only ever runs once, at startup.
            int frameBytes = (int)displayBuffer.Length;
            frameBuffers = new[] { new byte[frameBytes], new byte[frameBytes] };
            bufferFree?[0]?.Dispose();
            bufferFree?[1]?.Dispose();
            bufferReady?[0]?.Dispose();
            bufferReady?[1]?.Dispose();
            bufferFree = new[] { new SemaphoreSlim(1, 1), new SemaphoreSlim(1, 1) };
            bufferReady = new[] { new SemaphoreSlim(0, 1), new SemaphoreSlim(0, 1) };
            writeIndex = 0;

            denoiserIntensity?.Dispose();
            denoiserIntensity = accelerator.Allocate1D<byte>(sizeof(float));

            denoiserState?.Dispose();
            denoiserScratch?.Dispose();
            var denoiserSizes = denoiser.ComputeMemoryResources((uint)width, (uint)height);
            denoiserState = accelerator.Allocate1D<byte>((long)denoiserSizes.StateSizeInBytes);
            ulong denoiserScratchSize = Math.Max(denoiserSizes.WithOverlapScratchSizeInBytes, denoiserSizes.WithoutOverlapScratchSizeInBytes);
            denoiserScratch = accelerator.Allocate1D<byte>((long)denoiserScratchSize);
            denoiser.Setup(
                DefaultStreamPtr,
                (uint)width,
                (uint)height,
                denoiserState.NativePtr,
                (ulong)denoiserState.LengthInBytes,
                denoiserScratch.NativePtr,
                (ulong)denoiserScratch.LengthInBytes);

            launchParams.NumPixelSamples = NumPixelSamples;
            launchParams.ColorBuffer = (Vec4*)hdrColorBuffer.NativePtr;
            launchParams.AlbedoBuffer = (Vec4*)albedoBuffer.NativePtr;
            launchParams.NormalBuffer = (Vec4*)normalBuffer.NativePtr;
        }

        // Triangle geometry and custom-primitive geometry cannot be combined as
        // multiple build inputs within a single GAS (confirmed against the OptiX SDK's
        // own optixSimpleMotionBlur sample - it builds one GAS per build-input type and
        // combines them with an IAS instead). So triangles and custom primitives each
        // get their own GAS here (all 7 custom-primitive kinds share ONE GAS as multiple
        // build inputs of the same type), and an IAS with one instance per GAS ties them
        // together; each OptixInstance.SbtOffset picks up where the previous instance's
        // hitgroup records left off, matching BuildHitgroupSbt's record layout.
        public unsafe IntPtr buildAccel(SceneData scene)
        {
            bool hasTriangles = scene.Vertices.Length > 0 && scene.Indices.Length > 0;
            bool hasCustomPrimitives = scene.Spheres.Length > 0 || scene.Boxes.Length > 0
                || scene.CylindersY.Length > 0 || scene.Disks.Length > 0
                || scene.XYRects.Length > 0 || scene.XZRects.Length > 0 || scene.YZRects.Length > 0
                || scene.VoxelMaterialIds.Length > 0;

            IntPtr trianglesGasHandle = hasTriangles ? BuildTrianglesGas(scene) : IntPtr.Zero;
            IntPtr customPrimitivesGasHandle = hasCustomPrimitives ? BuildOrUpdateCustomPrimitivesGas(scene, OptixBuildOperation.OPTIX_BUILD_OPERATION_BUILD) : IntPtr.Zero;

            return BuildInstanceAccel(scene, trianglesGasHandle, hasTriangles, customPrimitivesGasHandle, hasCustomPrimitives);
        }

        // One build input per mesh (see SceneData.MeshRanges) instead of one merged
        // build input for the whole scene - mirrors BuildOrUpdateCustomPrimitivesGas's
        // multiple-build-inputs-in-one-GAS pattern below. Every mesh shares the same
        // device vertex/material-id buffers (only ever offset, never duplicated) and
        // the same global/absolute Materials[] palette (no per-mesh SBT-index
        // remapping - see GetTriangleMeshRanges's comment), so vertexBuffers/
        // sharedFlags below are declared once and reused by every build input, unlike
        // BuildOrUpdateCustomPrimitivesGas's per-kind AABB buffers which genuinely
        // differ per build input.
        unsafe IntPtr BuildTrianglesGas(SceneData scene)
        {
            MeshRange[] ranges = GetTriangleMeshRanges(scene);

            var vertexBuffers = stackalloc IntPtr[1];
            vertexBuffers[0] = d_vertices.NativePtr;
            var sharedFlags = stackalloc uint[scene.Materials.Length];

            var buildInputs = new OptixBuildInput[ranges.Length];
            for (int i = 0; i < ranges.Length; i++)
            {
                MeshRange range = ranges[i];
                OptixBuildInput input = new OptixBuildInput()
                {
                    Type = OptixBuildInputType.OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
                };

                input.TriangleArray.VertexFormat = OptixVertexFormat.OPTIX_VERTEX_FORMAT_FLOAT3;
                input.TriangleArray.VertexStrideInBytes = (uint)sizeof(Vec3);
                input.TriangleArray.NumVerticies = (uint)scene.Vertices.Length;
                input.TriangleArray.VertexBuffers = new IntPtr(vertexBuffers);

                input.TriangleArray.IndexFormat = OptixIndicesFormat.OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                input.TriangleArray.IndexStrideInBytes = (uint)sizeof(Vec3i);
                input.TriangleArray.NumIndexTriplets = (uint)range.IndexCount;
                input.TriangleArray.IndexBuffer = IntPtr.Add(d_indices.NativePtr, range.IndexStart * sizeof(Vec3i));

                input.TriangleArray.Flags = sharedFlags;
                input.TriangleArray.NumSbtRecords = (uint)scene.Materials.Length;
                input.TriangleArray.SbtIndexOffsetBuffer = IntPtr.Add(d_materialIds.NativePtr, range.IndexStart * sizeof(uint));
                input.TriangleArray.SbtIndexOffsetSizeInBytes = sizeof(uint);
                input.TriangleArray.SbtIndexOffsetStrideInBytes = 0;

                buildInputs[i] = input;
            }

            OptixAccelBuildOptions accelOptions = new OptixAccelBuildOptions()
            {
                BuildFlags = OptixBuildFlags.OPTIX_BUILD_FLAG_NONE,
                Operation = OptixBuildOperation.OPTIX_BUILD_OPERATION_BUILD
            };
            accelOptions.MotionOptions.NumKeys = 1;

            OptixAccelBufferSizes bufferSizes = deviceContext.AccelComputeMemoryUsage(accelOptions, buildInputs);

            using MemoryBuffer1D<byte, Stride1D.Dense> tempBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.TempSizeInBytes);
            trianglesGasBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.OutputSizeInBytes);

            return deviceContext.AccelBuild(accelerator.DefaultStream, accelOptions, buildInputs, tempBuffer, trianglesGasBuffer, ReadOnlySpan<OptixAccelEmitDesc>.Empty);
        }

        // UseMergedTrianglesGas==true (the default) always returns a single range
        // spanning the whole Indices array, regardless of what SceneData.MeshRanges
        // tracked - the original single-build-input behavior. Otherwise, empty/unset
        // SceneData.MeshRanges still means "treat the whole Indices array as a single
        // implicit mesh" (every scene that doesn't explicitly track mesh boundaries -
        // procedural/test/CSG scenes, BuildMeshScene). Shared by BuildTrianglesGas,
        // BuildHitgroupSbt, and BuildInstanceAccel so this fallback logic lives in one
        // place.
        MeshRange[] GetTriangleMeshRanges(SceneData scene) =>
            (!UseMergedTrianglesGas && scene.MeshRanges != null && scene.MeshRanges.Length > 0)
                ? scene.MeshRanges
                : new[] { new MeshRange { IndexStart = 0, IndexCount = scene.Indices.Length } };

        // Builds ONE GAS from multiple OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES build
        // inputs (one per active primitive kind, in canonical HitKind order) - this is
        // supported (same build-input TYPE repeated, like multiple triangle meshes
        // already sharing one GAS), unlike mixing triangles with custom primitives.
        // Every stackalloc'd AABB-buffer-pointer array and the shared flags array must
        // stay alive for this whole method body (stackalloc's lifetime is the enclosing
        // method's stack frame), so this can't be split into per-kind helper methods
        // the way BuildTrianglesGas is.
        //
        // Also doubles as the per-frame refit path for scenes with
        // SceneData.HasAnimatedGeometry (bobbing spheres): called again every frame
        // with operation=OPTIX_BUILD_OPERATION_UPDATE after the caller has re-uploaded
        // fresh contents into the SAME d_sphereAabbs/d_* device buffers (their addresses
        // never change between frames, only their contents) - OptiX's update operation
        // requires the exact same build-input topology/buffer pointers as the original
        // build, just refreshed data, and always returns the same traversable handle it
        // was given at the original BUILD (see docs/SAMPLE13_PLAN.md's AS-update design
        // note), so the IAS instance pointing at this GAS's handle never needs touching.
        unsafe IntPtr BuildOrUpdateCustomPrimitivesGas(SceneData scene, OptixBuildOperation operation)
        {
            var buildInputsList = new List<OptixBuildInput>();

            // Every custom-primitive build input can share this one all-zero
            // (OPTIX_GEOMETRY_FLAG_NONE) flags array - none of our materials need
            // per-geometry build flags, and "size must match numSbtRecords" is
            // satisfied for every kind since NumSbtRecords is always scene.Materials.Length.
            var sharedFlags = stackalloc uint[scene.Materials.Length];

            unsafe OptixBuildInput MakeCustomPrimitiveInput(IntPtr* aabbBufferSlot, IntPtr aabbBufferPtr, uint numPrimitives, IntPtr sbtIndexOffsetBuffer)
            {
                aabbBufferSlot[0] = aabbBufferPtr;
                var input = new OptixBuildInput { Type = OptixBuildInputType.OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES };
                input.CustomPrimitiveArray.AabbBuffers = new IntPtr(aabbBufferSlot);
                input.CustomPrimitiveArray.NumPrimitives = numPrimitives;
                input.CustomPrimitiveArray.Stride = 0; // tightly packed, sizeof(OptixAabb)
                input.CustomPrimitiveArray.Flags = sharedFlags;
                input.CustomPrimitiveArray.NumSbtRecords = (uint)scene.Materials.Length;
                input.CustomPrimitiveArray.SbtIndexOffsetBuffer = sbtIndexOffsetBuffer;
                input.CustomPrimitiveArray.SbtIndexOffsetSizeInBytes = sizeof(uint);
                input.CustomPrimitiveArray.SbtIndexOffsetStrideInBytes = 0;
                return input;
            }

            var sphereAabbSlot = stackalloc IntPtr[1];
            if (scene.Spheres.Length > 0)
                buildInputsList.Add(MakeCustomPrimitiveInput(sphereAabbSlot, d_sphereAabbs.NativePtr, (uint)scene.Spheres.Length, d_sphereMaterialIds.NativePtr));

            var boxAabbSlot = stackalloc IntPtr[1];
            if (scene.Boxes.Length > 0)
                buildInputsList.Add(MakeCustomPrimitiveInput(boxAabbSlot, d_boxAabbs.NativePtr, (uint)scene.Boxes.Length, d_boxMaterialIds.NativePtr));

            var cylinderYAabbSlot = stackalloc IntPtr[1];
            if (scene.CylindersY.Length > 0)
                buildInputsList.Add(MakeCustomPrimitiveInput(cylinderYAabbSlot, d_cylinderYAabbs.NativePtr, (uint)scene.CylindersY.Length, d_cylinderYMaterialIds.NativePtr));

            var diskAabbSlot = stackalloc IntPtr[1];
            if (scene.Disks.Length > 0)
                buildInputsList.Add(MakeCustomPrimitiveInput(diskAabbSlot, d_diskAabbs.NativePtr, (uint)scene.Disks.Length, d_diskMaterialIds.NativePtr));

            var xyRectAabbSlot = stackalloc IntPtr[1];
            if (scene.XYRects.Length > 0)
                buildInputsList.Add(MakeCustomPrimitiveInput(xyRectAabbSlot, d_xyRectAabbs.NativePtr, (uint)scene.XYRects.Length, d_xyRectMaterialIds.NativePtr));

            var xzRectAabbSlot = stackalloc IntPtr[1];
            if (scene.XZRects.Length > 0)
                buildInputsList.Add(MakeCustomPrimitiveInput(xzRectAabbSlot, d_xzRectAabbs.NativePtr, (uint)scene.XZRects.Length, d_xzRectMaterialIds.NativePtr));

            var yzRectAabbSlot = stackalloc IntPtr[1];
            if (scene.YZRects.Length > 0)
                buildInputsList.Add(MakeCustomPrimitiveInput(yzRectAabbSlot, d_yzRectAabbs.NativePtr, (uint)scene.YZRects.Length, d_yzRectMaterialIds.NativePtr));

            // Volume grid: NumPrimitives=1 (one AABB for the whole grid), NumSbtRecords=1
            // (its record's custom data is unused - ShadeVolumeGrid looks up materials
            // directly via LaunchParams.Materials/VoxelMaterialIds instead, since a
            // single primitive can't otherwise carry a per-voxel material through the
            // SbtIndexOffsetBuffer mechanism - see devicePrograms.cs). A null
            // SbtIndexOffsetBuffer is valid per optix_types.h ("May be NULL" - every
            // entry is then sbt index 0), so no dedicated device buffer is needed for it.
            var volumeGridAabbSlot = stackalloc IntPtr[1];
            if (scene.VoxelMaterialIds.Length > 0)
            {
                volumeGridAabbSlot[0] = d_volumeGridAabb.NativePtr;
                var volumeGridInput = new OptixBuildInput { Type = OptixBuildInputType.OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES };
                volumeGridInput.CustomPrimitiveArray.AabbBuffers = new IntPtr(volumeGridAabbSlot);
                volumeGridInput.CustomPrimitiveArray.NumPrimitives = 1;
                volumeGridInput.CustomPrimitiveArray.Stride = 0;
                volumeGridInput.CustomPrimitiveArray.Flags = sharedFlags;
                volumeGridInput.CustomPrimitiveArray.NumSbtRecords = 1;
                volumeGridInput.CustomPrimitiveArray.SbtIndexOffsetBuffer = IntPtr.Zero;
                buildInputsList.Add(volumeGridInput);
            }

            OptixAccelBuildOptions accelOptions = new OptixAccelBuildOptions()
            {
                BuildFlags = scene.HasAnimatedGeometry ? OptixBuildFlags.OPTIX_BUILD_FLAG_ALLOW_UPDATE : OptixBuildFlags.OPTIX_BUILD_FLAG_NONE,
                Operation = operation
            };
            accelOptions.MotionOptions.NumKeys = 1;

            OptixBuildInput[] buildInputs = buildInputsList.ToArray();

            if (operation == OptixBuildOperation.OPTIX_BUILD_OPERATION_UPDATE)
            {
                // Same output buffer as the original BUILD (required by OptiX for
                // update), and the persistent update-sized temp buffer allocated back
                // when that BUILD ran.
                return deviceContext.AccelBuild(accelerator.DefaultStream, accelOptions, buildInputs, customPrimitivesUpdateTempBuffer, customPrimitivesGasBuffer, ReadOnlySpan<OptixAccelEmitDesc>.Empty);
            }

            OptixAccelBufferSizes bufferSizes = deviceContext.AccelComputeMemoryUsage(accelOptions, buildInputs);

            using MemoryBuffer1D<byte, Stride1D.Dense> tempBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.TempSizeInBytes);
            customPrimitivesGasBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.OutputSizeInBytes);

            if (scene.HasAnimatedGeometry)
            {
                customPrimitivesUpdateTempBuffer?.Dispose();
                customPrimitivesUpdateTempBuffer = accelerator.Allocate1D<byte>((long)Math.Max(1UL, bufferSizes.TempUpdateSizeInBytes));
            }

            return deviceContext.AccelBuild(accelerator.DefaultStream, accelOptions, buildInputs, tempBuffer, customPrimitivesGasBuffer, ReadOnlySpan<OptixAccelEmitDesc>.Empty);
        }

        unsafe IntPtr BuildInstanceAccel(SceneData scene, IntPtr trianglesGasHandle, bool hasTriangles, IntPtr customPrimitivesGasHandle, bool hasCustomPrimitives)
        {
            // Triangle records (if present) occupy [0, triangleMeshCount *
            // materials.Length*RAY_TYPE_COUNT) in BuildHitgroupSbt's flat array
            // (RAY_TYPE_COUNT=2, radiance+shadow per material - matches
            // devicePrograms.cs's RAY_TYPE_COUNT; triangleMeshCount is >1 whenever the
            // triangles GAS has multiple per-mesh build inputs, see
            // GetTriangleMeshRanges/BuildTrianglesGas); the custom-primitives
            // instance's records start right after (or at 0, if there's no triangle
            // instance - e.g. the volume-grid-only test scene). All triangle build
            // inputs still live in the ONE triangles GAS, so only one IAS instance is
            // needed for them here, regardless of how many mesh build inputs it has.
            var instances = new List<OptixInstance>();
            if (hasTriangles)
                instances.Add(MakeInstance(trianglesGasHandle, sbtOffset: 0));
            if (hasCustomPrimitives)
            {
                int triangleMeshCount = hasTriangles ? GetTriangleMeshRanges(scene).Length : 0;
                uint customPrimitivesSbtOffset = hasTriangles ? (uint)(triangleMeshCount * scene.Materials.Length * 2) : 0;
                instances.Add(MakeInstance(customPrimitivesGasHandle, customPrimitivesSbtOffset));
            }

            instancesBuffer = accelerator.Allocate1D(instances.ToArray());

            OptixBuildInput instanceInput = new OptixBuildInput()
            {
                Type = OptixBuildInputType.OPTIX_BUILD_INPUT_TYPE_INSTANCES,
            };
            instanceInput.InstanceArray.Instances = instancesBuffer.NativePtr;
            instanceInput.InstanceArray.NumInstances = (uint)instances.Count;
            instanceInput.InstanceArray.InstanceStride = 0;

            OptixAccelBuildOptions accelOptions = new OptixAccelBuildOptions()
            {
                BuildFlags = OptixBuildFlags.OPTIX_BUILD_FLAG_NONE,
                Operation = OptixBuildOperation.OPTIX_BUILD_OPERATION_BUILD
            };
            accelOptions.MotionOptions.NumKeys = 1;

            OptixBuildInput[] buildInputs = { instanceInput };
            OptixAccelBufferSizes bufferSizes = deviceContext.AccelComputeMemoryUsage(accelOptions, buildInputs);

            using MemoryBuffer1D<byte, Stride1D.Dense> tempBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.TempSizeInBytes);
            iasBuffer = accelerator.Allocate1D<byte>((long)bufferSizes.OutputSizeInBytes);

            return deviceContext.AccelBuild(accelerator.DefaultStream, accelOptions, buildInputs, tempBuffer, iasBuffer, ReadOnlySpan<OptixAccelEmitDesc>.Empty);
        }

        static unsafe OptixInstance MakeInstance(IntPtr gasHandle, uint sbtOffset)
        {
            OptixInstance instance = default;
            for (int i = 0; i < 12; i++)
                instance.Transform[i] = OptixInstance.IdentityTransform[i];
            instance.InstanceId = 0;
            instance.SbtOffset = sbtOffset;
            instance.VisibilityMask = 0xFF;
            instance.Flags = 0; // OPTIX_INSTANCE_FLAG_NONE
            instance.TraversableHandle = unchecked((ulong)gasHandle.ToInt64());
            return instance;
        }

        // Guards every access to launchParams, traversable, and the d_* GPU buffers -
        // taken by render() around its whole body, and by SwitchToScene()/setCamera()
        // (both only ever called from the UI thread, on I/U/D/Space/A key presses or
        // mouse-drag camera moves) around theirs. Without this, SwitchToScene's buffer
        // disposal/reallocation and GAS/IAS rebuild could run concurrently with an
        // in-flight OptixLaunch reading the very pointers/traversable handle being torn
        // down - undefined behavior at the CUDA/OptiX driver level (observed as NaN ray
        // origins and a sticky "unspecified launch failure" CUDA error). This race
        // always existed but was rarely hit while the render loop was throttled by the
        // old fixed Thread.Sleep(10) + blocking per-frame Dispatcher.Invoke; removing
        // both (see TryPresentFrame/FinishPresent's double-buffered handoff) made the
        // render loop run fast enough to hit it almost every scene switch. This lock
        // is unrelated to - and does not reintroduce - the old per-frame UI-thread
        // block: the presenter thread only ever touches frameBuffers/bufferFree/
        // bufferReady via SemaphoreSlim, never launchParams or GPU buffers, so it
        // never waits on gpuLock.
        //
        // Critically, render()'s lock scope does NOT extend over its bufferFree.Wait()
        // publish step (see render()'s comment right after tonemapMs) - an earlier
        // version did, and it deadlocked the whole app on the first mouse drag:
        // setCamera() takes this same lock synchronously on the UI thread, so once
        // render() was blocked in bufferFree.Wait() *while holding gpuLock*, the only
        // thing that could unblock it - MainWindow's OnCompositionRendering calling
        // FinishPresent - could never run, because the UI thread was itself stuck
        // wanting gpuLock inside setCamera(). Keeping the buffer-publish step outside
        // this lock (safe, since it never touches launchParams/traversable/d_* buffers)
        // breaks that cycle.
        readonly object gpuLock = new object();

        // Per-frame animation for the museum/radial-museum scenes - direct port of the
        // reference's OrbitingLightEntity/PulsingLightEntity/BobbingSphereEntity update
        // formulas (RayTracing/Scenes/TestScenesRandom.cs), evaluated from elapsed
        // wall-clock time (animationClock) rather than accumulated per-frame dt: each
        // entity's own "t += dt" accumulator, started at 0 and monotonically advanced
        // once per Update call, is mathematically just elapsed time since the scene
        // started, so this is equivalent while being immune to any frame-timing drift.
        // Called every render() frame under gpuLock, before OptixLaunch.
        unsafe void ApplyAnimation(SceneData scene, double elapsedSeconds)
        {
            float t = (float)elapsedSeconds;
            bool lightsChanged = false;

            foreach (var anim in scene.PulsingLights)
            {
                float s = 0.5f + (0.5f * XMath.Sin(anim.Speed * t));
                float mult = Math.Max(0f, anim.MinMult + ((anim.MaxMult - anim.MinMult) * s));
                var light = animatedLightsHost[anim.LightIndex];
                light.Intensity = anim.BaseIntensity * mult;
                animatedLightsHost[anim.LightIndex] = light;
                lightsChanged = true;
            }

            foreach (var anim in scene.OrbitingLights)
            {
                float a = (anim.AngularSpeed * t) + anim.Phase;
                var light = animatedLightsHost[anim.LightIndex];
                light.Position = new Vec3(
                    anim.Pivot.x + (anim.Radius * XMath.Cos(a)),
                    anim.Height,
                    anim.Pivot.z + (anim.Radius * XMath.Sin(a)));
                animatedLightsHost[anim.LightIndex] = light;
                lightsChanged = true;
            }

            if (lightsChanged)
                d_lights.CopyFromCPU(animatedLightsHost);

            if (scene.BobbingSpheres.Length > 0)
            {
                foreach (var anim in scene.BobbingSpheres)
                {
                    float y = anim.BaseY + (anim.Amplitude * XMath.Sin((anim.Speed * t) + anim.Phase));
                    var sphere = animatedSpheresHost[anim.SphereIndex];
                    sphere.Center = new Vec3(sphere.Center.x, y, sphere.Center.z);
                    animatedSpheresHost[anim.SphereIndex] = sphere;
                }

                d_spheres.CopyFromCPU(animatedSpheresHost);
                d_sphereAabbs.CopyFromCPU(ComputeSphereAabbs(animatedSpheresHost));
                BuildOrUpdateCustomPrimitivesGas(scene, OptixBuildOperation.OPTIX_BUILD_OPERATION_UPDATE);
            }

            foreach (var videoTexture in activeVideoTextures)
                videoTexture.Refresh();
        }

        public void setCamera(Camera camera)
        {
            lock (gpuLock)
            {
                this.camera = camera;
                launchParams.camera = new Camera(camera, width, height);
                launchParams.FrameID = 0;
            }
        }

        public unsafe void render()
        {
            double traceMs, denoiseMs, tonemapMs;
            int samplesAccumulated;
            lock (gpuLock)
            {
                long frameStartTicks = Stopwatch.GetTimestamp();
                if (hasRenderedFrame)
                {
                    double sinceLastFrameMs = (frameStartTicks - lastFrameStartTicks) * 1000.0 / Stopwatch.Frequency;
                    // Exponential moving average - smooths out per-frame jitter so the
                    // displayed FPS doesn't flicker every frame; alpha=0.1 settles to a
                    // new steady-state rate within roughly 20-30 frames of a scene switch.
                    smoothedFrameMs = (smoothedFrameMs * 0.9) + (sinceLastFrameMs * 0.1);
                }
                lastFrameStartTicks = frameStartTicks;
                hasRenderedFrame = true;

                bool sceneIsAnimated = currentScene != null && (currentScene.HasAnyAnimation || activeVideoTextures.Count > 0);
                if (sceneIsAnimated)
                    ApplyAnimation(currentScene, animationClock.Elapsed.TotalSeconds);

                // Animated content invalidates progressive accumulation every frame
                // (lights/geometry moved since the last accumulated sample), so an
                // animated scene always renders as a fresh 1-sample frame + denoiser,
                // regardless of the user's Accumulate toggle - matching the reference's
                // own always-dynamic radial museum, and avoiding motion-ghosting
                // artifacts that a stale accumulation average would otherwise bake in.
                if (!Accumulate || sceneIsAnimated)
                    launchParams.FrameID = 0;
                launchParams.NumPixelSamples = NumPixelSamples;

                stepStopwatch.Restart();
                accelerator.OptixLaunch(
                    accelerator.DefaultStream,
                    pipeline,
                    launchParams,
                    sbt,
                    (uint)width,
                    (uint)height,
                    1);
                accelerator.Synchronize();
                traceMs = stepStopwatch.Elapsed.TotalMilliseconds;

                launchParams.FrameID++;

                var colorImage = new OptixImage2D
                {
                    Data = (ulong)hdrColorBuffer.NativePtr.ToInt64(),
                    Width = (uint)width,
                    Height = (uint)height,
                    RowStrideInBytes = (uint)(width * sizeof(Vec4)),
                    PixelStrideInBytes = (uint)sizeof(Vec4),
                    Format = OptixPixelFormat.OPTIX_PIXEL_FORMAT_FLOAT4
                };
                var albedoImage = new OptixImage2D
                {
                    Data = (ulong)albedoBuffer.NativePtr.ToInt64(),
                    Width = (uint)width,
                    Height = (uint)height,
                    RowStrideInBytes = (uint)(width * sizeof(Vec4)),
                    PixelStrideInBytes = (uint)sizeof(Vec4),
                    Format = OptixPixelFormat.OPTIX_PIXEL_FORMAT_FLOAT4
                };
                var normalImage = new OptixImage2D
                {
                    Data = (ulong)normalBuffer.NativePtr.ToInt64(),
                    Width = (uint)width,
                    Height = (uint)height,
                    RowStrideInBytes = (uint)(width * sizeof(Vec4)),
                    PixelStrideInBytes = (uint)sizeof(Vec4),
                    Format = OptixPixelFormat.OPTIX_PIXEL_FORMAT_FLOAT4
                };
                var outputImage = new OptixImage2D
                {
                    Data = (ulong)denoisedColorBuffer.NativePtr.ToInt64(),
                    Width = (uint)width,
                    Height = (uint)height,
                    RowStrideInBytes = (uint)(width * sizeof(Vec4)),
                    PixelStrideInBytes = (uint)sizeof(Vec4),
                    Format = OptixPixelFormat.OPTIX_PIXEL_FORMAT_FLOAT4
                };

                MemoryBuffer1D<Vec4, Stride1D.Dense> tonemapSource;
                denoiseMs = 0;
                if (DenoiserOn)
                {
                    stepStopwatch.Restart();
                    denoiser.ComputeIntensity(
                        DefaultStreamPtr,
                        colorImage,
                        denoiserIntensity.NativePtr,
                        denoiserScratch.NativePtr,
                        (ulong)denoiserScratch.LengthInBytes);

                    var denoiserParams = new OptixDenoiserParams
                    {
                        HdrIntensity = (ulong)denoiserIntensity.NativePtr.ToInt64(),
                        HdrAverageColor = 0,
                        BlendFactor = Accumulate ? 1f / launchParams.FrameID : 0f,
                        TemporalModeUsePreviousLayers = 0
                    };
                    var guideLayer = new OptixDenoiserGuideLayer { Albedo = albedoImage, Normal = normalImage };
                    var layer = new OptixDenoiserLayer { Input = colorImage, Output = outputImage };

                    denoiser.Invoke(
                        DefaultStreamPtr,
                        denoiserParams,
                        denoiserState.NativePtr,
                        (ulong)denoiserState.LengthInBytes,
                        guideLayer,
                        new[] { layer },
                        denoiserScratch.NativePtr,
                        (ulong)denoiserScratch.LengthInBytes);
                    accelerator.Synchronize();
                    denoiseMs = stepStopwatch.Elapsed.TotalMilliseconds;
                    tonemapSource = denoisedColorBuffer;
                }
                else
                {
                    tonemapSource = hdrColorBuffer;
                }

                stepStopwatch.Restart();
                tonemapAndFlip(new Index1D(width * height), width, height, tonemapSource.View, displayBuffer.View);
                accelerator.Synchronize();
                tonemapMs = stepStopwatch.Elapsed.TotalMilliseconds;

                samplesAccumulated = launchParams.FrameID;
                // gpuLock ends here, deliberately - everything below only touches
                // frameBuffers/bufferFree/bufferReady/displayBuffer (the last one is a
                // fixed GPU buffer allocated once in resize(), never touched by
                // SwitchToScene), never launchParams/traversable/d_* scene buffers, so
                // it doesn't need gpuLock's protection. This split is not optional:
                // bufferFree.Wait() below can block waiting for the UI thread's
                // OnCompositionRendering to run FinishPresent, and OnCompositionRendering
                // can only run when the UI thread is free to pump messages. Holding
                // gpuLock across that wait deadlocked the whole app the instant a mouse
                // drag called setCamera() (also a `lock (gpuLock)`, synchronously on the
                // UI thread) while render() happened to be waiting here - the UI thread
                // would then be stuck wanting gpuLock, unable to reach the very
                // OnCompositionRendering call render() was waiting on to release it.
            }

            stepStopwatch.Restart();
            int publishIndex = writeIndex;
            // Wait until the UI thread is done reading this slot from two frames ago
            // (see bufferFree's field comment) - in the overwhelmingly common case
            // this is immediate, since MainWindow.draw's single Marshal.Copy is far
            // cheaper than a GPU trace/denoise/tonemap pass; it only actually blocks
            // if presentation has fallen more than one frame behind, which is exactly
            // the DX12-style backpressure that couples the two rates together.
            // Bounded retries (rather than one blocking Wait) keep shutdown responsive
            // if MainWindow stops pumping CompositionTarget.Rendering (window closing)
            // before draining this slot.
            bool acquired;
            for (; ; )
            {
                if (bufferFree[publishIndex].Wait(50)) { acquired = true; break; }
                if (!window.run) { acquired = false; break; }
            }
            if (acquired)
                displayBuffer.CopyToCPU(frameBuffers[publishIndex]);
            double readbackMs = stepStopwatch.Elapsed.TotalMilliseconds;

            stepStopwatch.Restart();
            if (acquired)
            {
                bufferReady[publishIndex].Release();
                writeIndex = 1 - publishIndex;
            }
            double publishMs = stepStopwatch.Elapsed.TotalMilliseconds;

            LastStats = new FrameStats
            {
                TraceMs = traceMs,
                DenoiseMs = denoiseMs,
                TonemapMs = tonemapMs,
                ReadbackMs = readbackMs,
                PublishMs = publishMs,
                TotalFrameMs = smoothedFrameMs,
                Fps = 1000.0 / smoothedFrameMs,
                SamplesAccumulated = samplesAccumulated,
            };
        }

        // Reader side of the double-buffered async present handoff - called once per
        // WPF-composed frame from MainWindow's CompositionTarget.Rendering handler.
        // readIndex is the caller's private "next buffer to read" index (starts at 0,
        // see resize()); passing it by ref lets this update it in place. Non-blocking:
        // if the render thread hasn't published a fresh frame into this slot yet,
        // returns false and the caller just leaves whatever's already on screen (WPF
        // retains the WriteableBitmap's last contents automatically). Frames are
        // always consumed in the exact order render() produced them (0,1,0,1,...),
        // never "whichever's newest" - that in-order handoff plus render()'s own
        // bufferFree.Wait() backpressure is what couples the two threads' rates
        // together, unlike a fully decoupled latest-wins scheme.
        public bool TryPresentFrame(ref int readIndex, out byte[] data)
        {
            if (!bufferReady[readIndex].Wait(0))
            {
                data = null;
                return false;
            }

            data = frameBuffers[readIndex];
            return true;
        }

        // Called once the caller (MainWindow.draw) has finished copying the buffer
        // TryPresentFrame just handed back - frees the slot for the render thread to
        // reuse two frames from now, and advances readIndex to the next slot.
        public void FinishPresent(ref int readIndex)
        {
            bufferFree[readIndex].Release();
            readIndex = 1 - readIndex;
        }
    }
}
