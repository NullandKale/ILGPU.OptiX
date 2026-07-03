using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.Interop;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace Sample15
{
    /// <summary>
    /// The renderer coordinator: owns the launch params, the camera, the scene
    /// switcher, and the render loop, wiring together the components that do the
    /// actual work - <see cref="GpuContext"/>, <see cref="RendererPipeline"/>,
    /// <see cref="SceneGpuBuffers"/>, <see cref="SbtBuilder"/>,
    /// <see cref="AccelStructureBuilder"/>, <see cref="TextureCache"/>,
    /// <see cref="FrameOutput"/>, <see cref="SceneAnimator"/>.
    ///
    /// Unlike Sample13's SampleRenderer, there is no MainWindow coupling and no
    /// PresentQueue - render() runs entirely on the single GL/compute thread (see
    /// docs/SAMPLE14_PLAN.md's threading-model section) and DenoiseAndTonemap's
    /// Map/tonemap/Unmap sequence (see FrameOutput.cs) is the last GPU-touching step,
    /// with no further cross-thread handoff needed.
    /// </summary>
    public class SampleRenderer
    {
        int width;
        int height;

        readonly GpuContext gpu;
        readonly RendererPipeline pipeline;
        readonly SceneGpuBuffers buffers;
        readonly TextureCache textures;
        readonly SbtBuilder sbtBuilder;
        readonly AccelStructureBuilder accel;
        readonly SceneAnimator animator;
        readonly FrameOutput frameOutput;

        // HDRI environment maps (docs/SAMPLE15_PLAN.md Milestone M5) - scene-dependent
        // (SceneData.EnvMapPath), cached per unique path so scenes sharing the same
        // HDRI don't reload/reupload it on every switch (unlike SceneGpuBuffers, whose
        // per-scene buffers are torn down and rebuilt on every SwitchToScene call).
        sealed class EnvMapGpuData : IDisposable
        {
            public MemoryBuffer1D<Vec3, Stride1D.Dense> Pixels;
            public MemoryBuffer1D<float, Stride1D.Dense> Cdf;
            public MemoryBuffer1D<float, Stride1D.Dense> PdfUv;
            public int Width;
            public int Height;
            public float TotalWeight;

            public void Dispose()
            {
                Pixels?.Dispose();
                Cdf?.Dispose();
                PdfUv?.Dispose();
            }
        }

        readonly Dictionary<string, EnvMapGpuData> envMapCache = new Dictionary<string, EnvMapGpuData>();

        EnvMapGpuData GetOrLoadEnvMap(string path)
        {
            if (envMapCache.TryGetValue(path, out EnvMapGpuData cached))
                return cached;

            EnvironmentMap.EnvMapData envMap = EnvironmentMap.Build(Path.Combine(AppContext.BaseDirectory, path));
            var data = new EnvMapGpuData
            {
                Pixels = gpu.Accelerator.Allocate1D(envMap.Pixels),
                Cdf = gpu.Accelerator.Allocate1D(envMap.Cdf),
                PdfUv = gpu.Accelerator.Allocate1D(envMap.PdfUv),
                Width = envMap.Width,
                Height = envMap.Height,
                TotalWeight = envMap.TotalWeight,
            };
            envMapCache[path] = data;
            return data;
        }

        OptixShaderBindingTable sbt;
        IntPtr traversable;
        LaunchParams launchParams;

        public Camera camera { get; private set; }

        public bool DenoiserOn { get; set; } = true;
        public int NumPixelSamples { get; set; } = 1;
        public bool Accumulate { get; set; } = true;
        // Unified bounce budget (docs/SAMPLE15_PLAN.md Milestone M3) - Russian
        // roulette (RaygenProgram.cs) terminates most paths well before this ceiling.
        public int MaxBounces { get; set; } = 8;

        public bool UseMergedTrianglesGas { get; set; } = true;

        // Tonemap controls (docs/SAMPLE15_PLAN.md Milestone M8). ExposureStops
        // defaults to -1 (2^-1 = 0.5x) to reproduce the previous hardcoded-Exposure
        // constant's default look exactly - the UI/keyboard can move it from there.
        public float ExposureStops { get; set; } = -1f;
        public int TonemapOperator { get; set; } = TonemapKernel.OperatorReinhard;

        // NEE/MIS on/off toggle (docs/SAMPLE15_PLAN.md Milestone M8) - for A/B
        // variance comparison per the plan's own M4/M5 verification bar. Disabling
        // this makes ShadeSurface skip NextEventEstimation.SampleDirectLighting
        // entirely and always show a BSDF-sampled ray's own emission at full weight
        // (no MIS competitor), i.e. reverts to pre-M4 indirect-only lighting.
        public bool NeeEnabled { get; set; } = true;

        // Env-map controls (docs/SAMPLE15_PLAN.md Milestone M8) - Rotation is in
        // radians, applied uniformly by EnvironmentMapSampling's direction<->uv
        // conversion (both the NEE-sampling and miss-program-evaluation sides, so
        // they stay consistent for MIS). IntensityMultiplier is applied on top of
        // (not instead of) the active scene's own SceneData.EnvMapIntensity baseline,
        // so a scene-authored default survives a live UI tweak reset back to 1.
        public float EnvMapRotation { get; set; } = 0f;
        public float EnvMapIntensityMultiplier { get; set; } = 1f;

        readonly Func<SceneData>[] sceneBuilders;
        readonly Dictionary<int, SceneData> sceneCache = new Dictionary<int, SceneData>();
        int currentSceneIndex;
        SceneData currentScene;

        public string CurrentSceneName { get; private set; } = "";

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

        public FrameStats LastStats { get; private set; }

        public int GlTextureHandle => frameOutput.GlTextureHandle;

        readonly Stopwatch stepStopwatch = new Stopwatch();
        long lastFrameStartTicks;
        bool hasRenderedFrame;
        double smoothedFrameMs = 16.0;

        // Guards every access to launchParams, traversable, and the scene GPU buffers.
        // Kept even though Sample14 is single-threaded (unlike Sample13, where this
        // lock exists specifically to protect against a dedicated render thread racing
        // scene-switch/camera-move calls from the UI thread) - cheap insurance when
        // uncontended, and removing it isn't a meaningful simplification on its own
        // (see docs/SAMPLE14_PLAN.md's note on this).
        readonly object gpuLock = new object();

        public unsafe SampleRenderer(int width, int height)
        {
            gpu = new GpuContext();
            pipeline = new RendererPipeline(gpu);
            buffers = new SceneGpuBuffers(gpu.Accelerator);
            textures = new TextureCache();
            sbtBuilder = new SbtBuilder(gpu.Accelerator, pipeline, textures);
            accel = new AccelStructureBuilder(gpu.Accelerator, gpu.DeviceContext, buffers);
            animator = new SceneAnimator(buffers, accel, textures);
            frameOutput = new FrameOutput(gpu);

            sceneBuilders = SceneTable.Build();

            resize(width, height);

            SwitchToScene(0);
        }

        public void NextScene() => SwitchToScene((currentSceneIndex + 1) % sceneBuilders.Length);
        public void PreviousScene() => SwitchToScene((currentSceneIndex - 1 + sceneBuilders.Length) % sceneBuilders.Length);

        public void RebuildCurrentScene() => SwitchToScene(currentSceneIndex);

        string BuildAccelStructureSummary(SceneData scene, bool hasTriangles, bool hasCustomPrimitives)
        {
            var gasParts = new List<string>();
            if (hasTriangles)
            {
                int triangleMeshCount = SbtLayout.GetTriangleMeshRanges(scene, UseMergedTrianglesGas).Length;
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
                animator.OnSceneSwitched(scene);

                // Scene-dependent HDRI environment map (docs/SAMPLE15_PLAN.md
                // Milestone M5) - null/empty EnvMapPath means this scene keeps the
                // flat BackgroundTop/Bottom gradient sky instead. Must happen before
                // buffers.Upload(scene) below, which reads scene.NeeLights/
                // NeeLightCdf/NeeLightAreaPdf.
                EnvMapGpuData envMapData = string.IsNullOrEmpty(scene.EnvMapPath) ? null : GetOrLoadEnvMap(scene.EnvMapPath);
                float envMapPower = envMapData?.TotalWeight ?? 0f;
                (scene.NeeLights, scene.NeeLightCdf, scene.NeeLightAreaPdf, float envMapLightPdf) = LightList.Build(scene, envMapPower);

                textures.Clear();
                accel.DisposeBuffers();
                sbtBuilder.DisposeBuffers();
                buffers.DisposeAll();

                buffers.Upload(scene);

                MeshRange[] triangleMeshRanges = SbtLayout.GetTriangleMeshRanges(scene, UseMergedTrianglesGas);
                sbt = sbtBuilder.Build(scene, triangleMeshRanges);
                traversable = accel.Build(scene, triangleMeshRanges);

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
                launchParams.Vertices = (Vec3*)SceneGpuBuffers.NativePtrOrZero(buffers.Vertices);
                launchParams.Normals = (Vec3*)SceneGpuBuffers.NativePtrOrZero(buffers.Normals);
                launchParams.TexCoords = (Vec2*)SceneGpuBuffers.NativePtrOrZero(buffers.TexCoords);
                launchParams.Tangents = (Vec3*)SceneGpuBuffers.NativePtrOrZero(buffers.Tangents);
                launchParams.Indices = (Vec3i*)SceneGpuBuffers.NativePtrOrZero(buffers.Indices);
                launchParams.Spheres = (SphereData*)SceneGpuBuffers.NativePtrOrZero(buffers.Spheres);
                launchParams.Boxes = (BoxData*)SceneGpuBuffers.NativePtrOrZero(buffers.Boxes);
                launchParams.CylindersY = (CylinderYData*)SceneGpuBuffers.NativePtrOrZero(buffers.CylindersY);
                launchParams.Disks = (DiskData*)SceneGpuBuffers.NativePtrOrZero(buffers.Disks);
                launchParams.XYRects = (RectData*)SceneGpuBuffers.NativePtrOrZero(buffers.XYRects);
                launchParams.XZRects = (RectData*)SceneGpuBuffers.NativePtrOrZero(buffers.XZRects);
                launchParams.YZRects = (RectData*)SceneGpuBuffers.NativePtrOrZero(buffers.YZRects);
                launchParams.VoxelMaterialIds = (uint*)SceneGpuBuffers.NativePtrOrZero(buffers.VoxelMaterialIds);
                launchParams.VolumeGridMin = scene.VolumeGridMin;
                launchParams.VolumeVoxelSize = scene.VolumeVoxelSize;
                launchParams.VolumeDims = scene.VolumeDims;
                launchParams.Materials = (MaterialSbtData*)SceneGpuBuffers.NativePtrOrZero(buffers.Materials);
                launchParams.PointLights = (PointLightGpu*)SceneGpuBuffers.NativePtrOrZero(buffers.Lights);
                launchParams.NumPointLights = scene.Lights.Length;
                launchParams.NeeLights = (LightGpu*)SceneGpuBuffers.NativePtrOrZero(buffers.NeeLights);
                launchParams.NeeLightCdf = (float*)SceneGpuBuffers.NativePtrOrZero(buffers.NeeLightCdf);
                launchParams.NumNeeLights = scene.NeeLights.Length;
                launchParams.NeeLightAreaPdf = (float*)SceneGpuBuffers.NativePtrOrZero(buffers.NeeLightAreaPdf);
                launchParams.EnvMapPixels = envMapData != null ? (Vec3*)envMapData.Pixels.NativePtr : null;
                launchParams.EnvMapCdf = envMapData != null ? (float*)envMapData.Cdf.NativePtr : null;
                launchParams.EnvMapPdfUv = envMapData != null ? (float*)envMapData.PdfUv.NativePtr : null;
                launchParams.EnvMapWidth = envMapData?.Width ?? 0;
                launchParams.EnvMapHeight = envMapData?.Height ?? 0;
                launchParams.EnvMapIntensity = scene.EnvMapIntensity;
                launchParams.EnvMapLightPdf = envMapLightPdf;
                launchParams.AmbientColor = scene.AmbientColor;
                launchParams.AmbientIntensity = scene.AmbientIntensity;
                launchParams.BackgroundTop = scene.BackgroundTop;
                launchParams.BackgroundBottom = scene.BackgroundBottom;
                launchParams.FrameID = 0;

                frameOutput.ResetAccumulation();

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

        public void Dispose()
        {
            lock (gpuLock)
            {
                textures.Clear();
                accel.DisposeBuffers();
                sbtBuilder.DisposeBuffers();
                buffers.DisposeAll();

                foreach (EnvMapGpuData envMapData in envMapCache.Values)
                    envMapData.Dispose();
                envMapCache.Clear();
            }

            frameOutput.Dispose();
            pipeline.Dispose();
            gpu.Dispose();
        }

        public unsafe void resize(int width, int height)
        {
            if (width == 0 || height == 0)
                return;

            this.width = width;
            this.height = height;

            frameOutput.Resize(width, height);

            launchParams.NumPixelSamples = NumPixelSamples;
            launchParams.MaxBounces = MaxBounces;
            launchParams.ColorBuffer = (Vec4*)frameOutput.HdrColorBuffer.NativePtr;
            launchParams.AlbedoBuffer = (Vec4*)frameOutput.AlbedoBuffer.NativePtr;
            launchParams.NormalBuffer = (Vec4*)frameOutput.NormalBuffer.NativePtr;
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

        // Traces, denoises, tonemaps, and writes the finished frame straight into the
        // GL-owned interop texture (see FrameOutput.DenoiseAndTonemap/BlitToTexture) -
        // no CPU byte[] handoff, no second thread. Call BlitToTexture() (or draw the
        // fullscreen quad against GlTextureHandle) from the window after this returns.
        public unsafe void render()
        {
            lock (gpuLock)
            {
                long frameStartTicks = Stopwatch.GetTimestamp();
                if (hasRenderedFrame)
                {
                    double sinceLastFrameMs = (frameStartTicks - lastFrameStartTicks) * 1000.0 / Stopwatch.Frequency;
                    smoothedFrameMs = (smoothedFrameMs * 0.9) + (sinceLastFrameMs * 0.1);
                }
                lastFrameStartTicks = frameStartTicks;
                hasRenderedFrame = true;

                bool sceneIsAnimated = currentScene != null && animator.IsAnimated(currentScene);
                if (sceneIsAnimated)
                    animator.Update(currentScene);

                if (!Accumulate || sceneIsAnimated)
                    launchParams.FrameID = 0;
                launchParams.NumPixelSamples = NumPixelSamples;
                launchParams.MaxBounces = MaxBounces;
                launchParams.NeeEnabled = NeeEnabled ? 1 : 0;
                launchParams.EnvMapRotation = EnvMapRotation;
                launchParams.EnvMapIntensity = currentScene.EnvMapIntensity * EnvMapIntensityMultiplier;

                stepStopwatch.Restart();
                gpu.Accelerator.OptixLaunch(
                    gpu.Accelerator.DefaultStream,
                    pipeline.Pipeline,
                    launchParams,
                    sbt,
                    (uint)width,
                    (uint)height,
                    1);
                gpu.Accelerator.Synchronize();
                double traceMs = stepStopwatch.Elapsed.TotalMilliseconds;

                launchParams.FrameID++;

                frameOutput.DenoiseAndTonemap(DenoiserOn, Accumulate, launchParams.FrameID, ExposureStops, TonemapOperator, out double denoiseMs, out double tonemapMs);

                int samplesAccumulated = launchParams.FrameID;

                LastStats = new FrameStats
                {
                    TraceMs = traceMs,
                    DenoiseMs = denoiseMs,
                    TonemapMs = tonemapMs,
                    TotalFrameMs = smoothedFrameMs,
                    Fps = 1000.0 / smoothedFrameMs,
                    SamplesAccumulated = samplesAccumulated,
                };
            }
        }

        // GPU-internal PBO -> texture blit (no CPU copy) - call once per frame after
        // render(), before drawing the fullscreen quad.
        public void BlitToTexture() => frameOutput.BlitToTexture();
    }
}
