using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.Device;
using ILGPU.OptiX.Interop;
using ILGPU.OptiX.Pipeline;
using ILGPU.OptiX.DeviceApi;
using ILGPU.OptiX.AccelStructures;
using MeshRange = ILGPU.OptiX.Pipeline.OptixMeshRange;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Sample13
{
    /// <summary>
    /// The radiance/shadow ray types' shared 19-register payload struct (see
    /// Payloads.cs's own doc comment: color(3) + flag(1) + new-ray-origin(3) +
    /// new-ray-direction(3) + throughput-tint(3) + normal(3) + albedo(3)). Declaring
    /// the same struct on both ray types drives the pipeline-wide payload register
    /// count; the shadow ray type only ever uses the first 4 registers (transmittance
    /// xyz + hit count, see MissAndShadowPrograms.cs's raw OptixPayload.Payload0-3
    /// access), but the payload count is a shared pipeline-wide register count. Never
    /// Read/Write against this struct directly - every device program in this sample
    /// uses OptixPayloadInterop/OptixPayload's raw per-register API (Payloads.cs);
    /// this type exists purely so RayTypeBuilder&lt;T&gt;.Payload&lt;T&gt;() can size
    /// the pipeline's NumPayloadValues correctly.
    /// </summary>
    public struct RadianceShadowPayload
    {
        public uint Slot0, Slot1, Slot2, Slot3, Slot4, Slot5, Slot6, Slot7, Slot8,
            Slot9, Slot10, Slot11, Slot12, Slot13, Slot14, Slot15, Slot16, Slot17, Slot18;
    }

    /// <summary>
    /// The renderer coordinator: owns the launch params, the camera, the scene
    /// switcher, and the render loop, wiring together the components that do the
    /// actual work - <see cref="GpuContext"/>, <see cref="RayTracingPipeline{TLaunchParams}"/>,
    /// <see cref="SceneGpuBuffers"/>, <see cref="SbtBuilder"/>,
    /// <see cref="AccelStructureBuilder"/>, <see cref="TextureCache"/>,
    /// <see cref="FrameOutput"/>, <see cref="SceneAnimator"/>.
    ///
    /// render() runs entirely on the single GL/compute thread, with no separate
    /// present thread or cross-thread handoff - DenoiseAndTonemap's
    /// Map/tonemap/Unmap sequence (see FrameOutput.cs) is the last GPU-touching step.
    /// </summary>
    public class SampleRenderer
    {
        int width;
        int height;

        readonly GpuContext gpu;
        readonly RayTracingPipeline<LaunchParams> pipeline;
        readonly SceneGpuBuffers buffers;
        readonly TextureCache textures;
        readonly SbtBuilder sbtBuilder;
        readonly AccelStructureBuilder accel;
        readonly SceneAnimator animator;
        readonly FrameOutput frameOutput;

        IntPtr traversable;
        LaunchParams launchParams;

        public Camera camera { get; private set; }

        public bool DenoiserOn { get; set; } = true;
        public int NumPixelSamples { get; set; } = 1;
        public bool Accumulate { get; set; } = true;
        public int MaxMirrorBounces { get; set; } = 2;
        public int MaxRefractionBounces { get; set; } = 3;
        public int MaxDiffuseBounces { get; set; } = 1;

        public bool UseMergedTrianglesGas { get; set; } = true;

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
        // Kept even though this sample is single-threaded - cheap insurance when
        // uncontended, and removing it isn't a meaningful simplification on its own.
        readonly object gpuLock = new object();

        public SampleRenderer(int width, int height)
        {
            gpu = new GpuContext();

            // No module/pipeline compile options, no SetStackSize magic numbers, no
            // hand-rolled SBT record structs. Ray
            // type declaration order (radiance, then shadow) fixes their SBT index at
            // 0/1, matching Payloads.RADIANCE_RAY_TYPE/SHADOW_RAY_TYPE. Triangle
            // geometry and every custom-primitive kind (plus the volume grid) share
            // each ray type's closest-hit/any-hit pair via named hit groups (one
            // .HitGroup(kind, ...) call per intersection program) - see SbtBuilder.cs's
            // HitGroupKinds and Apply(). WithTraversableGraphFlags(...
            // ALLOW_SINGLE_LEVEL_INSTANCING) is required here (unlike every other
            // sample's single-GAS default) since AccelStructureBuilder always combines
            // this sample's GAS(es) via an IAS.
            pipeline = gpu.RayTracer.CreatePipeline<LaunchParams>(b => b
                .Raygen(RaygenProgram.__raygen__renderFrame)
                .RayType("radiance", r => r
                    .Payload<RadianceShadowPayload>()
                    .Miss(MissAndShadowPrograms.__miss__radiance)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.Triangle, ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.CustomPrimitiveKinds[0], ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance, IntersectionPrograms.__intersection__sphere)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.CustomPrimitiveKinds[1], ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance, IntersectionPrograms.__intersection__box)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.CustomPrimitiveKinds[2], ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance, IntersectionPrograms.__intersection__cylinderY)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.CustomPrimitiveKinds[3], ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance, IntersectionPrograms.__intersection__disk)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.CustomPrimitiveKinds[4], ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance, IntersectionPrograms.__intersection__xyRect)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.CustomPrimitiveKinds[5], ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance, IntersectionPrograms.__intersection__xzRect)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.CustomPrimitiveKinds[6], ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance, IntersectionPrograms.__intersection__yzRect)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.VolumeGrid, ClosestHitProgram.__closest__radiance, MissAndShadowPrograms.__anyhit__radiance, IntersectionPrograms.__intersection__volumeGrid))
                .RayType("shadow", r => r
                    .Payload<RadianceShadowPayload>()
                    .Miss(MissAndShadowPrograms.__miss__shadow)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.Triangle, MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.CustomPrimitiveKinds[0], MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow, IntersectionPrograms.__intersection__sphere)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.CustomPrimitiveKinds[1], MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow, IntersectionPrograms.__intersection__box)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.CustomPrimitiveKinds[2], MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow, IntersectionPrograms.__intersection__cylinderY)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.CustomPrimitiveKinds[3], MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow, IntersectionPrograms.__intersection__disk)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.CustomPrimitiveKinds[4], MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow, IntersectionPrograms.__intersection__xyRect)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.CustomPrimitiveKinds[5], MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow, IntersectionPrograms.__intersection__xzRect)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.CustomPrimitiveKinds[6], MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow, IntersectionPrograms.__intersection__yzRect)
                    .HitGroup<MaterialSbtData>(HitGroupKinds.VolumeGrid, MissAndShadowPrograms.__closesthit__shadow, MissAndShadowPrograms.__anyhit__shadow, IntersectionPrograms.__intersection__volumeGrid))
                .MaxTraceDepth(2)
                .WithTraversableGraphFlags(OptixTraversableGraphFlags.AllowSingleLevelInstancing));

            buffers = new SceneGpuBuffers(gpu.Accelerator);
            textures = new TextureCache();
            sbtBuilder = new SbtBuilder(textures);
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

        public void SwitchToScene(int index)
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

                textures.Clear();
                accel.DisposeBuffers();
                buffers.DisposeAll();

                buffers.Upload(scene);

                MeshRange[] triangleMeshRanges = SbtLayout.GetTriangleMeshRanges(scene, UseMergedTrianglesGas);
                // SetHitRecords (inside sbtBuilder.Apply) synchronizes the accelerator
                // and disposes the previous scene's hitgroup buffer itself - no
                // separate DisposeBuffers() call needed before this.
                sbtBuilder.Apply(pipeline, scene, triangleMeshRanges);
                traversable = accel.Build(scene, triangleMeshRanges);

                // Cross-checks that the accel structure's NumSbtRecords values (per
                // AddTriangleMesh()/AddCustomPrimitives() call) still agree with the
                // SBT's own actual hitgroup record count.
                Debug.Assert(
                    pipeline.HitgroupRecordCount == accel.TotalHitgroupRecordsUsed,
                    $"SBT hitgroup record count ({pipeline.HitgroupRecordCount}) doesn't match " +
                    $"what the accel structure's NumSbtRecords values imply " +
                    $"({accel.TotalHitgroupRecordsUsed}) - AccelStructureBuilder and SbtBuilder " +
                    "have drifted out of sync.");

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
                launchParams.Vertices = OptixDeviceView<Vec3>.From(buffers.Vertices);
                launchParams.Normals = OptixDeviceView<Vec3>.From(buffers.Normals);
                launchParams.TexCoords = OptixDeviceView<Vec2>.From(buffers.TexCoords);
                launchParams.Indices = OptixDeviceView<Vec3i>.From(buffers.Indices);
                launchParams.Spheres = OptixDeviceView<SphereData>.From(buffers.Spheres);
                launchParams.Boxes = OptixDeviceView<BoxData>.From(buffers.Boxes);
                launchParams.CylindersY = OptixDeviceView<CylinderYData>.From(buffers.CylindersY);
                launchParams.Disks = OptixDeviceView<DiskData>.From(buffers.Disks);
                launchParams.XYRects = OptixDeviceView<RectData>.From(buffers.XYRects);
                launchParams.XZRects = OptixDeviceView<RectData>.From(buffers.XZRects);
                launchParams.YZRects = OptixDeviceView<RectData>.From(buffers.YZRects);
                launchParams.VoxelMaterialIds = OptixDeviceView<uint>.From(buffers.VoxelMaterialIds);
                launchParams.VolumeGridMin = scene.VolumeGridMin;
                launchParams.VolumeVoxelSize = scene.VolumeVoxelSize;
                launchParams.VolumeDims = scene.VolumeDims;
                launchParams.Materials = OptixDeviceView<MaterialSbtData>.From(buffers.Materials);
                launchParams.PointLights = OptixDeviceView<PointLightGpu>.From(buffers.Lights);
                launchParams.NumPointLights = scene.Lights.Length;
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
                buffers.DisposeAll();
            }

            frameOutput.Dispose();
            pipeline.Dispose();
            gpu.Dispose();
        }

        public void resize(int width, int height)
        {
            if (width == 0 || height == 0)
                return;

            lock (gpuLock)
            {
                this.width = width;
                this.height = height;

                frameOutput.Resize(width, height);

                launchParams.NumPixelSamples = NumPixelSamples;
                launchParams.MaxMirrorBounces = MaxMirrorBounces;
                launchParams.MaxRefractionBounces = MaxRefractionBounces;
                launchParams.MaxDiffuseBounces = MaxDiffuseBounces;
                launchParams.ColorBuffer = OptixDeviceView<Vec4>.From(frameOutput.HdrColorBuffer);
                launchParams.AlbedoBuffer = OptixDeviceView<Vec4>.From(frameOutput.AlbedoBuffer);
                launchParams.NormalBuffer = OptixDeviceView<Vec4>.From(frameOutput.NormalBuffer);

                // Refresh the camera's own width/height (and therefore aspectRatio) to
                // the new resolution - without this, a resize alone (no subsequent
                // camera move) left the image stretched/squashed until the next WASD/
                // mouse-look frame happened to call setCamera. Guarded on
                // currentScene != null since the constructor's own initial resize()
                // call runs before SwitchToScene(0) has set up a real camera to refresh.
                if (currentScene != null)
                {
                    camera = new Camera(camera, width, height);
                    launchParams.camera = camera;
                }
            }
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
        public void render()
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
                launchParams.MaxMirrorBounces = MaxMirrorBounces;
                launchParams.MaxRefractionBounces = MaxRefractionBounces;
                launchParams.MaxDiffuseBounces = MaxDiffuseBounces;

                stepStopwatch.Restart();
                // Persistent launch-params buffer, reused every frame - no per-call
                // allocate/free (the leak OptixLaunchExtensions.OptixLaunch had).
                pipeline.Launch(launchParams, width, height);
                gpu.Accelerator.Synchronize();
                double traceMs = stepStopwatch.Elapsed.TotalMilliseconds;

                launchParams.FrameID++;

                frameOutput.DenoiseAndTonemap(DenoiserOn, Accumulate, launchParams.FrameID, out double denoiseMs, out double tonemapMs);

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
