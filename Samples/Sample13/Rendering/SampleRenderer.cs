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
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Sample13
{
    /// <summary>
    /// The renderer coordinator: owns the launch params, the camera, the scene
    /// switcher, and the render loop, wiring together the components that do the
    /// actual work - <see cref="GpuContext"/>, <see cref="RendererPipeline"/>,
    /// <see cref="SceneGpuBuffers"/>, <see cref="SbtBuilder"/>,
    /// <see cref="AccelStructureBuilder"/>, <see cref="TextureCache"/>,
    /// <see cref="FrameOutput"/>, <see cref="SceneAnimator"/>, and
    /// <see cref="PresentQueue"/>.
    /// </summary>
    public class SampleRenderer
    {
        int width;
        int height;
        readonly MainWindow window;

        readonly GpuContext gpu;
        readonly RendererPipeline pipeline;
        readonly SceneGpuBuffers buffers;
        readonly TextureCache textures;
        readonly SbtBuilder sbtBuilder;
        readonly AccelStructureBuilder accel;
        readonly SceneAnimator animator;
        readonly FrameOutput frameOutput;
        PresentQueue presentQueue;

        OptixShaderBindingTable sbt;
        IntPtr traversable;
        LaunchParams launchParams;

        public Camera camera { get; private set; }

        public bool DenoiserOn { get; set; } = true;
        public int NumPixelSamples { get; set; } = 1;
        public bool Accumulate { get; set; } = true;

        // Defaults to the original single-merged-build-input triangles GAS (fast) -
        // splitting into one build input per mesh (see SbtLayout.GetTriangleMeshRanges)
        // multiplies SBT hitgroup record count by (build-input count * Materials.Length),
        // which gets very large for scenes like the radial museum that also scatter
        // dozens of one-off AddTriangle calls (each its own per-triangle material - see
        // RadialMuseumScene's AddTriangleField) between AddMeshAutoGround calls, turning
        // into many small "gap" build inputs. Toggled live with the M key
        // (MainWindow_KeyDown), which forces a scene rebuild since it changes the
        // GAS/SBT shape.
        public bool UseMergedTrianglesGas { get; set; } = true;

        // Scene-switcher state - mirrors the reference's RaytraceEntity.BuildSceneTable
        // (lazily built, cached per index) - see docs/SAMPLE13_PLAN.md. currentScene is
        // the renderer's own reference to the active (cached) SceneData.
        readonly Func<SceneData>[] sceneBuilders;
        readonly Dictionary<int, SceneData> sceneCache = new Dictionary<int, SceneData>();
        int currentSceneIndex;
        SceneData currentScene;

        public string CurrentSceneName { get; private set; } = "";

        // Scene-static HUD info - recomputed once per SwitchToScene (not per frame,
        // unlike LastStats below), since none of it changes between frames of the same
        // scene. Mirrors the geometry counts already known to the SBT/accel builders,
        // just cached on public properties so MainWindow's overlay can read them
        // without re-deriving anything from SceneData itself.
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

        // Guards every access to launchParams, traversable, and the scene GPU buffers -
        // taken by render() around its whole body, and by SwitchToScene()/setCamera()
        // (both only ever called from the UI thread, on I/U/D/Space/A key presses or
        // mouse-drag camera moves) around theirs. Without this, SwitchToScene's buffer
        // disposal/reallocation and GAS/IAS rebuild could run concurrently with an
        // in-flight OptixLaunch reading the very pointers/traversable handle being torn
        // down - undefined behavior at the CUDA/OptiX driver level (observed as NaN ray
        // origins and a sticky "unspecified launch failure" CUDA error). This race
        // always existed but was rarely hit while the render loop was throttled by the
        // old fixed Thread.Sleep(10) + blocking per-frame Dispatcher.Invoke; removing
        // both (see PresentQueue's double-buffered handoff) made the render loop run
        // fast enough to hit it almost every scene switch. This lock is unrelated to -
        // and does not reintroduce - the old per-frame UI-thread block: the presenter
        // thread only ever touches the PresentQueue's buffers/semaphores, never
        // launchParams or GPU buffers, so it never waits on gpuLock.
        //
        // Critically, render()'s lock scope does NOT extend over its
        // PresentQueue.TryBeginWrite publish step (see render()'s comment right after
        // tonemapMs) - an earlier version did, and it deadlocked the whole app on the
        // first mouse drag: setCamera() takes this same lock synchronously on the UI
        // thread, so once render() was blocked in the buffer wait *while holding
        // gpuLock*, the only thing that could unblock it - MainWindow's
        // OnCompositionRendering calling FinishPresent - could never run, because the
        // UI thread was itself stuck wanting gpuLock inside setCamera(). Keeping the
        // buffer-publish step outside this lock (safe, since it never touches
        // launchParams/traversable/scene buffers) breaks that cycle.
        readonly object gpuLock = new object();

        public unsafe SampleRenderer(int width, int height, MainWindow window)
        {
            this.window = window;

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

        // Re-runs SwitchToScene against the same scene index - used after toggling
        // UseMergedTrianglesGas, which changes the triangles GAS/SBT shape and so
        // needs a full rebuild, not just a launchParams/camera tweak.
        public void RebuildCurrentScene() => SwitchToScene(currentSceneIndex);

        // HUD summary of the actual GAS/IAS shape built for this scene - mirrors the
        // fixed triangles-GAS -> custom-primitives-GAS -> IAS pipeline (see
        // AccelStructureBuilder.Build), reporting which of the two GASes are actually
        // present this scene (a scene can have either, both, or - in principle -
        // neither) and their per-kind primitive counts.
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

                // Tear down the previous scene's GPU state, then rebuild - same order
                // as the old monolithic DisposeSceneBuffers.
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
            }

            presentQueue.Dispose();
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

            // Sample13's window is ResizeMode="NoResize" so this only ever runs once,
            // at startup.
            presentQueue?.Dispose();
            presentQueue = new PresentQueue(frameOutput.DisplayBytes);

            launchParams.NumPixelSamples = NumPixelSamples;
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

                bool sceneIsAnimated = currentScene != null && animator.IsAnimated(currentScene);
                if (sceneIsAnimated)
                    animator.Update(currentScene);

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
                gpu.Accelerator.OptixLaunch(
                    gpu.Accelerator.DefaultStream,
                    pipeline.Pipeline,
                    launchParams,
                    sbt,
                    (uint)width,
                    (uint)height,
                    1);
                gpu.Accelerator.Synchronize();
                traceMs = stepStopwatch.Elapsed.TotalMilliseconds;

                launchParams.FrameID++;

                frameOutput.DenoiseAndTonemap(DenoiserOn, Accumulate, launchParams.FrameID, out denoiseMs, out tonemapMs);

                samplesAccumulated = launchParams.FrameID;
                // gpuLock ends here, deliberately - everything below only touches the
                // PresentQueue's buffers/semaphores and the display buffer (a fixed GPU
                // buffer allocated once in resize(), never touched by SwitchToScene),
                // never launchParams/traversable/scene buffers, so it doesn't need
                // gpuLock's protection. This split is not optional: the buffer wait in
                // TryBeginWrite below can block waiting for the UI thread's
                // OnCompositionRendering to run FinishPresent, and OnCompositionRendering
                // can only run when the UI thread is free to pump messages. Holding
                // gpuLock across that wait deadlocked the whole app the instant a mouse
                // drag called setCamera() (also a `lock (gpuLock)`, synchronously on the
                // UI thread) while render() happened to be waiting here - the UI thread
                // would then be stuck wanting gpuLock, unable to reach the very
                // OnCompositionRendering call render() was waiting on to release it.
            }

            stepStopwatch.Restart();
            bool acquired = presentQueue.TryBeginWrite(() => window.run, out byte[] target);
            if (acquired)
                frameOutput.ReadbackDisplay(target);
            double readbackMs = stepStopwatch.Elapsed.TotalMilliseconds;

            stepStopwatch.Restart();
            if (acquired)
                presentQueue.EndWrite();
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

        // Reader side of the double-buffered async present handoff - see PresentQueue.
        public bool TryPresentFrame(ref int readIndex, out byte[] data) =>
            presentQueue.TryPresentFrame(ref readIndex, out data);

        public void FinishPresent(ref int readIndex) =>
            presentQueue.FinishPresent(ref readIndex);
    }
}
