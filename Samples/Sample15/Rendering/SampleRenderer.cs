using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.Interop;
using ILGPU.Runtime;
using MeshRange = ILGPU.OptiX.Pipeline.OptixMeshRange;
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
        // Per-pixel reprojected accumulation (ColorBuffer/AccumCountBuffer,
        // RaygenProgram.cs) - converges the image toward a noise-free ground truth over
        // many frames, whether the camera is static OR moving (reprojected via the same
        // Flow data used by the temporal denoiser below), unlike the old global
        // FrameID-keyed running mean this replaced (which could only blend by pixel
        // index and so had to hard-reset on any camera move). Toggling this off (or an
        // animated scene, which changes *lighting* our purely-geometric reprojection
        // can't compensate for) sets MaxHistoryFrames down to 1 each frame (see
        // render()) - always a fresh single sample, no blending. Independent of
        // TemporalDenoiseEnabled below - that's a separate, secondary smoothing pass.
        public bool Accumulate { get; set; } = true;
        // The cap on AccumCountBuffer's per-pixel history length when Accumulate is on
        // (see LaunchParams.MaxHistoryFrames's own doc comment) - once a pixel's history
        // reaches this, the blend settles into a small new-sample weight
        // (1/MaxHistoryFrames, further aged by HistoryDecayHalfLifeSeconds below) rather
        // than continuing to shrink like an unbounded running mean would. Not "how many
        // frames until it looks perfect" - just where the bounded-EMA steady-state
        // noise floor gets set.
        // Kept well below the old default of 256 - a long-lived history survives many
        // small per-frame reprojection resamples during gradual camera motion (each
        // individually correct, but a very long chain of them can still soften fine
        // detail over time, "reprojecting too far"), and a much shorter cap bounds how
        // many times a pixel's color can be recursively resampled before it's forced
        // to refresh from a fully fresh sample.
        public int MaxHistoryFrames { get; set; } = 96;
        // Real-time half-life for AccumCount's decay (see LaunchParams.
        // HistoryDecayHalfLifeSeconds's own doc comment) - even a static, perfectly
        // reprojected pixel keeps slowly forgetting old contributions on this
        // wall-clock schedule instead of freezing once it hits MaxHistoryFrames.
        // Smaller = adapts to subtle scene changes faster but noisier at steady state;
        // 0 or negative disables decay entirely (the old "freeze at the cap" behavior).
        public float HistoryDecayHalfLifeSeconds { get; set; } = 3f;
        // Disocclusion rejection threshold (see LaunchParams.DepthRejectionThreshold's
        // own doc comment) - a relative depth-difference fraction; smaller rejects more
        // aggressively (less ghosting/smearing at moving edges, but discards history -
        // and thus noise reduction - more readily), larger tolerates more before
        // rejecting (smoother but more prone to smearing at disocclusions).
        public float DepthRejectionThreshold { get; set; } = 0.05f;
        // Motion-vector-guided temporal denoising (OPTIX_DENOISER_MODEL_KIND_TEMPORAL,
        // FrameOutput.cs) - lets the denoiser reuse the *previous* frame's own
        // denoised output, reprojected via RaygenProgram's per-pixel Flow buffer, as a
        // secondary smoothing pass on top of Accumulate's own reprojected history
        // (mainly useful on freshly-disoccluded/reset pixels, which are still raw noise
        // before Accumulate's own history has had a chance to build back up).
        // Independent of Accumulate - this only gates whether the denoiser is told to
        // trust that history (TemporalModeUsePreviousLayers); RaygenProgram still
        // computes Flow every frame either way (cheap, and needed again the instant
        // this is re-enabled).
        public bool TemporalDenoiseEnabled { get; set; } = true;
        // Unified bounce budget (docs/SAMPLE15_PLAN.md Milestone M3) - Russian
        // roulette (RaygenProgram.cs) terminates most paths well before this ceiling.
        public int MaxBounces { get; set; } = 8;

        public bool UseMergedTrianglesGas { get; set; } = true;

        // Render-resolution scaling: the traced/denoised/tonemapped resolution
        // (width/height, everywhere else in this class) can be lower than the actual
        // OS window's client size (windowWidth/windowHeight, tracked separately here) -
        // the GL fullscreen quad already stretches whatever texture resolution to fill
        // the viewport (linear-filtered, see CudaGlInteropDisplayBuffer's texture
        // params), so a smaller render resolution just means a softer (not letterboxed
        // or distorted) image, at proportionally lower trace cost. Both axes always
        // scale together off the window's own aspect ratio, so this never distorts it.
        public const float MinRenderScale = 0.1f;
        int windowWidth;
        int windowHeight;
        float renderScaleField = 1f;

        public float RenderScale
        {
            get => renderScaleField;
            set
            {
                renderScaleField = Math.Clamp(value, MinRenderScale, 1f);
                ApplyRenderResolution();
            }
        }

        public int RenderWidth => width;
        public int RenderHeight => height;
        public int WindowWidth => windowWidth;
        public int WindowHeight => windowHeight;

        // Auto-scale: adjusts RenderScale every AutoScaleCheckIntervalFrames frames to
        // keep the renderer's own GPU work (trace+denoise+tonemap) within the frame
        // budget implied by TargetFrameRate - see UpdateAutoScale's own comment. Off
        // by default; the manual RenderScale slider is the primary control until this
        // is enabled. MeasuredRaysPerSecond is a separate, read-only display metric
        // (see its own comment) - not itself a target, just shown next to the slider.
        public bool AutoScaleEnabled { get; set; } = true;
        public float TargetFrameRate { get; set; } = 30f;

        // Primary-ray throughput readout for the UI (width*height*NumPixelSamples /
        // TraceMs) - purely informational. Raw ray throughput isn't a usable
        // resolution-scaling *target* on its own: at reasonable GPU occupancy,
        // per-ray cost is roughly resolution-independent, so this number stays
        // roughly flat regardless of RenderScale - you can't "hit a throughput
        // target" by changing resolution, since the ratio is already
        // resolution-invariant. UpdateAutoScale targets frame rate instead, which
        // resolution actually does control.
        public double MeasuredRaysPerSecond { get; private set; }

        const int AutoScaleCheckIntervalFrames = 45;
        const float AutoScaleDeadbandFraction = 0.05f;
        const float AutoScaleMaxStepPerCheck = 0.15f;
        int framesSinceAutoScaleCheck;

        // Tonemap controls (docs/SAMPLE15_PLAN.md Milestone M8). ExposureStops
        // defaults to -1 (2^-1 = 0.5x) to reproduce the previous hardcoded-Exposure
        // constant's default look exactly - the UI/keyboard can move it from there.
        public float ExposureStops { get; set; } = -1f;
        public int TonemapOperator { get; set; } = TonemapKernel.OperatorReinhard;

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
        public int PointLightCount { get; private set; }
        public int EmissiveTriangleLightCount { get; private set; }
        public bool HasEnvMapLight { get; private set; }
        public string AccelStructureSummary { get; private set; } = "";

        public FrameStats LastStats { get; private set; }

        public int GlTextureHandle => frameOutput.GlTextureHandle;

        readonly Stopwatch stepStopwatch = new Stopwatch();
        long lastFrameStartTicks;
        bool hasRenderedFrame;
        double smoothedFrameMs = 16.0;

        // The camera used for the *previously rendered* frame - tracked separately
        // from `camera` (this frame's) so a camera move doesn't discard it, unlike
        // launchParams.FrameID's reset in setCamera() (see LaunchParams.PrevCamera*'s
        // own doc comment). hasValidPreviousFrame is false right after a scene switch
        // or resize (the previous frame's content/resolution doesn't apply) and true
        // once a frame has actually been rendered since.
        Camera previousRenderedCamera;
        bool hasValidPreviousFrame;

        // Whether the OptiX temporal denoiser actually ran last frame - the ping-ponged
        // denoisedColorBuffers/currentDenoisedIndex in FrameOutput.cs (its
        // PreviousOutput history) is only written to when DenoiserOn is true, so a
        // DenoiserOn transition false->true must not tell the denoiser to trust that
        // history on its first frame back (it would be reprojecting whatever stale
        // frame/camera/scene state existed the last time the denoiser ran, producing a
        // one-frame ghost). Reset alongside hasValidPreviousFrame for the same reasons
        // (scene switch/resize).
        bool wasDenoiserOnLastFrame;

        // Guards every access to launchParams, traversable, and the scene GPU buffers.
        // Kept even though Sample14 is single-threaded (unlike Sample13, where this
        // lock exists specifically to protect against a dedicated render thread racing
        // scene-switch/camera-move calls from the UI thread) - cheap insurance when
        // uncontended, and removing it isn't a meaningful simplification on its own
        // (see docs/SAMPLE14_PLAN.md's note on this).
        readonly object gpuLock = new object();

        public unsafe SampleRenderer(int windowWidth, int windowHeight)
        {
            gpu = new GpuContext();
            pipeline = new RendererPipeline(gpu);
            buffers = new SceneGpuBuffers(gpu.Accelerator);
            textures = new TextureCache();
            sbtBuilder = new SbtBuilder(gpu.Accelerator, pipeline, textures);
            accel = new AccelStructureBuilder(gpu.Accelerator, gpu.DeviceContext, buffers);
            animator = new SceneAnimator(buffers);
            frameOutput = new FrameOutput(gpu);

            sceneBuilders = SceneTable.Build();

            this.windowWidth = windowWidth;
            this.windowHeight = windowHeight;
            resize(windowWidth, windowHeight);

            SwitchToScene(0);
        }

        public void NextScene() => SwitchToScene((currentSceneIndex + 1) % sceneBuilders.Length);
        public void PreviousScene() => SwitchToScene((currentSceneIndex - 1 + sceneBuilders.Length) % sceneBuilders.Length);

        public void RebuildCurrentScene() => SwitchToScene(currentSceneIndex);

        string BuildAccelStructureSummary(SceneData scene, bool hasTriangles)
        {
            if (!hasTriangles)
                return "1 GAS: (empty)";

            int triangleMeshCount = SbtLayout.GetTriangleMeshRanges(scene, UseMergedTrianglesGas).Length;
            return $"1 GAS[Triangles: {triangleMeshCount} input(s), {scene.Indices.Length} tris]";
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

                // Cross-checks that the accel structure's NumSbtRecords values (per
                // AddTriangleMesh() call) still agree with the SBT's own actual hitgroup
                // record count - see Sample13's identical assert for the bug class this
                // guards against.
                Debug.Assert(
                    sbt.HitgroupRecordCount == accel.TotalHitgroupRecordsUsed,
                    $"SBT hitgroup record count ({sbt.HitgroupRecordCount}) doesn't match " +
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
                launchParams.Vertices = (Vec3*)SceneGpuBuffers.NativePtrOrZero(buffers.Vertices);
                launchParams.Normals = (Vec3*)SceneGpuBuffers.NativePtrOrZero(buffers.Normals);
                launchParams.TexCoords = (Vec2*)SceneGpuBuffers.NativePtrOrZero(buffers.TexCoords);
                launchParams.Tangents = (Vec3*)SceneGpuBuffers.NativePtrOrZero(buffers.Tangents);
                launchParams.Indices = (Vec3i*)SceneGpuBuffers.NativePtrOrZero(buffers.Indices);
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

                // The previous frame's rendered image (and its ping-ponged denoised
                // history) belongs to a different scene/resolution now - invalidate
                // reprojection until a frame has actually been rendered against this
                // one (see previousRenderedCamera's own doc comment).
                hasValidPreviousFrame = false;
                launchParams.PrevFrameValid = 0;
                wasDenoiserOnLastFrame = false;

                frameOutput.ResetAccumulation();

                bool hasTriangles = scene.Vertices.Length > 0 && scene.Indices.Length > 0;

                TriangleCount = scene.Indices.Length;
                MaterialCount = scene.Materials.Length;

                int pointLightCount = 0, emissiveTriangleLightCount = 0;
                bool hasEnvMapLight = false;
                foreach (var light in scene.NeeLights)
                {
                    if (light.Kind == LightGpu.KindPoint) pointLightCount++;
                    else if (light.Kind == LightGpu.KindTriangle) emissiveTriangleLightCount++;
                    else if (light.Kind == LightGpu.KindEnvMap) hasEnvMapLight = true;
                }
                PointLightCount = pointLightCount;
                EmissiveTriangleLightCount = emissiveTriangleLightCount;
                HasEnvMapLight = hasEnvMapLight;

                AccelStructureSummary = BuildAccelStructureSummary(scene, hasTriangles);
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

        // Sets the actual traced/denoised resolution directly - called both by
        // ApplyRenderResolution below (the normal path, window-size * RenderScale) and
        // by the constructor (before a scene/camera exists yet, see the currentScene
        // guard below).
        public unsafe void resize(int width, int height)
        {
            if (width == 0 || height == 0)
                return;

            lock (gpuLock)
            {
                this.width = width;
                this.height = height;

                frameOutput.Resize(width, height);

                launchParams.NumPixelSamples = NumPixelSamples;
                launchParams.MaxBounces = MaxBounces;
                // ColorBuffer/AccumCountBuffer/PrevColorBuffer/PrevAccumCountBuffer are NOT
                // wired here - which physical buffer is "current" vs "previous" flips every
                // frame (FrameOutput.SwapColorHistory), so they're re-wired at the top of
                // render() instead of once here.
                launchParams.AlbedoBuffer = (Vec4*)frameOutput.AlbedoBuffer.NativePtr;
                launchParams.NormalBuffer = (Vec4*)frameOutput.NormalBuffer.NativePtr;
                launchParams.FlowBuffer = (Vec2*)frameOutput.FlowBuffer.NativePtr;
                launchParams.FlowTrustworthinessBuffer = (float*)frameOutput.FlowTrustworthinessBuffer.NativePtr;

                // Refresh the camera's own width/height (and therefore aspectRatio/
                // reciprocalWidth/reciprocalHeight) to the new render resolution -
                // Camera(camera, width, height) preserves origin/lookAt/up/fov/
                // worldScale and only recomputes the resolution-derived fields, same
                // copy-constructor setCamera already uses for a camera move. Without
                // this, a resize alone (no subsequent camera move) left the aspect
                // ratio stale - stretching/squashing the image until the next WASD/
                // mouse-look frame happened to call setCamera. Guarded on
                // currentScene != null since the constructor's own initial resize()
                // call runs before SwitchToScene(0) has set up a real camera to refresh
                // (that call builds one from scratch already at the right resolution).
                if (currentScene != null)
                {
                    camera = new Camera(camera, width, height);
                    launchParams.camera = camera;
                }

                // The ping-ponged denoised history and the just-resized framebuffers no
                // longer correspond to anything - same invalidation as a scene switch (see
                // previousRenderedCamera's own doc comment).
                hasValidPreviousFrame = false;
                launchParams.PrevFrameValid = 0;
                wasDenoiserOnLastFrame = false;
            }
        }

        // Called by RenderWindow whenever the OS window's client size changes (on
        // launch, on manual resize/maximize/restore) - stores the raw window size and
        // re-derives the traced resolution from it via the current RenderScale, rather
        // than tracing at the window's exact pixel size. e.Width/e.Height can be 0
        // while the window is minimized; resize() below already no-ops for either
        // dimension being 0, so ApplyRenderResolution inherits that guard for free.
        public void OnWindowResized(int newWindowWidth, int newWindowHeight)
        {
            windowWidth = newWindowWidth;
            windowHeight = newWindowHeight;
            ApplyRenderResolution();
        }

        // Re-derives the traced resolution from windowWidth/windowHeight * RenderScale
        // and applies it (if it actually changed) - both axes scale off the window's
        // own aspect ratio, so RenderScale never distorts the image, only softens it.
        void ApplyRenderResolution()
        {
            int newWidth = Math.Max(1, (int)MathF.Round(windowWidth * renderScaleField));
            int newHeight = Math.Max(1, (int)MathF.Round(windowHeight * renderScaleField));
            if (newWidth == width && newHeight == height)
                return;

            resize(newWidth, newHeight);
        }

        // Auto-scale control law: standard dynamic-resolution scaling, targeting a
        // frame rate directly rather than a throughput number. gpuWorkMs is last
        // frame's own trace+denoise+tonemap cost (LastStats, all three of which scale
        // with resolution) - deliberately not the raw wall-clock frame delta, which
        // would also include vsync/present wait that resolution has no effect on and
        // would otherwise mask the actual headroom/deficit. Comparing that against
        // the frame-time budget TargetFrameRate implies gives a direct time ratio;
        // since GPU work scales ~linearly with pixel count, sqrt'ing that ratio gives
        // the RenderScale multiplier (pixel count ~ RenderScale²). Iterating this
        // (checked periodically, not every frame) converges toward whatever
        // resolution keeps gpuWorkMs at the target budget - if the target is higher
        // than this GPU/scene can sustain even at full resolution, it simply settles
        // at RenderScale=1 (best effort) rather than anything unstable.
        //
        // Checked every AutoScaleCheckIntervalFrames frames (not every frame) with a
        // deadband and a max step per check - every actual resize() call resets
        // temporal accumulation/denoise history (see resize()'s own comment), so
        // adjusting continuously would mean the image never gets a chance to
        // converge, defeating the whole point of the accumulation system.
        void UpdateAutoScale()
        {
            if (!AutoScaleEnabled)
            {
                framesSinceAutoScaleCheck = 0;
                return;
            }

            framesSinceAutoScaleCheck++;
            if (framesSinceAutoScaleCheck < AutoScaleCheckIntervalFrames)
                return;
            framesSinceAutoScaleCheck = 0;

            double gpuWorkMs = LastStats.TraceMs + LastStats.DenoiseMs + LastStats.TonemapMs;
            if (gpuWorkMs <= 0 || TargetFrameRate <= 0)
                return;

            double targetFrameMs = 1000.0 / TargetFrameRate;
            double timeRatio = targetFrameMs / gpuWorkMs;
            if (Math.Abs(timeRatio - 1.0) < AutoScaleDeadbandFraction)
                return;

            float desiredScale = renderScaleField * (float)Math.Sqrt(timeRatio);
            float step = Math.Clamp(desiredScale - renderScaleField, -AutoScaleMaxStepPerCheck, AutoScaleMaxStepPerCheck);
            RenderScale = renderScaleField + step;
        }

        public void setCamera(Camera camera)
        {
            lock (gpuLock)
            {
                this.camera = camera;
                launchParams.camera = new Camera(camera, width, height);
                // FrameID no longer resets on camera move - it's now a simple "frames
                // since scene load" counter (RaygenProgram.cs's own per-pixel
                // reprojected accumulation is what actually converges the image now,
                // see MaxHistoryFrames). Leaving FrameID monotonic also fixes a latent
                // bug: RaygenProgram.cs seeds its per-pixel RNG from FrameID, so
                // resetting it to 0 on every move frame meant continuous camera motion
                // reused the exact same jittered sample pattern every frame.
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
                double sinceLastFrameMs = 1000.0 / 60.0;
                if (hasRenderedFrame)
                {
                    sinceLastFrameMs = (frameStartTicks - lastFrameStartTicks) * 1000.0 / Stopwatch.Frequency;
                    smoothedFrameMs = (smoothedFrameMs * 0.9) + (sinceLastFrameMs * 0.1);
                }
                lastFrameStartTicks = frameStartTicks;
                hasRenderedFrame = true;

                // Runs before this frame's launch (using last frame's measurements) so
                // any resulting resize() takes effect for THIS frame's trace, not one
                // frame late - see UpdateAutoScale's own doc comment for the control law.
                UpdateAutoScale();

                bool sceneIsAnimated = currentScene != null && animator.IsAnimated(currentScene);
                if (sceneIsAnimated)
                    animator.Update(currentScene);

                launchParams.NumPixelSamples = NumPixelSamples;
                launchParams.MaxBounces = MaxBounces;
                launchParams.EnvMapRotation = EnvMapRotation;
                launchParams.EnvMapIntensity = currentScene.EnvMapIntensity * EnvMapIntensityMultiplier;
                // See LaunchParams.MaxHistoryFrames's own doc comment - 1 means "always
                // a fresh single sample, no blending", matching the old scenes' visual
                // behavior exactly for the off/animated cases.
                launchParams.MaxHistoryFrames = (Accumulate && !sceneIsAnimated) ? MaxHistoryFrames : 1;
                launchParams.DeltaTimeSeconds = (float)(sinceLastFrameMs / 1000.0);
                launchParams.HistoryDecayHalfLifeSeconds = HistoryDecayHalfLifeSeconds;
                launchParams.DepthRejectionThreshold = DepthRejectionThreshold;

                // The camera used for the frame *before* this one - captured before
                // overwriting below, so a camera move this frame still reprojects
                // correctly against where the camera actually was last frame (see
                // previousRenderedCamera's own doc comment).
                bool trustPreviousFrame = hasValidPreviousFrame;
                launchParams.PrevFrameValid = trustPreviousFrame ? 1 : 0;
                launchParams.PrevCameraOrigin = previousRenderedCamera.origin;
                launchParams.PrevCameraAxisX = previousRenderedCamera.axis.x;
                launchParams.PrevCameraAxisY = previousRenderedCamera.axis.y;
                launchParams.PrevCameraAxisZ = previousRenderedCamera.axis.z;
                launchParams.PrevCameraAspectRatio = previousRenderedCamera.aspectRatio;
                launchParams.PrevCameraPlaneDist = previousRenderedCamera.cameraPlaneDist;

                // Which physical buffer is "current" (this frame's write target) vs
                // "previous" (read-only history to reproject) flips every frame - see
                // FrameOutput.SwapColorHistory, called after DenoiseAndTonemap below.
                launchParams.ColorBuffer = (Vec4*)frameOutput.CurrentColorBuffer.NativePtr;
                launchParams.AccumCountBuffer = (float*)frameOutput.CurrentAccumCountBuffer.NativePtr;
                launchParams.PrevColorBuffer = (Vec4*)frameOutput.PreviousColorBuffer.NativePtr;
                launchParams.PrevAccumCountBuffer = (float*)frameOutput.PreviousAccumCountBuffer.NativePtr;

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

                // Primary-ray throughput readout (see MeasuredRaysPerSecond's own doc
                // comment) - purely informational, the UI's live number next to the
                // render-scale slider. UpdateAutoScale targets frame rate instead, not
                // this.
                MeasuredRaysPerSecond = traceMs > 0
                    ? (double)width * height * Math.Max(1, NumPixelSamples) / (traceMs / 1000.0)
                    : 0.0;

                previousRenderedCamera = launchParams.camera;
                hasValidPreviousFrame = true;

                launchParams.FrameID++;

                // wasDenoiserOnLastFrame guards against a DenoiserOn false->true toggle:
                // without it, the very first frame back would tell the denoiser to
                // trust denoisedColorBuffers' PreviousOutput history, which is stale
                // (last written whenever the denoiser was last on, possibly a different
                // camera position or scene) - see wasDenoiserOnLastFrame's own doc
                // comment.
                bool useTemporalHistory = trustPreviousFrame && TemporalDenoiseEnabled && wasDenoiserOnLastFrame;
                frameOutput.DenoiseAndTonemap(DenoiserOn, Accumulate, launchParams.FrameID, useTemporalHistory, ExposureStops, TonemapOperator, out double denoiseMs, out double tonemapMs);
                wasDenoiserOnLastFrame = DenoiserOn;

                // Flip current/previous for next frame's reprojection - must happen
                // after DenoiseAndTonemap above, which reads CurrentColorBuffer as its
                // raw input.
                frameOutput.SwapColorHistory();

                int samplesAccumulated = launchParams.FrameID;

                LastStats = new FrameStats
                {
                    TraceMs = traceMs,
                    DenoiseMs = denoiseMs,
                    TonemapMs = tonemapMs,
                    TotalFrameMs = smoothedFrameMs,
                    Fps = 1000.0 / smoothedFrameMs,
                    SamplesAccumulated = samplesAccumulated,
                    RaysPerSecond = MeasuredRaysPerSecond,
                };
            }
        }

        // GPU-internal PBO -> texture blit (no CPU copy) - call once per frame after
        // render(), before drawing the fullscreen quad.
        public void BlitToTexture() => frameOutput.BlitToTexture();
    }
}
