using ILGPU;
using ILGPU.OptiX;
using ILGPU.OptiX.CooperativeVectors;
using ILGPU.OptiX.Device;
using ILGPU.OptiX.Pipeline;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.OptiX.DeviceApi;
using MeshRange = ILGPU.OptiX.Pipeline.OptixMeshRange;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
// batchIndex, then learningRate/weightDecay/adamBiasCorrection1/adamBiasCorrection2/
// emaNewCoeff/emaOldCoeff - the latter 4 are precomputed once per optimizer step on
// the host (RunNrcTrainingAndComposite) instead of every weight-thread redundantly
// calling XMath.Pow to derive the same stepCount-only values (see AdamKernel.AdamStep's
// own note).
using OptimizerKernel = System.Action<ILGPU.Index1D, ILGPU.ArrayView<float>, ILGPU.ArrayView<float>, ILGPU.ArrayView<float>, ILGPU.ArrayView<float>, ILGPU.ArrayView<float>, ILGPU.ArrayView<int>, int, float, float, float, float, float, float>;

namespace Sample21
{
    /// <summary>
    /// The renderer coordinator: owns the launch params, the camera, the scene
    /// switcher, and the render loop, wiring together the components that do the
    /// actual work - <see cref="GpuContext"/>, <see cref="RayTracingPipeline{TLaunchParams}"/>,
    /// <see cref="SceneGpuBuffers"/>, <see cref="SbtBuilder"/>,
    /// <see cref="AccelStructureBuilder"/>, <see cref="TextureCache"/>,
    /// <see cref="FrameOutput"/>, <see cref="SceneAnimator"/>.
    ///
    /// There is no MainWindow coupling and no PresentQueue - render() runs entirely on
    /// the single GL/compute thread and TonemapToDisplay's Map/tonemap/Unmap sequence
    /// (see FrameOutput.cs) is the last GPU-touching step, with no further cross-thread
    /// handoff needed.
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

        // NRC - see Device/RaygenProgram.cs's NRC comment. Entirely separate from the
        // fields above: its own pipelines/weights/scratch/records, touching none of
        // the primary rendering path.
        readonly Rendering.WeightBuffers nrcWeights;
        readonly Rendering.NrcScratchBuffers nrcScratch;
        readonly Rendering.NrcPipelines nrcPipelines;
        readonly Rendering.TrainRecordBuffers nrcTrainBuffers;
        // AdamW only - this sample never exposed an optimizer switcher (no UI control,
        // nothing else ever changed it from the default), so the SGD/RMSProp kernel
        // variants that used to sit behind an Optimizer enum were removed outright
        // rather than kept as unreachable configuration.
        readonly OptimizerKernel nrcAdamKernel;
        // Network depth - fixed for the lifetime of this renderer (no UI control ever
        // changed it either; the runtime network-reconfiguration path that used to let
        // it vary was removed). Depth defaults to 3, not NrcConstants.HiddenLayerCount's VkNRC-parity 5 - part
        // of this sample's "lean" runtime defaults (matches or beats depth-5 quality at
        // every checkpoint while cutting both training and per-pixel composite-inference
        // cost).
        int nrcHiddenLayerCount = 3;
        int nrcLayerWidth = (int)Core.NrcConstants.LayerWidth;
        MemoryBuffer1D<Vec4, Stride1D.Dense> nrcPreviewBuffer;
        int nrcTrainStepCount;

        /// <summary>Master enable for the NRC side channel - on by default; toggle via the UI (UI/UiPanel.cs).</summary>
        public bool NrcEnabled { get; set; } = true;

        /// <summary>When true (and <see cref="NrcEnabled"/>), the displayed image is the cache's own predictions instead of the primary render - render() feeds <see cref="FrameOutput.TonemapToDisplay"/> the preview buffer and skips the denoiser outright.</summary>
        public bool ShowNrcPreview { get; set; }

        // NRC network settings - runtime-tunable live via the UI (UI/UiPanel.cs),
        // defaulted from Core/NrcConstants.cs.
        public float NrcLearningRate { get; set; } = Core.NrcConstants.AdamLearningRate;
        public float NrcWeightDecay { get; set; } = Core.NrcConstants.AdamWeightDecay;
        public float NrcEmaAlpha { get; set; } = Core.NrcConstants.EmaAlpha;
        // 0.01, not NrcConstants.TrainProbability's VkNRC-parity 0.03 - part of this
        // sample's "lean" runtime defaults; NrcConstants keeps the VkNRC-parity
        // reference values.
        public float NrcTrainProbability { get; set; } = 0.01f;

        /// <summary>
        /// VkNRC's "Use EMA" debug checkbox, finally implemented (see
        /// WeightBuffers.ConvertMasterToTrainingAndInference's own note): when true,
        /// composite/self-training-target inference reads the bias-corrected EMA weight
        /// snapshot instead of the live master weights. Off by default, matching VkNRC's
        /// m_use_ema_weights{false}. This is also the switch that makes
        /// <see cref="NrcEmaAlpha"/> actually affect the rendered output - with it off,
        /// the EMA snapshot is computed every step but sampled by nothing.
        /// </summary>
        public bool NrcUseEmaInference { get; set; }

        /// <summary>
        /// How many disjoint training batches run per trained frame (1..
        /// NrcConstants.TrainBatchesPerFrame) - performance knob. Raygen concentrates
        /// this frame's
        /// records into exactly this many batch slices (LaunchParams.NrcTrainBatchCount),
        /// so lowering it cuts gradient/optimize launches without discarding gathered
        /// records. Default 2, not VkNRC's 4 - part of this sample's "lean" runtime
        /// defaults (see NrcTrainProbability's own note).
        /// </summary>
        public int NrcTrainBatchesPerFrame { get; set; } = 2;

        /// <summary>
        /// Train the cache only every Nth frame (1 = every frame, VkNRC's behavior) -
        /// performance knob. Off-frames skip the whole training side (record gathering
        /// via NrcTrainProbability=0, self-training resolution, gradient/optimize) but
        /// still composite from the last trained weights; keyed off FrameID, so frame 0
        /// after any scene switch/reset always trains (the composite never reads a
        /// never-converted InferenceWeights buffer). Default 2, part of this sample's
        /// "lean" runtime defaults.
        /// </summary>
        public int NrcTrainEveryNFrames { get; set; } = 2;

        /// <summary>VkNRC's "C" adaptive-handoff footprint constant - see RaygenProgram.cs's adaptive test.</summary>
        public float NrcFootprintConstant { get; set; } = Core.NrcConstants.FootprintConstant;

        // HDRI environment maps - scene-dependent
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

        IntPtr traversable;
        LaunchParams launchParams;

        public Camera camera { get; private set; }

        public bool DenoiserOn { get; set; } = true;
        public int NumPixelSamples { get; set; } = 1;
        // Per-pixel reprojected accumulation (ColorBuffer/AccumCountBuffer,
        // RaygenProgram.cs) - converges the image toward a noise-free ground truth over
        // many frames, whether the camera is static OR moving. Toggling this off (or an
        // animated scene, which changes *lighting* our purely-geometric reprojection
        // can't compensate for) sets MaxHistoryFrames down to 1 each frame (see
        // render()) - always a fresh single sample, no blending. This is the sample's
        // only temporal integrator - FrameOutput's OptiX denoiser runs in HDR (not
        // TEMPORAL) model kind and does purely spatial cleanup on top of this.
        public bool Accumulate { get; set; } = true;
        // The cap on AccumCountBuffer's per-pixel history length **while the camera is
        // actually moving** - once a pixel's history reaches this, the blend settles
        // into a small new-sample weight (1/MaxHistoryFrames, further aged by
        // HistoryDecayHalfLifeSeconds below) rather than continuing to shrink like an
        // unbounded running mean would, so the image stays responsive to motion
        // instead of getting sluggish as history piles up. **Does not apply once the
        // camera has been stationary for a full frame** - render() detects that (via
        // CamerasEqual against previousRenderedCamera) and switches to true unbounded
        // accumulation (no cap, no decay) so a static view actually converges to a
        // clean image over time instead of plateauing at this cap's noise floor
        // forever (see render()'s own comment).
        // Kept low - a long-lived history survives many small per-frame reprojection
        // resamples during gradual camera motion (each individually correct, but a
        // very long chain of them can still soften fine detail over time, "reprojecting
        // too far"), and a much shorter cap bounds how many times a pixel's color can
        // be recursively resampled before it's forced to refresh from a fully fresh
        // sample.
        public int MaxHistoryFrames { get; set; } = 96;
        // Real-time half-life for AccumCount's decay **while the camera is actually
        // moving** - even a static-relative-to-last-frame, perfectly reprojected pixel
        // keeps slowly forgetting old contributions on this wall-clock schedule during
        // continuous motion, instead of freezing once it hits MaxHistoryFrames.
        // Smaller = adapts to subtle scene changes faster but noisier at steady state;
        // 0 or negative disables decay entirely, freezing at the cap.
        // Same static-camera exception as MaxHistoryFrames above: render() passes 0
        // here once the camera has been stationary for a full frame, disabling decay
        // entirely so history can grow unbounded and actually converge.
        public float HistoryDecayHalfLifeSeconds { get; set; } = 3f;
        // Disocclusion rejection threshold (see LaunchParams.DepthRejectionThreshold's
        // own doc comment) - a relative depth-difference fraction; smaller rejects more
        // aggressively (less ghosting/smearing at moving edges, but discards history -
        // and thus noise reduction - more readily), larger tolerates more before
        // rejecting (smoother but more prone to smearing at disocclusions).
        public float DepthRejectionThreshold { get; set; } = 0.05f;
        // Unified bounce budget - Russian
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

        // Averaged (not single-frame) GPU work over the current check window - path
        // tracing's own frame-to-frame cost variance (Russian roulette bounce counts,
        // mixed diffuse/metal/glass materials with very different average path
        // lengths) means whichever single frame happens to land on a
        // AutoScaleCheckIntervalFrames boundary is not a reliable sample of "the"
        // cost at this resolution - it can spuriously cross AutoScaleDeadbandFraction
        // on its own and trigger a resize() that isn't actually warranted. Every such
        // resize fully resets the per-pixel TAA/temporal-accumulation history (see
        // resize()'s own comment), so a falsely-triggered one is not just wasted GPU
        // work rebuilding buffers/denoiser - it also restarts convergence from
        // scratch. Accumulated every frame regardless of AutoScaleEnabled (cheap) so
        // toggling it on mid-session doesn't start from a stale/empty window.
        double autoScaleGpuWorkAccumMs;
        int autoScaleGpuWorkSampleCount;

        // Tonemap controls. ExposureStops
        // defaults to -1 (2^-1 = 0.5x) to reproduce the previous hardcoded-Exposure
        // constant's default look exactly - the UI/keyboard can move it from there.
        public float ExposureStops { get; set; } = -1f;
        public int TonemapOperator { get; set; } = TonemapKernel.OperatorReinhard;

        // Env-map controls - Rotation is in
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

        // Guards every access to launchParams, traversable, and the scene GPU buffers.
        // Kept even though this sample is single-threaded - cheap insurance when
        // uncontended, and removing it isn't a meaningful simplification on its own.
        readonly object gpuLock = new object();

        public SampleRenderer(int windowWidth, int windowHeight)
        {
            gpu = new GpuContext();

            // No module/pipeline compile options, no SetStackSize magic numbers, no
            // hand-rolled SBT record structs. Ray
            // type declaration order (radiance, then shadow) fixes their SBT index at
            // 0/1, matching Payloads.RADIANCE_RAY_TYPE/SHADOW_RAY_TYPE, which every
            // OptixTrace.Trace(...) call site (RaygenProgram.cs, Rays/ShadowRay.cs)
            // still addresses directly - migrating those off raw payload registers is
            // a separate, GPU-verification-gated change.
            pipeline = gpu.RayTracer.CreatePipeline<LaunchParams>(b => b
                .Raygen(RaygenProgram.__raygen__renderFrame)
                .RayType("radiance", r => r
                    .Payload<Payloads.RadiancePayload>()
                    .Miss(RadianceRay.__miss__radiance)
                    .HitGroup<MaterialSbtData>(RadianceRay.__closest__radiance, RadianceRay.__anyhit__radiance))
                .RayType("shadow", r => r
                    .Payload<Payloads.ShadowPayload>()
                    .Miss(ShadowRay.__miss__shadow)
                    .HitGroup<MaterialSbtData>(ShadowRay.__closesthit__shadow, ShadowRay.__anyhit__shadow))
                .MaxTraceDepth(2));

            buffers = new SceneGpuBuffers(gpu.Accelerator);
            textures = new TextureCache();
            sbtBuilder = new SbtBuilder(textures);
            accel = new AccelStructureBuilder(gpu.Accelerator, gpu.DeviceContext, buffers);
            animator = new SceneAnimator(buffers);
            frameOutput = new FrameOutput(gpu);

            nrcWeights = new Rendering.WeightBuffers(gpu.DeviceContext, gpu.Accelerator, new Random(1234), nrcHiddenLayerCount, nrcLayerWidth);
            nrcTrainBuffers = new Rendering.TrainRecordBuffers(gpu.Accelerator);
            nrcScratch = new Rendering.NrcScratchBuffers(gpu.Accelerator, nrcHiddenLayerCount, nrcLayerWidth);
            nrcPipelines = new Rendering.NrcPipelines(gpu.DeviceContext, gpu.Accelerator);
            nrcAdamKernel = gpu.Accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<int>, int, float, float, float, float, float, float>(
                Device.Network.AdamKernel.AdamStep);

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

                // Scene-dependent HDRI environment map - null/empty EnvMapPath means this scene keeps the
                // flat BackgroundTop/Bottom gradient sky instead. Must happen before
                // buffers.Upload(scene) below, which reads scene.NeeLights/
                // NeeLightCdf/NeeLightAreaPdf.
                EnvMapGpuData envMapData = string.IsNullOrEmpty(scene.EnvMapPath) ? null : GetOrLoadEnvMap(scene.EnvMapPath);
                float envMapPower = envMapData?.TotalWeight ?? 0f;
                (scene.NeeLights, scene.NeeLightCdf, scene.NeeLightAreaPdf, float envMapLightPdf) = LightList.Build(scene, envMapPower);

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
                // AddTriangleMesh() call) still agree with the SBT's own actual hitgroup
                // record count.
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
                launchParams.Tangents = OptixDeviceView<Vec3>.From(buffers.Tangents);
                launchParams.Indices = OptixDeviceView<Vec3i>.From(buffers.Indices);
                launchParams.PointLights = OptixDeviceView<PointLightGpu>.From(buffers.Lights);
                launchParams.NeeLights = OptixDeviceView<LightGpu>.From(buffers.NeeLights);
                launchParams.NeeLightCdf = OptixDeviceView<float>.From(buffers.NeeLightCdf);
                launchParams.NumNeeLights = scene.NeeLights.Length;
                launchParams.NeeLightAreaPdf = OptixDeviceView<float>.From(buffers.NeeLightAreaPdf);
                launchParams.EnvMapPixels = OptixDeviceView<Vec3>.From(envMapData?.Pixels);
                launchParams.EnvMapCdf = OptixDeviceView<float>.From(envMapData?.Cdf);
                launchParams.EnvMapPdfUv = OptixDeviceView<float>.From(envMapData?.PdfUv);
                launchParams.EnvMapWidth = envMapData?.Width ?? 0;
                launchParams.EnvMapHeight = envMapData?.Height ?? 0;
                launchParams.EnvMapIntensity = scene.EnvMapIntensity;
                launchParams.EnvMapLightPdf = envMapLightPdf;
                launchParams.AmbientColor = scene.AmbientColor;
                launchParams.AmbientIntensity = scene.AmbientIntensity;
                launchParams.BackgroundTop = scene.BackgroundTop;
                launchParams.BackgroundBottom = scene.BackgroundBottom;
                launchParams.FrameID = 0;

                // NRC scene-position normalization (LaunchParams.NrcSceneCenter's own
                // doc comment): VkNRC normalizes every model's vertices into [-1,1] at
                // load; this port keeps world-space geometry and remaps positions at
                // record-write time instead, so the remap constants are per-scene state.
                if (scene.Vertices.Length > 0)
                {
                    SceneBuilder.ComputeBounds(scene.Vertices, out Vec3 sceneMin, out Vec3 sceneMax);
                    Vec3 sceneExtent = sceneMax - sceneMin;
                    float maxExtent = Math.Max(sceneExtent.x, Math.Max(sceneExtent.y, sceneExtent.z));
                    launchParams.NrcSceneCenter = (sceneMin + sceneMax) / 2f;
                    launchParams.NrcSceneInvExtent = maxExtent > 0f ? 2f / maxExtent : 1f;
                }
                else
                {
                    launchParams.NrcSceneCenter = default;
                    launchParams.NrcSceneInvExtent = 1f;
                }

                // The radiance cache is a function of one scene's geometry/lighting -
                // start each scene from a fresh network instead of retraining across a
                // distribution shift (WeightBuffers.ResetTrainingState's own doc
                // comment). Same fixed seed as construction, so every scene starts from
                // the identical, reproducible init. nrcTrainStepCount restarts the
                // Adam/EMA bias corrections alongside the zeroed moments.
                nrcWeights.ResetTrainingState(new Random(1234));
                nrcTrainStepCount = 0;

                // The previous frame's rendered image (and its ping-ponged denoised
                // history) belongs to a different scene/resolution now - invalidate
                // reprojection until a frame has actually been rendered against this
                // one (see previousRenderedCamera's own doc comment).
                hasValidPreviousFrame = false;
                launchParams.PrevFrameValid = 0;

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
                buffers.DisposeAll();

                foreach (EnvMapGpuData envMapData in envMapCache.Values)
                    envMapData.Dispose();
                envMapCache.Clear();
            }

            frameOutput.Dispose();
            pipeline.Dispose();

            nrcPipelines.Dispose();
            nrcScratch.Dispose();
            nrcTrainBuffers.Dispose();
            nrcWeights.Dispose();
            nrcPreviewBuffer?.Dispose();

            gpu.Dispose();
        }

        // Sets the actual traced/denoised resolution directly - called both by
        // ApplyRenderResolution below (the normal path, window-size * RenderScale) and
        // by the constructor (before a scene/camera exists yet, see the currentScene
        // guard below).
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
                launchParams.MaxBounces = MaxBounces;
                launchParams.NrcEnabled = NrcEnabled ? 1 : 0;
                launchParams.NrcFootprintConstant = NrcFootprintConstant;
                launchParams.NrcTrainProbability = NrcTrainProbability;
                // PrevColorBuffer/PrevAccumCountBuffer are NOT wired here - which physical
                // buffer is "previous" flips every frame (FrameOutput.SwapColorHistory), so
                // they're re-wired at the top of render() instead of once here.
                launchParams.AlbedoBuffer = OptixDeviceView<Vec4>.From(frameOutput.AlbedoBuffer);
                launchParams.NormalBuffer = OptixDeviceView<Vec4>.From(frameOutput.NormalBuffer);
                // Scratch (not ping-ponged, so wired once here rather than every frame
                // in render() the way ColorBuffer/PrevColorBuffer's ping-pong is) - see
                // LaunchParams.RawColorBuffer's own doc comment.
                launchParams.RawColorBuffer = OptixDeviceView<Vec4>.From(frameOutput.RawColorBuffer);
                launchParams.ReprojCoordBuffer = OptixDeviceView<Vec2>.From(frameOutput.ReprojCoordBuffer);

                // NRC - one eval-record/preview-pixel slot per
                // screen pixel, resized alongside every other resolution-dependent
                // buffer above. NrcScratchBuffers must cover whichever is larger -
                // MaxTrainRecords or the pixel count - since CompositeProgram indexes
                // it per-pixel while SelfTrainingTargetProgram/GradientProgram index it
                // per-train-record (bug found on a real GPU run: the composite launch
                // wrote out of bounds when scratch was only ever sized for
                // MaxTrainRecords, causing "illegal memory access").
                nrcTrainBuffers.ResizeEvalRecords(gpu.Accelerator, width, height);
                nrcScratch.EnsureCapacity(gpu.Accelerator, width * height, nrcHiddenLayerCount, nrcLayerWidth);
                // Lazy - RunNrcComposite reallocates it at the new resolution the next
                // time the preview is actually displayed; the composite kernel skips
                // its preview writes entirely while the view is invalid.
                nrcPreviewBuffer?.Dispose();
                nrcPreviewBuffer = null;
                launchParams.NrcEvalRecords = nrcTrainBuffers.EvalRecordsView;
                launchParams.NrcTrainRecords = nrcTrainBuffers.TrainRecordsView;
                launchParams.NrcTrainCounter = nrcTrainBuffers.CounterView;

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

            double gpuWorkMs = autoScaleGpuWorkSampleCount > 0
                ? autoScaleGpuWorkAccumMs / autoScaleGpuWorkSampleCount
                : 0;
            autoScaleGpuWorkAccumMs = 0;
            autoScaleGpuWorkSampleCount = 0;
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

        // Bit-exact comparison, not epsilon-based - reliable here because launchParams.camera
        // only ever changes via setCamera(), which is only called in response to actual user
        // input (see RenderWindow.cs's ApplyCameraLook/UpdateWasdMovement); with no input this
        // frame, launchParams.camera is the literal same struct value as last frame's, not a
        // recomputation that could drift by a rounding error. Used by render() to decide
        // whether to fall back to unbounded TAA accumulation - see its own comment.
        static bool CamerasEqual(Camera a, Camera b) =>
            a.origin.x == b.origin.x && a.origin.y == b.origin.y && a.origin.z == b.origin.z &&
            a.lookAt.x == b.lookAt.x && a.lookAt.y == b.lookAt.y && a.lookAt.z == b.lookAt.z &&
            a.up.x == b.up.x && a.up.y == b.up.y && a.up.z == b.up.z;

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
        // GL-owned interop texture (see FrameOutput.TonemapToDisplay/BlitToTexture) -
        // no CPU byte[] handoff, no second thread. Call BlitToTexture() (or draw the
        // fullscreen quad against GlTextureHandle) from the window after this returns.
        public void render()
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
                launchParams.NrcEnabled = NrcEnabled ? 1 : 0;
                launchParams.NrcFootprintConstant = NrcFootprintConstant;
                // Train-cadence gate (NrcTrainEveryNFrames' own doc comment): off-frames
                // zero the train probability so raygen doesn't trace training suffixes
                // whose records nothing would consume - the gathering cost is part of
                // what the cadence knob exists to save. Keyed off FrameID (pre-increment,
                // 0 right after any scene switch/reset), so the first frame always trains.
                bool nrcTrainThisFrame = NrcEnabled &&
                    (NrcTrainEveryNFrames <= 1 || launchParams.FrameID % NrcTrainEveryNFrames == 0);
                launchParams.NrcTrainProbability = nrcTrainThisFrame ? NrcTrainProbability : 0f;
                launchParams.NrcTrainBatchCount = Math.Clamp(NrcTrainBatchesPerFrame, 1, Core.NrcConstants.TrainBatchesPerFrame);
                launchParams.EnvMapRotation = EnvMapRotation;
                launchParams.EnvMapIntensity = currentScene.EnvMapIntensity * EnvMapIntensityMultiplier;
                // The camera used for the frame *before* this one - captured before
                // overwriting below, so a camera move this frame still reprojects
                // correctly against where the camera actually was last frame (see
                // previousRenderedCamera's own doc comment).
                bool trustPreviousFrame = hasValidPreviousFrame;
                launchParams.PrevFrameValid = trustPreviousFrame ? 1 : 0;

                // See SampleRenderer.MaxHistoryFrames's own doc comment - 1 means "always
                // a fresh single sample, no blending" for the off/animated cases. Passed
                // straight to
                // FrameOutput.ResolveTaa below, not through LaunchParams - only
                // RaygenProgram's own DepthRejectionThreshold check needs to happen
                // inside the OptiX launch itself (see LaunchParams.RawColorBuffer's own
                // doc comment).
                //
                // Bounded-EMA vs. unbounded convergence: MaxHistoryFrames/
                // HistoryDecayHalfLifeSeconds define a steady-state noise floor that
                // never fully clears, by design, so a *moving* camera keeps converging
                // responsively without an ever-growing divisor making it sluggish to
                // adapt. But that same floor applied to a perfectly static camera means
                // the image plateaus at ~MaxHistoryFrames-worth of noise forever, no
                // matter how long it sits still - not what "let it accumulate" should
                // mean. So once the camera has been unchanged for a full frame (compared
                // against previousRenderedCamera, which is bit-identical unless setCamera
                // was actually called since - see its own doc comment), drop back to true
                // unbounded accumulation (no cap, no decay) so a static view keeps
                // converging toward a clean image indefinitely. Any camera move
                // reactivates the bounded/decayed scheme for the frame(s) where
                // reprojection actually needs it.
                bool cameraUnchangedSinceLastFrame = trustPreviousFrame && CamerasEqual(launchParams.camera, previousRenderedCamera);
                int effectiveMaxHistoryFrames;
                float effectiveHistoryDecayHalfLifeSeconds;
                if (!Accumulate || sceneIsAnimated)
                {
                    effectiveMaxHistoryFrames = 1;
                    effectiveHistoryDecayHalfLifeSeconds = HistoryDecayHalfLifeSeconds;
                }
                else if (cameraUnchangedSinceLastFrame)
                {
                    effectiveMaxHistoryFrames = int.MaxValue;
                    effectiveHistoryDecayHalfLifeSeconds = 0f;
                }
                else
                {
                    effectiveMaxHistoryFrames = MaxHistoryFrames;
                    effectiveHistoryDecayHalfLifeSeconds = HistoryDecayHalfLifeSeconds;
                }
                float deltaTimeSeconds = (float)(sinceLastFrameMs / 1000.0);
                launchParams.DepthRejectionThreshold = DepthRejectionThreshold;
                launchParams.PrevCameraOrigin = previousRenderedCamera.origin;
                launchParams.PrevCameraAxisX = previousRenderedCamera.axis.x;
                launchParams.PrevCameraAxisY = previousRenderedCamera.axis.y;
                launchParams.PrevCameraAxisZ = previousRenderedCamera.axis.z;
                launchParams.PrevCameraAspectRatio = previousRenderedCamera.aspectRatio;
                launchParams.PrevCameraPlaneDist = previousRenderedCamera.cameraPlaneDist;

                // "Previous" (read-only history to reproject) flips every frame - see
                // FrameOutput.SwapColorHistory, called after the tonemap below.
                // RawColorBuffer/ReprojCoordBuffer (this frame's own raw output) were
                // already wired once in resize() - see their own doc comment there.
                launchParams.PrevColorBuffer = OptixDeviceView<Vec4>.From(frameOutput.PreviousColorBuffer);

                // NRC - reset the train-record counter before this
                // frame's raygen launch writes into it (see Device/RaygenProgram.cs's
                // NRC comment). Trained frames only: off-frames gather no records
                // (train probability zeroed above) and nothing reads the counters until
                // the next trained frame resets them here first.
                if (nrcTrainThisFrame)
                    nrcTrainBuffers.ResetCounter(gpu.Accelerator.DefaultStream);

                // Single-sync frame: every stage below is ENQUEUED on the default
                // stream without waiting - the one hard sync lives inside
                // frameOutput.TonemapToDisplay (mandatory there anyway: the GL interop
                // buffer must not be unmapped with GPU work still touching it). Stage
                // costs are measured GPU-side with profiling markers (CUDA events,
                // enabled via Context.Profiling() in GpuContext) and read out after
                // that sync, when the events are already complete - a driver query, not
                // a CPU/GPU stall.
                var stream = gpu.Accelerator.DefaultStream;
                using var frameStartMarker = stream.AddProfilingMarker();

                // Persistent launch-params buffer, reused every frame - no per-call
                // allocate/free (the leak OptixLaunchExtensions.OptixLaunch had).
                pipeline.Launch(launchParams, width, height);
                using var traceEndMarker = stream.AddProfilingMarker();

                if (NrcEnabled)
                    RunNrcTrainingAndComposite(nrcTrainThisFrame);
                using var nrcEndMarker = stream.AddProfilingMarker();

                previousRenderedCamera = launchParams.camera;
                hasValidPreviousFrame = true;

                launchParams.FrameID++;

                // Neighborhood-clamped reprojection/blend - runs after the OptiX
                // launch has written RawColorBuffer/ReprojCoordBuffer for every pixel
                // (stream order guarantees that; see FrameOutput.ResolveTaa's own doc
                // comment), and before the denoise/tonemap below, which read its
                // output (CurrentColorBuffer).
                frameOutput.ResolveTaa(effectiveMaxHistoryFrames, effectiveHistoryDecayHalfLifeSeconds, deltaTimeSeconds);

                // NRC preview replaces the displayed image outright, so the denoiser
                // and a primary tonemap would both be dead work while it's up - feed
                // the tonemap the preview buffer directly instead. TAA above still
                // ran either way, keeping the accumulation history converging for
                // when the preview toggles back off.
                bool showPreview = NrcEnabled && ShowNrcPreview;
                MemoryBuffer1D<Vec4, Stride1D.Dense> tonemapSource = showPreview
                    ? nrcPreviewBuffer
                    : frameOutput.Denoise(DenoiserOn, Accumulate, launchParams.FrameID);
                using var denoiseEndMarker = stream.AddProfilingMarker();

                frameOutput.TonemapToDisplay(tonemapSource, ExposureStops, TonemapOperator, out double tonemapMs);

                // Flip current/previous for next frame's reprojection - must happen
                // after the denoise/tonemap enqueues above, which read
                // CurrentColorBuffer as their raw input.
                frameOutput.SwapColorHistory();

                // TonemapToDisplay's sync has returned, so every marker's event has
                // completed - measuring them here is a driver query, not a stall.
                double traceMs = (traceEndMarker - frameStartMarker).TotalMilliseconds;
                double nrcMs = NrcEnabled ? (nrcEndMarker - traceEndMarker).TotalMilliseconds : 0;
                // TAA resolve + denoiser span.
                double denoiseMs = (denoiseEndMarker - nrcEndMarker).TotalMilliseconds;

                // Primary-ray throughput readout (see MeasuredRaysPerSecond's own doc
                // comment) - purely informational, the UI's live number next to the
                // render-scale slider. UpdateAutoScale targets frame rate instead, not
                // this.
                MeasuredRaysPerSecond = traceMs > 0
                    ? (double)width * height * Math.Max(1, NumPixelSamples) / (traceMs / 1000.0)
                    : 0.0;

                int samplesAccumulated = launchParams.FrameID;

                // Accumulated for UpdateAutoScale's own averaged read (see
                // autoScaleGpuWorkAccumMs's own doc comment) - unconditional
                // (not gated on AutoScaleEnabled) so re-enabling it mid-session
                // doesn't start from an empty window.
                autoScaleGpuWorkAccumMs += traceMs + nrcMs + denoiseMs + tonemapMs;
                autoScaleGpuWorkSampleCount++;

                LastStats = new FrameStats
                {
                    TraceMs = traceMs,
                    NrcMs = nrcMs,
                    DenoiseMs = denoiseMs,
                    TonemapMs = tonemapMs,
                    TotalFrameMs = smoothedFrameMs,
                    Fps = 1000.0 / smoothedFrameMs,
                    SamplesAccumulated = samplesAccumulated,
                };
            }
        }

        // NRC: self-training-target resolution -> 4x
        // {gradient, optimize} (one disjoint batch per iteration, chained weight
        // updates - matching VkNRC's own scheme, see TrainRecordBuffers's own doc
        // comment) -> composite (adds the cache's prediction into RawColorBuffer where
        // RaygenProgram.cs's adaptive handoff stopped tracing, see
        // Device/CompositeProgram.cs's own doc comment). Runs once per frame, after the
        // main path-tracing launch above has finished writing this frame's
        // TrainRecords/EvalRecords.
        void RunNrcTrainingAndComposite(bool trainThisFrame)
        {
            var accelerator = gpu.Accelerator;
            var stream = accelerator.DefaultStream;

            // Off-frames (NrcTrainEveryNFrames' cadence) skip the whole training side
            // and composite from the last trained frame's InferenceWeights conversion -
            // the weights haven't changed since, so reconverting would be pure waste.
            if (trainThisFrame)
                TrainNrc(accelerator, stream);

            RunNrcComposite(accelerator);
        }

        // The training side of the per-frame NRC pass, split out of
        // RunNrcTrainingAndComposite so the NrcTrainEveryNFrames cadence can skip it
        // wholesale : master->TrainingOptimal
        // conversion, self-training-target resolution, then NrcTrainBatchesPerFrame x
        // {gradient, optimize} over the disjoint batch slices raygen filled this frame.
        void TrainNrc(CudaAccelerator accelerator, AcceleratorStream stream)
        {
            // Only the slices raygen actually distributed records across this frame
            // (LaunchParams.NrcTrainBatchCount, already clamped by render()) - both the
            // self-training launch size and the gradient loop below scale down with it.
            int batchCount = launchParams.NrcTrainBatchCount;

            nrcWeights.ConvertMasterToTrainingAndInference(stream, NrcUseEmaInference);

            var selfTrainingParams = new Device.SelfTrainingTargetLaunchParams
            {
                TrainRecords = nrcTrainBuffers.TrainRecordsView,
                TrainCounter = nrcTrainBuffers.CounterView,
                InferenceWeightsBase = nrcWeights.InferenceWeights.GetDeviceAddress(),
                HiddenLayerStrideInBytes = nrcWeights.HiddenLayerStrideInBytes,
                OutputLayerOffsetInBytes = nrcWeights.OutputLayerOffsetInBytes,
                ZeroBias64Base = nrcScratch.ZeroBias64.GetDeviceAddress(),
                ZeroBias3Base = nrcScratch.ZeroBias3.GetDeviceAddress(),
                ZeroVec64Base = nrcScratch.ZeroBias64.GetDeviceAddress(),
                ActScratchBase = nrcScratch.ActScratch.GetDeviceAddress(),
                ActScratch = nrcScratch.ActScratchView,
                OutputScratchBase = nrcScratch.OutputScratch.GetDeviceAddress(),
                OutputScratch = nrcScratch.OutputScratchView,
                HiddenLayerCount = nrcHiddenLayerCount,
                LayerWidth = nrcLayerWidth,
            };
            nrcPipelines.RunSelfTrainingTarget(accelerator, selfTrainingParams, (uint)(batchCount * Core.NrcConstants.TrainBatchSize));

            for (int i = 0; i < batchCount; i++)
            {
                // A single full-TrainBatchSize gradient accumulation, not chunked into
                // smaller sub-batches flushed to FP32 between each - chunking only
                // reduces random FP16 rounding noise, not the weight-magnitude growth
                // itself, while adding extra launch/convert/add cycles that measurably
                // slow training. NrcConstants.AdamWeightDecay/AdamKernel.AdamStep's own
                // note: decoupled weight decay is what actually controls weight growth,
                // since it's a structural bias, not accumulation noise.
                nrcWeights.ZeroDw();

                var gradientParams = new Device.GradientLaunchParams
                {
                    TrainRecords = nrcTrainBuffers.TrainRecordsView,
                    TrainCounter = nrcTrainBuffers.CounterView,
                    BatchIndex = i,
                    TrainingWeightsBase = nrcWeights.TrainingWeights.GetDeviceAddress(),
                    DwBase = nrcWeights.DwConverted.GetDeviceAddress(),
                    HiddenLayerStrideInBytes = nrcWeights.HiddenLayerStrideInBytes,
                    OutputLayerOffsetInBytes = nrcWeights.OutputLayerOffsetInBytes,
                    ZeroBias64Base = nrcScratch.ZeroBias64.GetDeviceAddress(),
                    ZeroBias3Base = nrcScratch.ZeroBias3.GetDeviceAddress(),
                    ZeroVec64Base = nrcScratch.ZeroBias64.GetDeviceAddress(),
                    ActScratchBase = nrcScratch.ActScratch.GetDeviceAddress(),
                    ActScratch = nrcScratch.ActScratchView,
                    OutputScratchBase = nrcScratch.OutputScratch.GetDeviceAddress(),
                    OutputScratch = nrcScratch.OutputScratchView,
                    DaOutScratchBase = nrcScratch.DaOutScratch.GetDeviceAddress(),
                    DaOutScratch = nrcScratch.DaOutScratchView,
                    DaScratchBase = nrcScratch.DaScratch.GetDeviceAddress(),
                    DInScratchBase = nrcScratch.DInScratch.GetDeviceAddress(),
                    MaskScratchBase = nrcScratch.MaskScratch.GetDeviceAddress(),
                    HiddenLayerCount = nrcHiddenLayerCount,
                    LayerWidth = nrcLayerWidth,
                };
                nrcPipelines.RunGradient(accelerator, gradientParams, (uint)Core.NrcConstants.TrainBatchSize);

                nrcWeights.ConvertDwToRowMajor(stream);

                nrcTrainStepCount++;

                // Bias-correction/EMA coefficients depend only on nrcTrainStepCount/
                // NrcEmaAlpha (uniform across the whole weight vector this step), not
                // per-weight - computed once here on the host instead of every one of
                // TotalWeightCount threads independently calling XMath.Pow 2-4 times
                // for the identical result (see AdamKernel.AdamStep's own note).
                float adamBiasCorrection1 = 1f - MathF.Pow(Core.NrcConstants.AdamBeta1, nrcTrainStepCount);
                float adamBiasCorrection2 = 1f - MathF.Pow(Core.NrcConstants.AdamBeta2, nrcTrainStepCount);
                float emaAlphaT = MathF.Pow(NrcEmaAlpha, nrcTrainStepCount);
                float emaAlphaT1 = MathF.Pow(NrcEmaAlpha, nrcTrainStepCount - 1);
                float emaEtaT = 1f - emaAlphaT;
                float emaEtaT1 = 1f - emaAlphaT1;
                float emaNewCoeff = (1f - NrcEmaAlpha) / emaEtaT;
                float emaOldCoeff = NrcEmaAlpha * emaEtaT1;

                nrcAdamKernel((Index1D)nrcWeights.TotalWeightCount,
                    nrcWeights.DwRowMajor.View, nrcWeights.AdamM.View, nrcWeights.AdamV.View,
                    nrcWeights.MasterWeights.View, nrcWeights.EmaWeights.View,
                    nrcTrainBuffers.Counter.View, i,
                    NrcLearningRate, NrcWeightDecay,
                    adamBiasCorrection1, adamBiasCorrection2, emaNewCoeff, emaOldCoeff);
            }

            // Refresh InferenceWeights once more so the composite pass queries the
            // weights this frame's training just produced, not the pre-training
            // snapshot from the top of this method. Inference-only: TrainingWeights
            // isn't read again until the next trained frame's own full conversion
            // (WeightBuffers.ConvertMasterToInference's own doc comment).
            nrcWeights.ConvertMasterToInference(stream, NrcUseEmaInference);
        }

        // Forward-only composite (+preview) over every active EvalRecord - runs every
        // frame regardless of the training cadence, against whatever InferenceWeights
        // conversion the most recent trained frame produced.
        void RunNrcComposite(CudaAccelerator accelerator)
        {
            // The preview buffer (16 bytes/pixel) only exists while the preview is
            // actually displayed - allocated here on first use (and after any resize,
            // which disposes it), freed again when the preview is toggled off. With no
            // buffer, CompositeProgram gets an invalid view and skips its per-pixel
            // preview store, saving the VRAM and the write bandwidth both.
            if (NrcEnabled && ShowNrcPreview)
            {
                if (nrcPreviewBuffer == null)
                    nrcPreviewBuffer = accelerator.Allocate1D<Vec4>(width * height);
            }
            else if (nrcPreviewBuffer != null)
            {
                // Freeing device memory implies a sync, but this runs once per toggle,
                // not per frame - and last frame's composite (the only toucher of this
                // buffer) drained at that frame's TonemapToDisplay sync; nothing queued
                // since reads it.
                nrcPreviewBuffer.Dispose();
                nrcPreviewBuffer = null;
            }

            var compositeParams = new Device.CompositeLaunchParams
            {
                EvalRecords = nrcTrainBuffers.EvalRecordsView,
                NrcPreviewBuffer = OptixDeviceView<Vec4>.From(nrcPreviewBuffer),
                RawColorBuffer = OptixDeviceView<Vec4>.From(frameOutput.RawColorBuffer),
                PixelCount = width * height,
                NumPixelSamples = NumPixelSamples,
                InferenceWeightsBase = nrcWeights.InferenceWeights.GetDeviceAddress(),
                HiddenLayerStrideInBytes = nrcWeights.HiddenLayerStrideInBytes,
                OutputLayerOffsetInBytes = nrcWeights.OutputLayerOffsetInBytes,
                ZeroBias64Base = nrcScratch.ZeroBias64.GetDeviceAddress(),
                ZeroBias3Base = nrcScratch.ZeroBias3.GetDeviceAddress(),
                ZeroVec64Base = nrcScratch.ZeroBias64.GetDeviceAddress(),
                ActScratchBase = nrcScratch.ActScratch.GetDeviceAddress(),
                ActScratch = nrcScratch.ActScratchView,
                OutputScratchBase = nrcScratch.OutputScratch.GetDeviceAddress(),
                OutputScratch = nrcScratch.OutputScratchView,
                HiddenLayerCount = nrcHiddenLayerCount,
                LayerWidth = nrcLayerWidth,
            };
            nrcPipelines.RunComposite(accelerator, compositeParams, (uint)(width * height));
        }

        // GPU-internal PBO -> texture blit (no CPU copy) - call once per frame after
        // render(), before drawing the fullscreen quad.
        public void BlitToTexture() => frameOutput.BlitToTexture();
    }
}
