// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: LaunchParams.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace Sample15
{
    // ILGPU compiles device kernels against one concrete unmanaged struct layout, so this
    // must stay a single superset shape across every scene the runtime scene-switcher can
    // select (docs/SAMPLE13_PLAN.md, "LaunchParams superset design"). Unused buffers on a
    // given scene are left null/zero-length rather than changing the struct itself.
    //
    // Triangle geometry, multi-point-light shading, unified GGX metallic-roughness
    // materials (docs/SAMPLE15_PLAN.md Milestone M2) + perfect-specular dielectric
    // materials, mesh scenes, and AOV/denoiser buffers are all wired up. Custom
    // primitives (spheres/boxes/cylinders/disks/rects) and the voxel volume grid were
    // removed in favor of an all-triangle-mesh geometry path (single GAS, no IAS).
    public unsafe struct LaunchParams
    {
        public int NumPixelSamples;
        public int FrameID;

        // Unified bounce budget (docs/SAMPLE15_PLAN.md Milestone M3) - replaces the
        // old per-material-kind MaxMirrorBounces/MaxRefractionBounces/
        // MaxDiffuseBounces counters now that every material kind shares the same
        // raygen bounce loop; Russian roulette (RaygenProgram.cs) is what actually
        // terminates most paths early in practice, this is just the hard ceiling.
        public int MaxBounces;

        // Ping-ponged raw HDR accumulation (see FrameOutput.cs's hdrColorBuffers/
        // accumCountBuffers and SwapColorHistory) - ColorBuffer/AccumCountBuffer are
        // THIS frame's write targets; PrevColorBuffer/PrevAccumCountBuffer are the
        // PREVIOUS frame's already-finished values, read back in RaygenProgram.cs by
        // reprojecting this pixel's primary-hit world position against PrevCamera*
        // (the same reprojection already computed for FlowBuffer below) and nearest-
        // neighbor-sampling at the resulting (generally fractional) previous-frame
        // pixel coordinates (deliberately not bilinear - see RaygenProgram.cs's
        // SampleNearestColor doc comment). ColorBuffer.w carries this frame's own
        // view-space depth (not alpha - alpha is otherwise unused), read back next
        // frame for DepthRejectionThreshold's disocclusion check below. AccumCount is
        // a per-pixel history length, capped at MaxHistoryFrames (bounded/capped
        // incremental mean - "SMA-then-EMA", the standard production-TAA convergence
        // technique) - this replaces the old single global FrameID-keyed running mean,
        // which could only blend by pixel index and so had to hard-reset on any camera
        // move/scene animation instead of reprojecting history the way this per-pixel
        // scheme does.
        public Vec4* ColorBuffer;
        public float* AccumCountBuffer;
        public Vec4* PrevColorBuffer;
        public float* PrevAccumCountBuffer;
        // Per-pixel history cap - a real cap when SampleRenderer.Accumulate is true and
        // the scene isn't animated, or 1 (always a fresh single sample, no blending) when
        // Accumulate is off or the scene is animated (orbiting/pulsing lights change the
        // *lighting*, which this purely-geometric reprojection can't compensate for).
        public int MaxHistoryFrames;

        // Disocclusion rejection: a reprojected pixel is untrusted if its previous
        // frame's stored depth (ColorBuffer.w) differs from this frame's own depth at
        // the same location by more than this fraction of the larger of the two (e.g.
        // 0.05 = 5%) - catches a foreground object moving to reveal background (or vice
        // versa), which reprojects to a screen location that used to show a very
        // different distance and would otherwise ghost/smear if blended in.
        public float DepthRejectionThreshold;

        // Real-time (not frame-count) exponential decay on AccumCount, applied every
        // frame before the min(...,MaxHistoryFrames) cap - even a perfectly static,
        // perfectly reprojected pixel would otherwise sit at a fixed weight
        // (1/MaxHistoryFrames) forever once capped, never adapting further. Halving
        // the effective count every HistoryDecayHalfLifeSeconds keeps the image
        // continuously (if slowly) forgetting old contributions on a wall-clock
        // schedule independent of frame rate, at the cost of a slightly higher noise
        // floor than an undecayed cap would reach. DeltaTimeSeconds is this frame's own
        // elapsed wall-clock time (see SampleRenderer.render()).
        public float DeltaTimeSeconds;
        public float HistoryDecayHalfLifeSeconds;

        // AOV guide buffers for the denoiser (see SampleRenderer.cs's OptixDenoiser
        // wiring) - overwritten fresh every frame from the primary ray's own hit, not
        // blended across frames like ColorBuffer (see devicePrograms.cs's raygen).
        public Vec4* NormalBuffer;
        public Vec4* AlbedoBuffer;

        // Motion-vector guide buffers for the OptiX temporal denoiser (see
        // FrameOutput.cs's OPTIX_DENOISER_MODEL_KIND_TEMPORAL wiring). FlowBuffer is a
        // per-pixel screen-space displacement (in pixels) from this frame back to the
        // corresponding pixel in the previous frame, computed in RaygenProgram.cs by
        // reprojecting the primary hit's world position (carried back via Payloads
        // 21-23) against PrevCamera* - the same reprojection ColorBuffer's own blend
        // above reuses. FlowTrustworthinessBuffer is 1 where that reprojection is valid
        // (real primary hit, in front of the previous camera, on-screen, previous frame
        // exists) and 0 otherwise (background pixel, disocclusion, or no previous frame
        // yet - e.g. right after a scene switch/resize), telling the denoiser not to
        // trust history there.
        public Vec2* FlowBuffer;
        public float* FlowTrustworthinessBuffer;

        // The camera used for the *previous* rendered frame - kept distinct from
        // `camera` below (this frame's camera) specifically so a camera move doesn't
        // invalidate reprojection the way resetting FrameID invalidates the running-
        // mean accumulator; SampleRenderer.render() updates this once per frame,
        // independently of setCamera()'s FrameID reset. PrevFrameValid is 0 right after
        // a scene switch or resize (previous frame's content/resolution doesn't apply)
        // and 1 otherwise.
        public Vec3 PrevCameraOrigin;
        public Vec3 PrevCameraAxisX;
        public Vec3 PrevCameraAxisY;
        public Vec3 PrevCameraAxisZ;
        public float PrevCameraAspectRatio;
        public float PrevCameraPlaneDist;
        public int PrevFrameValid;

        public Camera camera;
        public ulong traversable;

        public Vec3* Vertices;
        public Vec3* Normals;
        public Vec2* TexCoords;
        public Vec3* Tangents;
        public Vec3i* Indices;

        public PointLightGpu* PointLights;
        public int NumPointLights;
        public Vec3 AmbientColor;
        public float AmbientIntensity;

        // Unified NEE light list (docs/SAMPLE15_PLAN.md Milestone M4) - see
        // Scenes/LightList.cs. NeeLightAreaPdf is parallel to Indices/
        // TriangleMaterialIds (one entry per triangle, 0 = not a registered light) -
        // read directly by primitive index in __closest__radiance's MIS reweighting,
        // no light-list search needed on that hot path.
        public LightGpu* NeeLights;
        public float* NeeLightCdf;
        public int NumNeeLights;
        public float* NeeLightAreaPdf;

        // HDRI environment map (docs/SAMPLE15_PLAN.md Milestone M5) - scene-dependent
        // (see SceneData.EnvMapPath); EnvMapWidth == 0 means "no environment map for
        // this scene, fall back to the flat BackgroundTop/Bottom gradient" (checked by
        // __miss__radiance and NextEventEstimation before touching the other EnvMap*
        // pointers, which are null in that case). EnvMapLightPdf is this scene's own
        // NEE light-list picking probability for the environment map (computed by
        // Scenes/LightList.cs, 0 if EnvMapWidth == 0) - kept separate from NeeLights
        // since there's no per-triangle-style array to carry one scalar light's pdf in.
        public Vec3* EnvMapPixels;
        public float* EnvMapCdf;
        public float* EnvMapPdfUv;
        public int EnvMapWidth;
        public int EnvMapHeight;
        public float EnvMapIntensity;
        public float EnvMapLightPdf;
        // Azimuthal rotation in radians (docs/SAMPLE15_PLAN.md Milestone M8) - applied
        // uniformly by EnvironmentMapSampling's direction<->uv conversion.
        public float EnvMapRotation;

        public Vec3 BackgroundTop;
        public Vec3 BackgroundBottom;
    }
}
