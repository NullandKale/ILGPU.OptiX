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

using ILGPU.OptiX.Device;

namespace Sample14
{
    // ILGPU compiles device kernels against one concrete unmanaged struct layout, so this
    // must stay a single superset shape across every scene the runtime scene-switcher can
    // select. Unused buffers on a given scene are left invalid/zero-length
    // (OptixDeviceView<T>.IsValid == false) rather than changing the struct itself.
    //
    // Covers triangle geometry, multi-point-light shading, unified GGX
    // metallic-roughness materials plus perfect-specular dielectric materials, mesh
    // scenes, and AOV/denoiser buffers. All-triangle-mesh geometry path (single GAS,
    // no IAS) - no custom primitives or voxel volume grid.
    public struct LaunchParams
    {
        public int NumPixelSamples;
        public int FrameID;

        // Unified bounce budget - every material kind shares the same raygen bounce
        // loop; Russian roulette (RaygenProgram.cs) is what actually terminates most
        // paths early in practice, this is just the hard ceiling.
        public int MaxBounces;

        // Ping-ponged raw HDR accumulation history (see FrameOutput.cs's
        // hdrColorBuffers/accumCountBuffers and SwapColorHistory) - PrevColorBuffer is
        // the PREVIOUS frame's already-finished color, read back here only for the
        // depth-based disocclusion check below (RaygenProgram.cs reprojects this
        // pixel's primary-hit world position against PrevCamera* and bilinear-samples
        // PrevColorBuffer at the resulting, generally fractional, previous-frame pixel
        // coordinates - see RaygenProgram.cs's SampleBilinearColor doc comment). The
        // actual color blend (and the neighborhood clamp that guards it, and the
        // matching PreviousAccumCountBuffer read) happens one step later, in
        // TaaResolveKernel.cs - OptiX raygen threads have no ordering/synchronization
        // guarantee across pixels in the same launch, so a neighborhood read isn't safe
        // to do here; see RaygenProgram.cs's own note at its reprojection block.
        // PrevColorBuffer.w carries the previous frame's own view-space depth (not
        // alpha - alpha is otherwise unused).
        public OptixDeviceView<Vec4> PrevColorBuffer;

        // This frame's raw (per-pixel-sample-averaged, not yet history-blended) output -
        // RawColorBuffer.xyz is the radiance, .w this frame's own view-space depth;
        // ReprojCoordBuffer is the previous-frame pixel coordinate this pixel reprojects
        // to, or TaaResolveKernel.NoHistorySentinel if untrustworthy (background,
        // disocclusion, off-screen, specular, or no previous frame yet). Consumed by
        // TaaResolveKernel.cs, which runs after this OptiX launch has finished writing
        // every pixel (see PrevColorBuffer's own doc comment above for why the blend
        // can't happen inline here).
        public OptixDeviceView<Vec4> RawColorBuffer;
        public OptixDeviceView<Vec2> ReprojCoordBuffer;

        // Disocclusion rejection: a reprojected pixel is untrusted if its previous
        // frame's stored depth (PrevColorBuffer.w) differs from this frame's own depth
        // at the same location by more than this fraction of the larger of the two
        // (e.g. 0.05 = 5%) - catches a foreground object moving to reveal background
        // (or vice versa), which reprojects to a screen location that used to show a
        // very different distance and would otherwise ghost/smear if blended in.
        public float DepthRejectionThreshold;

        // AOV guide buffers for the denoiser (see SampleRenderer.cs's OptixDenoiser
        // wiring) - overwritten fresh every frame from the primary ray's own hit, not
        // blended across frames like ColorBuffer (see devicePrograms.cs's raygen).
        public OptixDeviceView<Vec4> NormalBuffer;
        public OptixDeviceView<Vec4> AlbedoBuffer;

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

        public OptixDeviceView<Vec3> Vertices;
        public OptixDeviceView<Vec3> Normals;
        public OptixDeviceView<Vec2> TexCoords;
        public OptixDeviceView<Vec3> Tangents;
        public OptixDeviceView<Vec3i> Indices;

        public OptixDeviceView<PointLightGpu> PointLights;
        public Vec3 AmbientColor;
        public float AmbientIntensity;

        // Unified NEE light list - see
        // Scenes/LightList.cs. NeeLightAreaPdf is parallel to Indices/
        // TriangleMaterialIds (one entry per triangle, 0 = not a registered light) -
        // read directly by primitive index in __closest__radiance's MIS reweighting,
        // no light-list search needed on that hot path.
        public OptixDeviceView<LightGpu> NeeLights;
        public OptixDeviceView<float> NeeLightCdf;
        public int NumNeeLights;
        public OptixDeviceView<float> NeeLightAreaPdf;

        // HDRI environment map - scene-dependent
        // (see SceneData.EnvMapPath); EnvMapWidth == 0 means "no environment map for
        // this scene, fall back to the flat BackgroundTop/Bottom gradient" (checked by
        // __miss__radiance and NextEventEstimation before touching the other EnvMap*
        // views, which are invalid in that case). EnvMapLightPdf is this scene's own
        // NEE light-list picking probability for the environment map (computed by
        // Scenes/LightList.cs, 0 if EnvMapWidth == 0) - kept separate from NeeLights
        // since there's no per-triangle-style array to carry one scalar light's pdf in.
        public OptixDeviceView<Vec3> EnvMapPixels;
        public OptixDeviceView<float> EnvMapCdf;
        public OptixDeviceView<float> EnvMapPdfUv;
        public int EnvMapWidth;
        public int EnvMapHeight;
        public float EnvMapIntensity;
        public float EnvMapLightPdf;
        // Azimuthal rotation in radians - applied
        // uniformly by EnvironmentMapSampling's direction<->uv conversion.
        public float EnvMapRotation;

        public Vec3 BackgroundTop;
        public Vec3 BackgroundBottom;
    }
}
