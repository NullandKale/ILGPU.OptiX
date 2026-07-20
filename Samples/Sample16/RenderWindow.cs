// ---------------------------------------------------------------------------------------
//                                    ILGPU Samples
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: RenderWindow.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

using ILGPU;
using ILGPU.Algorithms;
using ILGPU.OptiX;
using ILGPU.OptiX.AccelStructures;
using ILGPU.OptiX.Device;
using ILGPU.OptiX.DeviceApi;
using ILGPU.OptiX.Native;
using ILGPU.OptiX.Pipeline;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.GraphicsLibraryFramework;
using System;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Sample16
{
    /// <summary>
    /// Pipeline configuration API surface sample. Covers:
    ///  - typed payload dispatch (OptixPayloadType / RayTypeBuilder.Payload&lt;T&gt;(typeId) /
    ///    OptixTrace.Typed0/Typed1) instead of the untyped max-registers path - left
    ///    tile, live;
    ///  - direct/continuation callable programs and a user exception program - right
    ///    tile, live.
    ///
    /// Left tile: two ray types share one pipeline, each with its own typed payload -
    /// "radiance" (Id0, 3-uint RGB) determines the base hit color, and "shadow" (Id1,
    /// 1-uint occlusion flag) determines whether a second probe ray also hits the
    /// mesh, darkening the pixel.
    ///
    /// Right tile: a box rendered through a custom intersection program (2 attribute
    /// registers, read back via OptixGetAttribute for checkerboard shading),
    /// background gradients computed by two direct callable programs (left/right
    /// half of the tile), scanline stripes drawn by a continuation callable, and a
    /// user exception always thrown inside a screen circle whose exception program
    /// paints those pixels magenta - visible every frame, no key needed.
    ///
    /// [Controls] Hold Left Mouse + drag to orbit (orbits both tiles together). Esc
    /// to quit.
    /// </summary>
    public sealed class RenderWindow : GameWindow
    {
        private struct CallableLaunchParams
        {
            // ColorBuffer/XOffset/Stride: this tile's TileWidth x TileHeight launch
            // writes into its column of a single shared full-window CUDA-GL interop
            // buffer (see RenderWindow's interopBuffer field) - no per-tile buffer,
            // no host round-trip.
            public OptixDeviceView<uint> ColorBuffer;
            public int XOffset;
            public int Stride;
            public OptixDeviceView<uint> ExceptionBuffer;
            public ulong Traversable;
            public int CallableLeft;
            public int CallableRight;
            public int CallableStripes;
            public int ThrowInCircle;
            public float OriginX, OriginY, OriginZ;
            public float ForwardX, ForwardY, ForwardZ;
            public float RightX, RightY, RightZ;
            public float UpX, UpY, UpZ;
            public float Focal;
        }

        private struct CallableRadiancePayload
        {
            public uint Hit, R, G, B;
        }

        private static void CallableRaygen(CallableLaunchParams launchParams)
        {
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;
            var (width, height, _) = OptixGetLaunchDimensions.Value;

            float aspect = (float)width / height;
            float ndcX = (2f * ((ix + 0.5f) / width) - 1f) * aspect;
            float ndcY = 1f - 2f * ((iy + 0.5f) / height);

            float dx = launchParams.ForwardX * launchParams.Focal + launchParams.RightX * ndcX + launchParams.UpX * ndcY;
            float dy = launchParams.ForwardY * launchParams.Focal + launchParams.RightY * ndcX + launchParams.UpY * ndcY;
            float dz = launchParams.ForwardZ * launchParams.Focal + launchParams.RightZ * ndcX + launchParams.UpZ * ndcY;
            float invLen = 1f / XMath.Sqrt(dx * dx + dy * dy + dz * dz);
            dx *= invLen;
            dy *= invLen;
            dz *= invLen;

            uint hit = 0, r = 0, g = 0, b = 0;
            OptixTrace.Trace(
                launchParams.Traversable,
                (launchParams.OriginX, launchParams.OriginY, launchParams.OriginZ),
                (dx, dy, dz),
                0.05f,
                1e6f,
                0f,
                0xff,
                OptixRayFlags.DisableAnyHit,
                0,
                1,
                0,
                ref hit,
                ref r,
                ref g,
                ref b);

            long pixel = (ix + launchParams.XOffset) + (long)iy * launchParams.Stride;
            if (hit != 0)
            {
                // Box hit - shaded from the intersection attributes in the CH.
                launchParams.ColorBuffer[pixel] = 0xff000000 | (b << 0) | (g << 8) | (r << 16);
            }
            else
            {
                // Background - computed and written by a DIRECT CALLABLE program
                // (left/right half use different callables, so the seam down the
                // middle is the visual proof both run).
                OptixCallables.DirectCall(
                    ix < width / 2
                        ? (uint)launchParams.CallableLeft
                        : (uint)launchParams.CallableRight);
            }

            // Scanline stripes - drawn by a CONTINUATION CALLABLE over whatever is
            // already in the color buffer.
            if (iy % 48 < 2)
                OptixCallables.ContinuationCall((uint)launchParams.CallableStripes);

            // Exception disc - pixels inside the circle throw a user exception; the
            // EXCEPTION PROGRAM paints them magenta (this launch index aborts, so
            // the magenta comes from the exception program alone).
            if (launchParams.ThrowInCircle != 0)
            {
                float cx = ix - width * 0.5f;
                float cy = iy - height * 0.5f;
                if (cx * cx + cy * cy < (height * 0.18f) * (height * 0.18f))
                    OptixThrowException.Throw(42, 0xDEADBEEF);
            }
        }

        private static void CallableMiss(CallableLaunchParams launchParams) { }

        private static void CallableIntersection(CallableLaunchParams launchParams)
        {
            // Analytic ray/box intersection against the unit box [-1,1]^3 in object
            // space (exercises optixGetObjectRayOrigin/Direction), reporting the
            // face index as the hit kind and the in-face (u,v) as 2 attribute
            // registers (_optix_report_intersection_2).
            var o = OptixGetObjectRayOrigin.AsVector3;
            var d = OptixGetObjectRayDirection.AsVector3;

            float invDx = 1f / d.X;
            float invDy = 1f / d.Y;
            float invDz = 1f / d.Z;

            float tx1 = (-1f - o.X) * invDx, tx2 = (1f - o.X) * invDx;
            float ty1 = (-1f - o.Y) * invDy, ty2 = (1f - o.Y) * invDy;
            float tz1 = (-1f - o.Z) * invDz, tz2 = (1f - o.Z) * invDz;

            float tminX = XMath.Min(tx1, tx2), tmaxX = XMath.Max(tx1, tx2);
            float tminY = XMath.Min(ty1, ty2), tmaxY = XMath.Max(ty1, ty2);
            float tminZ = XMath.Min(tz1, tz2), tmaxZ = XMath.Max(tz1, tz2);

            float tEnter = XMath.Max(tminX, XMath.Max(tminY, tminZ));
            float tExit = XMath.Min(tmaxX, XMath.Min(tmaxY, tmaxZ));

            if (tExit < tEnter || tEnter < OptixGetRayTmin.Value)
                return;

            var p = o + d * tEnter;
            uint face;
            float u, v;
            if (tEnter == tminX)
            {
                face = d.X > 0f ? 0u : 1u;
                u = p.Y * 0.5f + 0.5f;
                v = p.Z * 0.5f + 0.5f;
            }
            else if (tEnter == tminY)
            {
                face = d.Y > 0f ? 2u : 3u;
                u = p.X * 0.5f + 0.5f;
                v = p.Z * 0.5f + 0.5f;
            }
            else
            {
                face = d.Z > 0f ? 4u : 5u;
                u = p.X * 0.5f + 0.5f;
                v = p.Y * 0.5f + 0.5f;
            }

            OptixReportIntersection.Report(
                tEnter,
                face,
                Interop.FloatAsInt(u),
                Interop.FloatAsInt(v));
        }

        private static void CallableClosestHit(CallableLaunchParams launchParams)
        {
            // Read the attributes back (optixGetAttribute_0/1) and shade a
            // checkerboard from them; the face (hit kind) picks the base color.
            float u = Interop.IntAsFloat(OptixGetAttribute.Attribute0);
            float v = Interop.IntAsFloat(OptixGetAttribute.Attribute1);
            uint face = OptixGetHitKind.Value;

            float r = 0.2f, g = 0.2f, b = 0.2f;
            if (face == 0) { r = 0.95f; g = 0.35f; b = 0.30f; }
            else if (face == 1) { r = 0.30f; g = 0.85f; b = 0.35f; }
            else if (face == 2) { r = 0.30f; g = 0.45f; b = 0.95f; }
            else if (face == 3) { r = 0.95f; g = 0.85f; b = 0.30f; }
            else if (face == 4) { r = 0.85f; g = 0.35f; b = 0.90f; }
            else { r = 0.30f; g = 0.85f; b = 0.90f; }

            int checker = ((int)(u * 4f) + (int)(v * 4f)) & 1;
            float shade = checker == 0 ? 1f : 0.55f;

            OptixPayload.Set(0, 1);
            OptixPayload.Set(1, (uint)(XMath.Clamp(r * shade, 0f, 1f) * 255f));
            OptixPayload.Set(2, (uint)(XMath.Clamp(g * shade, 0f, 1f) * 255f));
            OptixPayload.Set(3, (uint)(XMath.Clamp(b * shade, 0f, 1f) * 255f));
        }

        private static void CallableBackgroundLeft(CallableLaunchParams launchParams)
        {
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;
            var (width, height, _) = OptixGetLaunchDimensions.Value;

            uint b = (uint)(120f + 120f * iy / height);
            uint g = (uint)(40f + 60f * ix / width);
            launchParams.ColorBuffer[(ix + launchParams.XOffset) + (long)iy * launchParams.Stride] =
                0xff000000 | (b << 0) | (g << 8) | 20u << 16;
        }

        private static void CallableBackgroundRight(CallableLaunchParams launchParams)
        {
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;
            var (width, height, _) = OptixGetLaunchDimensions.Value;

            uint r = (uint)(140f + 100f * iy / height);
            uint g = (uint)(60f + 60f * (width - ix) / width);
            launchParams.ColorBuffer[(ix + launchParams.XOffset) + (long)iy * launchParams.Stride] =
                0xff000000 | 20u << 0 | (g << 8) | (r << 16);
        }

        private static void CallableStripes(CallableLaunchParams launchParams)
        {
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;

            long pixel = (ix + launchParams.XOffset) + (long)iy * launchParams.Stride;
            uint px = launchParams.ColorBuffer[pixel];
            uint b = (px & 0xffu) / 2;
            uint g = ((px >> 8) & 0xffu) / 2;
            uint r = ((px >> 16) & 0xffu) / 2;
            launchParams.ColorBuffer[pixel] = 0xff000000 | b | (g << 8) | (r << 16);
        }

        private static void CallableExceptionProgram(CallableLaunchParams launchParams)
        {
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;

            // Paint the aborted pixel magenta and record the code/detail once.
            launchParams.ColorBuffer[(ix + launchParams.XOffset) + (long)iy * launchParams.Stride] = 0xffff00ff;
            launchParams.ExceptionBuffer[0] = (uint)OptixGetExceptionInfo.Code;
            launchParams.ExceptionBuffer[1] = OptixGetExceptionInfo.Detail0;
        }

        private struct LaunchParams
        {
            public OptixDeviceView<uint> ColorBuffer;
            public int XOffset;
            public int Stride;
            public int Width;
            public int Height;
            public ulong Traversable;
            public float OriginX, OriginY, OriginZ;
            public float ForwardX, ForwardY, ForwardZ;
            public float RightX, RightY, RightZ;
            public float UpX, UpY, UpZ;
            public float Focal;
        }

        // Typed payload for the "radiance" ray type (OptixPayloadTypeID.Id0) - base hit
        // color (P0-P2) plus the hit distance (P3, bit-cast float via
        // Interop.FloatAsInt/IntAsFloat, -1 if the ray missed) - carried back to raygen
        // so it can reconstruct the hit point and fire a real shadow ray from there,
        // since this device API layer has no separate attribute/hit-point readback.
        private struct RadiancePayload
        {
            public uint P0, P1, P2, P3;
        }

        // Typed payload for the "shadow" ray type (OptixPayloadTypeID.Id1) - 1 if the
        // probe ray hit the mesh, 0 otherwise.
        private struct ShadowPayload
        {
            public uint P0;
        }

        private static void RaygenRenderFrame(LaunchParams launchParams)
        {
            var ix = OptixGetLaunchIndex.X;
            var iy = OptixGetLaunchIndex.Y;

            float aspect = (float)launchParams.Width / launchParams.Height;
            float ndcX = (2f * ((ix + 0.5f) / launchParams.Width) - 1f) * aspect;
            float ndcY = 1f - 2f * ((iy + 0.5f) / launchParams.Height);

            float dx = launchParams.ForwardX * launchParams.Focal + launchParams.RightX * ndcX + launchParams.UpX * ndcY;
            float dy = launchParams.ForwardY * launchParams.Focal + launchParams.RightY * ndcX + launchParams.UpY * ndcY;
            float dz = launchParams.ForwardZ * launchParams.Focal + launchParams.RightZ * ndcX + launchParams.UpZ * ndcY;
            float invLen = 1f / XMath.Sqrt(dx * dx + dy * dy + dz * dz);
            dx *= invLen;
            dy *= invLen;
            dz *= invLen;

            // "radiance" ray type - typed payload OptixPayloadTypeID.Id0.
            uint r0 = 0, g0 = 0, b0 = 0, t0 = 0;
            OptixTrace.Typed0.Trace(
                launchParams.Traversable,
                (launchParams.OriginX, launchParams.OriginY, launchParams.OriginZ),
                (dx, dy, dz),
                1e-3f,
                1e6f,
                0f,
                0xff,
                OptixRayFlags.DisableAnyHit,
                0,
                2,
                0,
                ref r0,
                ref g0,
                ref b0,
                ref t0);

            float hitDistance = Interop.IntAsFloat(t0);

            // "shadow" ray type - typed payload OptixPayloadTypeID.Id1 - a real shadow
            // ray fired from the radiance ray's actual hit point (reconstructed from
            // hitDistance, above) toward a fixed light direction, offset a small
            // epsilon along the surface-independent ray direction to avoid immediately
            // re-hitting the same triangle. Skipped entirely for background pixels
            // (hitDistance <= 0, i.e. the radiance ray missed) - nothing to shadow.
            uint occluded = 0;
            if (hitDistance > 0f)
            {
                float hitX = launchParams.OriginX + dx * hitDistance;
                float hitY = launchParams.OriginY + dy * hitDistance;
                float hitZ = launchParams.OriginZ + dz * hitDistance;

                const float lightDirX = 0.4f, lightDirY = 0.85f, lightDirZ = 0.35f;
                // length(0.4, 0.85, 0.35) = sqrt(0.16 + 0.7225 + 0.1225) = sqrt(1.005) ~= 1.0025
                const float lightInvLen = 1f / 1.0025f;

                float shadowOriginX = hitX + dx * 1e-3f;
                float shadowOriginY = hitY + dy * 1e-3f;
                float shadowOriginZ = hitZ + dz * 1e-3f;

                OptixTrace.Typed1.Trace(
                    launchParams.Traversable,
                    (shadowOriginX, shadowOriginY, shadowOriginZ),
                    (lightDirX * lightInvLen, lightDirY * lightInvLen, lightDirZ * lightInvLen),
                    1e-3f,
                    1e6f,
                    0f,
                    0xff,
                    OptixRayFlags.TerminateOnFirstHit,
                    1,
                    2,
                    1,
                    ref occluded);
            }

            float shadowFactor = occluded == 1u ? 0.35f : 1.0f;

            float r = Interop.IntAsFloat(r0) * shadowFactor;
            float g = Interop.IntAsFloat(g0) * shadowFactor;
            float b = Interop.IntAsFloat(b0) * shadowFactor;

            uint ru = (uint)(XMath.Clamp(r, 0f, 1f) * 255f);
            uint gu = (uint)(XMath.Clamp(g, 0f, 1f) * 255f);
            uint bu = (uint)(XMath.Clamp(b, 0f, 1f) * 255f);

            uint rgba = 0xff000000 | (bu << 0) | (gu << 8) | (ru << 16);

            long pixel = (ix + launchParams.XOffset) + (long)iy * launchParams.Stride;
            launchParams.ColorBuffer[pixel] = rgba;
        }

        private static void MissRadiance(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.08f);
            OptixPayloadInterop.SetFloat(1, 0.08f);
            OptixPayloadInterop.SetFloat(2, 0.12f);
            OptixPayloadInterop.SetFloat(3, -1f); // sentinel: no hit
        }

        private static void ClosestHitRadiance(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.85f);
            OptixPayloadInterop.SetFloat(1, 0.55f);
            OptixPayloadInterop.SetFloat(2, 0.25f);
            OptixPayloadInterop.SetFloat(3, OptixGetRayTmax.Value);
        }

        private static void AnyHitRadiance(LaunchParams launchParams) { }

        private static void MissShadow(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetUint(0, 0);
        }

        private static void ClosestHitShadow(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetUint(0, 1);
        }

        private static void AnyHitShadow(LaunchParams launchParams) { }

        private const int Width = 1024;
        private const int Height = 768;
        private const int TileWidth = Width / 2;
        private const int TileHeight = Height;

        private ILGPU.Context context;
        private CudaAccelerator accelerator;
        private OptixRayTracer rayTracer;
        private RayTracingPipeline<LaunchParams> pipeline;

        // Startup-only scratch buffer + host array for the one-time typed-payload
        // and callable/exception correctness checks (not the live display path).
        private MemoryBuffer1D<uint, Stride1D.Dense> checkBuffer;
        private uint[] checkPixels;

        // Zero-copy CUDA-GL interop: both tiles' pipelines write directly into this
        // single full-window buffer every frame (see LaunchParams.XOffset/Stride) -
        // no host round-trip anywhere in the live render loop.
        private CudaGlInteropDisplayBuffer interopBuffer;

        private BuiltAccelStructure accel;
        private LaunchParams launchParams;

        private RayTracingPipeline<CallableLaunchParams> callablePipeline;
        private CallableLaunchParams callableLaunchParams;
        private MemoryBuffer1D<uint, Stride1D.Dense> exceptionBuffer;
        private MemoryBuffer1D<OptixAabb, Stride1D.Dense> aabbBuffer;
        private BuiltAccelStructure callableGas;
        private BuiltAccelStructure callableIas;
        private (Vector3 Center, float Radius) meshBounds;

        private FullscreenQuad quad;

        private OptixLogCallback logCallback;

        // Only cameraYaw/cameraPitch are shared (one mouse drag orbits both tiles in
        // sync); each tile keeps its own fixed scene center/orbit distance.
        private Vector3 meshCenter;
        private float meshOrbitDistance;
        private static readonly Vector3 CallableCenter = Vector3.Zero;
        private const float CallableOrbitDistance = 5.5f;
        private float cameraYaw;
        private float cameraPitch;
        private float cameraFocal;
        private bool wasLeftMouseDown;
        private float lastMouseX;
        private float lastMouseY;
        private const float MouseSensitivity = 0.3f;

        private int frameCount;

        public RenderWindow(GameWindowSettings gameWindowSettings, NativeWindowSettings nativeWindowSettings)
            : base(gameWindowSettings, nativeWindowSettings)
        {
        }

        protected override void OnLoad()
        {
            base.OnLoad();
            GL.ClearColor(0.05f, 0.05f, 0.08f, 1.0f);

            Console.WriteLine("Sample16: typed payload dispatch (radiance=Id0, shadow=Id1)");
            Console.WriteLine("Initializing CUDA + OptiX (validation mode ALL)...");

            context = ILGPU.Context.Create(b => b.Cuda().EnableAlgorithms());
            accelerator = context.CreateCudaAccelerator(0);

            logCallback = (level, tag, message, _) =>
                Console.Error.WriteLine($"[OptiX][{level}][{Marshal.PtrToStringAnsi(tag)}] {Marshal.PtrToStringAnsi(message)}");
            rayTracer = OptixRayTracer.Create(accelerator, new OptixDeviceContextOptions
            {
                LogCallbackFunction = Marshal.GetFunctionPointerForDelegate(logCallback),
                LogCallbackLevel = 4,
                ValidationMode = OptixDeviceContextValidationMode.All,
            });

            Console.WriteLine("Compiling pipeline (two typed-payload ray types)...");
            pipeline = rayTracer.CreatePipeline<LaunchParams>(b => b
                .Raygen(RaygenRenderFrame)
                .RayType("radiance", r => r
                    .Payload<RadiancePayload>(OptixPayloadTypeID.Id0)
                    .Miss(MissRadiance)
                    .HitGroup<byte>(ClosestHitRadiance, AnyHitRadiance))
                .RayType("shadow", r => r
                    .Payload<ShadowPayload>(OptixPayloadTypeID.Id1)
                    .Miss(MissShadow)
                    .HitGroup<byte>(ClosestHitShadow, AnyHitShadow))
                .MaxTraceDepth(2));
            pipeline.SetHitRecords<byte>(new byte[] { 0 });
            Console.WriteLine("Typed payload dispatch: PASS (pipeline compiled and linked successfully).");

            Console.WriteLine("Loading mesh (cow.obj)...");
            var (vertices, indices) = Model.LoadPositionsOnly("models/meshes/cow.obj");
            Console.WriteLine($"Loaded {vertices.Length} vertices, {indices.Length} triangles.");

            using var vertexBuffer = accelerator.Allocate1D(vertices);
            using var indexBuffer = accelerator.Allocate1D(indices);

            Console.WriteLine("Building GAS...");
            accel = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .AddTriangleMesh(vertexBuffer, indexBuffer)
                .AllowCompaction()
                .Build();

            var (center, radius) = ComputeBounds(vertices);
            SetupCamera(center, radius);

            checkBuffer = accelerator.Allocate1D<uint>(TileWidth * TileHeight);
            checkPixels = new uint[TileWidth * TileHeight];
            launchParams.ColorBuffer = OptixDeviceView<uint>.From(checkBuffer);
            launchParams.XOffset = 0;
            launchParams.Stride = TileWidth;
            launchParams.Traversable = unchecked((ulong)accel.TraversableHandle.ToInt64());

            // One render to sanity-check that both typed payloads actually produced
            // varying output, not just "it compiled" - if either payload were broken
            // (e.g. silently reading the wrong registers), every pixel would either be
            // pure background or never darken.
            pipeline.Launch(launchParams, TileWidth, TileHeight);
            accelerator.Synchronize();
            checkBuffer.CopyToCPU(checkPixels);
            int litPixels = 0, shadowedPixels = 0, backgroundPixels = 0;
            foreach (var px in checkPixels)
            {
                byte pr = (byte)((px >> 16) & 0xff);
                byte pg = (byte)((px >> 8) & 0xff);
                byte pb = (byte)(px & 0xff);
                if (pr < 30 && pg < 30 && pb > pr) backgroundPixels++;
                else if (pr > 150) litPixels++;
                else shadowedPixels++;
            }
            Console.WriteLine($"Startup frame: {backgroundPixels} background, {litPixels} fully-lit hit, {shadowedPixels} shadowed-hit pixels.");
            Console.WriteLine(litPixels > 0 && shadowedPixels > 0
                ? "Both typed payloads produced distinguishable output: PASS"
                : "Expected both lit and shadowed hit pixels but didn't see both: FAIL (check log above for OptiX warnings)");

            interopBuffer = new CudaGlInteropDisplayBuffer(Width, Height, accelerator);
            launchParams.XOffset = 0;
            launchParams.Stride = Width;

            meshBounds = (center, radius);
            SetupCallableScene();
            ApplyCameraOrbit();

            quad = new FullscreenQuad();

            Console.WriteLine("[Controls] Hold Left Mouse + drag to orbit (orbits both tiles together). " +
                "Left tile = typed payloads, right tile = callables/exceptions. Esc to quit.");
        }

        private void SetupCallableScene()
        {
            Console.WriteLine("Compiling pipeline (custom IS + exception program + 3 callables)...");
            callablePipeline = rayTracer.CreatePipeline<CallableLaunchParams>(b => b
                .Raygen(CallableRaygen)
                .RayType("radiance", r => r
                    .Payload<CallableRadiancePayload>()
                    .Miss(CallableMiss)
                    .HitGroup<byte>(CallableClosestHit, null, CallableIntersection))
                .WithExceptionProgram(CallableExceptionProgram)
                .WithExceptionFlags(OptixExceptionFlags.User)
                .WithDirectCallable("backgroundLeft", CallableBackgroundLeft)
                .WithDirectCallable("backgroundRight", CallableBackgroundRight)
                .WithContinuationCallable("stripes", CallableStripes)
                .WithTraversableGraphFlags(OptixTraversableGraphFlags.AllowSingleLevelInstancing)
                .MaxTraceDepth(2));
            callablePipeline.SetHitRecords<byte>(new byte[] { 0 });
            Console.WriteLine("Pipeline with exception + callable program groups: PASS.");

            // One custom primitive: the unit box (its AABB; the IS does the exact test).
            aabbBuffer = accelerator.Allocate1D(new[]
            {
                new OptixAabb(new Vector3(-1f, -1f, -1f), new Vector3(1f, 1f, 1f)),
            });
            callableGas = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .AddCustomPrimitives(aabbBuffer)
                .Build();
            callableIas = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .BuildInstanceAccelFromHandles(
                    new[] { callableGas.TraversableHandle },
                    new uint[] { 0 });

            exceptionBuffer = accelerator.Allocate1D<uint>(2);
            exceptionBuffer.MemSetToZero();

            // Startup check only - uses the same check buffer as the typed-payload
            // check above, at this tile's column offset (0 for the check; the live
            // loop re-points ColorBuffer/XOffset/Stride at the shared interop buffer).
            callableLaunchParams.ColorBuffer = OptixDeviceView<uint>.From(checkBuffer);
            callableLaunchParams.XOffset = 0;
            callableLaunchParams.Stride = TileWidth;
            callableLaunchParams.ExceptionBuffer = OptixDeviceView<uint>.From(exceptionBuffer);
            callableLaunchParams.Traversable = unchecked((ulong)callableIas.TraversableHandle.ToInt64());
            callableLaunchParams.CallableLeft = callablePipeline.CallableIndex("backgroundLeft");
            callableLaunchParams.CallableRight = callablePipeline.CallableIndex("backgroundRight");
            callableLaunchParams.CallableStripes = callablePipeline.CallableIndex("stripes");
            // Always on - the exception program's magenta pixels must be visible
            // every frame with no key press, same as every other feature here.
            callableLaunchParams.ThrowInCircle = 1;

            VerifyCallableFirstFrame();
        }

        private void VerifyCallableFirstFrame()
        {
            var (origin, forward, right, up) = OrbitBasis(cameraYaw, cameraPitch, CallableCenter, CallableOrbitDistance);
            callableLaunchParams.OriginX = origin.X;
            callableLaunchParams.OriginY = origin.Y;
            callableLaunchParams.OriginZ = origin.Z;
            callableLaunchParams.ForwardX = forward.X;
            callableLaunchParams.ForwardY = forward.Y;
            callableLaunchParams.ForwardZ = forward.Z;
            callableLaunchParams.RightX = right.X;
            callableLaunchParams.RightY = right.Y;
            callableLaunchParams.RightZ = right.Z;
            callableLaunchParams.UpX = up.X;
            callableLaunchParams.UpY = up.Y;
            callableLaunchParams.UpZ = up.Z;
            callableLaunchParams.Focal = cameraFocal;

            callablePipeline.Launch(callableLaunchParams, TileWidth, TileHeight);
            accelerator.Synchronize();
            checkBuffer.CopyToCPU(checkPixels);

            int failures = 0;
            void Check(string name, bool condition, string detail)
            {
                Console.WriteLine($"  {(condition ? "PASS" : "FAIL")}  {name} ({detail})");
                if (!condition)
                    failures++;
            }

            uint center = checkPixels[(TileHeight / 2) * TileWidth + TileWidth / 2];
            Check("box hit via custom IS + attributes",
                (center & 0x00ffffff) != 0 && center != 0xffff00ff,
                $"center=0x{center:X8}");

            uint left = checkPixels[10 * TileWidth + 10];
            uint right2 = checkPixels[10 * TileWidth + (TileWidth - 10)];
            Check("direct callable A background (left, blue-ish)",
                (left & 0xff) > ((left >> 16) & 0xff),
                $"left=0x{left:X8}");
            Check("direct callable B background (right, red-ish)",
                ((right2 >> 16) & 0xff) > (right2 & 0xff),
                $"right={right2:X8}");

            uint striped = checkPixels[0 * TileWidth + 10];
            uint unstriped = checkPixels[24 * TileWidth + 10];
            Check("continuation callable stripes",
                (striped & 0xff) < (unstriped & 0xff),
                $"striped=0x{striped:X8} vs 0x{unstriped:X8}");

            var exceptionResults = exceptionBuffer.GetAsArray1D();
            Check("user exception code reached the exception program",
                exceptionResults[0] == 42, $"code={exceptionResults[0]}");
            Check("exception detail value",
                exceptionResults[1] == 0xDEADBEEF, $"0x{exceptionResults[1]:X8}");

            Console.WriteLine(failures == 0
                ? "Callable/exception startup checks: ALL PASSED"
                : $"Callable/exception startup checks: {failures} FAILED");

            // Restore the live-loop column offset (the check above ran with both
            // tiles at XOffset=0/Stride=TileWidth against the shared check buffer).
            callableLaunchParams.XOffset = TileWidth;
            callableLaunchParams.Stride = Width;
        }

        private static (Vector3 Center, float Radius) ComputeBounds(Vector3[] vertices)
        {
            var min = vertices[0];
            var max = vertices[0];
            foreach (var v in vertices)
            {
                min = Vector3.Min(min, v);
                max = Vector3.Max(max, v);
            }
            var center = (min + max) * 0.5f;
            var radius = (max - min).Length() * 0.5f;
            return (center, radius);
        }

        private void SetupCamera(Vector3 center, float radius)
        {
            meshCenter = center;
            meshOrbitDistance = radius * 1.6f;

            const float verticalFovDegrees = 45f;
            cameraFocal = 1f / MathF.Tan(verticalFovDegrees * MathF.PI / 180f * 0.5f);

            var initialDir = Vector3.Normalize(new Vector3(0.9f, 0.5f, 1.4f));
            cameraYaw = MathF.Atan2(initialDir.X, initialDir.Z) * 180f / MathF.PI;
            cameraPitch = MathF.Asin(Math.Clamp(initialDir.Y, -1f, 1f)) * 180f / MathF.PI;

            launchParams.Width = TileWidth;
            launchParams.Height = TileHeight;
            ApplyCameraOrbit();
        }

        private static (Vector3 Origin, Vector3 Forward, Vector3 Right, Vector3 Up) OrbitBasis(
            float yawDegrees, float pitchDegrees, Vector3 center, float distance)
        {
            float yawRad = yawDegrees * MathF.PI / 180f;
            float pitchRad = pitchDegrees * MathF.PI / 180f;

            var dir = new Vector3(
                MathF.Cos(pitchRad) * MathF.Sin(yawRad),
                MathF.Sin(pitchRad),
                MathF.Cos(pitchRad) * MathF.Cos(yawRad));

            var origin = center + dir * distance;
            var forward = Vector3.Normalize(center - origin);
            var worldUp = new Vector3(0f, 1f, 0f);
            var right = Vector3.Normalize(Vector3.Cross(forward, worldUp));
            var up = Vector3.Cross(right, forward);
            return (origin, forward, right, up);
        }

        // Recomputes both tiles' camera basis from the shared yaw/pitch - both tiles
        // always render, so both always need up-to-date camera params.
        private void ApplyCameraOrbit()
        {
            var (origin, forward, right, up) = OrbitBasis(cameraYaw, cameraPitch, meshCenter, meshOrbitDistance);
            launchParams.OriginX = origin.X;
            launchParams.OriginY = origin.Y;
            launchParams.OriginZ = origin.Z;
            launchParams.ForwardX = forward.X;
            launchParams.ForwardY = forward.Y;
            launchParams.ForwardZ = forward.Z;
            launchParams.RightX = right.X;
            launchParams.RightY = right.Y;
            launchParams.RightZ = right.Z;
            launchParams.UpX = up.X;
            launchParams.UpY = up.Y;
            launchParams.UpZ = up.Z;
            launchParams.Focal = cameraFocal;

            var (cOrigin, cForward, cRight, cUp) = OrbitBasis(cameraYaw, cameraPitch, CallableCenter, CallableOrbitDistance);
            callableLaunchParams.OriginX = cOrigin.X;
            callableLaunchParams.OriginY = cOrigin.Y;
            callableLaunchParams.OriginZ = cOrigin.Z;
            callableLaunchParams.ForwardX = cForward.X;
            callableLaunchParams.ForwardY = cForward.Y;
            callableLaunchParams.ForwardZ = cForward.Z;
            callableLaunchParams.RightX = cRight.X;
            callableLaunchParams.RightY = cRight.Y;
            callableLaunchParams.RightZ = cRight.Z;
            callableLaunchParams.UpX = cUp.X;
            callableLaunchParams.UpY = cUp.Y;
            callableLaunchParams.UpZ = cUp.Z;
            callableLaunchParams.Focal = cameraFocal;
        }

        private void UpdateMouseOrbit()
        {
            bool leftDown = MouseState.IsButtonDown(MouseButton.Left);
            float x = MouseState.X;
            float y = MouseState.Y;

            if (leftDown && !wasLeftMouseDown)
            {
                lastMouseX = x;
                lastMouseY = y;
            }
            else if (leftDown)
            {
                float dx = (x - lastMouseX) * MouseSensitivity;
                float dy = (y - lastMouseY) * MouseSensitivity;
                lastMouseX = x;
                lastMouseY = y;

                cameraYaw += dx;
                cameraPitch = Math.Clamp(cameraPitch - dy, -89f, 89f);
                ApplyCameraOrbit();
            }

            wasLeftMouseDown = leftDown;
        }

        protected override void OnUpdateFrame(FrameEventArgs args)
        {
            base.OnUpdateFrame(args);
            if (KeyboardState.IsKeyDown(Keys.Escape))
            {
                Close();
                return;
            }
            UpdateMouseOrbit();
        }

        protected override unsafe void OnRenderFrame(FrameEventArgs args)
        {
            base.OnRenderFrame(args);

            var stream = (CudaStream)accelerator.DefaultStream;
            interopBuffer.MapCuda(stream);
            interopBuffer.GetCudaArrayView(); // sets interopBuffer.NativePtr to the mapped device pointer
            var fullWindowView = new OptixDeviceView<uint>((uint*)interopBuffer.NativePtr, Width * (long)Height);

            launchParams.ColorBuffer = fullWindowView;
            pipeline.Launch(launchParams, TileWidth, TileHeight);

            callableLaunchParams.ColorBuffer = fullWindowView;
            callablePipeline.Launch(callableLaunchParams, TileWidth, TileHeight);

            accelerator.Synchronize();
            interopBuffer.UnmapCuda(stream);
            interopBuffer.BlitToTexture();

            GL.Clear(ClearBufferMask.ColorBufferBit);
            quad.Draw(interopBuffer.GlTextureHandle);
            SwapBuffers();

            frameCount++;
            if (frameCount % 300 == 0)
                Console.WriteLine($"[Sample16] frame {frameCount}");
        }

        protected override void OnResize(ResizeEventArgs e)
        {
            base.OnResize(e);
            GL.Viewport(0, 0, e.Width, e.Height);
        }

        protected override void OnUnload()
        {
            quad?.Dispose();
            interopBuffer?.Dispose();

            callableIas?.Dispose();
            callableGas?.Dispose();
            aabbBuffer?.Dispose();
            exceptionBuffer?.Dispose();
            callablePipeline?.Dispose();

            accel?.Dispose();
            checkBuffer?.Dispose();
            pipeline?.Dispose();
            rayTracer?.Dispose();
            accelerator?.Dispose();
            context?.Dispose();

            base.OnUnload();
        }
    }
}
