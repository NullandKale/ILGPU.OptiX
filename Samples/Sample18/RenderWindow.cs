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

namespace Sample18
{
    /// <summary>
    /// Exercises OptiX typed payload dispatch
    /// (OptixPayloadType / RayTypeBuilder.Payload&lt;T&gt;(typeId) / the new
    /// OptixTrace.Typed0/Typed1 nested classes) instead of the untyped
    /// max-registers path every other sample uses.
    ///
    /// Two ray types share this pipeline, each with its own distinct typed payload:
    /// "radiance" (OptixPayloadTypeID.Id0, a 3-uint RGB payload) determines the base
    /// hit color, and "shadow" (OptixPayloadTypeID.Id1, a 1-uint occlusion flag)
    /// determines whether a second, differently-angled probe ray from the same pixel
    /// also hits the mesh - if so the pixel is darkened. Both typed payloads are
    /// fired from the same raygen program and their independent results are combined
    /// into the final color, proving both compile and trace correctly together on one
    /// pipeline (not just compile - matching Sample16/17's precedent of also
    /// exercising the feature during live rendering, not only at startup).
    /// </summary>
    public sealed class RenderWindow : GameWindow
    {
        private struct LaunchParams
        {
            public OptixDeviceView<uint> ColorBuffer;
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

            long pixel = ix + (long)iy * launchParams.Width;
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

        private ILGPU.Context context;
        private CudaAccelerator accelerator;
        private OptixRayTracer rayTracer;
        private RayTracingPipeline<LaunchParams> pipeline;

        private MemoryBuffer1D<uint, Stride1D.Dense> colorBuffer;
        private uint[] pixelHost;

        private BuiltAccelStructure accel;
        private LaunchParams launchParams;

        private int glTexture;
        private FullscreenQuad quad;

        private OptixLogCallback logCallback;

        private Vector3 sceneCenter;
        private float orbitDistance;
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

            Console.WriteLine("Sample18: typed payload dispatch (radiance=Id0, shadow=Id1)");
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

            colorBuffer = accelerator.Allocate1D<uint>(Width * Height);
            pixelHost = new uint[Width * Height];
            launchParams.ColorBuffer = OptixDeviceView<uint>.From(colorBuffer);
            launchParams.Traversable = unchecked((ulong)accel.TraversableHandle.ToInt64());

            glTexture = GL.GenTexture();
            GL.BindTexture(TextureTarget.Texture2D, glTexture);
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, Width, Height, 0,
                PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            GL.BindTexture(TextureTarget.Texture2D, 0);

            quad = new FullscreenQuad();

            // One render to sanity-check that both typed payloads actually produced
            // varying output, not just "it compiled" - if either payload were broken
            // (e.g. silently reading the wrong registers), every pixel would either be
            // pure background or never darken.
            pipeline.Launch(launchParams, Width, Height);
            accelerator.Synchronize();
            colorBuffer.CopyToCPU(pixelHost);
            int litPixels = 0, shadowedPixels = 0, backgroundPixels = 0;
            foreach (var px in pixelHost)
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

            Console.WriteLine("[Controls] Hold Left Mouse + drag to orbit, Esc to quit.");
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
            sceneCenter = center;
            orbitDistance = radius * 1.6f;

            const float verticalFovDegrees = 45f;
            cameraFocal = 1f / MathF.Tan(verticalFovDegrees * MathF.PI / 180f * 0.5f);

            var initialDir = Vector3.Normalize(new Vector3(0.9f, 0.5f, 1.4f));
            cameraYaw = MathF.Atan2(initialDir.X, initialDir.Z) * 180f / MathF.PI;
            cameraPitch = MathF.Asin(Math.Clamp(initialDir.Y, -1f, 1f)) * 180f / MathF.PI;

            launchParams.Width = Width;
            launchParams.Height = Height;
            ApplyCameraOrbit();
        }

        private void ApplyCameraOrbit()
        {
            float yawRad = cameraYaw * MathF.PI / 180f;
            float pitchRad = cameraPitch * MathF.PI / 180f;

            var dir = new Vector3(
                MathF.Cos(pitchRad) * MathF.Sin(yawRad),
                MathF.Sin(pitchRad),
                MathF.Cos(pitchRad) * MathF.Cos(yawRad));

            var origin = sceneCenter + dir * orbitDistance;
            var forward = Vector3.Normalize(sceneCenter - origin);
            var worldUp = new Vector3(0f, 1f, 0f);
            var right = Vector3.Normalize(Vector3.Cross(forward, worldUp));
            var up = Vector3.Cross(right, forward);

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

        protected override void OnRenderFrame(FrameEventArgs args)
        {
            base.OnRenderFrame(args);

            pipeline.Launch(launchParams, Width, Height);
            accelerator.Synchronize();
            colorBuffer.CopyToCPU(pixelHost);

            GL.BindTexture(TextureTarget.Texture2D, glTexture);
            GL.TexSubImage2D(TextureTarget.Texture2D, 0, 0, 0, Width, Height,
                PixelFormat.Bgra, PixelType.UnsignedByte, pixelHost);
            GL.BindTexture(TextureTarget.Texture2D, 0);

            GL.Clear(ClearBufferMask.ColorBufferBit);
            quad.Draw(glTexture);
            SwapBuffers();

            frameCount++;
            if (frameCount % 300 == 0)
                Console.WriteLine($"[Sample18] frame {frameCount}");
        }

        protected override void OnResize(ResizeEventArgs e)
        {
            base.OnResize(e);
            GL.Viewport(0, 0, e.Width, e.Height);
        }

        protected override void OnUnload()
        {
            quad?.Dispose();
            if (glTexture != 0)
                GL.DeleteTexture(glTexture);

            accel?.Dispose();
            colorBuffer?.Dispose();
            pipeline?.Dispose();
            rayTracer?.Dispose();
            accelerator?.Dispose();
            context?.Dispose();

            base.OnUnload();
        }
    }
}
