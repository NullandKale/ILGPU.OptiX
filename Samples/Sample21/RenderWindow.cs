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
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Sample21
{
    /// <summary>
    /// Exercises opacity micromaps
    /// (optixOpacityMicromapArrayComputeMemoryUsage/optixOpacityMicromapArrayBuild,
    /// wired through OptixOpacityMicromapBuilder.BuildUniformStateArray +
    /// OptixAccelBuilder.AddTriangleMesh's opacityMicromapArray parameter) - lets
    /// triangles be marked fully transparent/opaque so the driver never needs to
    /// invoke an any-hit program for them.
    ///
    /// Scene: a flat ground plane (ordinary opaque triangles) plus ~40 small upright
    /// "cutout cards" - each a 2-triangle quad where one triangle is opaque (a green
    /// "leaf") and the other fully transparent via its opacity micromap, with no
    /// any-hit program declared for that hit group at all. Ground and cards share one
    /// GAS as two triangle build inputs (opacity micromaps don't change the build
    /// input's type, unlike curves in Sample20).
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

        private struct RadiancePayload
        {
            public uint P0, P1, P2;
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

            uint p0 = 0, p1 = 0, p2 = 0;
            OptixTrace.Trace(
                launchParams.Traversable,
                (launchParams.OriginX, launchParams.OriginY, launchParams.OriginZ),
                (dx, dy, dz),
                1e-3f,
                1e6f,
                0f,
                0xff,
                OptixRayFlags.None,
                0,
                1,
                0,
                ref p0,
                ref p1,
                ref p2);

            float r = Interop.IntAsFloat(p0);
            float g = Interop.IntAsFloat(p1);
            float b = Interop.IntAsFloat(p2);

            uint ru = (uint)(XMath.Clamp(r, 0f, 1f) * 255f);
            uint gu = (uint)(XMath.Clamp(g, 0f, 1f) * 255f);
            uint bu = (uint)(XMath.Clamp(b, 0f, 1f) * 255f);

            uint rgba = 0xff000000 | (bu << 0) | (gu << 8) | (ru << 16);

            long pixel = ix + (long)iy * launchParams.Width;
            launchParams.ColorBuffer[pixel] = rgba;
        }

        private static void MissRadiance(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.45f);
            OptixPayloadInterop.SetFloat(1, 0.55f);
            OptixPayloadInterop.SetFloat(2, 0.75f);
        }

        private static void ClosestHitGround(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.3f);
            OptixPayloadInterop.SetFloat(1, 0.25f);
            OptixPayloadInterop.SetFloat(2, 0.2f);
        }

        private static void ClosestHitCard(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.2f);
            OptixPayloadInterop.SetFloat(1, 0.6f);
            OptixPayloadInterop.SetFloat(2, 0.25f);
        }

        private const int Width = 1024;
        private const int Height = 768;
        private const int CardCount = 40;
        private const float GroundHalfExtent = 6f;
        private const float PatchHalfExtent = 5f;

        private ILGPU.Context context;
        private CudaAccelerator accelerator;
        private OptixRayTracer rayTracer;
        private RayTracingPipeline<LaunchParams> pipeline;

        private MemoryBuffer1D<uint, Stride1D.Dense> colorBuffer;
        private uint[] pixelHost;

        private BuiltOpacityMicromapArray opacityMicromapArray;
        private BuiltAccelStructure gas;
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

            Console.WriteLine("Sample21: opacity micromaps (cutout cards, no any-hit program)");
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

            Console.WriteLine("Compiling pipeline (ground + opacity-micromap cutout cards)...");
            pipeline = rayTracer.CreatePipeline<LaunchParams>(b => b
                .Raygen(RaygenRenderFrame)
                .RayType("radiance", r => r
                    .Payload<RadiancePayload>()
                    .Miss(MissRadiance)
                    .HitGroup<byte>("ground", ClosestHitGround)
                    .HitGroup<byte>("cards", ClosestHitCard))
                .AllowOpacityMicromaps()
                .MaxTraceDepth(2));
            pipeline.SetHitRecords<byte>(
                new[]
                {
                    new HitGroupEntry<byte>("ground", 0),
                    new HitGroupEntry<byte>("cards", 0),
                },
                pipeline.RayTypeCount);
            Console.WriteLine("Opacity micromap pipeline: PASS (compiled and linked successfully).");

            Console.WriteLine("Building ground plane + cutout cards...");
            var (groundVertices, groundIndices) = MakeGroundPlane();
            var (cardVertices, cardIndices, cardStates) = MakeCutoutCards();
            Console.WriteLine($"{groundVertices.Length} ground vertices, {cardVertices.Length} card vertices, {cardIndices.Length} card triangles.");

            using var groundVertexBuffer = accelerator.Allocate1D(groundVertices);
            using var groundIndexBuffer = accelerator.Allocate1D(groundIndices);
            using var cardVertexBuffer = accelerator.Allocate1D(cardVertices);
            using var cardIndexBuffer = accelerator.Allocate1D(cardIndices);

            Console.WriteLine("Building opacity micromap array...");
            opacityMicromapArray = OptixOpacityMicromapBuilder.BuildUniformStateArray(
                rayTracer.DeviceContext, accelerator, cardStates);
            Console.WriteLine("Opacity micromap array build: PASS.");

            Console.WriteLine("Building GAS (ground + cutout cards, two triangle build inputs)...");
            gas = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .AddTriangleMesh(groundVertexBuffer, groundIndexBuffer)
                .AddTriangleMesh(cardVertexBuffer, cardIndexBuffer, opacityMicromapArray: opacityMicromapArray)
                .AllowCompaction()
                .Build();
            Console.WriteLine("GAS build: PASS.");

            SetupCamera(Vector3.Zero, GroundHalfExtent);

            colorBuffer = accelerator.Allocate1D<uint>(Width * Height);
            pixelHost = new uint[Width * Height];
            launchParams.ColorBuffer = OptixDeviceView<uint>.From(colorBuffer);
            launchParams.Traversable = unchecked((ulong)gas.TraversableHandle.ToInt64());

            glTexture = GL.GenTexture();
            GL.BindTexture(TextureTarget.Texture2D, glTexture);
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, Width, Height, 0,
                PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            GL.BindTexture(TextureTarget.Texture2D, 0);

            quad = new FullscreenQuad();

            // One render + pixel scan: proves the cutout half of each card is
            // actually invisible (ground/sky shows through) while the other half is
            // opaque green, not just that the build calls returned success.
            pipeline.Launch(launchParams, Width, Height);
            accelerator.Synchronize();
            colorBuffer.CopyToCPU(pixelHost);
            int skyPixels = 0, groundPixels = 0, cardPixels = 0;
            foreach (var px in pixelHost)
            {
                byte pr = (byte)((px >> 16) & 0xff);
                byte pg = (byte)((px >> 8) & 0xff);
                byte pb = (byte)(px & 0xff);
                if (pb > pr && pb > 100) skyPixels++;
                else if (pg > pr && pg > pb) cardPixels++;
                else groundPixels++;
            }
            Console.WriteLine($"Startup frame: {skyPixels} sky, {groundPixels} ground, {cardPixels} card (opaque-half) pixels.");
            Console.WriteLine(cardPixels > 0 && groundPixels > 0
                ? "Opaque triangles rendered and cutout triangles let ground/sky through: PASS"
                : "Expected both card and ground/sky pixels but didn't see both: FAIL (check log above for OptiX warnings)");

            Console.WriteLine("[Controls] Hold Left Mouse + drag to orbit, Esc to quit.");
        }

        private static (Vector3[] Vertices, Tri[] Indices) MakeGroundPlane()
        {
            var vertices = new[]
            {
                new Vector3(-GroundHalfExtent, 0f, -GroundHalfExtent),
                new Vector3( GroundHalfExtent, 0f, -GroundHalfExtent),
                new Vector3( GroundHalfExtent, 0f,  GroundHalfExtent),
                new Vector3(-GroundHalfExtent, 0f,  GroundHalfExtent),
            };
            var indices = new[]
            {
                new Tri(0, 1, 2),
                new Tri(0, 2, 3),
            };
            return (vertices, indices);
        }

        // CardCount small upright quads scattered on the ground, each 2 triangles:
        // triangle 0 (bottom-left half) opaque, triangle 1 (top-right half)
        // transparent via its opacity micromap - visually a triangular "leaf" cutout.
        private static (Vector3[] Vertices, Tri[] Indices, OptixOpacityMicromapState[] States) MakeCutoutCards()
        {
            var rng = new Random(7);
            var vertices = new List<Vector3>(CardCount * 4);
            var indices = new List<Tri>(CardCount * 2);
            var states = new List<OptixOpacityMicromapState>(CardCount * 2);

            for (int c = 0; c < CardCount; c++)
            {
                float rx = (float)(rng.NextDouble() * 2 - 1) * PatchHalfExtent;
                float rz = (float)(rng.NextDouble() * 2 - 1) * PatchHalfExtent;
                float yaw = (float)(rng.NextDouble() * Math.PI * 2);
                float halfWidth = 0.2f + (float)rng.NextDouble() * 0.15f;
                float height = 0.35f + (float)rng.NextDouble() * 0.3f;

                float cosY = MathF.Cos(yaw);
                float sinY = MathF.Sin(yaw);
                Vector3 Rotate(float lx, float ly, float lz) =>
                    new Vector3(rx + lx * cosY - lz * sinY, ly, rz + lx * sinY + lz * cosY);

                uint baseIndex = (uint)vertices.Count;
                vertices.Add(Rotate(-halfWidth, 0f, 0f));
                vertices.Add(Rotate(halfWidth, 0f, 0f));
                vertices.Add(Rotate(halfWidth, height, 0f));
                vertices.Add(Rotate(-halfWidth, height, 0f));

                indices.Add(new Tri(baseIndex, baseIndex + 1, baseIndex + 2)); // opaque half
                indices.Add(new Tri(baseIndex, baseIndex + 2, baseIndex + 3)); // cutout half

                states.Add(OptixOpacityMicromapState.Opaque);
                states.Add(OptixOpacityMicromapState.Transparent);
            }

            return (vertices.ToArray(), indices.ToArray(), states.ToArray());
        }

        private void SetupCamera(Vector3 center, float radius)
        {
            sceneCenter = center + new Vector3(0f, radius * 0.15f, 0f);
            orbitDistance = radius * 2f;

            const float verticalFovDegrees = 45f;
            cameraFocal = 1f / MathF.Tan(verticalFovDegrees * MathF.PI / 180f * 0.5f);

            var initialDir = Vector3.Normalize(new Vector3(0.7f, 0.55f, 1.2f));
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
                Console.WriteLine($"[Sample21] frame {frameCount}");
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

            gas?.Dispose();
            opacityMicromapArray?.Dispose();
            colorBuffer?.Dispose();
            pipeline?.Dispose();
            rayTracer?.Dispose();
            accelerator?.Dispose();
            context?.Dispose();

            base.OnUnload();
        }
    }
}
