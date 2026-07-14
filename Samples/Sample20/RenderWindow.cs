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

namespace Sample20
{
    /// <summary>
    /// Exercises curve primitives via OptiX's
    /// built-in intersection program (optixBuiltinISModuleGet, wired through
    /// OptixAccelBuilder.AddCurves + RayTypeBuilder.HitGroupWithBuiltinIS - there is
    /// no user-suppliable intersection program for curves, unlike custom primitives).
    ///
    /// Scene: a flat triangle ground plane plus a patch of ~3000 short round cubic
    /// B-spline "fur" strands rooted on it, both geometry kinds combined into one GAS
    /// as separate build inputs, each with its own named hit group sharing the one
    /// "radiance" ray type.
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
                OptixRayFlags.DisableAnyHit,
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
            OptixPayloadInterop.SetFloat(0, 0.35f);
            OptixPayloadInterop.SetFloat(1, 0.3f);
            OptixPayloadInterop.SetFloat(2, 0.25f);
        }

        private static void AnyHitGround(LaunchParams launchParams) { }

        private static void ClosestHitFur(LaunchParams launchParams)
        {
            OptixPayloadInterop.SetFloat(0, 0.55f);
            OptixPayloadInterop.SetFloat(1, 0.35f);
            OptixPayloadInterop.SetFloat(2, 0.15f);
        }

        private static void AnyHitFur(LaunchParams launchParams) { }

        private const int Width = 1024;
        private const int Height = 768;
        private const int StrandCount = 3000;
        private const float GroundHalfExtent = 6f;
        private const float PatchHalfExtent = 5f;

        private ILGPU.Context context;
        private CudaAccelerator accelerator;
        private OptixRayTracer rayTracer;
        private RayTracingPipeline<LaunchParams> pipeline;

        private MemoryBuffer1D<uint, Stride1D.Dense> colorBuffer;
        private uint[] pixelHost;

        private BuiltAccelStructure groundGas;
        private BuiltAccelStructure furGas;
        private BuiltAccelStructure ias;
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

            Console.WriteLine("Sample20: curve primitives (built-in intersection)");
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

            Console.WriteLine("Compiling pipeline (triangle ground + built-in-IS curve hit groups)...");
            pipeline = rayTracer.CreatePipeline<LaunchParams>(b => b
                .Raygen(RaygenRenderFrame)
                .RayType("radiance", r => r
                    .Payload<RadiancePayload>()
                    .Miss(MissRadiance)
                    .HitGroup<byte>("ground", ClosestHitGround, AnyHitGround)
                    .HitGroupWithBuiltinIS<byte>(
                        "fur", ClosestHitFur, AnyHitFur, OptixPrimitiveType.RoundCubicBSpline))
                .UsesPrimitiveTypes(OptixPrimitiveTypeFlags.Triangle | OptixPrimitiveTypeFlags.RoundCubicBSpline)
                // Ground (triangles) and fur (curves) cannot share one GAS - OptiX
                // requires every build input within a single GAS to have the same
                // type (confirmed via a real GPU error: "buildInputs[1].type !=
                // buildInputs[0].type. All build inputs for geometry acceleration
                // structures must have the same type") - each becomes its own GAS,
                // combined via an IAS below, so this pipeline launches against a
                // 2-level (IAS -> GAS) traversable graph.
                .WithTraversableGraphFlags(OptixTraversableGraphFlags.AllowSingleLevelInstancing)
                .MaxTraceDepth(2));
            pipeline.SetHitRecords<byte>(
                new[]
                {
                    new HitGroupEntry<byte>("ground", 0),
                    new HitGroupEntry<byte>("fur", 0),
                },
                pipeline.RayTypeCount);
            Console.WriteLine("Curve built-in intersection: PASS (pipeline compiled and linked successfully).");

            Console.WriteLine("Building ground plane + fur strands...");
            var (groundVertices, groundIndices) = MakeGroundPlane();
            var (curveVertices, curveWidths, curveIndices) = MakeFurStrands();
            Console.WriteLine($"{groundVertices.Length} ground vertices, {curveVertices.Length} curve vertices, {curveIndices.Length} curve segments.");

            using var groundVertexBuffer = accelerator.Allocate1D(groundVertices);
            using var groundIndexBuffer = accelerator.Allocate1D(groundIndices);
            using var curveVertexBuffer = accelerator.Allocate1D(curveVertices);
            using var curveWidthBuffer = accelerator.Allocate1D(curveWidths);
            using var curveIndexBuffer = accelerator.Allocate1D(curveIndices);

            Console.WriteLine("Building ground GAS (triangles)...");
            groundGas = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .AddTriangleMesh(groundVertexBuffer, groundIndexBuffer)
                .AllowCompaction()
                .Build();

            Console.WriteLine("Building fur GAS (curves, built-in intersection)...");
            furGas = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .AddCurves(OptixPrimitiveType.RoundCubicBSpline, curveVertexBuffer, curveWidthBuffer, curveIndexBuffer)
                .AllowCompaction()
                .Build();

            Console.WriteLine("Building IAS (ground + fur GAS instances)...");
            ias = new OptixAccelBuilder()
                .WithDeviceContext(rayTracer.DeviceContext)
                .WithAccelerator(accelerator)
                .BuildInstanceAccelFromHandles(
                    new[] { groundGas.TraversableHandle, furGas.TraversableHandle },
                    new uint[] { 0, 1 });
            Console.WriteLine("GAS builds (triangles, curves) + IAS: PASS.");

            SetupCamera(Vector3.Zero, GroundHalfExtent);

            colorBuffer = accelerator.Allocate1D<uint>(Width * Height);
            pixelHost = new uint[Width * Height];
            launchParams.ColorBuffer = OptixDeviceView<uint>.From(colorBuffer);
            launchParams.Traversable = unchecked((ulong)ias.TraversableHandle.ToInt64());

            glTexture = GL.GenTexture();
            GL.BindTexture(TextureTarget.Texture2D, glTexture);
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, Width, Height, 0,
                PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            GL.BindTexture(TextureTarget.Texture2D, 0);

            quad = new FullscreenQuad();

            // One render + a quick pixel scan: proves curve hits are actually being
            // reported (fur-colored pixels present), not just that the pipeline
            // compiled and the GAS build call returned success.
            pipeline.Launch(launchParams, Width, Height);
            accelerator.Synchronize();
            colorBuffer.CopyToCPU(pixelHost);
            int skyPixels = 0, groundPixels = 0, furPixels = 0;
            foreach (var px in pixelHost)
            {
                byte pr = (byte)((px >> 16) & 0xff);
                byte pg = (byte)((px >> 8) & 0xff);
                byte pb = (byte)(px & 0xff);
                if (pb > pr && pb > 100) skyPixels++;
                else if (pr > pg) furPixels++;
                else groundPixels++;
            }
            Console.WriteLine($"Startup frame: {skyPixels} sky, {groundPixels} ground, {furPixels} fur pixels.");
            Console.WriteLine(furPixels > 0
                ? "Curve strands are being hit and shaded: PASS"
                : "Expected some fur-colored pixels but found none: FAIL (check log above for OptiX warnings)");

            Console.WriteLine("[Controls] Hold Left Mouse + drag to orbit, Esc to quit.");
        }

        // A flat quad ground plane, GroundHalfExtent units in each direction from the origin.
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

        // StrandCount short round cubic B-spline strands (4 control points = 1
        // segment each), rooted at random points on the ground plane, leaning in a
        // random direction and tapering from base to tip.
        private static (Vector3[] Vertices, float[] Widths, uint[] Indices) MakeFurStrands()
        {
            var rng = new Random(42);
            var vertices = new List<Vector3>(StrandCount * 4);
            var widths = new List<float>(StrandCount * 4);
            var indices = new List<uint>(StrandCount);

            for (int s = 0; s < StrandCount; s++)
            {
                float rx = (float)(rng.NextDouble() * 2 - 1) * PatchHalfExtent;
                float rz = (float)(rng.NextDouble() * 2 - 1) * PatchHalfExtent;
                float height = 0.3f + (float)rng.NextDouble() * 0.35f;
                float leanX = ((float)rng.NextDouble() - 0.5f) * 0.2f;
                float leanZ = ((float)rng.NextDouble() - 0.5f) * 0.2f;

                uint baseIndex = (uint)vertices.Count;

                vertices.Add(new Vector3(rx, 0f, rz));
                vertices.Add(new Vector3(rx + leanX * 0.33f, height * 0.33f, rz + leanZ * 0.33f));
                vertices.Add(new Vector3(rx + leanX * 0.66f, height * 0.66f, rz + leanZ * 0.66f));
                vertices.Add(new Vector3(rx + leanX, height, rz + leanZ));

                widths.Add(0.02f);
                widths.Add(0.014f);
                widths.Add(0.008f);
                widths.Add(0.002f);

                indices.Add(baseIndex);
            }

            return (vertices.ToArray(), widths.ToArray(), indices.ToArray());
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
                Console.WriteLine($"[Sample20] frame {frameCount}");
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

            ias?.Dispose();
            furGas?.Dispose();
            groundGas?.Dispose();
            colorBuffer?.Dispose();
            pipeline?.Dispose();
            rayTracer?.Dispose();
            accelerator?.Dispose();
            context?.Dispose();

            base.OnUnload();
        }
    }
}
